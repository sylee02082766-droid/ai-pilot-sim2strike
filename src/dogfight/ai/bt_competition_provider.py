from __future__ import annotations

import math

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.sim.state_schema import StateIndex


def _nose(state: np.ndarray) -> np.ndarray:
    pitch = math.radians(float(state[StateIndex.PITCH]))
    yaw = math.radians(float(state[StateIndex.YAW]))
    return np.asarray(
        [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            -math.sin(pitch),
        ],
        dtype=np.float64,
    )


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1.0e-6)
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _ned_to_body_matrix(state: np.ndarray) -> np.ndarray:
    roll = math.radians(float(state[StateIndex.ROLL]))
    pitch = math.radians(float(state[StateIndex.PITCH]))
    yaw = math.radians(float(state[StateIndex.YAW]))
    roll_matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(roll), math.sin(roll)],
            [0.0, -math.sin(roll), math.cos(roll)],
        ]
    )
    pitch_matrix = np.asarray(
        [
            [math.cos(pitch), 0.0, -math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [math.sin(pitch), 0.0, math.cos(pitch)],
        ]
    )
    yaw_matrix = np.asarray(
        [
            [math.cos(yaw), math.sin(yaw), 0.0],
            [-math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return roll_matrix @ pitch_matrix @ yaw_matrix


def _velocity_ned(state: np.ndarray) -> np.ndarray:
    return _ned_to_body_matrix(state).T @ np.asarray(state[6:9], dtype=np.float64)


def _body_los_deg(state: np.ndarray, relative: np.ndarray) -> tuple[float, float]:
    unit = relative / max(float(np.linalg.norm(relative)), 1.0e-6)
    body = _ned_to_body_matrix(state) @ unit
    azimuth = math.degrees(math.atan2(float(body[1]), float(body[0])))
    elevation = -math.degrees(math.asin(float(np.clip(body[2], -1.0, 1.0))))
    return azimuth, elevation


class BTCompetitionProvider(ActionProvider):
    """Keep the stable BT in charge and intervene only at narrow opportunities."""

    def __init__(
        self,
        primary: ActionProvider,
        debug: bool = False,
        reference_rudder_scale: float = 0.45,
        max_attack_closure_mps: float = 180.0,
        reattack_duration_s: float = 3.8,
        reattack_cooldown_s: float = 25.0,
        energy_dive_enter_speed_mps: float = 200.0,
        energy_dive_exit_speed_mps: float = 235.0,
        defense_duration_s: float = 0.0,
        defense_cooldown_s: float = 2.5,
    ) -> None:
        self.primary = primary
        self.debug = bool(debug)
        self.reference_rudder_scale = float(np.clip(reference_rudder_scale, 0.0, 1.0))
        self.max_attack_closure_mps = float(max_attack_closure_mps)
        self.reattack_duration_s = max(0.0, float(reattack_duration_s))
        self.reattack_cooldown_s = max(0.0, float(reattack_cooldown_s))
        self.energy_dive_enter_speed_mps = float(energy_dive_enter_speed_mps)
        self.energy_dive_exit_speed_mps = max(
            self.energy_dive_enter_speed_mps,
            float(energy_dive_exit_speed_mps),
        )
        self.defense_duration_s = max(0.0, float(defense_duration_s))
        self.defense_cooldown_s = max(0.0, float(defense_cooldown_s))
        self.reset()

    def reset(self, context: ActionContext | None = None) -> None:
        self._closure_mps: float | None = None
        self._previous_distance_m: float | None = None
        self._previous_time_s: float | None = None
        self._approach_active = False
        self._approach_peak_closure_mps = 0.0
        self._approach_min_distance_m = float("inf")
        self._approach_best_ata_deg = 180.0
        self._reattack_start_s = -1.0
        self._reattack_start_altitude_m = 0.0
        self._reattack_turn_sign = 1.0
        self._reattack_cooldown_until_s = -1.0
        self._energy_dive_start_s = -1.0
        self._energy_dive_cooldown_until_s = -1.0
        self._energy_turn_sign = -1.0
        self._last_energy_turn_sign = 1.0
        self._defense_start_s = -1.0
        self._defense_cooldown_until_s = -1.0
        self._defense_turn_sign = -1.0
        self._last_defense_turn_sign = 1.0
        self._defense_activation_count = 0
        self._start_time_s: float | None = None
        self._next_status_s = 0.0
        self._last_stage: str | None = None
        self.primary.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        primary_result = self.primary.compute_action(context)
        primary_stage = str(
            primary_result.info.get(
                "bt32_stage", primary_result.info.get("stage", primary_result.source)
            )
        )
        action = np.asarray(primary_result.action, dtype=np.float32).copy()

        own = np.asarray(context.ownship_state, dtype=np.float64)
        target = np.asarray(context.target_state, dtype=np.float64)
        relative = target[:3] - own[:3]
        distance_m = float(np.linalg.norm(relative))
        relative_unit = relative / max(distance_m, 1.0e-6)
        own_ata_deg = _angle(_nose(own), relative)
        target_ata_deg = _angle(_nose(target), -relative)
        sim_time_s = float(own[StateIndex.SIM_TIME])
        raw_closure_mps = 0.0
        if self._previous_distance_m is not None and self._previous_time_s is not None:
            dt_s = sim_time_s - self._previous_time_s
            if 1.0e-4 <= dt_s <= 0.5:
                raw_closure_mps = (self._previous_distance_m - distance_m) / dt_s
        self._previous_distance_m = distance_m
        self._previous_time_s = sim_time_s
        raw_closure_mps = float(np.clip(raw_closure_mps, -800.0, 800.0))
        if self._closure_mps is None:
            self._closure_mps = raw_closure_mps
        else:
            self._closure_mps = 0.50 * self._closure_mps + 0.50 * raw_closure_mps
        closure_mps = float(self._closure_mps)

        altitude_m = float(own[StateIndex.ALT])
        speed_mps = float(np.linalg.norm(_velocity_ned(own)))
        roll_deg = float(own[StateIndex.ROLL])
        pitch_deg = float(own[StateIndex.PITCH])
        los_az_deg, los_el_deg = _body_los_deg(own, relative)

        # The reconstructed BT's full rudder command costs too much energy.  A
        # 0.45 scale was the most stable V1.2 baseline and remains the default.
        action[2] *= self.reference_rudder_scale
        stage = "reference_bt"

        if primary_stage.startswith("deck_recovery"):
            self._reattack_start_s = -1.0
            self._energy_dive_start_s = -1.0
            self._defense_start_s = -1.0
            self._approach_active = False
            action = np.asarray(primary_result.action, dtype=np.float32).copy()
            stage = primary_stage
        else:
            self._update_pass_detector(
                sim_time_s=sim_time_s,
                distance_m=distance_m,
                closure_mps=closure_mps,
                own_ata_deg=own_ata_deg,
                los_az_deg=los_az_deg,
                roll_deg=roll_deg,
                altitude_m=altitude_m,
                speed_mps=speed_mps,
            )

            reattack_active = self._reattack_start_s >= 0.0
            if reattack_active:
                elapsed_s = sim_time_s - self._reattack_start_s
                reattack_finished = (
                    elapsed_s >= self.reattack_duration_s
                    or altitude_m <= 1200.0
                    or speed_mps <= 195.0
                    or altitude_m >= self._reattack_start_altitude_m + 450.0
                    or (elapsed_s >= 0.8 and own_ata_deg <= 45.0)
                )
                if reattack_finished:
                    self._reattack_start_s = -1.0
                    reattack_active = False

            energy_dive_active = self._energy_dive_start_s >= 0.0
            immediate_threat = (
                distance_m <= 2500.0
                and own_ata_deg >= 60.0
                and target_ata_deg <= 25.0
            )
            if energy_dive_active:
                energy_elapsed_s = sim_time_s - self._energy_dive_start_s
                energy_dive_finished = (
                    speed_mps >= self.energy_dive_exit_speed_mps
                    or altitude_m <= 1500.0
                    or energy_elapsed_s >= 10.0
                    or immediate_threat
                )
                if energy_dive_finished:
                    self._energy_dive_start_s = -1.0
                    self._energy_dive_cooldown_until_s = sim_time_s + 2.0
                    energy_dive_active = False

            defense_active = self._defense_start_s >= 0.0
            if defense_active:
                defense_elapsed_s = sim_time_s - self._defense_start_s
                defense_finished = (
                    defense_elapsed_s >= self.defense_duration_s
                    or altitude_m <= 850.0
                    or speed_mps <= 185.0
                    or (defense_elapsed_s >= 1.0 and target_ata_deg >= 55.0)
                )
                if defense_finished:
                    self._defense_start_s = -1.0
                    self._defense_cooldown_until_s = (
                        sim_time_s + self.defense_cooldown_s
                    )
                    defense_active = False

            defense_requested = (
                250.0 <= distance_m <= 3000.0
                and own_ata_deg >= 20.0
                and target_ata_deg <= 25.0
                and own_ata_deg >= target_ata_deg + 15.0
                and altitude_m >= 1000.0
                and speed_mps >= 205.0
                and self.defense_duration_s > 0.0
            )
            if (
                not defense_active
                and defense_requested
                and sim_time_s >= self._defense_cooldown_until_s
            ):
                established_turn = abs(roll_deg) >= 25.0
                if established_turn:
                    preferred_sign = 1.0 if roll_deg > 0.0 else -1.0
                elif abs(los_az_deg) >= 5.0:
                    preferred_sign = 1.0 if los_az_deg > 0.0 else -1.0
                else:
                    preferred_sign = -self._last_defense_turn_sign
                if (
                    not established_turn
                    and self._defense_activation_count > 0
                    and preferred_sign == self._last_defense_turn_sign
                ):
                    preferred_sign = -preferred_sign
                self._defense_turn_sign = preferred_sign
                self._last_defense_turn_sign = preferred_sign
                self._defense_activation_count += 1
                self._defense_start_s = sim_time_s
                self._energy_dive_start_s = -1.0
                self._reattack_start_s = -1.0
                energy_dive_active = False
                reattack_active = False
                defense_active = True

            energy_dive_requested = (
                (
                    speed_mps <= self.energy_dive_enter_speed_mps
                    and altitude_m >= 2200.0
                )
                or (altitude_m >= 6500.0 and pitch_deg >= 10.0)
            )
            if (
                not energy_dive_active
                and not defense_active
                and not immediate_threat
                and energy_dive_requested
                and sim_time_s >= self._energy_dive_cooldown_until_s
            ):
                if 15.0 <= abs(los_az_deg) <= 150.0:
                    self._energy_turn_sign = 1.0 if los_az_deg > 0.0 else -1.0
                else:
                    self._energy_turn_sign = -self._last_energy_turn_sign
                self._last_energy_turn_sign = self._energy_turn_sign
                self._energy_dive_start_s = sim_time_s
                self._reattack_start_s = -1.0
                reattack_active = False
                energy_dive_active = True

            official_line = (
                152.4 <= distance_m <= 914.4
                and own_ata_deg <= 1.5
                and target_ata_deg >= 90.0
                and closure_mps <= self.max_attack_closure_mps
            )
            near_attack = (
                152.4 <= distance_m <= 1200.0
                and own_ata_deg <= 8.0
                and target_ata_deg >= 90.0
                and closure_mps <= self.max_attack_closure_mps
            )

            if official_line:
                action = self._firing_line_micro(action, los_az_deg, los_el_deg)
                stage = "firing_line_micro"
                self._reattack_start_s = -1.0
            elif near_attack:
                action = self._near_attack_assist(action, los_az_deg, los_el_deg)
                stage = "near_attack_assist"
                self._reattack_start_s = -1.0
            elif defense_active:
                defense_elapsed_s = sim_time_s - self._defense_start_s
                action = self._defensive_break_action(
                    action,
                    roll_deg,
                    speed_mps,
                    defense_elapsed_s,
                )
                direction = "right" if self._defense_turn_sign > 0.0 else "left"
                bank_error_deg = _wrap_deg(
                    82.0 * self._defense_turn_sign - roll_deg
                )
                phase = "roll" if abs(bank_error_deg) > 35.0 else "pull"
                stage = f"defense_{phase}_{direction}"
            elif energy_dive_active:
                action = self._energy_dive_action(action, roll_deg, pitch_deg)
                direction = "right" if self._energy_turn_sign > 0.0 else "left"
                stage = f"energy_dive_{direction}"
            elif reattack_active:
                action = self._post_pass_reattack(action, roll_deg, speed_mps)
                stage = "post_pass_reattack"
            elif distance_m <= 1200.0 and closure_mps > self.max_attack_closure_mps:
                # A high-speed crossing is not a stable gun opportunity.  Let
                # the proven BT maneuver through it without an aiming override.
                stage = "fast_closure_guard"

        result = ActionResult(
            action=clip_action(action),
            source=stage,
            confidence=primary_result.confidence,
            info=dict(primary_result.info),
        )
        result.info.update(
            {
                "bt_competition_stage": stage,
                "distance_m": distance_m,
                "own_ata_deg": own_ata_deg,
                "target_ata_deg": target_ata_deg,
                "closure_mps": closure_mps,
            }
        )
        self._debug_status(
            sim_time_s,
            stage,
            distance_m,
            own_ata_deg,
            target_ata_deg,
            closure_mps,
            speed_mps,
            altitude_m,
            result.action,
        )
        return result

    def _update_pass_detector(
        self,
        *,
        sim_time_s: float,
        distance_m: float,
        closure_mps: float,
        own_ata_deg: float,
        los_az_deg: float,
        roll_deg: float,
        altitude_m: float,
        speed_mps: float,
    ) -> None:
        if closure_mps >= 25.0:
            if not self._approach_active:
                self._approach_active = True
                self._approach_peak_closure_mps = closure_mps
                self._approach_min_distance_m = distance_m
                self._approach_best_ata_deg = own_ata_deg
            else:
                self._approach_peak_closure_mps = max(
                    self._approach_peak_closure_mps, closure_mps
                )
                self._approach_min_distance_m = min(
                    self._approach_min_distance_m, distance_m
                )
                self._approach_best_ata_deg = min(
                    self._approach_best_ata_deg, own_ata_deg
                )
            return

        passed = self._approach_active and closure_mps <= -20.0
        if not passed:
            return

        eligible = (
            self._approach_peak_closure_mps >= 80.0
            and self._approach_min_distance_m <= 850.0
            and self._approach_best_ata_deg <= 45.0
            and sim_time_s >= self._reattack_cooldown_until_s
            and altitude_m >= 1400.0
            and speed_mps >= 220.0
            and self.reattack_duration_s > 0.0
        )
        if eligible:
            # Preserve the established turn through the pass.  Reversing on a
            # noisy rear-hemisphere LOS sign gave the reference opponent an
            # immediate tail shot in V1.2 testing.
            if abs(roll_deg) >= 15.0:
                self._reattack_turn_sign = 1.0 if roll_deg > 0.0 else -1.0
            elif abs(los_az_deg) >= 5.0:
                self._reattack_turn_sign = 1.0 if los_az_deg > 0.0 else -1.0
            self._reattack_start_s = sim_time_s
            self._reattack_start_altitude_m = altitude_m
            self._reattack_cooldown_until_s = sim_time_s + self.reattack_cooldown_s

        self._approach_active = False
        self._approach_peak_closure_mps = 0.0
        self._approach_min_distance_m = float("inf")
        self._approach_best_ata_deg = 180.0

    @staticmethod
    def _firing_line_micro(
        base_action: np.ndarray,
        los_az_deg: float,
        los_el_deg: float,
    ) -> np.ndarray:
        az_error = float(np.clip(los_az_deg / 2.0, -1.0, 1.0))
        el_error = float(np.clip(los_el_deg / 2.0, -1.0, 1.0))
        action = np.asarray(base_action, dtype=np.float32).copy()
        action[0] = np.clip(0.40 * action[0] + 0.06 * az_error, -0.25, 0.25)
        action[1] = np.clip(0.55 * action[1] - 0.10 * el_error, -0.35, 0.20)
        action[2] = np.clip(0.25 * action[2] - 0.18 * az_error, -0.25, 0.25)
        return clip_action(action)

    @staticmethod
    def _near_attack_assist(
        base_action: np.ndarray,
        los_az_deg: float,
        los_el_deg: float,
    ) -> np.ndarray:
        az_error = float(np.clip(los_az_deg / 8.0, -1.0, 1.0))
        el_error = float(np.clip(los_el_deg / 6.0, -1.0, 1.0))
        action = np.asarray(base_action, dtype=np.float32).copy()
        action[0] = 0.90 * action[0] + 0.04 * az_error
        action[1] = 0.90 * action[1] - 0.08 * el_error
        action[2] = 0.90 * action[2] - 0.12 * az_error
        return clip_action(action)

    def _post_pass_reattack(
        self,
        base_action: np.ndarray,
        roll_deg: float,
        speed_mps: float,
    ) -> np.ndarray:
        desired_bank_deg = 70.0 * self._reattack_turn_sign
        bank_error_deg = _wrap_deg(desired_bank_deg - roll_deg)
        bank_hold_cmd = float(np.clip(0.025 * bank_error_deg, -0.35, 0.35))
        action = np.asarray(base_action, dtype=np.float32).copy()
        action[0] = 0.80 * action[0] + 0.20 * bank_hold_cmd
        action[3] = 1.0
        return clip_action(action)

    def _energy_dive_action(
        self,
        base_action: np.ndarray,
        roll_deg: float,
        pitch_deg: float,
    ) -> np.ndarray:
        desired_bank_deg = 55.0 * self._energy_turn_sign
        bank_error_deg = _wrap_deg(desired_bank_deg - roll_deg)
        roll_cmd = float(np.clip(0.035 * bank_error_deg, -0.75, 0.75))
        if pitch_deg >= 20.0:
            pitch_cmd = 0.65
        elif pitch_deg >= 5.0:
            pitch_cmd = 0.45
        else:
            pitch_cmd = 0.20
        action = np.asarray(base_action, dtype=np.float32).copy()
        action[0] = roll_cmd
        action[1] = pitch_cmd
        action[2] = -0.10 * self._energy_turn_sign
        action[3] = 1.0
        return clip_action(action)

    def _defensive_break_action(
        self,
        base_action: np.ndarray,
        roll_deg: float,
        speed_mps: float,
        elapsed_s: float,
    ) -> np.ndarray:
        desired_bank_deg = 82.0 * self._defense_turn_sign
        bank_error_deg = _wrap_deg(desired_bank_deg - roll_deg)
        bank_hold_cmd = float(np.clip(0.050 * bank_error_deg, -1.0, 1.0))
        action = np.asarray(base_action, dtype=np.float32).copy()
        action[0] = 0.75 * action[0] + 0.25 * bank_hold_cmd
        if speed_mps >= 215.0:
            pitch_cmd = -1.0
        elif speed_mps >= 195.0:
            pitch_cmd = -0.65
        else:
            pitch_cmd = 0.20
        action[1] = pitch_cmd
        action[2] = np.clip(2.0 * action[2], -1.0, 1.0)
        action[3] = 1.0
        return clip_action(action)

    def _debug_status(
        self,
        sim_time_s: float,
        stage: str,
        distance_m: float,
        own_ata_deg: float,
        target_ata_deg: float,
        closure_mps: float,
        speed_mps: float,
        altitude_m: float,
        action: np.ndarray,
    ) -> None:
        if not self.debug:
            return
        if self._start_time_s is None or sim_time_s < self._start_time_s:
            self._start_time_s = sim_time_s
            self._next_status_s = 0.0
        elapsed_s = sim_time_s - self._start_time_s
        if stage != self._last_stage:
            print(
                f"[bt-competition] stage={stage} distance={distance_m:.0f}m "
                f"ata={own_ata_deg:.2f} target_ata={target_ata_deg:.1f} "
                f"closure={closure_mps:.1f}mps",
                flush=True,
            )
            self._last_stage = stage
        if elapsed_s + 1.0e-6 < self._next_status_s:
            return
        self._next_status_s = elapsed_s + 1.0
        print(
            f"[bt-competition-status] t={elapsed_s:.1f}s stage={stage} "
            f"distance={distance_m:.0f}m ata={own_ata_deg:.2f} "
            f"target_ata={target_ata_deg:.1f} closure={closure_mps:.1f}mps "
            f"speed={speed_mps:.1f}mps alt={altitude_m:.0f}m "
            f"action={np.asarray(action).round(3).tolist()}",
            flush=True,
        )

    def close(self) -> None:
        self.primary.close()


__all__ = ["BTCompetitionProvider"]
