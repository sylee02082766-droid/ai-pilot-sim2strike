from __future__ import annotations

import math

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.sim.state_schema import StateIndex


def _nose(state: np.ndarray) -> np.ndarray:
    pitch = math.radians(float(state[StateIndex.PITCH]))
    yaw = math.radians(float(state[StateIndex.YAW]))
    return np.asarray([
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        -math.sin(pitch),
    ])


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1.0e-6)
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _ned_to_body_matrix(state: np.ndarray) -> np.ndarray:
    phi = math.radians(float(state[StateIndex.ROLL]))
    theta = math.radians(float(state[StateIndex.PITCH]))
    psi = math.radians(float(state[StateIndex.YAW]))
    tx = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(phi), math.sin(phi)],
        [0.0, -math.sin(phi), math.cos(phi)],
    ])
    ty = np.asarray([
        [math.cos(theta), 0.0, -math.sin(theta)],
        [0.0, 1.0, 0.0],
        [math.sin(theta), 0.0, math.cos(theta)],
    ])
    tz = np.asarray([
        [math.cos(psi), math.sin(psi), 0.0],
        [-math.sin(psi), math.cos(psi), 0.0],
        [0.0, 0.0, 1.0],
    ])
    return tx @ ty @ tz


def _body_los_deg(state: np.ndarray, relative: np.ndarray) -> tuple[float, float]:
    unit = relative / max(float(np.linalg.norm(relative)), 1.0e-6)
    body = _ned_to_body_matrix(state) @ unit
    azimuth = math.degrees(math.atan2(float(body[1]), float(body[0])))
    elevation = -math.degrees(math.asin(float(np.clip(body[2], -1.0, 1.0))))
    return azimuth, elevation


def _velocity_ned(state: np.ndarray) -> np.ndarray:
    return _ned_to_body_matrix(state).T @ np.asarray(state[6:9], dtype=np.float64)


class BTTailPrecisionProvider(ActionProvider):
    """Use the BT for maneuvering and a small closed loop only for the gun solution."""

    def __init__(
        self,
        primary: ActionProvider,
        debug: bool = False,
        precision_enter_ata_deg: float = 45.0,
        merge_yoyo: bool = False,
        merge_yoyo_duration_s: float = 5.0,
        merge_brake_s: float = 0.0,
        opportunity_gun: bool = False,
        opportunity_gun_brake: bool = False,
        direct_los_assist: bool = False,
    ) -> None:
        self.primary = primary
        self.debug = bool(debug)
        self.precision_enter_ata_deg = float(precision_enter_ata_deg)
        self.merge_yoyo = bool(merge_yoyo)
        self.merge_yoyo_duration_s = max(1.0, float(merge_yoyo_duration_s))
        self.merge_brake_s = max(0.0, float(merge_brake_s))
        self.opportunity_gun = bool(opportunity_gun)
        self.opportunity_gun_brake = bool(opportunity_gun_brake)
        self.direct_los_assist = bool(direct_los_assist)
        self.reset()

    def reset(self, context: ActionContext | None = None) -> None:
        self._precision_latched = False
        self._gun_latched = False
        self._energy_recovering = False
        self._defense_until_s = -1.0
        self._defense_start_s = -1.0
        self._defense_turn_sign = 1.0
        self._defense_activation_count = 0
        self._last_stage: str | None = None
        self._merge_yoyo_used = False
        self._merge_yoyo_start_s: float | None = None
        self._merge_yoyo_sign = 1.0
        self._merge_yoyo_start_altitude_m = 0.0
        self._merge_brake_until_s = -1.0
        self._merge_brake_used = False
        self._los_micro_latched = False
        self._start_time_s: float | None = None
        self._next_status_s = 0.0
        self.primary.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        result = self.primary.compute_action(context)
        primary_stage = str(result.info.get("bt32_stage", "reference_bt"))
        if primary_stage.startswith("deck_recovery"):
            return result

        own = np.asarray(context.ownship_state, dtype=np.float64)
        target = np.asarray(context.target_state, dtype=np.float64)
        observation = np.asarray(context.observation, dtype=np.float32).reshape(-1)
        relative = target[:3] - own[:3]
        distance_m = float(np.linalg.norm(relative))
        own_ata_deg = _angle(_nose(own), relative)
        target_ata_deg = _angle(_nose(target), -relative)
        speed_mps = float(np.linalg.norm(own[6:9]))
        sim_time_s = float(own[StateIndex.SIM_TIME])
        altitude_m = float(own[StateIndex.ALT])

        tactical_capture = (
            300.0 <= distance_m <= 1500.0
            and own_ata_deg <= 20.0
            and target_ata_deg >= 120.0
        )
        defensive_danger = (
            180.0 <= distance_m <= 1600.0
            and own_ata_deg >= 70.0
            and target_ata_deg <= 25.0
        )
        merge_yoyo_trigger = (
            self.merge_yoyo
            and not self._merge_yoyo_used
            and 650.0 <= distance_m <= 1400.0
            and own_ata_deg <= 45.0
            and target_ata_deg <= 45.0
            and speed_mps >= 220.0
            and altitude_m >= 1500.0
        )
        merge_brake_trigger = (
            self.merge_brake_s > 0.0
            and not self._merge_brake_used
            and 650.0 <= distance_m <= 1450.0
            and own_ata_deg <= 50.0
            and target_ata_deg <= 50.0
            and speed_mps >= 220.0
        )
        if merge_brake_trigger:
            self._merge_brake_used = True
            self._merge_brake_until_s = sim_time_s + self.merge_brake_s
        merge_brake_active = sim_time_s < self._merge_brake_until_s
        if merge_yoyo_trigger:
            self._merge_yoyo_used = True
            self._merge_yoyo_start_s = sim_time_s
            current_roll_deg = float(own[StateIndex.ROLL])
            self._merge_yoyo_sign = -1.0 if current_roll_deg < 0.0 else 1.0
            self._merge_yoyo_start_altitude_m = altitude_m
        merge_yoyo_elapsed_s = (
            sim_time_s - self._merge_yoyo_start_s
            if self._merge_yoyo_start_s is not None
            else 0.0
        )
        merge_yoyo_active = (
            self._merge_yoyo_start_s is not None
            and merge_yoyo_elapsed_s < self.merge_yoyo_duration_s
            and altitude_m < self._merge_yoyo_start_altitude_m + 1800.0
            and speed_mps >= 165.0
        )
        if defensive_danger and sim_time_s >= self._defense_until_s:
            direct_az_deg = float(observation[11]) * 180.0
            if self._defense_activation_count % 2 == 0:
                self._defense_turn_sign = -1.0 if direct_az_deg < 0.0 else 1.0
            else:
                self._defense_turn_sign *= -1.0
            self._defense_activation_count += 1
            self._defense_start_s = sim_time_s
            self._defense_until_s = sim_time_s + 4.0
        defense_active = sim_time_s < self._defense_until_s

        if (
            not self._energy_recovering
            and not tactical_capture
            and (altitude_m >= 8500.0 or (speed_mps <= 180.0 and altitude_m >= 3000.0))
        ):
            self._energy_recovering = True
        elif self._energy_recovering and tactical_capture:
            self._energy_recovering = False

        enter_precision = (
            350.0 <= distance_m <= 2000.0
            and own_ata_deg <= self.precision_enter_ata_deg
            and (target_ata_deg >= 100.0 or self.opportunity_gun)
            and speed_mps >= 190.0
        )
        leave_precision = (
            distance_m < 300.0
            or distance_m > 2400.0
            or own_ata_deg >= 115.0
            or (target_ata_deg <= 70.0 and not self.opportunity_gun)
            or speed_mps <= 145.0
        )
        if merge_yoyo_active or merge_brake_active or defense_active:
            self._precision_latched = False
        elif self._energy_recovering:
            self._precision_latched = False
        elif enter_precision:
            self._precision_latched = True
        elif leave_precision:
            self._precision_latched = False
        if not self._precision_latched:
            self._gun_latched = False
        elif (
            distance_m <= 1400.0
            and own_ata_deg <= 20.0
            and (target_ata_deg >= 120.0 or self.opportunity_gun)
        ):
            self._gun_latched = True
        elif (
            distance_m >= 1700.0
            or own_ata_deg >= 35.0
            or (target_ata_deg <= 95.0 and not self.opportunity_gun)
        ):
            self._gun_latched = False

        action = np.asarray(result.action, dtype=np.float32)
        stage = "reference_bt"
        guidance_az_deg = float(observation[11]) * 180.0
        guidance_el_deg = float(observation[12]) * 90.0
        own_velocity_ned = _velocity_ned(own)
        target_velocity_ned = _velocity_ned(target)
        relative_unit = relative / max(distance_m, 1.0e-6)
        closure_mps = float(np.dot(
            own_velocity_ned - target_velocity_ned,
            relative_unit,
        ))
        if merge_yoyo_active:
            current_roll_deg = float(own[StateIndex.ROLL])
            current_pitch_deg = float(own[StateIndex.PITCH])
            desired_bank_deg = 35.0 * self._merge_yoyo_sign
            bank_error_deg = _wrap_deg(desired_bank_deg - current_roll_deg)
            roll_cmd = float(np.clip(0.055 * bank_error_deg, -1.0, 1.0))
            if merge_yoyo_elapsed_s < 1.2:
                pitch_cmd = -0.30
                stage = "merge_yoyo_roll_set"
            else:
                pitch_cmd = -1.0 if current_pitch_deg < 35.0 else -0.45
                stage = "merge_yoyo_pull"
            rudder_cmd = -0.25 * self._merge_yoyo_sign
            action = clip_action([roll_cmd, pitch_cmd, rudder_cmd, 1.0])
        elif defense_active:
            current_roll_deg = float(own[StateIndex.ROLL])
            desired_bank_deg = 75.0 * self._defense_turn_sign
            bank_error_deg = _wrap_deg(current_roll_deg - desired_bank_deg)
            roll_cmd = float(np.clip(-0.055 * bank_error_deg, -1.0, 1.0))
            pitch_cmd = -0.55 if speed_mps >= 220.0 else -0.18
            rudder_cmd = -0.35 * self._defense_turn_sign
            braking = sim_time_s - self._defense_start_s < 2.0
            throttle_cmd = -1.0 if braking else 1.0
            action = clip_action([roll_cmd, pitch_cmd, rudder_cmd, throttle_cmd])
            stage = "defensive_brake" if braking else "defensive_break"
        elif self._energy_recovering:
            current_roll_deg = float(own[StateIndex.ROLL])
            current_pitch_deg = float(own[StateIndex.PITCH])
            direct_az_deg = float(observation[11]) * 180.0
            az_error = float(np.clip(direct_az_deg / 20.0, -1.0, 1.0))
            roll_cmd = float(np.clip(
                -0.025 * current_roll_deg + 0.30 * az_error,
                -0.45,
                0.45,
            ))
            if altitude_m <= 1800.0:
                pitch_cmd = -1.0
                stage = "energy_deck_pullout"
            elif current_pitch_deg > -5.0:
                pitch_cmd = 1.0
                stage = "energy_nose_down"
            elif current_pitch_deg < -20.0:
                pitch_cmd = -0.80
                stage = "energy_pullout"
            else:
                pitch_cmd = 0.0
                stage = "energy_unloaded"
                if speed_mps >= 245.0 and altitude_m <= 5500.0:
                    self._energy_recovering = False
            rudder_cmd = float(np.clip(-0.65 * az_error, -0.65, 0.65))
            action = clip_action([roll_cmd, pitch_cmd, rudder_cmd, 1.0])
        elif self._precision_latched and observation.size >= 13:
            gun_track = self._gun_latched
            if gun_track:
                if distance_m > 1050.0:
                    lead_time_s = float(np.clip(
                        1.5 + (distance_m - 930.0) / 120.0,
                        1.5,
                        3.0,
                    ))
                    guidance_relative = relative + lead_time_s * target_velocity_ned
                    guidance_relative[2] += 250.0
                    stage = "lead_track"
                else:
                    guidance_relative = relative
                    stage = "gun_track"
            else:
                desired_closure_mps = float(np.clip(
                    25.0 + 0.12 * max(distance_m - 560.0, 0.0),
                    25.0,
                    70.0,
                ))
                desired_velocity = (
                    target_velocity_ned + desired_closure_mps * relative_unit
                )
                # A small downward bias trades altitude for speed before firing.
                desired_velocity[2] += 12.0
                desired_heading_deg = math.degrees(math.atan2(
                    float(desired_velocity[1]),
                    float(desired_velocity[0]),
                ))
                own_track_heading_deg = math.degrees(math.atan2(
                    float(own_velocity_ned[1]),
                    float(own_velocity_ned[0]),
                ))
                heading_error_deg = _wrap_deg(
                    desired_heading_deg - own_track_heading_deg
                )
                desired_bank_deg = float(np.clip(
                    2.0 * heading_error_deg,
                    -70.0,
                    70.0,
                ))
                bank_error_deg = _wrap_deg(
                    desired_bank_deg - float(own[StateIndex.ROLL])
                )
                horizontal_speed = max(
                    float(np.linalg.norm(desired_velocity[:2])),
                    1.0,
                )
                desired_pitch_deg = math.degrees(math.atan2(
                    -float(desired_velocity[2]),
                    horizontal_speed,
                ))
                bank_fraction = abs(desired_bank_deg) / 70.0
                roll_cmd = float(np.clip(0.04 * bank_error_deg, -1.0, 1.0))
                pitch_cmd = float(np.clip(
                    0.04 * (float(own[StateIndex.PITCH]) - desired_pitch_deg)
                    - 0.55 * bank_fraction,
                    -0.85,
                    0.65,
                ))
                direct_az_deg = float(observation[11]) * 180.0
                rudder_cmd = float(np.clip(-0.025 * direct_az_deg, -0.45, 0.45))
                action = clip_action([roll_cmd, pitch_cmd, rudder_cmd, 1.0])
                guidance_relative = desired_velocity
                stage = "tail_close"

            guidance_az_deg, guidance_el_deg = _body_los_deg(own, guidance_relative)
            if gun_track:
                gun_snap = (
                    stage == "gun_track"
                    and distance_m <= 930.0
                    and own_ata_deg <= 12.0
                )
                high_energy_snap = (
                    stage == "gun_track"
                    and distance_m <= 1000.0
                    and own_ata_deg >= 30.0
                    and target_ata_deg >= 120.0
                    and speed_mps >= 220.0
                )
                if high_energy_snap:
                    error_scale_deg = 8.0
                elif gun_snap:
                    error_scale_deg = 4.0
                else:
                    error_scale_deg = 12.0
                az_error = float(np.clip(guidance_az_deg / error_scale_deg, -1.0, 1.0))
                el_error = float(np.clip(guidance_el_deg / error_scale_deg, -1.0, 1.0))
                base_weight = 0.03
                if high_energy_snap:
                    roll_gain, pitch_gain, rudder_gain = 1.00, 0.65, 1.00
                elif gun_snap:
                    roll_gain, pitch_gain, rudder_gain = 0.50, 0.90, 1.00
                elif stage == "lead_track" and speed_mps >= 220.0:
                    roll_gain, pitch_gain, rudder_gain = 0.60, 0.45, 0.85
                else:
                    roll_gain, pitch_gain, rudder_gain = 0.38, 0.45, 0.80
                if high_energy_snap:
                    roll_limit = 1.00
                elif stage == "lead_track" and speed_mps >= 220.0:
                    roll_limit = 0.75
                else:
                    roll_limit = 0.35 if speed_mps < 205.0 else 0.55
                if high_energy_snap:
                    pull_limit = 0.75
                elif gun_snap:
                    pull_limit = 0.60
                else:
                    pull_limit = 0.20 if speed_mps < 205.0 else 0.48
                if high_energy_snap:
                    bank_pull = 0.75
                elif stage == "lead_track" and speed_mps >= 220.0:
                    bank_pull = 0.55
                else:
                    bank_pull = 0.20
                roll_cmd = float(np.clip(
                    base_weight * float(action[0]) + roll_gain * az_error,
                    -roll_limit,
                    roll_limit,
                ))
                pitch_cmd = float(np.clip(
                    base_weight * float(action[1])
                    - pitch_gain * el_error
                    - bank_pull * abs(az_error),
                    -pull_limit,
                    0.55,
                ))
                rudder_cmd = float(np.clip(
                    base_weight * float(action[2]) - rudder_gain * az_error,
                    -1.0,
                    1.0,
                ))
                throttle_cmd = 0.35 if distance_m < 350.0 else 1.0
                action = clip_action([
                    roll_cmd,
                    pitch_cmd,
                    rudder_cmd,
                    throttle_cmd,
                ])

        if merge_brake_active and not merge_yoyo_active:
            action = np.asarray(action, dtype=np.float32).copy()
            action[3] = -1.0
            stage = "merge_brake"
        elif (
            self.opportunity_gun_brake
            and self._precision_latched
            and distance_m <= 1500.0
            and own_ata_deg <= 20.0
        ):
            action = np.asarray(action, dtype=np.float32).copy()
            action[3] = -1.0
            stage = f"{stage}_brake"

        if (
            self.direct_los_assist
            and 152.4 <= distance_m <= 1500.0
            and own_ata_deg <= 25.0
            and observation.size >= 13
        ):
            los_az_deg = float(observation[11]) * 180.0
            los_el_deg = float(observation[12]) * 90.0
            if own_ata_deg <= 0.8:
                self._los_micro_latched = True
            elif own_ata_deg >= 3.0:
                self._los_micro_latched = False
            if self._los_micro_latched:
                az_error = float(np.clip(los_az_deg / 1.3, -1.0, 1.0))
                el_error = float(np.clip(los_el_deg / 1.1, -1.0, 1.0))
                action = np.asarray(action, dtype=np.float32).copy()
                action[2] += -1.5 * az_error
                action[0] += 0.15 * az_error
                action[1] += -0.15 * el_error
                stage = "direct_los_micro"
            else:
                az_error = float(np.clip(los_az_deg / 1.3, -1.0, 1.0))
                el_error = float(np.clip(los_el_deg / 1.1, -1.0, 1.0))
                action = np.asarray(action, dtype=np.float32).copy()
                action[2] += -10.2 * az_error
                action[0] += 0.75 * az_error
                action[1] += -0.52 * el_error
                stage = "direct_los_assist"
            action = clip_action(action)
        elif self._los_micro_latched:
            self._los_micro_latched = False

        result.action = clip_action(action)
        result.source = stage
        result.info = dict(result.info)
        result.info["bt_tail_precision_stage"] = stage

        if self.debug:
            if self._start_time_s is None or sim_time_s < self._start_time_s:
                self._start_time_s = sim_time_s
                self._next_status_s = 0.0
            elapsed_s = sim_time_s - self._start_time_s
            if stage != self._last_stage:
                print(
                    "[bt-tail-precision] "
                    f"stage={stage} distance={distance_m:.0f}m "
                    f"own_ata={own_ata_deg:.1f} target_ata={target_ata_deg:.1f}",
                    flush=True,
                )
                self._last_stage = stage
            if elapsed_s + 1.0e-6 >= self._next_status_s:
                self._next_status_s = elapsed_s + 2.0
                print(
                    "[bt-tail-status] "
                    f"t={elapsed_s:.1f}s stage={stage} distance={distance_m:.0f}m "
                    f"own_ata={own_ata_deg:.1f} target_ata={target_ata_deg:.1f} "
                    f"los_az={float(observation[11]) * 180.0:.1f} "
                    f"los_el={float(observation[12]) * 90.0:.1f} "
                    f"guide_az={guidance_az_deg:.1f} guide_el={guidance_el_deg:.1f} "
                    f"closure={closure_mps:.1f} "
                    f"speed={speed_mps:.1f}mps alt={float(own[StateIndex.ALT]):.0f}m "
                    f"own_hp={float(own[StateIndex.HEALTH]):.3f} "
                    f"target_hp={float(target[StateIndex.HEALTH]):.3f} "
                    f"roll={float(own[StateIndex.ROLL]):.1f}deg "
                    f"pitch={float(own[StateIndex.PITCH]):.1f}deg "
                    f"climb={-float(own_velocity_ned[2]):.1f}mps "
                    f"action={np.asarray(result.action).round(3).tolist()}",
                    flush=True,
                )
        return result

    def close(self) -> None:
        self.primary.close()


__all__ = ["BTTailPrecisionProvider"]
