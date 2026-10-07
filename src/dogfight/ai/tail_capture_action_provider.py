from __future__ import annotations

import math

import numpy as np

from dogfight.ai.action_provider import (
    ActionContext,
    ActionProvider,
    ActionResult,
    clip_action,
)
from dogfight.sim.state_schema import StateIndex


def _wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator < 1.0e-6:
        return 180.0
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _nose_vector_ned(state: np.ndarray) -> np.ndarray:
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


def _body_speed_mps(state: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(state[6:9], dtype=np.float64)))


def _energy_limited_turn(
    speed_mps: float,
    requested_bank_deg: float,
    maximum_pull: float,
) -> tuple[float, float]:
    """Reduce bank and pull before the aircraft reaches an unrecoverable low-energy state."""
    energy = float(np.clip((speed_mps - 150.0) / 90.0, 0.0, 1.0))
    bank_limit_deg = 35.0 + energy * max(abs(requested_bank_deg) - 35.0, 0.0)
    pull_magnitude = 0.28 + energy * max(float(maximum_pull) - 0.28, 0.0)
    return math.copysign(min(abs(requested_bank_deg), bank_limit_deg), requested_bank_deg), pull_magnitude


class TailCaptureActionProvider(ActionProvider):
    """Acquire the target's rear quarter before handing control to a policy."""

    def __init__(
        self,
        delegate: ActionProvider | None = None,
        hard_turn_until_ata_deg: float = 45.0,
        hard_turn_max_s: float = 18.0,
        max_bank_deg: float = 88.0,
        pursuit_max_bank_deg: float = 88.0,
        bank_per_heading_error: float = 2.2,
        roll_gain: float = 0.040,
        pull_gain: float = 1.30,
        pitch_gain: float = 0.040,
        altitude_gain: float = 0.0025,
        hard_turn_rudder: float = 0.35,
        initial_turn_sign: float = 0.0,
        capture_min_range_m: float = 500.0,
        capture_max_range_m: float = 2200.0,
        capture_own_ata_deg: float = 25.0,
        capture_target_ata_deg: float = 90.0,
        capture_hold_s: float = 0.4,
        delegate_release_ata_deg: float = 55.0,
        delegate_release_distance_m: float = 2800.0,
        max_acquisition_s: float = 50.0,
        recovery_enter_altitude_m: float = 800.0,
        recovery_exit_altitude_m: float = 1400.0,
        recovery_bank_deg: float = 45.0,
        emergency_altitude_m: float = 800.0,
        recovery_pitch_deg: float = 14.0,
        recovery_reengage_s: float = 6.0,
        recovery_level_pitch_deg: float = 4.0,
        tail_aim_offset_m: float = 700.0,
        altitude_ceiling_above_start_m: float = 2000.0,
        threat_distance_m: float = 2200.0,
        threat_enemy_ata_deg: float = 45.0,
        threat_own_ata_deg: float = 60.0,
        evasion_bank_deg: float = 65.0,
        evasion_min_s: float = 6.0,
        evasion_max_s: float = 15.0,
        evasion_switch_s: float = 4.0,
        evasion_random_seed: int | None = None,
        merge_brake_s: float = 0.0,
        merge_brake_distance_m: float = 1700.0,
        debug: bool = False,
    ):
        self.delegate = delegate
        self.hard_turn_until_ata_deg = float(hard_turn_until_ata_deg)
        self.hard_turn_max_s = max(0.1, float(hard_turn_max_s))
        self.max_bank_deg = float(max_bank_deg)
        self.pursuit_max_bank_deg = abs(float(pursuit_max_bank_deg))
        self.bank_per_heading_error = float(bank_per_heading_error)
        self.roll_gain = float(roll_gain)
        self.pull_gain = float(pull_gain)
        self.pitch_gain = float(pitch_gain)
        self.altitude_gain = float(altitude_gain)
        self.hard_turn_rudder = float(hard_turn_rudder)
        self.initial_turn_sign = float(np.sign(initial_turn_sign))
        self.capture_min_range_m = float(capture_min_range_m)
        self.capture_max_range_m = float(capture_max_range_m)
        self.capture_own_ata_deg = float(capture_own_ata_deg)
        self.capture_target_ata_deg = float(capture_target_ata_deg)
        self.capture_hold_s = max(0.0, float(capture_hold_s))
        self.delegate_release_ata_deg = float(delegate_release_ata_deg)
        self.delegate_release_distance_m = float(delegate_release_distance_m)
        self.max_acquisition_s = max(0.1, float(max_acquisition_s))
        self.recovery_enter_altitude_m = float(recovery_enter_altitude_m)
        self.recovery_exit_altitude_m = max(
            self.recovery_enter_altitude_m,
            float(recovery_exit_altitude_m),
        )
        self.recovery_bank_deg = abs(float(recovery_bank_deg))
        self.emergency_altitude_m = float(emergency_altitude_m)
        self.recovery_pitch_deg = float(recovery_pitch_deg)
        self.recovery_reengage_s = max(0.0, float(recovery_reengage_s))
        self.recovery_level_pitch_deg = float(recovery_level_pitch_deg)
        self.tail_aim_offset_m = max(0.0, float(tail_aim_offset_m))
        self.altitude_ceiling_above_start_m = max(
            500.0,
            float(altitude_ceiling_above_start_m),
        )
        self.threat_distance_m = float(threat_distance_m)
        self.threat_enemy_ata_deg = float(threat_enemy_ata_deg)
        self.threat_own_ata_deg = float(threat_own_ata_deg)
        self.evasion_bank_deg = abs(float(evasion_bank_deg))
        self.evasion_min_s = max(0.0, float(evasion_min_s))
        self.evasion_max_s = max(self.evasion_min_s, float(evasion_max_s))
        self.evasion_switch_s = float(np.clip(
            evasion_switch_s,
            0.0,
            self.evasion_max_s,
        ))
        self._evasion_rng = np.random.default_rng(evasion_random_seed)
        self.merge_brake_s = max(0.0, float(merge_brake_s))
        self.merge_brake_distance_m = max(0.0, float(merge_brake_distance_m))
        self.debug = bool(debug)
        self.reset()

    def reset(self, context: ActionContext | None = None) -> None:
        self._reference_altitude_m: float | None = None
        self._hard_turn_complete = False
        self._hard_turn_sign = self.initial_turn_sign
        self._start_sim_time_s: float | None = None
        self._capture_start_time_s: float | None = None
        self._delegating = False
        self._recovering = False
        self._recovery_exit_time_s: float | None = None
        self._suppress_evasion_until_s = -1.0
        self._evasion_start_time_s: float | None = None
        self._evasion_sign = 0.0
        self._last_evasion_sign = 0.0
        self._evasion_reversed = False
        self._merge_brake_used = False
        self._merge_brake_until_s = -1.0
        self.last_info: dict[str, float | bool | str] = {}
        self._last_debug_stage: str | None = None
        self._next_debug_status_s = 0.0
        if self.delegate is not None:
            self.delegate.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        own = np.asarray(context.ownship_state, dtype=np.float64)
        target = np.asarray(context.target_state, dtype=np.float64)
        if self._reference_altitude_m is None:
            self._reference_altitude_m = float(own[StateIndex.ALT])

        sim_time_s = float(own[StateIndex.SIM_TIME])
        if self._start_sim_time_s is None or sim_time_s < self._start_sim_time_s:
            self._start_sim_time_s = sim_time_s
            self._capture_start_time_s = None
        relative = np.asarray(target[:3] - own[:3], dtype=np.float64)
        distance_m = float(np.linalg.norm(relative))
        # The server LOS display and WEZ use the aircraft nose, not the flight
        # path. At high angle of attack these vectors can differ by many degrees.
        own_ata_deg = _angle_deg(_nose_vector_ned(own), relative)
        target_ata_deg = _angle_deg(_nose_vector_ned(target), -relative)
        pure_heading_deg = math.degrees(
            math.atan2(float(relative[1]), float(relative[0]))
        )
        heading_error_deg = _wrap_deg(
            pure_heading_deg - float(own[StateIndex.YAW])
        )
        speed_mps = _body_speed_mps(own)
        if self._hard_turn_sign == 0.0 and abs(heading_error_deg) > 1.0e-6:
            self._hard_turn_sign = math.copysign(1.0, heading_error_deg)
        if not self._hard_turn_complete and (
            own_ata_deg <= self.hard_turn_until_ata_deg
            or sim_time_s - self._start_sim_time_s >= self.hard_turn_max_s
        ):
            self._hard_turn_complete = True

        captured = (
            self.capture_min_range_m <= distance_m <= self.capture_max_range_m
            and own_ata_deg <= self.capture_own_ata_deg
            and target_ata_deg >= self.capture_target_ata_deg
        )
        if captured:
            if self._capture_start_time_s is None:
                self._capture_start_time_s = sim_time_s
        else:
            self._capture_start_time_s = None
        capture_duration_s = (
            sim_time_s - self._capture_start_time_s
            if self._capture_start_time_s is not None
            else 0.0
        )
        elapsed_s = sim_time_s - self._start_sim_time_s
        timed_out = elapsed_s >= self.max_acquisition_s
        altitude_m = float(own[StateIndex.ALT])
        was_recovering = self._recovering
        if altitude_m <= self.recovery_enter_altitude_m:
            self._recovering = True
        elif altitude_m >= self.recovery_exit_altitude_m:
            self._recovering = False
        if self._recovering and not was_recovering:
            self._recovery_exit_time_s = None
            self._evasion_start_time_s = None
            self._evasion_sign = 0.0
            self._evasion_reversed = False
        elif was_recovering and not self._recovering:
            # Recovery is a temporary safety override. Once enough altitude is
            # restored, level out and reacquire the target instead of chaining
            # directly into another climbing defensive turn.
            self._recovery_exit_time_s = sim_time_s
            self._suppress_evasion_until_s = sim_time_s + self.recovery_reengage_s
            self._hard_turn_complete = True
            self._evasion_start_time_s = None
            self._evasion_sign = 0.0
            self._evasion_reversed = False
        reengaging_after_recovery = (
            self._recovery_exit_time_s is not None
            and sim_time_s < self._recovery_exit_time_s + self.recovery_reengage_s
        )

        threatened = (
            sim_time_s >= self._suppress_evasion_until_s
            and
            distance_m <= self.threat_distance_m
            and target_ata_deg <= self.threat_enemy_ata_deg
            and own_ata_deg >= self.threat_own_ata_deg
        )
        if threatened and self._evasion_start_time_s is None:
            self._evasion_start_time_s = sim_time_s
            # Turn toward an attacker that is clearly offset to one side.  Near
            # the exact six-o'clock line the bearing sign is numerically noisy,
            # so alternate sides instead of repeating a predictable break.
            if 15.0 <= abs(heading_error_deg) <= 165.0:
                self._evasion_sign = math.copysign(1.0, heading_error_deg)
            elif self._last_evasion_sign != 0.0:
                self._evasion_sign = -self._last_evasion_sign
            else:
                self._evasion_sign = float(self._evasion_rng.choice((-1.0, 1.0)))
            self._last_evasion_sign = self._evasion_sign
            self._evasion_reversed = False
        evasion_elapsed_s = (
            sim_time_s - self._evasion_start_time_s
            if self._evasion_start_time_s is not None
            else 0.0
        )
        threat_cleared = (
            target_ata_deg >= 75.0
            or distance_m >= 2600.0
        )
        still_locked = (
            target_ata_deg <= self.threat_enemy_ata_deg
            and own_ata_deg >= self.threat_own_ata_deg
            and distance_m <= self.threat_distance_m
        )
        if (
            self._evasion_start_time_s is not None
            and not self._evasion_reversed
            and evasion_elapsed_s >= self.evasion_switch_s
            and still_locked
            and altitude_m > self.recovery_exit_altitude_m
        ):
            self._evasion_sign = -self._evasion_sign
            self._last_evasion_sign = self._evasion_sign
            self._evasion_reversed = True
        if self._evasion_start_time_s is not None and (
            evasion_elapsed_s >= self.evasion_max_s
            or (evasion_elapsed_s >= self.evasion_min_s and threat_cleared)
        ):
            self._evasion_start_time_s = None
            self._evasion_sign = 0.0
            self._evasion_reversed = False
            evasion_elapsed_s = 0.0
        evading = self._evasion_start_time_s is not None

        if self._delegating and (
            own_ata_deg >= self.delegate_release_ata_deg
            or distance_m >= self.delegate_release_distance_m
        ):
            self._delegating = False
        if self.delegate is not None and (
            capture_duration_s >= self.capture_hold_s or timed_out
        ):
            self._delegating = True

        if self._recovering:
            emergency = altitude_m <= self.emergency_altitude_m
            desired_bank_deg = 0.0 if emergency else float(
                np.clip(
                    self.bank_per_heading_error * heading_error_deg,
                    -self.recovery_bank_deg,
                    self.recovery_bank_deg,
                )
            )
            bank_error_deg = _wrap_deg(desired_bank_deg - float(own[StateIndex.ROLL]))
            roll_cmd = float(np.clip(self.roll_gain * bank_error_deg, -1.0, 1.0))
            desired_pitch_deg = self.recovery_pitch_deg + (6.0 if emergency else 0.0)
            pitch_cmd = float(np.clip(
                self.pitch_gain * (float(own[StateIndex.PITCH]) - desired_pitch_deg),
                -1.0,
                0.25,
            ))
            stage = "emergency_recovery" if emergency else "climbing_turn"
            action = clip_action([roll_cmd, pitch_cmd, 0.0, 1.0])
        elif reengaging_after_recovery:
            tail_point = target[:3] - (
                _nose_vector_ned(target) * self.tail_aim_offset_m
            )
            pursuit_relative = np.asarray(tail_point - own[:3], dtype=np.float64)
            pursuit_heading_deg = math.degrees(math.atan2(
                float(pursuit_relative[1]), float(pursuit_relative[0])
            ))
            pursuit_heading_error_deg = _wrap_deg(
                pursuit_heading_deg - float(own[StateIndex.YAW])
            )
            desired_bank_deg = float(np.clip(
                self.bank_per_heading_error * pursuit_heading_error_deg,
                -self.recovery_bank_deg,
                self.recovery_bank_deg,
            ))
            bank_error_deg = _wrap_deg(
                desired_bank_deg - float(own[StateIndex.ROLL])
            )
            roll_cmd = float(np.clip(self.roll_gain * bank_error_deg, -1.0, 1.0))
            pitch_cmd = float(np.clip(
                self.pitch_gain * (
                    float(own[StateIndex.PITCH]) - self.recovery_level_pitch_deg
                ),
                -0.35,
                0.75,
            ))
            action = clip_action([roll_cmd, pitch_cmd, 0.0, 1.0])
            stage = "reengage_after_recovery"
        elif evading:
            desired_bank_deg, pull_magnitude = _energy_limited_turn(
                speed_mps,
                self._evasion_sign * self.evasion_bank_deg,
                0.88,
            )
            if target_ata_deg <= 15.0 and distance_m <= 1200.0:
                desired_bank_deg = self._evasion_sign * max(abs(desired_bank_deg), 55.0)
                pull_magnitude = max(pull_magnitude, 0.75)
            if float(own[StateIndex.PITCH]) >= 18.0:
                pull_magnitude = max(0.30, pull_magnitude - 0.20)
            bank_error_deg = _wrap_deg(desired_bank_deg - float(own[StateIndex.ROLL]))
            roll_cmd = float(np.clip(self.roll_gain * bank_error_deg, -1.0, 1.0))
            action = clip_action([roll_cmd, -pull_magnitude, 0.0, 1.0])
            direction = "right" if self._evasion_sign > 0.0 else "left"
            suffix = "_reversal" if self._evasion_reversed else ""
            stage = f"threat_evasion_{direction}{suffix}"
        elif self._delegating and self.delegate is not None:
            self._debug_stage(
                "delegate", elapsed_s, own_ata_deg, target_ata_deg, distance_m
            )
            result = self.delegate.compute_action(context)
            result.info = dict(result.info)
            result.info.update({
                "tail_capture_handoff": True,
                "tail_capture_timeout": timed_out,
            })
            self._debug_status(
                elapsed_s, "delegate", own, target, distance_m,
                own_ata_deg, target_ata_deg, speed_mps,
            )
            return result

        else:
            if not self._hard_turn_complete:
                desired_bank_deg, pull_magnitude = _energy_limited_turn(
                    speed_mps,
                    self._hard_turn_sign * self.max_bank_deg,
                    1.0,
                )
                bank_error_deg = _wrap_deg(
                    desired_bank_deg - float(own[StateIndex.ROLL])
                )
                action = clip_action([
                    self.roll_gain * bank_error_deg,
                    -pull_magnitude,
                    -self._hard_turn_sign * self.hard_turn_rudder,
                    1.0,
                ])
                stage = "hard_turn"
            else:
                # Aim behind the target instead of at its center. Pure pursuit
                # repeatedly overshoots a turning opponent and never captures
                # its rear quarter.
                tail_point = target[:3] - (
                    _nose_vector_ned(target) * self.tail_aim_offset_m
                )
                pursuit_relative = np.asarray(tail_point - own[:3], dtype=np.float64)
                pursuit_heading_deg = math.degrees(math.atan2(
                    float(pursuit_relative[1]),
                    float(pursuit_relative[0]),
                ))
                pursuit_heading_error_deg = _wrap_deg(
                    pursuit_heading_deg - float(own[StateIndex.YAW])
                )
                requested_bank_deg = float(np.clip(
                    self.bank_per_heading_error * pursuit_heading_error_deg,
                    -self.pursuit_max_bank_deg,
                    self.pursuit_max_bank_deg,
                ))
                desired_bank_deg, pull_limit = _energy_limited_turn(
                    speed_mps,
                    requested_bank_deg,
                    1.0,
                )
                bank_error_deg = _wrap_deg(
                    desired_bank_deg - float(own[StateIndex.ROLL])
                )
                roll_cmd = float(np.clip(
                    self.roll_gain * bank_error_deg, -1.0, 1.0
                ))
                horizontal_m = max(
                    float(np.linalg.norm(pursuit_relative[:2])),
                    1.0,
                )
                path_pitch_deg = math.degrees(
                    math.atan2(-float(pursuit_relative[2]), horizontal_m)
                )
                altitude_error_m = self._reference_altitude_m - altitude_m
                desired_pitch_deg = float(np.clip(
                    path_pitch_deg + self.altitude_gain * altitude_error_m,
                    -8.0,
                    12.0,
                ))
                bank_fraction = min(
                    abs(desired_bank_deg) / self.pursuit_max_bank_deg, 1.0
                )
                pitch_cmd = (
                    self.pitch_gain * (float(own[StateIndex.PITCH]) - desired_pitch_deg)
                    - self.pull_gain * bank_fraction
                )
                pitch_cmd = max(pitch_cmd, -pull_limit)
                pitch_error_deg = float(own[StateIndex.PITCH]) - desired_pitch_deg
                if pitch_error_deg >= 12.0:
                    pitch_cmd = max(pitch_cmd, 0.28)
                if float(own[StateIndex.PITCH]) >= 25.0:
                    pitch_cmd = max(pitch_cmd, 0.40)
                if altitude_m >= (
                    self._reference_altitude_m + self.altitude_ceiling_above_start_m
                ):
                    pitch_cmd = max(pitch_cmd, 0.45)
                action = clip_action([roll_cmd, pitch_cmd, 0.0, 1.0])
                stage = "pursuit"
                merge_crossing = (
                    self.merge_brake_s > 0.0
                    and not self._merge_brake_used
                    and distance_m <= self.merge_brake_distance_m
                    and own_ata_deg <= 70.0
                    and target_ata_deg <= 90.0
                )
                if merge_crossing:
                    self._merge_brake_used = True
                    self._merge_brake_until_s = sim_time_s + self.merge_brake_s
                if sim_time_s < self._merge_brake_until_s:
                    action[3] = -1.0
                    stage = "pursuit_brake"

        self.last_info = {
            "stage": stage,
            "distance_m": distance_m,
            "own_ata_deg": own_ata_deg,
            "target_ata_deg": target_ata_deg,
            "capture_duration_s": capture_duration_s,
            "elapsed_s": elapsed_s,
            "altitude_m": altitude_m,
            "speed_mps": speed_mps,
            "recovering": self._recovering,
            "threatened": threatened,
            "evading": evading,
            "evasion_elapsed_s": evasion_elapsed_s,
            "evasion_sign": self._evasion_sign,
            "evasion_reversed": self._evasion_reversed,
        }
        self._debug_stage(stage, elapsed_s, own_ata_deg, target_ata_deg, distance_m)
        self._debug_status(
            elapsed_s, stage, own, target, distance_m,
            own_ata_deg, target_ata_deg, speed_mps,
        )
        return ActionResult(
            action=np.asarray(action, dtype=np.float32),
            source=f"tail_capture_{stage}",
            confidence=0.98,
            info=dict(self.last_info),
        )

    def _debug_stage(
        self,
        stage: str,
        elapsed_s: float,
        own_ata_deg: float,
        target_ata_deg: float,
        distance_m: float,
    ) -> None:
        if not self.debug or stage == self._last_debug_stage:
            return
        self._last_debug_stage = stage
        print(
            "[tail-capture] "
            f"stage={stage} t={elapsed_s:.2f}s own_ata={own_ata_deg:.1f} "
            f"target_ata={target_ata_deg:.1f} distance={distance_m:.0f}m",
            flush=True,
        )

    def _debug_status(
        self,
        elapsed_s: float,
        stage: str,
        own: np.ndarray,
        target: np.ndarray,
        distance_m: float,
        own_ata_deg: float,
        target_ata_deg: float,
        speed_mps: float,
    ) -> None:
        if not self.debug or elapsed_s + 1.0e-6 < self._next_debug_status_s:
            return
        self._next_debug_status_s = elapsed_s + 5.0
        print(
            "[tail-status] "
            f"t={elapsed_s:.1f}s stage={stage} distance={distance_m:.0f}m "
            f"own_ata={own_ata_deg:.1f} target_ata={target_ata_deg:.1f} "
            f"speed={speed_mps:.1f}mps alt={float(own[StateIndex.ALT]):.0f}m "
            f"pitch={float(own[StateIndex.PITCH]):.1f}deg "
            f"roll={float(own[StateIndex.ROLL]):.1f}deg "
            f"own_hp={float(own[StateIndex.HEALTH]):.3f} "
            f"target_hp={float(target[StateIndex.HEALTH]):.3f}",
            flush=True,
        )

    def close(self) -> None:
        if self.delegate is not None:
            self.delegate.close()


__all__ = ["TailCaptureActionProvider"]
