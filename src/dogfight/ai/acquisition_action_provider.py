from __future__ import annotations

import math

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.sim.state_schema import StateIndex


def _wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _body_velocity_ned(state: np.ndarray) -> np.ndarray:
    roll = math.radians(float(state[StateIndex.ROLL]))
    pitch = math.radians(float(state[StateIndex.PITCH]))
    yaw = math.radians(float(state[StateIndex.YAW]))
    u, v, w = (float(value) for value in state[6:9])
    cphi, sphi = math.cos(roll), math.sin(roll)
    cth, sth = math.cos(pitch), math.sin(pitch)
    cpsi, spsi = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            cth * cpsi * u
            + (sphi * sth * cpsi - cphi * spsi) * v
            + (cphi * sth * cpsi + sphi * spsi) * w,
            cth * spsi * u
            + (sphi * sth * spsi + cphi * cpsi) * v
            + (cphi * sth * spsi - sphi * cpsi) * w,
            -sth * u + sphi * cth * v + cphi * cth * w,
        ],
        dtype=np.float64,
    )


def _positive_intercept_time(
    relative_position: np.ndarray,
    target_velocity: np.ndarray,
    own_speed_mps: float,
    max_time_s: float,
) -> float:
    a = float(np.dot(target_velocity, target_velocity) - own_speed_mps**2)
    b = 2.0 * float(np.dot(relative_position, target_velocity))
    c = float(np.dot(relative_position, relative_position))
    candidates: list[float] = []
    if abs(a) < 1.0e-6:
        if abs(b) > 1.0e-6:
            candidates.append(-c / b)
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= 0.0:
            root = math.sqrt(discriminant)
            candidates.extend(((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)))
    positive = [value for value in candidates if value > 0.0 and math.isfinite(value)]
    if positive:
        return float(np.clip(min(positive), 0.0, max_time_s))
    distance = math.sqrt(max(c, 0.0))
    return float(np.clip(distance / max(own_speed_mps, 1.0), 0.0, max_time_s))


class AcquisitionActionProvider(ActionProvider):
    """Classical lead-pursuit controller for high-ATA position acquisition."""

    def __init__(
        self,
        delegate: ActionProvider | None = None,
        handoff_ata_deg: float = 18.0,
        handoff_distance_m: float = 2500.0,
        handoff_frames: int = 30,
        max_lead_time_s: float = 2.0,
        max_bank_deg: float = 65.0,
        bank_per_heading_error: float = 1.15,
        roll_gain: float = 0.035,
        pitch_gain: float = 0.028,
        bank_load_gain: float = 0.34,
        altitude_gain: float = 0.002,
    ):
        self.delegate = delegate
        self.handoff_ata_deg = float(handoff_ata_deg)
        self.handoff_distance_m = float(handoff_distance_m)
        self.handoff_frames = max(1, int(handoff_frames))
        self.max_lead_time_s = float(max_lead_time_s)
        self.max_bank_deg = float(max_bank_deg)
        self.bank_per_heading_error = float(bank_per_heading_error)
        self.roll_gain = float(roll_gain)
        self.pitch_gain = float(pitch_gain)
        self.bank_load_gain = float(bank_load_gain)
        self.altitude_gain = float(altitude_gain)
        self._handoff_count = 0
        self._delegating = False
        self._reference_altitude_m: float | None = None
        self.last_info: dict[str, float] = {}

    def reset(self, context: ActionContext | None = None) -> None:
        self._handoff_count = 0
        self._delegating = False
        self._reference_altitude_m = None
        self.last_info = {}
        if self.delegate is not None:
            self.delegate.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        own = np.asarray(context.ownship_state, dtype=np.float64)
        target = np.asarray(context.target_state, dtype=np.float64)
        if self._reference_altitude_m is None:
            self._reference_altitude_m = float(own[StateIndex.ALT])

        relative = target[:3] - own[:3]
        distance_m = float(np.linalg.norm(relative))
        own_velocity = _body_velocity_ned(own)
        target_velocity = _body_velocity_ned(target)
        own_speed_mps = max(float(np.linalg.norm(own_velocity)), 1.0)
        intercept_time_s = _positive_intercept_time(
            relative, target_velocity, own_speed_mps, self.max_lead_time_s
        )
        intercept_vector = relative + target_velocity * intercept_time_s
        horizontal = max(float(np.linalg.norm(intercept_vector[:2])), 1.0)
        desired_yaw_deg = math.degrees(
            math.atan2(float(intercept_vector[1]), float(intercept_vector[0]))
        )
        heading_error_deg = _wrap_deg(desired_yaw_deg - float(own[StateIndex.YAW]))
        desired_path_pitch_deg = math.degrees(
            math.atan2(-float(intercept_vector[2]), horizontal)
        )

        own_ata_deg = math.degrees(
            math.acos(
                float(
                    np.clip(
                        np.dot(own_velocity, relative)
                        / max(float(np.linalg.norm(own_velocity)) * distance_m, 1.0e-6),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        if own_ata_deg <= self.handoff_ata_deg and distance_m <= self.handoff_distance_m:
            self._handoff_count += 1
        else:
            self._handoff_count = 0
        if self.delegate is not None and self._handoff_count >= self.handoff_frames:
            self._delegating = True
        if self._delegating and self.delegate is not None:
            result = self.delegate.compute_action(context)
            result.info = dict(result.info)
            result.info["acquisition_handoff"] = True
            return result

        desired_bank_deg = float(
            np.clip(
                self.bank_per_heading_error * heading_error_deg,
                -self.max_bank_deg,
                self.max_bank_deg,
            )
        )
        current_roll_deg = float(own[StateIndex.ROLL])
        bank_error_deg = _wrap_deg(desired_bank_deg - current_roll_deg)
        roll_cmd = float(np.clip(self.roll_gain * bank_error_deg, -0.85, 0.85))

        altitude_error_m = self._reference_altitude_m - float(own[StateIndex.ALT])
        desired_pitch_deg = float(
            np.clip(
                desired_path_pitch_deg + self.altitude_gain * altitude_error_m,
                -12.0,
                12.0,
            )
        )
        bank_radians = math.radians(min(abs(current_roll_deg), 75.0))
        bank_load = 1.0 / max(math.cos(bank_radians), 0.3) - 1.0
        pitch_cmd = self.pitch_gain * (
            float(own[StateIndex.PITCH]) - desired_pitch_deg
        ) - self.bank_load_gain * bank_load
        turn_pull_floor = 0.50 * abs(desired_bank_deg) / max(self.max_bank_deg, 1.0)
        pitch_cmd = min(pitch_cmd, -turn_pull_floor)
        pitch_cmd = float(np.clip(pitch_cmd, -0.72, 0.72))

        action = clip_action([roll_cmd, pitch_cmd, 0.0, 1.0])
        self.last_info = {
            "distance_m": distance_m,
            "own_ata_deg": own_ata_deg,
            "intercept_time_s": intercept_time_s,
            "heading_error_deg": heading_error_deg,
            "desired_bank_deg": desired_bank_deg,
            "handoff_count": float(self._handoff_count),
        }
        return ActionResult(
            action=action,
            source="lead_acquisition",
            confidence=0.95,
            info=dict(self.last_info),
        )

    def close(self) -> None:
        if self.delegate is not None:
            self.delegate.close()


__all__ = ["AcquisitionActionProvider"]
