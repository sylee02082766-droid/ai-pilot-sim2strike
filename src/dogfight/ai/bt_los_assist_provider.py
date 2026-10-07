from __future__ import annotations

import math

import numpy as np

from dogfight.ai.action_provider import (
    ActionContext,
    ActionProvider,
    ActionResult,
    _apply_tactical16_bank_turn_assist,
    _apply_tactical16_los_assist,
    clip_action,
)
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
    return math.degrees(math.acos(float(np.clip(
        np.dot(first, second) / denominator,
        -1.0,
        1.0,
    ))))


def _wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


class BTLOSAssistProvider(ActionProvider):
    """Keep the reference BT maneuvering and tighten its nose near a firing chance."""

    def __init__(self, primary: ActionProvider, debug: bool = False) -> None:
        self.primary = primary
        self.debug = bool(debug)
        self.reset()

    def reset(self, context: ActionContext | None = None) -> None:
        self._last_stage = None
        self._reference_altitude_m: float | None = None
        self._energy_recovery = False
        self._evasion_start_s: float | None = None
        self._evasion_sign = 0.0
        self._last_evasion_sign = -1.0
        self._evasion_reversed = False
        self._debug_start_s: float | None = None
        self._next_debug_s = 0.0
        self.primary.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        result = self.primary.compute_action(context)
        stage = str(result.info.get("bt32_stage", "reference_bt"))
        if stage == "deck_recovery":
            return result

        own = np.asarray(context.ownship_state, dtype=np.float64)
        target = np.asarray(context.target_state, dtype=np.float64)
        if self._reference_altitude_m is None:
            self._reference_altitude_m = float(own[StateIndex.ALT])
        relative = target[:3] - own[:3]
        distance_m = float(np.linalg.norm(relative))
        own_ata_deg = _angle(_nose(own), relative)
        target_ata_deg = _angle(_nose(target), -relative)
        speed_mps = float(np.linalg.norm(own[6:9]))
        altitude_m = float(own[StateIndex.ALT])
        pitch_deg = float(own[StateIndex.PITCH])
        roll_deg = float(own[StateIndex.ROLL])
        sim_time_s = float(own[StateIndex.SIM_TIME])
        observation = np.asarray(context.observation, dtype=np.float32)
        action = np.asarray(result.action, dtype=np.float32)

        pure_heading_deg = math.degrees(math.atan2(float(relative[1]), float(relative[0])))
        heading_error_deg = _wrap_deg(pure_heading_deg - float(own[StateIndex.YAW]))

        threatened = (
            distance_m <= 2200.0
            and target_ata_deg <= 32.0
            and own_ata_deg >= 58.0
        )
        if threatened and self._evasion_start_s is None:
            self._evasion_start_s = sim_time_s
            los_az_deg = float(observation[11]) * 180.0
            if 12.0 <= abs(los_az_deg) <= 165.0:
                self._evasion_sign = math.copysign(1.0, los_az_deg)
            else:
                self._evasion_sign = -self._last_evasion_sign
            self._last_evasion_sign = self._evasion_sign
            self._evasion_reversed = False

        evasion_elapsed_s = (
            sim_time_s - self._evasion_start_s
            if self._evasion_start_s is not None
            else 0.0
        )
        if (
            self._evasion_start_s is not None
            and not self._evasion_reversed
            and evasion_elapsed_s >= 4.0
            and threatened
            and altitude_m >= 1200.0
        ):
            self._evasion_sign = -self._evasion_sign
            self._last_evasion_sign = self._evasion_sign
            self._evasion_reversed = True
        if self._evasion_start_s is not None and (
            evasion_elapsed_s >= 10.0
            or (
                evasion_elapsed_s >= 3.0
                and (target_ata_deg >= 75.0 or distance_m >= 2600.0)
            )
        ):
            self._evasion_start_s = None
            self._evasion_sign = 0.0
            self._evasion_reversed = False
            evasion_elapsed_s = 0.0

        high_altitude = altitude_m >= self._reference_altitude_m + 1600.0
        low_energy = speed_mps <= 175.0
        steep_low_energy_climb = pitch_deg >= 25.0 and speed_mps <= 215.0
        if high_altitude or low_energy or steep_low_energy_climb:
            self._energy_recovery = True
        elif (
            speed_mps >= 225.0
            and altitude_m <= self._reference_altitude_m + 1200.0
            and pitch_deg <= 12.0
        ):
            self._energy_recovery = False

        if self._evasion_start_s is not None and speed_mps >= 145.0:
            energy = float(np.clip((speed_mps - 145.0) / 100.0, 0.0, 1.0))
            desired_bank_deg = self._evasion_sign * (40.0 + 28.0 * energy)
            bank_error_deg = _wrap_deg(desired_bank_deg - roll_deg)
            pull = 0.28 + 0.50 * energy
            if pitch_deg >= 18.0 or altitude_m >= self._reference_altitude_m + 1200.0:
                pull = max(0.28, pull - 0.22)
            if target_ata_deg <= 12.0 and distance_m <= 1200.0 and speed_mps >= 190.0:
                pull = max(pull, 0.65)
            action = clip_action([
                0.04 * bank_error_deg,
                -pull,
                0.0,
                1.0,
            ])
            direction = "right" if self._evasion_sign > 0.0 else "left"
            suffix = "_reversal" if self._evasion_reversed else ""
            stage = f"threat_evasion_{direction}{suffix}"
        elif self._energy_recovery:
            desired_bank_deg = float(np.clip(0.65 * heading_error_deg, -32.0, 32.0))
            bank_error_deg = _wrap_deg(desired_bank_deg - roll_deg)
            desired_pitch_deg = -10.0 if (
                altitude_m >= self._reference_altitude_m + 700.0 or speed_mps <= 200.0
            ) else 0.0
            pitch_cmd = float(np.clip(
                0.05 * (pitch_deg - desired_pitch_deg),
                -0.20,
                0.75,
            ))
            action = clip_action([
                0.04 * bank_error_deg,
                pitch_cmd,
                0.0,
                1.0,
            ])
            stage = "energy_recovery"

        attack_geometry = distance_m <= 3200.0 and target_ata_deg >= 75.0
        if stage == "reference_bt" and attack_geometry and own_ata_deg <= 55.0:
            action = _apply_tactical16_los_assist(action, observation, {
                "enabled": True,
                "max_ata_deg": 55.0,
                "rudder_gain": 2.8,
                "rudder_sign": -1.0,
                "roll_gain": 0.45,
                "roll_sign": 1.0,
                "pitch_gain": 0.28,
                "pitch_sign": -1.0,
                "az_ref_deg": 8.0,
                "el_ref_deg": 5.0,
                "deadband_deg": 0.0,
            })
            stage = "coarse_los_assist"

        if stage == "coarse_los_assist" and attack_geometry and own_ata_deg <= 25.0:
            action = _apply_tactical16_los_assist(action, observation, {
                "enabled": True,
                "max_ata_deg": 25.0,
                "rudder_gain": 9.5,
                "rudder_sign": -1.0,
                "roll_gain": 0.80,
                "roll_sign": 1.0,
                "pitch_gain": 0.50,
                "pitch_sign": -1.0,
                "az_ref_deg": 1.6,
                "el_ref_deg": 1.3,
                "deadband_deg": 0.0,
            })
            action = _apply_tactical16_bank_turn_assist(action, observation, {
                "enabled": True,
                "max_ata_deg": 25.0,
                "az_ref_deg": 10.0,
                "deadband_deg": 1.0,
                "disable_ata_deg": 0.0,
                "disable_in_wez": False,
                "roll_sign": -1.0,
                "roll_gain": 0.12,
                "pull_gain": 0.05,
            })
            stage = "precision_los_assist"

        if stage in ("coarse_los_assist", "precision_los_assist"):
            if speed_mps <= 220.0:
                action[1] = max(float(action[1]), 0.08)
            if altitude_m >= self._reference_altitude_m + 1000.0:
                action[1] = max(float(action[1]), 0.14)
            if pitch_deg >= 18.0:
                action[1] = max(float(action[1]), 0.22)
            action[3] = 1.0

        if self.debug and stage != self._last_stage:
            print(
                "[bt-los-assist] "
                f"stage={stage} distance={distance_m:.0f}m "
                f"own_ata={own_ata_deg:.1f} target_ata={target_ata_deg:.1f}",
                flush=True,
            )
            self._last_stage = stage
        result.action = clip_action(action)
        result.source = stage
        result.info = dict(result.info)
        result.info["bt_los_stage"] = stage

        if self.debug:
            if self._debug_start_s is None or sim_time_s < self._debug_start_s:
                self._debug_start_s = sim_time_s
                self._next_debug_s = 0.0
            elapsed_s = sim_time_s - self._debug_start_s
            if elapsed_s + 1.0e-6 >= self._next_debug_s:
                self._next_debug_s = elapsed_s + 5.0
                print(
                    "[bt-los-status] "
                    f"t={elapsed_s:.1f}s stage={stage} distance={distance_m:.0f}m "
                    f"own_ata={own_ata_deg:.1f} target_ata={target_ata_deg:.1f} "
                    f"speed={speed_mps:.1f}mps alt={altitude_m:.0f}m "
                    f"pitch={pitch_deg:.1f} roll={roll_deg:.1f} "
                    f"action={np.asarray(result.action).round(3).tolist()}",
                    flush=True,
                )
        return result

    def close(self) -> None:
        self.primary.close()


__all__ = ["BTLOSAssistProvider"]
