# -*- coding: utf-8 -*-
"""Stability-first reward for early dogfight training.

The first baseline crashed by rolling and descending before it learned useful
WEZ behavior. This hook keeps the tactical terms, but makes early flight
stability visible to the learner: stay alive, avoid low altitude, avoid extreme
roll/pitch, and keep a usable speed band.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.sim.state_schema import StateIndex


MY_REWARD_CONFIG = {
    "survival_bonus": 0.02,
    "step_penalty": -0.005,
    "damage_scale": 10.0,
    "pursuit_scale": 0.15,
    "pursuit_half_angle_deg": 60.0,
    "pursuit_range_m": 8000.0,
    "altitude_target_m": 7000.0,
    "altitude_band_m": 3500.0,
    "safe_altitude_m": 2500.0,
    "low_altitude_penalty": 1.5,
    "roll_penalty_scale": 0.04,
    "pitch_penalty_scale": 0.04,
    "speed_target_kcas": 300.0,
    "speed_band_kcas": 180.0,
    "speed_penalty_scale": 0.03,
    "win_reward": 100.0,
    "loss_reward": -120.0,
    "draw_reward": -20.0,
    "crash_penalty": -150.0,
    "guard_fail_penalty": -80.0,
}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _abs_state(state, index: StateIndex) -> float:
    return abs(float(state[index]))


def compute_reward(
    ownship_state,
    target_state,
    ownship_damage: float,
    target_damage: float,
    geo_info,
    wez_config: dict,
    reward_config: dict,
    terminated: bool,
    truncated: bool,
    end_condition: str,
) -> tuple[float, dict]:
    components: dict[str, float] = {}

    components["survival"] = float(reward_config.get("survival_bonus", 0.0))
    components["step"] = float(reward_config.get("step_penalty", -0.005))

    distance = float(geo_info._get_distance(ownship_state, target_state))
    ata = abs(float(geo_info._get_antenna_train_angle(ownship_state, target_state, False)))
    half_angle = float(reward_config.get("pursuit_half_angle_deg", 60.0))
    pursuit_range = float(reward_config.get("pursuit_range_m", 8000.0))
    ata_factor = _clip01(1.0 - ata / max(half_angle, 1.0))
    range_factor = _clip01(1.0 - distance / max(pursuit_range, 1.0))
    components["pursuit"] = (
        float(reward_config.get("pursuit_scale", 0.0)) * ata_factor * range_factor
    )

    components["damage"] = (
        float(reward_config.get("damage_scale", 10.0))
        * (float(target_damage) - float(ownship_damage))
    )

    altitude = float(ownship_state[StateIndex.ALT])
    target_alt = float(reward_config.get("altitude_target_m", 7000.0))
    altitude_band = max(float(reward_config.get("altitude_band_m", 3500.0)), 1.0)
    safe_altitude = max(float(reward_config.get("safe_altitude_m", 2500.0)), 1.0)
    altitude_hold = 0.02 * _clip01(1.0 - abs(altitude - target_alt) / altitude_band)
    low_altitude = 0.0
    if altitude < safe_altitude:
        low_altitude = -float(reward_config.get("low_altitude_penalty", 1.5)) * (
            1.0 - altitude / safe_altitude
        )
    components["altitude"] = altitude_hold + low_altitude

    roll = _abs_state(ownship_state, StateIndex.ROLL)
    pitch = _abs_state(ownship_state, StateIndex.PITCH)
    components["attitude"] = -(
        float(reward_config.get("roll_penalty_scale", 0.04)) * (roll / 180.0) ** 2
        + float(reward_config.get("pitch_penalty_scale", 0.04)) * (pitch / 90.0) ** 2
    )

    speed = float(ownship_state[StateIndex.KCAS])
    speed_target = float(reward_config.get("speed_target_kcas", 300.0))
    speed_band = max(float(reward_config.get("speed_band_kcas", 180.0)), 1.0)
    speed_error = max(0.0, abs(speed - speed_target) - speed_band)
    components["speed"] = -float(reward_config.get("speed_penalty_scale", 0.03)) * _clip01(
        speed_error / speed_band
    )

    terminal = 0.0
    if terminated or truncated:
        ownship_health = float(ownship_state[StateIndex.HEALTH])
        target_health = float(target_state[StateIndex.HEALTH])
        if end_condition in ("ownship altitude below min", "FDM Update Fail"):
            terminal = float(reward_config.get("crash_penalty", -150.0))
        elif end_condition == "two circle headon guard fail":
            terminal = float(reward_config.get("guard_fail_penalty", -80.0))
        elif target_health <= 0.0 < ownship_health:
            terminal = float(reward_config.get("win_reward", 100.0))
        elif ownship_health <= 0.0 < target_health:
            terminal = float(reward_config.get("loss_reward", -120.0))
        elif truncated:
            terminal = float(reward_config.get("draw_reward", -20.0))
        else:
            terminal = float(reward_config.get("draw_reward", -20.0))
    components["terminal"] = terminal

    return float(sum(components.values())), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
