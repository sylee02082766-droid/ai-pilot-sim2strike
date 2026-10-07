# -*- coding: utf-8 -*-
"""Guarded attack reward for PPO curriculum.

This reward keeps attack shaping, but treats target self-crash as a weak signal
and heavily penalizes ownship altitude failures. The intent is to avoid learning
"win because Red falls first" while still preserving WEZ/damage pressure.
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
    "step_penalty": -0.002,
    "survival_bonus": 0.002,
    "pursuit_scale": 0.45,
    "pursuit_half_angle_deg": 75.0,
    "pursuit_range_m": 10000.0,
    "close_range_bonus": 0.015,
    "damage_scale": 25.0,
    "altitude_target_m": 7000.0,
    "altitude_band_m": 5000.0,
    "safe_altitude_m": 2600.0,
    "low_altitude_penalty": 1.2,
    "attitude_penalty_scale": 0.025,
    "speed_target_kcas": 320.0,
    "speed_band_kcas": 220.0,
    "speed_penalty_scale": 0.02,
    "win_reward": 130.0,
    "loss_reward": -150.0,
    "draw_reward": -15.0,
    "own_crash_penalty": -220.0,
    "target_crash_reward": -5.0,
    "guard_fail_penalty": -110.0,
}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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
    components["step"] = float(reward_config.get("step_penalty", -0.002))

    distance = float(geo_info._get_distance(ownship_state, target_state))
    ata = abs(float(geo_info._get_antenna_train_angle(ownship_state, target_state, False)))
    aa = abs(float(geo_info._get_aspect_angle(ownship_state, target_state, False)))

    half_angle = max(float(reward_config.get("pursuit_half_angle_deg", 75.0)), 1.0)
    pursuit_range = max(float(reward_config.get("pursuit_range_m", 10000.0)), 1.0)
    nose_score = _clip01(1.0 - ata / half_angle)
    aspect_score = _clip01(1.0 - aa / 140.0)
    range_score = _clip01(1.0 - distance / pursuit_range)
    components["pursuit"] = float(reward_config.get("pursuit_scale", 0.45)) * (
        0.7 * nose_score + 0.3 * aspect_score
    ) * range_score
    if distance < 1200.0 and ata < 35.0:
        components["pursuit"] += float(reward_config.get("close_range_bonus", 0.015))

    components["damage"] = float(reward_config.get("damage_scale", 25.0)) * (
        float(target_damage) - float(ownship_damage)
    )

    altitude = float(ownship_state[StateIndex.ALT])
    target_alt = float(reward_config.get("altitude_target_m", 7000.0))
    altitude_band = max(float(reward_config.get("altitude_band_m", 5000.0)), 1.0)
    safe_altitude = max(float(reward_config.get("safe_altitude_m", 2600.0)), 1.0)
    altitude_hold = 0.015 * _clip01(1.0 - abs(altitude - target_alt) / altitude_band)
    low_altitude = 0.0
    if altitude < safe_altitude:
        low_altitude = -float(reward_config.get("low_altitude_penalty", 1.2)) * (
            1.0 - altitude / safe_altitude
        )
    components["altitude"] = altitude_hold + low_altitude

    roll = abs(float(ownship_state[StateIndex.ROLL]))
    pitch = abs(float(ownship_state[StateIndex.PITCH]))
    components["attitude"] = -float(reward_config.get("attitude_penalty_scale", 0.025)) * (
        (roll / 180.0) ** 2 + (pitch / 90.0) ** 2
    )

    speed = float(ownship_state[StateIndex.KCAS])
    speed_target = float(reward_config.get("speed_target_kcas", 320.0))
    speed_band = max(float(reward_config.get("speed_band_kcas", 220.0)), 1.0)
    speed_excess_error = max(0.0, abs(speed - speed_target) - speed_band)
    components["speed"] = -float(reward_config.get("speed_penalty_scale", 0.02)) * _clip01(
        speed_excess_error / speed_band
    )

    terminal = 0.0
    if terminated or truncated:
        ownship_health = float(ownship_state[StateIndex.HEALTH])
        target_health = float(target_state[StateIndex.HEALTH])
        if end_condition in ("ownship altitude below min", "FDM Update Fail"):
            terminal = float(reward_config.get("own_crash_penalty", -220.0))
        elif end_condition == "target altitude below min":
            terminal = float(reward_config.get("target_crash_reward", -5.0))
        elif end_condition == "two circle headon guard fail":
            terminal = float(reward_config.get("guard_fail_penalty", -110.0))
        elif target_health <= 0.0 < ownship_health:
            terminal = float(reward_config.get("win_reward", 130.0))
        elif ownship_health <= 0.0 < target_health:
            terminal = float(reward_config.get("loss_reward", -150.0))
        else:
            terminal = float(reward_config.get("draw_reward", -15.0))
    components["terminal"] = terminal

    return float(sum(components.values())), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
