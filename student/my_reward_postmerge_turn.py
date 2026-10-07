# -*- coding: utf-8 -*-
"""Gentle post-merge nose-alignment reward.

Use after a policy can already survive the first merge. The goal is not to
increase closeness further, but to reward turning the nose back toward the
target in the 700-2200 m post-merge band while preserving altitude safety.
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
    "pursuit_scale": 0.22,
    "pursuit_half_angle_deg": 50.0,
    "pursuit_range_m": 7000.0,
    "turn_band_scale": 0.18,
    "turn_band_min_m": 700.0,
    "turn_band_max_m": 2200.0,
    "range_band_scale": 0.0,
    "below_band_penalty_scale": 0.0,
    "above_band_penalty_scale": 0.0,
    "above_band_ref_m": 9000.0,
    "turn_half_angle_deg": 35.0,
    "tight_nose_bonus": 0.04,
    "tight_nose_angle_deg": 8.0,
    "too_close_penalty": 0.12,
    "damage_scale": 35.0,
    "altitude_target_m": 7000.0,
    "altitude_band_m": 5500.0,
    "safe_altitude_m": 2800.0,
    "low_altitude_penalty": 1.8,
    "attitude_penalty_scale": 0.018,
    "win_reward": 160.0,
    "loss_reward": -190.0,
    "draw_reward": -10.0,
    "own_crash_penalty": -260.0,
    "target_crash_reward": -8.0,
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

    half_angle = max(float(reward_config.get("pursuit_half_angle_deg", 50.0)), 1.0)
    pursuit_range = max(float(reward_config.get("pursuit_range_m", 7000.0)), 1.0)
    nose_score = _clip01(1.0 - ata / half_angle)
    aspect_score = _clip01(1.0 - aa / 150.0)
    range_score = _clip01(1.0 - distance / pursuit_range)
    components["pursuit"] = float(reward_config.get("pursuit_scale", 0.22)) * (
        0.82 * nose_score + 0.18 * aspect_score
    ) * range_score

    band_min = float(reward_config.get("turn_band_min_m", 700.0))
    band_max = max(float(reward_config.get("turn_band_max_m", 2200.0)), band_min + 1.0)
    band_center = 0.5 * (band_min + band_max)
    band_half_width = 0.5 * (band_max - band_min)
    band_score = _clip01(1.0 - abs(distance - band_center) / band_half_width)
    turn_half_angle = max(float(reward_config.get("turn_half_angle_deg", 35.0)), 1.0)
    turn_score = _clip01(1.0 - ata / turn_half_angle)
    components["range_band"] = float(reward_config.get("range_band_scale", 0.0)) * band_score
    below_band_penalty = 0.0
    if distance < band_min:
        below_band_penalty = -float(
            reward_config.get("below_band_penalty_scale", 0.0)
        ) * _clip01((band_min - distance) / max(band_min, 1.0))
    above_band_penalty = 0.0
    if distance > band_max:
        above_ref = max(float(reward_config.get("above_band_ref_m", 9000.0)), band_max + 1.0)
        above_band_penalty = -float(
            reward_config.get("above_band_penalty_scale", 0.0)
        ) * _clip01((distance - band_max) / max(above_ref - band_max, 1.0))
    components["range_penalty"] = below_band_penalty + above_band_penalty
    components["postmerge_turn"] = (
        float(reward_config.get("turn_band_scale", 0.18)) * band_score * turn_score
    )
    if band_min <= distance <= band_max and ata <= float(reward_config.get("tight_nose_angle_deg", 8.0)):
        components["tight_nose"] = float(reward_config.get("tight_nose_bonus", 0.04))
    else:
        components["tight_nose"] = 0.0

    min_range = float(wez_config.get("min_range_m", 152.4))
    if distance < min_range:
        components["too_close"] = -float(reward_config.get("too_close_penalty", 0.12)) * (
            1.0 - distance / max(min_range, 1.0)
        ) * _clip01(ata / 90.0)
    else:
        components["too_close"] = 0.0

    components["damage"] = float(reward_config.get("damage_scale", 35.0)) * (
        float(target_damage) - float(ownship_damage)
    )

    altitude = float(ownship_state[StateIndex.ALT])
    target_alt = float(reward_config.get("altitude_target_m", 7000.0))
    altitude_band = max(float(reward_config.get("altitude_band_m", 5500.0)), 1.0)
    safe_altitude = max(float(reward_config.get("safe_altitude_m", 2800.0)), 1.0)
    altitude_hold = 0.012 * _clip01(1.0 - abs(altitude - target_alt) / altitude_band)
    low_altitude = 0.0
    if altitude < safe_altitude:
        low_altitude = -float(reward_config.get("low_altitude_penalty", 1.8)) * (
            1.0 - altitude / safe_altitude
        )
    components["altitude"] = altitude_hold + low_altitude

    roll = abs(float(ownship_state[StateIndex.ROLL]))
    pitch = abs(float(ownship_state[StateIndex.PITCH]))
    components["attitude"] = -float(reward_config.get("attitude_penalty_scale", 0.018)) * (
        (roll / 180.0) ** 2 + (pitch / 90.0) ** 2
    )

    terminal = 0.0
    if terminated or truncated:
        ownship_health = float(ownship_state[StateIndex.HEALTH])
        target_health = float(target_state[StateIndex.HEALTH])
        if end_condition in ("ownship altitude below min", "FDM Update Fail"):
            terminal = float(reward_config.get("own_crash_penalty", -260.0))
        elif end_condition == "target altitude below min":
            terminal = float(reward_config.get("target_crash_reward", -8.0))
        elif target_health <= 0.0 < ownship_health:
            terminal = float(reward_config.get("win_reward", 160.0))
        elif ownship_health <= 0.0 < target_health:
            terminal = float(reward_config.get("loss_reward", -190.0))
        else:
            terminal = float(reward_config.get("draw_reward", -10.0))
    components["terminal"] = terminal

    return float(sum(components.values())), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
