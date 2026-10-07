# -*- coding: utf-8 -*-
"""WEZ-band reward for the second dogfight entry stage.

The stage80/entry40 policy can already survive and pass close to the target.
This reward tries to convert that close pass into a useful WEZ setup by
rewarding the official 500-3000 ft range band and nose-on geometry, while
penalizing undershooting inside the minimum range.
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
    "pursuit_scale": 0.26,
    "pursuit_half_angle_deg": 55.0,
    "pursuit_range_m": 6500.0,
    "band_scale": 0.55,
    "band_center_m": 520.0,
    "band_half_width_m": 390.0,
    "nose_band_half_angle_deg": 12.0,
    "near_wez_bonus": 0.10,
    "too_close_penalty": 0.30,
    "damage_scale": 45.0,
    "altitude_target_m": 7000.0,
    "altitude_band_m": 5200.0,
    "safe_altitude_m": 2600.0,
    "low_altitude_penalty": 1.4,
    "attitude_penalty_scale": 0.02,
    "win_reward": 170.0,
    "loss_reward": -190.0,
    "draw_reward": -10.0,
    "own_crash_penalty": -240.0,
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

    half_angle = max(float(reward_config.get("pursuit_half_angle_deg", 55.0)), 1.0)
    pursuit_range = max(float(reward_config.get("pursuit_range_m", 6500.0)), 1.0)
    nose_score = _clip01(1.0 - ata / half_angle)
    aspect_score = _clip01(1.0 - aa / 140.0)
    range_score = _clip01(1.0 - distance / pursuit_range)
    components["pursuit"] = float(reward_config.get("pursuit_scale", 0.26)) * (
        0.78 * nose_score + 0.22 * aspect_score
    ) * range_score

    band_center = float(reward_config.get("band_center_m", 520.0))
    band_half_width = max(float(reward_config.get("band_half_width_m", 390.0)), 1.0)
    band_score = _clip01(1.0 - abs(distance - band_center) / band_half_width)
    nose_band_half_angle = max(float(reward_config.get("nose_band_half_angle_deg", 12.0)), 1.0)
    nose_band_score = _clip01(1.0 - ata / nose_band_half_angle)
    components["wez_band"] = (
        float(reward_config.get("band_scale", 0.55))
        * band_score
        * (0.35 + 0.65 * nose_band_score)
    )

    min_range = float(wez_config.get("min_range_m", 152.4))
    max_range = float(wez_config.get("max_range_m", 914.4))
    if min_range <= distance <= max_range and ata <= 4.0:
        components["near_wez"] = float(reward_config.get("near_wez_bonus", 0.10))
    else:
        components["near_wez"] = 0.0

    if distance < min_range:
        components["too_close"] = -float(reward_config.get("too_close_penalty", 0.30)) * (
            1.0 - distance / max(min_range, 1.0)
        )
    else:
        components["too_close"] = 0.0

    components["damage"] = float(reward_config.get("damage_scale", 45.0)) * (
        float(target_damage) - float(ownship_damage)
    )

    altitude = float(ownship_state[StateIndex.ALT])
    target_alt = float(reward_config.get("altitude_target_m", 7000.0))
    altitude_band = max(float(reward_config.get("altitude_band_m", 5200.0)), 1.0)
    safe_altitude = max(float(reward_config.get("safe_altitude_m", 2600.0)), 1.0)
    altitude_hold = 0.010 * _clip01(1.0 - abs(altitude - target_alt) / altitude_band)
    low_altitude = 0.0
    if altitude < safe_altitude:
        low_altitude = -float(reward_config.get("low_altitude_penalty", 1.4)) * (
            1.0 - altitude / safe_altitude
        )
    components["altitude"] = altitude_hold + low_altitude

    roll = abs(float(ownship_state[StateIndex.ROLL]))
    pitch = abs(float(ownship_state[StateIndex.PITCH]))
    components["attitude"] = -float(reward_config.get("attitude_penalty_scale", 0.02)) * (
        (roll / 180.0) ** 2 + (pitch / 90.0) ** 2
    )

    terminal = 0.0
    if terminated or truncated:
        ownship_health = float(ownship_state[StateIndex.HEALTH])
        target_health = float(target_state[StateIndex.HEALTH])
        if end_condition in ("ownship altitude below min", "FDM Update Fail"):
            terminal = float(reward_config.get("own_crash_penalty", -240.0))
        elif end_condition == "target altitude below min":
            terminal = float(reward_config.get("target_crash_reward", -8.0))
        elif target_health <= 0.0 < ownship_health:
            terminal = float(reward_config.get("win_reward", 170.0))
        elif ownship_health <= 0.0 < target_health:
            terminal = float(reward_config.get("loss_reward", -190.0))
        else:
            terminal = float(reward_config.get("draw_reward", -10.0))
    components["terminal"] = terminal

    return float(sum(components.values())), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
