# -*- coding: utf-8 -*-
"""Attack-oriented reward after a survival warmup.

Use this only after the policy can already survive. It keeps basic safety
penalties, but shifts most reward pressure toward nose-on target geometry,
close range, damage, and avoiding passive timeout-only behavior.
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
    "survival_bonus": 0.001,
    "pursuit_scale": 1.0,
    "pursuit_half_angle_deg": 90.0,
    "pursuit_range_m": 10000.0,
    "close_range_bonus": 0.05,
    "damage_scale": 40.0,
    "safe_altitude_m": 2200.0,
    "low_altitude_penalty": 0.8,
    "attitude_penalty_scale": 0.02,
    "win_reward": 150.0,
    "loss_reward": -150.0,
    "draw_reward": -25.0,
    "crash_penalty": -180.0,
    "guard_fail_penalty": -100.0,
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
    half_angle = max(float(reward_config.get("pursuit_half_angle_deg", 90.0)), 1.0)
    pursuit_range = max(float(reward_config.get("pursuit_range_m", 10000.0)), 1.0)

    nose_score = _clip01(1.0 - ata / half_angle)
    tail_score = _clip01(1.0 - aa / 120.0)
    range_score = _clip01(1.0 - distance / pursuit_range)
    pursuit = float(reward_config.get("pursuit_scale", 1.0)) * (
        0.75 * nose_score + 0.25 * tail_score
    ) * range_score
    if distance < 1000.0 and ata < 30.0:
        pursuit += float(reward_config.get("close_range_bonus", 0.05))
    components["pursuit"] = pursuit

    components["damage"] = (
        float(reward_config.get("damage_scale", 40.0))
        * (float(target_damage) - float(ownship_damage))
    )

    altitude = float(ownship_state[StateIndex.ALT])
    safe_altitude = max(float(reward_config.get("safe_altitude_m", 2200.0)), 1.0)
    safety = 0.0
    if altitude < safe_altitude:
        safety -= float(reward_config.get("low_altitude_penalty", 0.8)) * (
            1.0 - altitude / safe_altitude
        )
    roll = abs(float(ownship_state[StateIndex.ROLL]))
    pitch = abs(float(ownship_state[StateIndex.PITCH]))
    safety -= float(reward_config.get("attitude_penalty_scale", 0.02)) * (
        (roll / 180.0) ** 2 + (pitch / 90.0) ** 2
    )
    components["safety"] = safety

    terminal = 0.0
    if terminated or truncated:
        ownship_health = float(ownship_state[StateIndex.HEALTH])
        target_health = float(target_state[StateIndex.HEALTH])
        if end_condition in ("ownship altitude below min", "FDM Update Fail"):
            terminal = float(reward_config.get("crash_penalty", -180.0))
        elif end_condition == "two circle headon guard fail":
            terminal = float(reward_config.get("guard_fail_penalty", -100.0))
        elif target_health <= 0.0 < ownship_health:
            terminal = float(reward_config.get("win_reward", 150.0))
        elif ownship_health <= 0.0 < target_health:
            terminal = float(reward_config.get("loss_reward", -150.0))
        else:
            terminal = float(reward_config.get("draw_reward", -25.0))
    components["terminal"] = terminal

    return float(sum(components.values())), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
