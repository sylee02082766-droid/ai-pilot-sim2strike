# -*- coding: utf-8 -*-
"""Lockheed-Martin-style CZ reward wrapper.

CZ here means the first low-level skill we want: approach and hold a useful
control-zone position before asking the agent to solve precise WEZ shooting.

This module intentionally reuses the already checked Table-I implementation in
student.my_reward_table1, and only changes the default component weights.
Per-experiment YAML files may still override any key under env_config.reward.
"""
from __future__ import annotations

from student.my_reward_table1 import compute_reward


MY_REWARD_CONFIG: dict[str, float | str] = {
    "mode": "lm_cz",

    # General time term. Kept small so geometry dominates.
    "step_penalty": -0.002,

    # CZ is mostly about position and closure. Exact one-degree WEZ holding is
    # left for the later CS shooter policy.
    "adverse_angle_scale": 0.0,
    "track_angle_scale": 0.10,
    "relative_position_scale": 1.00,
    "closure_scale": 0.50,
    "gunsnap_blue_scale": 0.35,
    "gunsnap_red_scale": 1.00,
    "deck_scale": 1.00,
    "too_close_scale": 1.00,

    "track_lambda": 1.0,
    "blue_beta": 3.0,
    "red_beta": 3.0,
    "deck_altitude_ft": 1300.0,
    "deck_transition_ft": 20.0,
    "damage_scale": 0.0,

    # Compatibility keys used by the training recorder.
    "pursuit_scale": 0.0,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 3000.0,
    "low_altitude_penalty": 0.1,

    # AI Pilot environment safety shaping. These are deliberately present in
    # the CZ default because our earlier Table-I-only attempts learned unstable
    # dives before learning useful geometry.
    "altitude_hold_scale": 0.15,
    "altitude_target_m": 7000.0,
    "altitude_band_m": 5000.0,
    "safe_altitude_m": 2600.0,
    "low_altitude_guard_scale": 2.0,
    "attitude_penalty_scale": 0.08,
    "speed_target_kcas": 320.0,
    "speed_band_kcas": 220.0,
    "speed_penalty_scale": 0.05,

    # Disabled for CZ; enabled in CS/AS shooting-school experiments.
    "wez_band_scale": 0.0,
    "wez_band_center_m": 520.0,
    "wez_band_half_width_m": 390.0,
    "wez_nose_half_angle_deg": 8.0,
    "wez_hold_bonus": 0.0,

    "win_reward": 100.0,
    "loss_reward": -100.0,
    "draw_reward": -30.0,
    "own_crash_penalty": -150.0,
    "target_crash_reward": -5.0,
    "guard_fail_penalty": -50.0,
}


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
