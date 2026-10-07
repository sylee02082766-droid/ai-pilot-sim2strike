# -*- coding: utf-8 -*-
"""Lockheed-Martin-style CS reward wrapper.

CS here means the second low-level skill after CZ: keep a conservative shooting
setup near the official WEZ band. It starts from a CZ policy, so the weights
still preserve safety and positional terms, but the WEZ band and blue gunsnap
terms are enabled.
"""
from __future__ import annotations

from student.my_reward_table1 import compute_reward


MY_REWARD_CONFIG: dict[str, float | str] = {
    "mode": "lm_cs",

    "step_penalty": -0.001,
    "survival_bonus": 0.003,

    # Keep enough CZ shaping so the policy does not forget how to arrive.
    "adverse_angle_scale": 0.02,
    "track_angle_scale": 0.16,
    "relative_position_scale": 0.45,
    "closure_scale": 0.04,

    # CS starts emphasizing a stable nose-on WEZ setup.
    "gunsnap_blue_scale": 0.45,
    "gunsnap_red_scale": 1.05,
    "wez_band_scale": 0.35,
    "wez_band_center_m": 520.0,
    "wez_band_half_width_m": 390.0,
    "wez_nose_half_angle_deg": 8.0,
    "wez_hold_bonus": 0.015,

    "deck_scale": 4.30,
    "too_close_scale": 1.55,
    "track_lambda": 1.0,
    "blue_beta": 3.0,
    "red_beta": 3.0,
    "deck_altitude_ft": 6200.0,
    "deck_transition_ft": 1000.0,
    "damage_scale": 0.0,

    # Compatibility keys used by dashboards/older experiments.
    "pursuit_scale": 0.0,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 3000.0,
    "low_altitude_penalty": 0.1,

    "altitude_hold_scale": 0.08,
    "altitude_target_m": 7000.0,
    "altitude_band_m": 5200.0,
    "safe_altitude_m": 4300.0,
    "low_altitude_guard_scale": 5.20,
    "attitude_penalty_scale": 0.16,
    "speed_target_kcas": 320.0,
    "speed_band_kcas": 220.0,
    "speed_penalty_scale": 0.08,

    "win_reward": 140.0,
    "loss_reward": -190.0,
    "draw_reward": -12.0,
    "own_crash_penalty": -540.0,
    "target_crash_reward": -8.0,
    "guard_fail_penalty": -130.0,
}


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
