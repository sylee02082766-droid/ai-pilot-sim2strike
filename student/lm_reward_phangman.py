# -*- coding: utf-8 -*-
"""PHANG-MAN-style reward components adapted to the AIP challenge API.

This module is intentionally narrower than ``student.my_reward_table1``.
It follows the Lockheed Martin PHANG-MAN paper structure as closely as the
public release state allows:

* CZ: relative position + closure + blue/red gunsnap + deck + too-close
* AS: track angle + blue/red gunsnap + deck
* CS: track angle + blue/red gunsnap + deck, with a flatter range reward
* selector: sparse WEZ damage differential + track-alignment signal

Known adaptation limits:
* The paper's environment exposes velocity, acceleration, alpha/beta, angular
  rates, thrust, and control-surface states. The challenge state schema does
  not, so closure is estimated from KCAS, pitch, yaw, and line-of-sight.
* The paper does not publish exact source code for every shaping curve. The
  formulas that are visible in the paper/table are used directly where possible;
  prose-only pieces are implemented as smooth gates with configurable constants.
* Distances in the paper are feet. The challenge environment stores position
  and WEZ distances in meters, so all paper distance constants are converted.
* AIP-specific stability guards are available, but disabled by default. They are
  used only by YAML files that explicitly set their scales above zero.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC, Path.cwd(), Path.cwd() / "src"):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from dogfight.sim.state_schema import StateIndex
except Exception:  # pragma: no cover
    from enum import IntEnum

    class StateIndex(IntEnum):
        N = 0
        E = 1
        D = 2
        ROLL = 3
        PITCH = 4
        YAW = 5
        KCAS = 12
        FUEL = 23
        SIM_TIME = 41
        LAT = 42
        LON = 43
        ALT = 44
        HEALTH = 45


FEET_TO_METER = 0.3048
KNOT_TO_METER_SEC = 0.51444


MY_REWARD_CONFIG: dict[str, float | str | bool] = {
    "mode": "lm_phangman",
    "policy_kind": "cz",

    # Global scalar applied after the policy-specific paper reward is built.
    "reward_scale": 1.0,
    "step_penalty": 0.0,

    # Component weights default to the paper formulas. YAML files can override
    # them for ablations without changing source code.
    "relative_position_scale": 1.0,
    "closure_scale": 1.0,
    "track_angle_scale": 1.0,
    "gunsnap_blue_scale": 1.0,
    "gunsnap_red_scale": 1.0,
    "deck_scale": 1.0,
    "too_close_scale": 1.0,

    # Paper/Table-I constants and smooth-gate widths.
    "track_lambda": 1.0,
    "blue_beta": 3.0,
    "red_beta": 3.0,
    "wez_min_ft": 500.0,
    "wez_max_ft": 3000.0,
    "wez_half_angle_deg": 1.0,
    "angle_gate_transition_deg": 0.10,
    "deck_altitude_ft": 1000.0,
    "deck_transition_ft": 50.0,
    "too_close_ft": 800.0,
    "too_close_transition_ft": 50.0,

    # AS rewards closer shots more strongly. CS keeps a flatter value inside
    # the valid WEZ band, matching the paper's "maintain scoring position" idea.
    "gunsnap_profile": "cz",  # cz | as | cs
    "cs_plateau_value": 1.0,

    # Selector reward: environment WEZ damage plus a per-step track signal.
    "selector_damage_scale": 1.0,
    "selector_track_scale": 0.25,

    # AIP/JSBSim stability guards. These are not PHANG-MAN paper terms, so all
    # scales default to zero and must be enabled explicitly by an experiment.
    "altitude_hold_scale": 0.0,
    "altitude_target_m": 7000.0,
    "altitude_band_m": 5000.0,
    "low_altitude_guard_scale": 0.0,
    "safe_altitude_m": 3000.0,
    "attitude_penalty_scale": 0.0,
    "speed_penalty_scale": 0.0,
    "speed_target_kcas": 300.0,
    "speed_band_kcas": 180.0,

    # Terminal values are disabled by default to stay close to the paper, which
    # discusses sparse WEZ damage and shaped geometry rather than outcome bonuses.
    "terminal_enabled": False,
    "win_reward": 0.0,
    "loss_reward": 0.0,
    "draw_reward": 0.0,
    "own_crash_penalty": 0.0,
    "target_crash_reward": 0.0,
    "guard_fail_penalty": 0.0,
}


def _ft(value: float) -> float:
    return float(value) * FEET_TO_METER


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _angle_abs_180(value: float) -> float:
    return float(np.clip(abs(float(value)), 0.0, 180.0))


def _sigmoid(x: float, alpha: float, x0: float) -> float:
    z = float(alpha) * (float(x) - float(x0))
    if z >= 60.0:
        return 1.0
    if z <= -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _falling_gate(x: float, center: float, transition: float) -> float:
    transition = max(float(transition), 1.0e-6)
    return 1.0 - _sigmoid(float(x), 1.0 / transition, float(center))


def _rising_gate(x: float, center: float, transition: float) -> float:
    transition = max(float(transition), 1.0e-6)
    return _sigmoid(float(x), 1.0 / transition, float(center))


def _range_gate(distance_m: float, min_m: float, max_m: float, transition_m: float) -> float:
    return _rising_gate(distance_m, min_m, transition_m) * _falling_gate(
        distance_m, max_m, transition_m
    )


def _aircraft_velocity_ned(state: Any) -> np.ndarray:
    speed_mps = float(state[StateIndex.KCAS]) * KNOT_TO_METER_SEC
    pitch_rad = math.radians(float(state[StateIndex.PITCH]))
    yaw_rad = math.radians(float(state[StateIndex.YAW]))
    cos_pitch = math.cos(pitch_rad)
    return np.array(
        [
            speed_mps * cos_pitch * math.cos(yaw_rad),
            speed_mps * cos_pitch * math.sin(yaw_rad),
            -speed_mps * math.sin(pitch_rad),
        ],
        dtype=np.float64,
    )


def _closure_rate_mps(ownship_state: Any, target_state: Any, distance_m: float) -> float:
    if distance_m <= 1.0e-6:
        return 0.0
    los_unit = (
        np.asarray(target_state[0:3], dtype=np.float64)
        - np.asarray(ownship_state[0:3], dtype=np.float64)
    ) / float(distance_m)
    relative_velocity = _aircraft_velocity_ned(ownship_state) - _aircraft_velocity_ned(
        target_state
    )
    return float(np.dot(relative_velocity, los_unit))


def _gamma_blue_table(distance_m: float, beta_blue: float) -> float:
    """Paper/Table-I blue gunsnap distance shaping."""
    if distance_m < _ft(1950.0):
        return float(beta_blue) * _sigmoid(distance_m, 1.0 / _ft(50.0), _ft(1000.0))
    return float(beta_blue) * (
        1.0 - _sigmoid(distance_m, 1.0 / _ft(50.0), _ft(2900.0))
    )


def _gamma_red_table(distance_m: float, beta_red: float) -> float:
    """Paper/Table-I red gunsnap distance shaping; caller applies penalty sign."""
    if distance_m < _ft(2250.0):
        return float(beta_red) * _sigmoid(distance_m, 1.0 / _ft(35.0), _ft(400.0))
    return float(beta_red) * (
        1.0 - _sigmoid(distance_m, 1.0 / _ft(200.0), _ft(4100.0))
    )


def _wez_damage_profile(distance_m: float, min_m: float, max_m: float) -> float:
    if max_m <= min_m:
        return 0.0
    if not (min_m <= distance_m <= max_m):
        return 0.0
    return _clip01((max_m - distance_m) / (max_m - min_m))


def _gunsnap_gain(
    distance_m: float,
    config: dict,
    profile: str,
    is_blue: bool,
) -> float:
    min_m = _ft(float(config.get("wez_min_ft", 500.0)))
    max_m = _ft(float(config.get("wez_max_ft", 3000.0)))
    transition_m = _ft(float(config.get("too_close_transition_ft", 50.0)))
    beta = float(config.get("blue_beta" if is_blue else "red_beta", 3.0))

    if profile == "as":
        # Aggressive shots prefer the closer end of the valid WEZ band.
        return beta * _wez_damage_profile(distance_m, min_m, max_m)
    if profile == "cs":
        # Conservative shots keep reward nearly flat throughout the range gate.
        plateau = float(config.get("cs_plateau_value", 1.0))
        return beta * plateau * _range_gate(distance_m, min_m, max_m, transition_m)

    return (
        _gamma_blue_table(distance_m, beta)
        if is_blue
        else _gamma_red_table(distance_m, beta)
    )


def _base_components(
    ownship_state: Any,
    target_state: Any,
    ownship_damage: float,
    target_damage: float,
    geo_info: Any,
    wez_config: dict,
    reward_config: dict,
) -> dict[str, float]:
    distance_m = float(geo_info._get_distance(ownship_state, target_state))
    track_angle_deg = _angle_abs_180(
        geo_info._get_antenna_train_angle(ownship_state, target_state, False)
    )
    adverse_angle_deg = _angle_abs_180(
        geo_info._get_aspect_angle(ownship_state, target_state, False)
    )
    red_track_angle_deg = _angle_abs_180(
        geo_info._get_antenna_train_angle(target_state, ownship_state, False)
    )

    theta_bar = track_angle_deg / 180.0
    phi_bar = adverse_angle_deg / 180.0
    red_theta_bar = red_track_angle_deg / 180.0
    closure_mps = _closure_rate_mps(ownship_state, target_state, distance_m)
    altitude_m = float(ownship_state[StateIndex.ALT])

    track_lambda = float(reward_config.get("track_lambda", 1.0))
    angle_gate_deg = float(reward_config.get("wez_half_angle_deg", 1.0))
    angle_transition_deg = float(reward_config.get("angle_gate_transition_deg", 0.10))
    angle_gate_blue = _falling_gate(track_angle_deg, angle_gate_deg, angle_transition_deg)
    angle_gate_red = _falling_gate(red_track_angle_deg, angle_gate_deg, angle_transition_deg)

    s_phi_half = _sigmoid(phi_bar, 18.0, 0.5)
    relative_position = (theta_bar - 2.0) * s_phi_half - theta_bar + 1.0
    closure = (
        (closure_mps / _ft(500.0))
        * (1.0 - s_phi_half)
        * _sigmoid(distance_m, 1.0 / _ft(500.0), _ft(2900.0))
    )
    track_angle = -(theta_bar**track_lambda)

    profile = str(reward_config.get("gunsnap_profile", "cz")).strip().lower()
    gunsnap_blue = _gunsnap_gain(distance_m, reward_config, profile, True) * angle_gate_blue
    gunsnap_red = -_gunsnap_gain(distance_m, reward_config, profile, False) * angle_gate_red

    deck_altitude_m = _ft(float(reward_config.get("deck_altitude_ft", 1000.0)))
    deck_transition_m = _ft(float(reward_config.get("deck_transition_ft", 50.0)))
    deck = -4.0 * (1.0 - _rising_gate(altitude_m, deck_altitude_m, deck_transition_m))

    too_close_m = _ft(float(reward_config.get("too_close_ft", 800.0)))
    too_close_transition_m = _ft(float(reward_config.get("too_close_transition_ft", 50.0)))
    too_close = -2.0 * (1.0 - s_phi_half) * _falling_gate(
        distance_m, too_close_m, too_close_transition_m
    )

    min_wez_m = float(wez_config.get("min_range_m", _ft(500.0)))
    max_wez_m = float(wez_config.get("max_range_m", _ft(3000.0)))
    official_damage_shape = _wez_damage_profile(distance_m, min_wez_m, max_wez_m)
    official_blue_wez = official_damage_shape if angle_gate_blue > 0.5 else 0.0
    official_red_wez = official_damage_shape if angle_gate_red > 0.5 else 0.0

    return {
        "relative_position_raw": float(relative_position),
        "closure_raw": float(closure),
        "track_angle_raw": float(track_angle),
        "gunsnap_blue_raw": float(gunsnap_blue),
        "gunsnap_red_raw": float(gunsnap_red),
        "deck_raw": float(deck),
        "too_close_raw": float(too_close),
        "selector_track_raw": float(1.0 - theta_bar),
        "selector_damage_raw": float(target_damage) - float(ownship_damage),
        "official_blue_wez_shape": float(official_blue_wez),
        "official_red_wez_shape": float(official_red_wez),
        "distance_m": distance_m,
        "track_angle_deg": track_angle_deg,
        "adverse_angle_deg": adverse_angle_deg,
        "red_track_angle_deg": red_track_angle_deg,
    }


def _scaled(name: str, raw: dict[str, float], config: dict) -> float:
    return float(config.get(f"{name}_scale", 1.0)) * raw[f"{name}_raw"]


def _terminal_reward(
    ownship_state: Any,
    target_state: Any,
    reward_config: dict,
    terminated: bool,
    truncated: bool,
    end_condition: str,
) -> float:
    if not bool(reward_config.get("terminal_enabled", False)):
        return 0.0
    if not (terminated or truncated):
        return 0.0

    ownship_health = float(ownship_state[StateIndex.HEALTH])
    target_health = float(target_state[StateIndex.HEALTH])
    if end_condition in ("ownship altitude below min", "FDM Update Fail"):
        return float(reward_config.get("own_crash_penalty", 0.0))
    if end_condition == "target altitude below min":
        return float(reward_config.get("target_crash_reward", 0.0))
    if end_condition == "two circle headon guard fail":
        return float(reward_config.get("guard_fail_penalty", 0.0))
    if target_health <= 0.0 < ownship_health:
        return float(reward_config.get("win_reward", 0.0))
    if ownship_health <= 0.0 < target_health:
        return float(reward_config.get("loss_reward", 0.0))
    return float(reward_config.get("draw_reward", 0.0))


def _aip_stability_components(ownship_state: Any, reward_config: dict) -> dict[str, float]:
    altitude_m = float(ownship_state[StateIndex.ALT])
    altitude_target_m = float(reward_config.get("altitude_target_m", 7000.0))
    altitude_band_m = max(float(reward_config.get("altitude_band_m", 5000.0)), 1.0)
    altitude_error = _clip01(abs(altitude_m - altitude_target_m) / altitude_band_m)
    altitude_hold = -float(reward_config.get("altitude_hold_scale", 0.0)) * (
        altitude_error**2
    )

    safe_altitude_m = max(float(reward_config.get("safe_altitude_m", 3000.0)), 1.0)
    low_altitude = 0.0
    if altitude_m < safe_altitude_m:
        low_altitude = -float(reward_config.get("low_altitude_guard_scale", 0.0)) * (
            (safe_altitude_m - altitude_m) / safe_altitude_m
        ) ** 2

    roll_deg = abs(float(ownship_state[StateIndex.ROLL]))
    pitch_deg = abs(float(ownship_state[StateIndex.PITCH]))
    attitude = -float(reward_config.get("attitude_penalty_scale", 0.0)) * (
        (roll_deg / 180.0) ** 2 + (pitch_deg / 90.0) ** 2
    )

    speed_kcas = float(ownship_state[StateIndex.KCAS])
    speed_target_kcas = float(reward_config.get("speed_target_kcas", 300.0))
    speed_band_kcas = max(float(reward_config.get("speed_band_kcas", 180.0)), 1.0)
    speed_error = _clip01(abs(speed_kcas - speed_target_kcas) / speed_band_kcas)
    speed = -float(reward_config.get("speed_penalty_scale", 0.0)) * (
        speed_error**2
    )

    return {
        "altitude_hold": float(altitude_hold),
        "low_altitude_guard": float(low_altitude),
        "attitude_guard": float(attitude),
        "speed_guard": float(speed),
    }


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
    raw = _base_components(
        ownship_state,
        target_state,
        ownship_damage,
        target_damage,
        geo_info,
        wez_config,
        reward_config,
    )
    policy_kind = str(reward_config.get("policy_kind", "cz")).strip().lower()
    components: dict[str, float] = {
        "step": float(reward_config.get("step_penalty", 0.0)),
        "relative_position": 0.0,
        "closure": 0.0,
        "track_angle": 0.0,
        "gunsnap_blue": 0.0,
        "gunsnap_red": 0.0,
        "deck": 0.0,
        "too_close": 0.0,
        "selector_damage": 0.0,
        "selector_track": 0.0,
        "altitude_hold": 0.0,
        "low_altitude_guard": 0.0,
        "attitude_guard": 0.0,
        "speed_guard": 0.0,
    }

    if policy_kind == "cz":
        components["relative_position"] = _scaled("relative_position", raw, reward_config)
        components["closure"] = _scaled("closure", raw, reward_config)
        components["gunsnap_blue"] = _scaled("gunsnap_blue", raw, reward_config)
        components["gunsnap_red"] = _scaled("gunsnap_red", raw, reward_config)
        components["deck"] = _scaled("deck", raw, reward_config)
        components["too_close"] = _scaled("too_close", raw, reward_config)
    elif policy_kind in {"as", "cs"}:
        components["track_angle"] = _scaled("track_angle", raw, reward_config)
        components["gunsnap_blue"] = _scaled("gunsnap_blue", raw, reward_config)
        components["gunsnap_red"] = _scaled("gunsnap_red", raw, reward_config)
        components["deck"] = _scaled("deck", raw, reward_config)
    elif policy_kind == "selector":
        components["selector_damage"] = float(
            reward_config.get("selector_damage_scale", 1.0)
        ) * raw["selector_damage_raw"]
        components["selector_track"] = float(
            reward_config.get("selector_track_scale", 0.25)
        ) * raw["selector_track_raw"]
    else:
        raise ValueError(
            "policy_kind must be one of 'cz', 'as', 'cs', or 'selector', "
            f"got {policy_kind!r}"
        )

    components.update(_aip_stability_components(ownship_state, reward_config))

    components["terminal"] = _terminal_reward(
        ownship_state,
        target_state,
        reward_config,
        terminated,
        truncated,
        end_condition,
    )

    # Diagnostics are intentionally zero-weight so they appear in callback
    # component logs without changing the optimization objective.
    components["diag_official_blue_wez"] = 0.0 * raw["official_blue_wez_shape"]
    components["diag_official_red_wez"] = 0.0 * raw["official_red_wez_shape"]

    reward_scale = float(reward_config.get("reward_scale", 1.0))
    total = reward_scale * float(sum(components.values()))
    return total, components


def describe_reward(reward_config: dict, wez_config: dict) -> dict:
    policy_kind = str(reward_config.get("policy_kind", "cz")).strip().lower()
    if policy_kind == "cz":
        formula = (
            "CZ = relative_position + closure + gunsnap_blue + "
            "gunsnap_red + deck + too_close"
        )
    elif policy_kind in {"as", "cs"}:
        formula = "AS/CS = track_angle + gunsnap_blue + gunsnap_red + deck"
    elif policy_kind == "selector":
        formula = "Selector = WEZ damage differential + track alignment"
    else:
        formula = "Unknown PHANG-MAN policy kind"

    return {
        "description": (
            "Lockheed Martin PHANG-MAN paper-close reward adapted to the "
            "AI Pilot challenge state/API. " + formula
        ),
        "policy_kind": policy_kind,
        "formula": formula,
        "step_penalty": reward_config.get("step_penalty", 0.0),
        "relative_position_scale": reward_config.get("relative_position_scale", 0.0),
        "closure_scale": reward_config.get("closure_scale", 0.0),
        "track_angle_scale": reward_config.get("track_angle_scale", 0.0),
        "gunsnap_blue_scale": reward_config.get("gunsnap_blue_scale", 0.0),
        "gunsnap_red_scale": reward_config.get("gunsnap_red_scale", 0.0),
        "deck_scale": reward_config.get("deck_scale", 0.0),
        "too_close_scale": reward_config.get("too_close_scale", 0.0),
        "selector_damage_scale": reward_config.get("selector_damage_scale", 0.0),
        "selector_track_scale": reward_config.get("selector_track_scale", 0.0),
        "aip_stability_guards": {
            "altitude_hold_scale": reward_config.get("altitude_hold_scale", 0.0),
            "altitude_target_m": reward_config.get("altitude_target_m", 7000.0),
            "altitude_band_m": reward_config.get("altitude_band_m", 5000.0),
            "low_altitude_guard_scale": reward_config.get(
                "low_altitude_guard_scale", 0.0
            ),
            "safe_altitude_m": reward_config.get("safe_altitude_m", 3000.0),
            "attitude_penalty_scale": reward_config.get(
                "attitude_penalty_scale", 0.0
            ),
            "speed_penalty_scale": reward_config.get("speed_penalty_scale", 0.0),
            "speed_target_kcas": reward_config.get("speed_target_kcas", 300.0),
            "speed_band_kcas": reward_config.get("speed_band_kcas", 180.0),
        },
        "wez": {
            "angle_deg": wez_config.get("angle_deg"),
            "min_range_m": wez_config.get("min_range_m"),
            "max_range_m": wez_config.get("max_range_m"),
            "paper_min_ft": reward_config.get("wez_min_ft", 500.0),
            "paper_max_ft": reward_config.get("wez_max_ft", 3000.0),
            "paper_half_angle_deg": reward_config.get("wez_half_angle_deg", 1.0),
        },
    }


__all__ = ["MY_REWARD_CONFIG", "compute_reward", "describe_reward"]
