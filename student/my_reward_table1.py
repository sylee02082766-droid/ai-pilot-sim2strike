# -*- coding: utf-8 -*-
"""Student reward: Table-I-style air-combat shaping.

This reward adapts the component equations in Table I of
"Hierarchical Reinforcement Learning for Air Combat at DARPA's
AlphaDogfight Trials" to the public AI Pilot Top Gun Challenge API.

Required contract:
  - MY_REWARD_CONFIG must be a dict.
  - compute_reward(...) must return (total_reward: float, components: dict).
  - Each item in components is recorded as ep_reward_<name> by the callbacks.

Implementation notes:
  * The paper's distances are in feet. The challenge environment stores NED
    position and WEZ ranges in meters, so all Table I distance constants are
    converted to meters here.
  * The API does not pass previous distance, so closure rate is approximated
    from the two aircraft's KCAS, pitch, yaw, and current line-of-sight vector.
  * The paper's prose says r_gunsnap(red) is a penalty when the opponent has a
    gun solution. In this release, ownship aspect angle is not a reliable proxy
    for that condition, so the opponent's ATA against us is computed directly.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# Also support quick checks from the release root after this file is copied into
# student/my_reward.py.
CWD = Path.cwd()
for path in (ROOT, SRC, CWD, CWD / "src"):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from dogfight.sim.state_schema import StateIndex
except Exception:  # pragma: no cover - fallback for standalone lint/tests
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


FEET_TO_METER = 0.30480
KNOT_TO_METER_SEC = 0.51444


MY_REWARD_CONFIG: dict[str, float | str] = {
    # Marker used only by this student reward module.
    "mode": "table1_adapted",

    # Small framework/time-efficiency term retained from the template.
    "step_penalty": -0.002,

    # Table I component weights. Set a component scale to 0.0 to disable it.
    "adverse_angle_scale": 0.25,      # r_phi_a
    "track_angle_scale": 0.25,        # r_theta_t
    "relative_position_scale": 1.0,   # r_rel_pos
    "closure_scale": 0.5,             # r_closure
    "gunsnap_blue_scale": 1.0,        # r_gunsnap(blue)
    "gunsnap_red_scale": 1.0,         # r_gunsnap(red), penalty
    "deck_scale": 1.0,                # r_deck
    "too_close_scale": 1.0,           # r_too_close

    # Parameters used by Table I / paper policy-selector style reward.
    "track_lambda": 1.0,
    "blue_beta": 3.0,
    "red_beta": 3.0,
    "deck_altitude_ft": 1300.0,
    "deck_transition_ft": 20.0,

    # Optional environment damage differential. Kept at zero because Table I
    # already supplies a dense WEZ/gunsnap signal; change only after testing.
    "damage_scale": 0.0,

    # Keys below are kept for compatibility with the release training recorder
    # and default reward description utility.
    "pursuit_scale": 0.0,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 3000.0,
    "low_altitude_penalty": 0.1,

    # Optional AI Pilot safety shaping. These are disabled by default so the
    # module can still reproduce the Table-I-style reward, then enabled from a
    # YAML when the aircraft needs a stronger anti-crash curriculum.
    "altitude_hold_scale": 0.0,
    "altitude_target_m": 7000.0,
    "altitude_band_m": 5000.0,
    "safe_altitude_m": 2600.0,
    "low_altitude_guard_scale": 0.0,
    "attitude_penalty_scale": 0.0,
    "nose_down_guard_scale": 0.0,
    "nose_down_pitch_ref_deg": 30.0,
    "descent_rate_guard_scale": 0.0,
    "descent_rate_ref_mps": 80.0,
    "low_altitude_roll_guard_scale": 0.0,
    "roll_guard_ref_deg": 90.0,
    "attitude_envelope_guard_scale": 0.0,
    "pitch_envelope_ref_deg": 35.0,
    "roll_envelope_ref_deg": 120.0,
    "action_pitch_down_guard_scale": 0.0,
    "action_pitch_guard_scale": 0.0,
    "action_roll_guard_scale": 0.0,
    "action_roll_global_guard_scale": 0.0,
    "action_guard_ref": 1.0,
    "speed_target_kcas": 320.0,
    "speed_band_kcas": 220.0,
    "speed_penalty_scale": 0.0,
    "wez_band_scale": 0.0,
    "wez_band_center_m": 520.0,
    "wez_band_half_width_m": 390.0,
    "wez_nose_half_angle_deg": 8.0,
    "wez_hold_bonus": 0.0,
    "official_wez_aim_scale": 0.0,
    "official_wez_aim_half_angle_deg": 2.5,
    "precision_aim_scale": 0.0,
    "precision_aim_sigma_deg": 5.0,
    "precision_aim_power": 1.0,
    "reacquire_scale": 0.0,
    "reacquire_window_steps": 60.0,
    "reacquire_range_ref_m": 180.0,
    "reacquire_angle_ref_deg": 8.0,

    # Terminal rewards retained from the challenge template/config.
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "draw_reward": -30.0,
    "own_crash_penalty": -150.0,
    "target_crash_reward": -5.0,
    "guard_fail_penalty": -50.0,
}


def _ft(value: float) -> float:
    """Convert feet to meters."""
    return float(value) * FEET_TO_METER


def _sigmoid(x: float, alpha: float, x0: float) -> float:
    """Numerically stable logistic S(x, alpha, x0)."""
    z = float(alpha) * (float(x) - float(x0))
    if z >= 60.0:
        return 1.0
    if z <= -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _clip_angle_180(value: float) -> float:
    return float(np.clip(abs(float(value)), 0.0, 180.0))


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _aircraft_velocity_ned(state: Any) -> np.ndarray:
    """Approximate NED velocity from KCAS, pitch, and yaw.

    The public state schema does not expose true velocity vector components,
    but it does expose KCAS and Euler attitude. This approximation is good
    enough to provide the sign and rough magnitude needed by r_closure.
    """
    speed_mps = float(state[StateIndex.KCAS]) * KNOT_TO_METER_SEC
    pitch = math.radians(float(state[StateIndex.PITCH]))
    yaw = math.radians(float(state[StateIndex.YAW]))

    cos_pitch = math.cos(pitch)
    return np.array(
        [
            speed_mps * cos_pitch * math.cos(yaw),
            speed_mps * cos_pitch * math.sin(yaw),
            -speed_mps * math.sin(pitch),
        ],
        dtype=np.float64,
    )


def _closure_rate_mps(ownship_state: Any, target_state: Any, distance_m: float) -> float:
    """Return positive closure rate when distance is decreasing."""
    if distance_m <= 1.0e-6:
        return 0.0
    los_unit = (
        np.asarray(target_state[0:3], dtype=np.float64)
        - np.asarray(ownship_state[0:3], dtype=np.float64)
    ) / float(distance_m)
    relative_velocity = _aircraft_velocity_ned(ownship_state) - _aircraft_velocity_ned(target_state)
    return float(np.dot(relative_velocity, los_unit))


def _gamma_blue(distance_m: float, beta_blue: float) -> float:
    """Table I Eq. (16), with feet constants converted to meters."""
    if distance_m < _ft(1950.0):
        return beta_blue * _sigmoid(distance_m, 1.0 / _ft(50.0), _ft(1000.0))
    return beta_blue * (1.0 - _sigmoid(distance_m, 1.0 / _ft(50.0), _ft(2900.0)))


def _gamma_red(distance_m: float, beta_red: float) -> float:
    """Table I Eq. (17), with positive beta; caller applies penalty sign."""
    if distance_m < _ft(2250.0):
        return beta_red * _sigmoid(distance_m, 1.0 / _ft(35.0), _ft(400.0))
    return beta_red * (1.0 - _sigmoid(distance_m, 1.0 / _ft(200.0), _ft(4100.0)))


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
    *,
    action=None,
    context=None,
) -> tuple[float, dict]:
    """Compute Table-I-style shaped reward.

    Returned components are intentionally fine-grained so TensorBoard/RLlib
    callbacks can show which shaping terms dominate training.
    """
    distance_m = float(geo_info._get_distance(ownship_state, target_state))
    track_angle_deg = _clip_angle_180(
        geo_info._get_antenna_train_angle(ownship_state, target_state, False)
    )
    adverse_angle_deg = _clip_angle_180(
        geo_info._get_aspect_angle(ownship_state, target_state, False)
    )
    red_track_angle_deg = _clip_angle_180(
        geo_info._get_antenna_train_angle(target_state, ownship_state, False)
    )
    theta_bar = track_angle_deg / 180.0
    phi_bar = adverse_angle_deg / 180.0
    closure_mps = _closure_rate_mps(ownship_state, target_state, distance_m)
    altitude_m = float(ownship_state[StateIndex.ALT])

    track_lambda = float(reward_config.get("track_lambda", 1.0))
    beta_blue = float(reward_config.get("blue_beta", 3.0))
    beta_red = float(reward_config.get("red_beta", 3.0))

    s_phi_half = _sigmoid(phi_bar, 18.0, 0.5)

    # Table I components, before optional scaling.
    r_phi_a = -phi_bar
    r_theta_t = -(theta_bar ** track_lambda)
    r_rel_pos = (theta_bar - 2.0) * s_phi_half - theta_bar + 1.0
    r_closure = (
        (closure_mps / _ft(500.0))
        * (1.0 - s_phi_half)
        * _sigmoid(distance_m, 1.0 / _ft(500.0), _ft(2900.0))
    )
    r_gunsnap_blue = _gamma_blue(distance_m, beta_blue) * (
        1.0 - _sigmoid(theta_bar, 1.0e5, 1.0 / 180.0)
    )

    # Penalize opponent gun solutions. The paper writes this term in terms of
    # the opponent's firing geometry; in the public challenge API, the most
    # direct equivalent is the target aircraft's ATA to ownship.
    red_theta_bar = red_track_angle_deg / 180.0
    r_gunsnap_red = -_gamma_red(distance_m, beta_red) * (
        1.0 - _sigmoid(red_theta_bar, 1.0e5, 1.0 / 180.0)
    )
    deck_altitude_ft = float(reward_config.get("deck_altitude_ft", 1300.0))
    deck_transition_ft = float(reward_config.get("deck_transition_ft", 20.0))
    r_deck = -4.0 * (
        1.0
        - _sigmoid(
            altitude_m,
            1.0 / max(_ft(deck_transition_ft), 1.0e-6),
            _ft(deck_altitude_ft),
        )
    )
    r_too_close = -2.0 * (1.0 - s_phi_half) * _sigmoid(
        distance_m, 1.0 / _ft(50.0), _ft(800.0)
    )

    altitude_target_m = float(reward_config.get("altitude_target_m", 7000.0))
    altitude_band_m = max(float(reward_config.get("altitude_band_m", 5000.0)), 1.0)
    safe_altitude_m = max(float(reward_config.get("safe_altitude_m", 2600.0)), 1.0)
    altitude_hold = float(reward_config.get("altitude_hold_scale", 0.0)) * _clip01(
        1.0 - abs(altitude_m - altitude_target_m) / altitude_band_m
    )
    low_altitude_guard = 0.0
    if altitude_m < safe_altitude_m:
        low_altitude_guard = -float(
            reward_config.get("low_altitude_guard_scale", 0.0)
        ) * (1.0 - altitude_m / safe_altitude_m)

    roll_deg = abs(float(ownship_state[StateIndex.ROLL]))
    pitch_deg = abs(float(ownship_state[StateIndex.PITCH]))
    attitude_guard = -float(reward_config.get("attitude_penalty_scale", 0.0)) * (
        (roll_deg / 180.0) ** 2 + (pitch_deg / 90.0) ** 2
    )
    pitch_envelope_ref_deg = min(
        max(float(reward_config.get("pitch_envelope_ref_deg", 35.0)), 0.0),
        89.0,
    )
    roll_envelope_ref_deg = min(
        max(float(reward_config.get("roll_envelope_ref_deg", 120.0)), 0.0),
        179.0,
    )
    pitch_excess = _clip01(
        max(0.0, pitch_deg - pitch_envelope_ref_deg)
        / max(90.0 - pitch_envelope_ref_deg, 1.0)
    )
    roll_excess = _clip01(
        max(0.0, roll_deg - roll_envelope_ref_deg)
        / max(180.0 - roll_envelope_ref_deg, 1.0)
    )
    attitude_envelope_guard = -float(
        reward_config.get("attitude_envelope_guard_scale", 0.0)
    ) * (pitch_excess**2 + roll_excess**2)
    altitude_deficit_score = _clip01(
        max(0.0, altitude_target_m - altitude_m) / altitude_band_m
    )
    pitch_down_deg = max(0.0, -float(ownship_state[StateIndex.PITCH]))
    pitch_ref_deg = max(float(reward_config.get("nose_down_pitch_ref_deg", 30.0)), 1.0)
    nose_down_guard = -float(reward_config.get("nose_down_guard_scale", 0.0)) * (
        altitude_deficit_score * (pitch_down_deg / pitch_ref_deg) ** 2
    )
    vertical_down_mps = max(0.0, float(_aircraft_velocity_ned(ownship_state)[2]))
    descent_ref_mps = max(float(reward_config.get("descent_rate_ref_mps", 80.0)), 1.0)
    descent_rate_guard = -float(reward_config.get("descent_rate_guard_scale", 0.0)) * (
        altitude_deficit_score * _clip01(vertical_down_mps / descent_ref_mps) ** 2
    )
    roll_ref_deg = max(float(reward_config.get("roll_guard_ref_deg", 90.0)), 1.0)
    low_altitude_roll_guard = -float(
        reward_config.get("low_altitude_roll_guard_scale", 0.0)
    ) * (altitude_deficit_score * (roll_deg / roll_ref_deg) ** 2)
    action_pitch_down_guard = 0.0
    action_pitch_guard = 0.0
    action_roll_guard = 0.0
    action_roll_global_guard = 0.0
    if action is not None:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        action_ref = max(float(reward_config.get("action_guard_ref", 1.0)), 1e-6)
        if action_arr.size > 1:
            # Simulator convention: positive pitch action drives the nose down,
            # negative pitch action pulls up.
            pitch_down_action = max(0.0, float(action_arr[1]))
            action_pitch_down_guard = -float(
                reward_config.get("action_pitch_down_guard_scale", 0.0)
            ) * (altitude_deficit_score * (pitch_down_action / action_ref) ** 2)
            pitch_action = abs(float(action_arr[1]))
            action_pitch_guard = -float(
                reward_config.get("action_pitch_guard_scale", 0.0)
            ) * (pitch_action / action_ref) ** 2
        if action_arr.size > 0:
            roll_action = abs(float(action_arr[0]))
            action_roll_guard = -float(
                reward_config.get("action_roll_guard_scale", 0.0)
            ) * (altitude_deficit_score * (roll_action / action_ref) ** 2)
            action_roll_global_guard = -float(
                reward_config.get("action_roll_global_guard_scale", 0.0)
            ) * (roll_action / action_ref) ** 2

    speed_kcas = float(ownship_state[StateIndex.KCAS])
    speed_target_kcas = float(reward_config.get("speed_target_kcas", 320.0))
    speed_band_kcas = max(float(reward_config.get("speed_band_kcas", 220.0)), 1.0)
    speed_excess_error = max(0.0, abs(speed_kcas - speed_target_kcas) - speed_band_kcas)
    speed_guard = -float(reward_config.get("speed_penalty_scale", 0.0)) * _clip01(
        speed_excess_error / speed_band_kcas
    )

    min_wez_range_m = float(wez_config.get("min_range_m", _ft(500.0)))
    max_wez_range_m = float(wez_config.get("max_range_m", _ft(3000.0)))
    default_band_center = 0.5 * (min_wez_range_m + max_wez_range_m)
    default_band_half_width = 0.5 * max(max_wez_range_m - min_wez_range_m, 1.0)
    wez_band_center_m = float(
        reward_config.get("wez_band_center_m", default_band_center)
    )
    wez_band_half_width_m = max(
        float(reward_config.get("wez_band_half_width_m", default_band_half_width)),
        1.0,
    )
    wez_nose_half_angle_deg = max(
        float(reward_config.get("wez_nose_half_angle_deg", 8.0)),
        1.0,
    )
    band_score = _clip01(1.0 - abs(distance_m - wez_band_center_m) / wez_band_half_width_m)
    nose_score = _clip01(1.0 - track_angle_deg / wez_nose_half_angle_deg)
    official_half_wez_angle_deg = float(wez_config.get("angle_deg", 2.0)) / 2.0
    in_official_wez = (
        min_wez_range_m <= distance_m <= max_wez_range_m
        and track_angle_deg <= official_half_wez_angle_deg
    )
    official_aim_half_angle_deg = max(
        float(
            reward_config.get(
                "official_wez_aim_half_angle_deg",
                max(official_half_wez_angle_deg * 2.5, 0.5),
            )
        ),
        0.1,
    )
    official_aim_score = _clip01(1.0 - track_angle_deg / official_aim_half_angle_deg)
    official_range_score = (
        band_score
        if min_wez_range_m <= distance_m <= max_wez_range_m
        else 0.0
    )
    precision_sigma_deg = max(
        float(reward_config.get("precision_aim_sigma_deg", 5.0)),
        0.1,
    )
    precision_power = max(
        float(reward_config.get("precision_aim_power", 1.0)),
        0.1,
    )
    precision_aim_score = math.exp(
        -0.5 * (track_angle_deg / precision_sigma_deg) ** 2
    ) ** precision_power
    reacquire_score = 0.0
    if isinstance(context, dict) and not in_official_wez:
        steps_since_wez = int(context.get("steps_since_wez", 10000) or 10000)
        reacquire_window_steps = int(
            reward_config.get("reacquire_window_steps", 60.0)
        )
        recently_lost_wez = (
            bool(context.get("previous_in_wez", False))
            or steps_since_wez <= reacquire_window_steps
        )
        if recently_lost_wez:
            distance_delta = context.get("distance_delta_m")
            track_delta = context.get("track_angle_delta_deg")
            range_ref = max(
                float(reward_config.get("reacquire_range_ref_m", 180.0)),
                1.0,
            )
            angle_ref = max(
                float(reward_config.get("reacquire_angle_ref_deg", 8.0)),
                0.1,
            )
            closing_score = (
                _clip01(-float(distance_delta) / range_ref)
                if distance_delta is not None
                else 0.0
            )
            aiming_score = (
                _clip01(-float(track_delta) / angle_ref)
                if track_delta is not None
                else 0.0
            )
            current_geometry_score = official_range_score * _clip01(
                1.0 - track_angle_deg / max(angle_ref * 2.0, 0.1)
            )
            reacquire_score = (
                0.35 * closing_score
                + 0.45 * aiming_score
                + 0.20 * current_geometry_score
            )

    components: dict[str, float] = {
        "survival": float(reward_config.get("survival_bonus", 0.0)),
        "step": float(reward_config.get("step_penalty", -0.002)),
        "adverse_angle": float(reward_config.get("adverse_angle_scale", 0.25)) * r_phi_a,
        "track_angle": float(reward_config.get("track_angle_scale", 0.25)) * r_theta_t,
        "relative_position": float(reward_config.get("relative_position_scale", 1.0)) * r_rel_pos,
        "closure": float(reward_config.get("closure_scale", 0.5)) * r_closure,
        "gunsnap_blue": float(reward_config.get("gunsnap_blue_scale", 1.0)) * r_gunsnap_blue,
        "gunsnap_red": float(reward_config.get("gunsnap_red_scale", 1.0)) * r_gunsnap_red,
        "deck": float(reward_config.get("deck_scale", 1.0)) * r_deck,
        "too_close": float(reward_config.get("too_close_scale", 1.0)) * r_too_close,
        "damage": float(reward_config.get("damage_scale", 0.0))
        * (float(target_damage) - float(ownship_damage)),
        "safety": altitude_hold + low_altitude_guard + attitude_guard + speed_guard,
        "attitude_envelope_guard": attitude_envelope_guard,
        "nose_down_guard": nose_down_guard,
        "descent_rate_guard": descent_rate_guard,
        "low_altitude_roll_guard": low_altitude_roll_guard,
        "action_pitch_down_guard": action_pitch_down_guard,
        "action_pitch_guard": action_pitch_guard,
        "action_roll_guard": action_roll_guard,
        "action_roll_global_guard": action_roll_global_guard,
        "wez_band": float(reward_config.get("wez_band_scale", 0.0))
        * band_score
        * (0.25 + 0.75 * nose_score),
        "wez_hold": float(reward_config.get("wez_hold_bonus", 0.0))
        if in_official_wez
        else 0.0,
        "official_wez_aim": float(reward_config.get("official_wez_aim_scale", 0.0))
        * official_range_score
        * (official_aim_score**2),
        "precision_aim": float(reward_config.get("precision_aim_scale", 0.0))
        * official_range_score
        * precision_aim_score,
        "reacquire": float(reward_config.get("reacquire_scale", 0.0))
        * reacquire_score,
    }

    terminal_reward = 0.0
    if terminated or truncated:
        ownship_health = float(ownship_state[StateIndex.HEALTH])
        target_health = float(target_state[StateIndex.HEALTH])
        if end_condition in ("ownship altitude below min", "FDM Update Fail"):
            terminal_reward = float(reward_config.get("own_crash_penalty", -150.0))
        elif end_condition == "target altitude below min":
            terminal_reward = float(reward_config.get("target_crash_reward", -5.0))
        elif end_condition == "two circle headon guard fail":
            terminal_reward = float(reward_config.get("guard_fail_penalty", -50.0))
        elif target_health <= 0.0 < ownship_health:
            terminal_reward = float(reward_config.get("win_reward", 100.0))
        elif ownship_health <= 0.0 < target_health:
            terminal_reward = float(reward_config.get("loss_reward", -100.0))
        else:
            terminal_reward = float(reward_config.get("draw_reward", -30.0))
    components["terminal"] = terminal_reward


    return float(sum(components.values())), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
