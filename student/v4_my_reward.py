# -*- coding: utf-8 -*-
"""Lockheed Martin / AlphaDogfight reward adapted to the competition Release.

Primary source
--------------
Pope et al., "Hierarchical Reinforcement Learning for Air Combat at
DARPA's AlphaDogfight Trials," IEEE Transactions on Artificial Intelligence,
2023, Table I and Eqs. (7)-(21).

Important design rule
---------------------
The paper did *not* use one universal low-level reward.  It trained separate
Control-Zone (CZ), Aggressive-Shooter (AS), Conservative-Shooter (CS), and
policy-selector policies.  This module preserves those roles and additionally
provides ``soft_hierarchy`` for the public Release when only one continuous
policy is available:

* far/acquisition region: CZ relative-position + closure terms dominate;
* near the official WEZ: CS track-angle + gun-snap terms dominate;
* positive gun-snap reward is paid only in the competition's official WEZ;
* a narrow precision term is a negative aim-error penalty, never a positive
  near-miss bonus;
* safe flight is neutral.  Only unsafe flight is penalized, preventing a
  profitable "survive until timeout" policy.

The module works with the original Release reward signature.  A compatible
optional environment patch may pass ``context`` containing exact distance
change and dt; otherwise closure is estimated from the state velocity vector.
"""
from __future__ import annotations

import math
import sys
from enum import IntEnum
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CWD = Path.cwd()
for _path in (ROOT, SRC, CWD, CWD / "src"):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from dogfight.sim.state_schema import StateIndex
except Exception:  # pragma: no cover - standalone unit-test fallback
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
EPS = 1.0e-9
BODY_U = 6
BODY_V = 7
BODY_W = 8


MY_REWARD_CONFIG: dict[str, float | str] = {
    "mode": "lockheed_alpha_dogfight_final_v3",

    # Paper-faithful roles: control_zone, aggressive_shooter,
    # conservative_shooter, policy_selector.  soft_hierarchy is the practical
    # single-policy bridge used before a true high-level selector is available.
    "policy_role": "soft_hierarchy",

    # Standard challenge RL transition: 60-Hz simulation, step_ratio=6 -> 0.1 s.
    # Table-I values are calibrated at this reference step and scaled by actual
    # dt when an environment context is supplied.
    "reward_dt_s": 0.10,
    "reference_dt_s": 0.10,

    # One common multiplier preserves the relative weighting of each Table-I
    # role.  At a perfect CS gun solution, beta=3 -> about +0.30 per 0.1-s step.
    "table1_step_scale": 0.10,

    # AS/CS Eq. (9).
    "shooter_track_lambda": 4.0 / 9.0,

    # CS/CZ constant aggressiveness factors.  Red beta is stored as a positive
    # magnitude; the penalty sign is applied exactly once in this code.
    "blue_beta": 3.0,
    "red_beta_magnitude": 3.0,

    # Soft single-policy selector.  With the official 914.4-m max range,
    # shooting starts to dominate around 1.14 km and transitions over ~137 m.
    "shooter_entry_range_factor": 1.25,
    "shooter_transition_range_factor": 0.15,

    # A pure CZ policy uses the exact Eq. (10) surface.  In soft_hierarchy, the
    # ideal non-scoring tail position is shifted from +1 to 0 so it cannot farm
    # reward indefinitely at long range.
    "soft_cz_relative_position_offset": 1.0,

    # Competition precision bridge.  The paper's track penalty is retained.
    # This additional term addresses the observed final 1-3 degree miss:
    # zero error is neutral, larger error is negative, and only official
    # gun-snap/damage provides positive scoring reward.
    "precision_aim_penalty_per_s": 1.50,
    "precision_bridge_angle_deg": 3.0,
    "pre_wez_bridge_m": 220.0,
    "pre_wez_bridge_fraction": 0.25,
    "pre_wez_closure_ref_mps": 80.0,
    "enable_precision_bridge": 1.0,

    # Actual competition score.  Full health differential is +/-50; final
    # win/loss remains dominant.  This is deliberately below the previous 80-
    # 120 scales that encouraged a closer fly-through instead of WEZ hold.
    "damage_scale": 50.0,

    # Weak time pressure.  No positive survival or altitude-hold bonus exists.
    "time_penalty_per_s": 0.05,

    # Paper deck term adapted relative to the challenge hard deck.  The paper
    # uses a 1000-ft hard deck, a 1300-ft reward midpoint, and 20-ft transition.
    "hard_deck_m": 300.0,
    "deck_warning_offset_m": 300.0 * FEET_TO_METER,
    "deck_transition_m": 20.0 * FEET_TO_METER,

    # Minimal additional low-altitude dynamic envelope.  It is inactive at
    # safe altitude and adds no positive level-flight reward.
    "safety_envelope_start_m": 1800.0,
    "safety_envelope_rate_per_s": 2.0,
    "safe_descent_mps": 25.0,
    "descent_span_mps": 75.0,
    "safe_nose_down_deg": 15.0,
    "nose_down_span_deg": 45.0,
    "safe_roll_deg": 90.0,
    "roll_span_deg": 90.0,

    # Terminal ordering.  A target hard-deck loss is rewarded less than a true
    # health kill so the policy does not learn to wait for a BT crash.
    "win_reward": 200.0,
    "loss_reward": -200.0,
    "draw_reward": -50.0,
    "own_crash_penalty": -300.0,
    "target_deck_reward": 50.0,
    "guard_fail_penalty": -150.0,
    "range_discipline_penalty": -150.0,
    "curriculum_success_reward": 100.0,
}


def _cfg(config: dict, key: str) -> float:
    return float(config.get(key, MY_REWARD_CONFIG[key]))


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _clip01(value: float) -> float:
    return _clip(value, 0.0, 1.0)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _safe_sigmoid_z(z: float) -> float:
    if z >= 60.0:
        return 1.0
    if z <= -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _sigmoid(x: float, alpha: float, x0: float) -> float:
    """Pope et al. Eq. (7): S(x, alpha, x0)."""
    return _safe_sigmoid_z(float(alpha) * (float(x) - float(x0)))


def _smooth_decreasing(x: float, center: float, width: float) -> float:
    width = max(abs(float(width)), 1.0e-6)
    return _safe_sigmoid_z((float(center) - float(x)) / width)


def _ft(value: float) -> float:
    return float(value) * FEET_TO_METER


def _vec_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(max(_dot(v, v), 0.0))


def _unit(v: tuple[float, float, float]) -> tuple[float, float, float]:
    magnitude = _norm(v)
    if magnitude <= EPS:
        return (1.0, 0.0, 0.0)
    return (v[0] / magnitude, v[1] / magnitude, v[2] / magnitude)


def _angle_deg(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.degrees(math.acos(_clip(_dot(_unit(a), _unit(b)), -1.0, 1.0)))


def _position_ned(state: Any) -> tuple[float, float, float]:
    return (
        _finite(state[StateIndex.N]),
        _finite(state[StateIndex.E]),
        _finite(state[StateIndex.D]),
    )


def _forward_ned(state: Any) -> tuple[float, float, float]:
    pitch = math.radians(_finite(state[StateIndex.PITCH]))
    yaw = math.radians(_finite(state[StateIndex.YAW]))
    cp = math.cos(pitch)
    return (cp * math.cos(yaw), cp * math.sin(yaw), -math.sin(pitch))


def _body_velocity_to_ned(state: Any) -> tuple[float, float, float]:
    """Transform Release body-axis u/v/w (indices 6:9) to NED velocity.

    FighterSim.py already stores u, v, w and KCAS in m/s.  No knot conversion
    is applied here.  If a shortened test/deployment state does not contain a
    valid body velocity, KCAS along the aircraft nose is used as fallback.
    """
    try:
        u = _finite(state[BODY_U], float("nan"))
        v = _finite(state[BODY_V], float("nan"))
        w = _finite(state[BODY_W], float("nan"))
    except (IndexError, TypeError):
        u = v = w = float("nan")

    if all(math.isfinite(x) for x in (u, v, w)) and (abs(u) + abs(v) + abs(w) > 1.0e-6):
        roll = math.radians(_finite(state[StateIndex.ROLL]))
        pitch = math.radians(_finite(state[StateIndex.PITCH]))
        yaw = math.radians(_finite(state[StateIndex.YAW]))

        cphi, sphi = math.cos(roll), math.sin(roll)
        cth, sth = math.cos(pitch), math.sin(pitch)
        cpsi, spsi = math.cos(yaw), math.sin(yaw)

        north = cth * cpsi * u + (sphi * sth * cpsi - cphi * spsi) * v + (cphi * sth * cpsi + sphi * spsi) * w
        east = cth * spsi * u + (sphi * sth * spsi + cphi * cpsi) * v + (cphi * sth * spsi - sphi * cpsi) * w
        down = -sth * u + sphi * cth * v + cphi * cth * w
        return (north, east, down)

    speed = max(0.0, _finite(state[StateIndex.KCAS]))
    forward = _forward_ned(state)
    return (speed * forward[0], speed * forward[1], speed * forward[2])


def _altitude_m(state: Any) -> float:
    altitude = _finite(state[StateIndex.ALT])
    if altitude > 1.0:
        return altitude
    return max(0.0, -_finite(state[StateIndex.D]))


def _geometry(ownship_state: Any, target_state: Any) -> dict[str, float]:
    """Compute track angle, adverse angle, red track, range, and closure."""
    own_pos = _position_ned(ownship_state)
    target_pos = _position_ned(target_state)
    los_own_to_target = _vec_sub(target_pos, own_pos)
    distance_m = _norm(los_own_to_target)
    los_unit = _unit(los_own_to_target)

    own_forward = _forward_ned(ownship_state)
    target_forward = _forward_ned(target_state)
    target_to_own = (-los_unit[0], -los_unit[1], -los_unit[2])
    target_tail = (-target_forward[0], -target_forward[1], -target_forward[2])

    own_track_deg = _angle_deg(own_forward, los_unit)
    adverse_deg = _angle_deg(target_tail, target_to_own)
    red_track_deg = _angle_deg(target_forward, target_to_own)

    own_velocity = _body_velocity_to_ned(ownship_state)
    target_velocity = _body_velocity_to_ned(target_state)
    relative_velocity = _vec_sub(own_velocity, target_velocity)
    # Positive closure means range is decreasing.
    closure_mps = _dot(relative_velocity, los_unit)

    return {
        "distance_m": float(distance_m),
        "own_track_deg": _clip(own_track_deg, 0.0, 180.0),
        "adverse_deg": _clip(adverse_deg, 0.0, 180.0),
        "red_track_deg": _clip(red_track_deg, 0.0, 180.0),
        "closure_mps": float(closure_mps),
        "own_down_mps": float(own_velocity[2]),
    }


def _gamma_blue(distance_m: float, beta_blue: float) -> float:
    """Pope et al. Eq. (16), feet constants converted to meters."""
    if distance_m < _ft(1950.0):
        return beta_blue * _sigmoid(distance_m, 1.0 / _ft(50.0), _ft(1000.0))
    return beta_blue * (1.0 - _sigmoid(distance_m, 1.0 / _ft(50.0), _ft(2900.0)))


def _gamma_red(distance_m: float, beta_red_magnitude: float) -> float:
    """Magnitude of Pope et al. Eq. (17); caller applies the minus sign."""
    if distance_m < _ft(2250.0):
        return beta_red_magnitude * _sigmoid(distance_m, 1.0 / _ft(35.0), _ft(400.0))
    return beta_red_magnitude * (1.0 - _sigmoid(distance_m, 1.0 / _ft(200.0), _ft(4100.0)))


def _aggressive_blue_beta(distance_m: float) -> float:
    """Pope et al. Eq. (19)."""
    distance_ft = distance_m / FEET_TO_METER
    if 1000.0 <= distance_ft <= 2900.0:
        return 2.0 * ((2900.0 - distance_ft) / 1900.0 + 1.0)
    return 2.0


def _aggressive_red_beta_magnitude(distance_m: float) -> float:
    """Magnitude of Pope et al. Eq. (20)."""
    distance_ft = distance_m / FEET_TO_METER
    if 400.0 <= distance_ft <= 4100.0:
        return 2.0 * ((4100.0 - distance_ft) / 3700.0 + 1.0)
    return 2.0


def _normalize_role(value: object) -> str:
    role = str(value or "soft_hierarchy").strip().lower().replace("-", "_")
    aliases = {
        "cz": "control_zone",
        "as": "aggressive_shooter",
        "cs": "conservative_shooter",
        "ps": "policy_selector",
        "selector": "policy_selector",
        "soft": "soft_hierarchy",
        "bt_bridge": "soft_hierarchy",
    }
    role = aliases.get(role, role)
    allowed = {
        "control_zone",
        "aggressive_shooter",
        "conservative_shooter",
        "policy_selector",
        "soft_hierarchy",
    }
    if role not in allowed:
        raise ValueError(f"Unsupported policy_role={value!r}; expected one of {sorted(allowed)}")
    return role


def _terminal_reward(
    ownship_state: Any,
    target_state: Any,
    terminated: bool,
    truncated: bool,
    end_condition: str,
    reward_config: dict,
) -> float:
    if not (terminated or truncated):
        return 0.0

    own_health = _finite(ownship_state[StateIndex.HEALTH], 1.0)
    target_health = _finite(target_state[StateIndex.HEALTH], 1.0)
    end = str(end_condition or "").strip().lower()

    # Referee/curriculum outcomes supplied by an extended environment.
    if "mutual" in end and "evasion" in end:
        return _cfg(reward_config, "draw_reward")
    if "target" in end and "evasion" in end:
        return _cfg(reward_config, "win_reward")
    if "ownship" in end and "evasion" in end:
        return _cfg(reward_config, "loss_reward")
    if end == "range discipline":
        return _cfg(reward_config, "range_discipline_penalty")
    if end in {"position acquired", "position capture", "curriculum success"}:
        return _cfg(reward_config, "curriculum_success_reward")
    if "two circle" in end and "guard" in end:
        return _cfg(reward_config, "guard_fail_penalty")

    # Physical combat outcome.
    if target_health <= 0.0 < own_health:
        return _cfg(reward_config, "win_reward")
    if own_health <= 0.0 < target_health:
        return _cfg(reward_config, "loss_reward")

    if "ownship altitude" in end or "fdm update fail" in end or "ownship fdm" in end:
        return _cfg(reward_config, "own_crash_penalty")
    if "target altitude" in end or "target fdm" in end:
        return _cfg(reward_config, "target_deck_reward")
    if "fuel fail" in end:
        return _cfg(reward_config, "loss_reward")

    return _cfg(reward_config, "draw_reward")


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
    """Return the one-step reward and named reward components."""
    del geo_info, action  # Robust geometry is computed directly; action is not penalized.

    geometry = _geometry(ownship_state, target_state)
    distance_m = geometry["distance_m"]
    own_track_deg = geometry["own_track_deg"]
    adverse_deg = geometry["adverse_deg"]
    red_track_deg = geometry["red_track_deg"]
    closure_mps = geometry["closure_mps"]

    dt_s = max(_cfg(reward_config, "reward_dt_s"), 1.0e-6)
    if isinstance(context, dict):
        if context.get("dt_s") is not None:
            dt_s = max(_finite(context.get("dt_s"), dt_s), 1.0e-6)
        if context.get("distance_delta_m") is not None:
            closure_mps = -_finite(context.get("distance_delta_m")) / dt_s

    reference_dt_s = max(_cfg(reward_config, "reference_dt_s"), 1.0e-6)
    step_time_scale = dt_s / reference_dt_s

    r_min = max(_finite(wez_config.get("min_range_m"), _ft(500.0)), 1.0)
    r_max = max(_finite(wez_config.get("max_range_m"), _ft(3000.0)), r_min + 1.0)
    official_half_angle_deg = max(_finite(wez_config.get("angle_deg"), 2.0) * 0.5, 0.05)

    theta_bar = own_track_deg / 180.0
    phi_bar = adverse_deg / 180.0
    s_phi_half = _sigmoid(phi_bar, 18.0, 0.5)

    # ------------------------- Pope et al. Table I -------------------------
    # Eq. (10): relative position.
    r_relative_position_raw = (theta_bar - 2.0) * s_phi_half - theta_bar + 1.0

    # Eq. (11): closure from behind.  500 ft/s is the paper normalization.
    closure_norm = _clip(closure_mps / _ft(500.0), -1.0, 1.0)
    r_closure = closure_norm * (1.0 - s_phi_half) * _sigmoid(
        distance_m, 1.0 / _ft(500.0), _ft(2900.0)
    )

    role = _normalize_role(reward_config.get("policy_role", MY_REWARD_CONFIG["policy_role"]))
    if role == "policy_selector":
        track_lambda = 1.0
        blue_beta = 5.0
        red_beta = 5.0
    elif role == "aggressive_shooter":
        track_lambda = 4.0 / 9.0
        blue_beta = _aggressive_blue_beta(distance_m)
        red_beta = _aggressive_red_beta_magnitude(distance_m)
    else:
        track_lambda = max(_cfg(reward_config, "shooter_track_lambda"), 0.05)
        blue_beta = _cfg(reward_config, "blue_beta")
        red_beta = _cfg(reward_config, "red_beta_magnitude")

    # Eq. (9): shooter track-angle penalty.
    r_track = -(theta_bar ** track_lambda)

    # Eq. (12): blue gun snap.  The paper's steep logistic is effectively a
    # one-degree gate.  Positive reward is additionally hard-gated by the
    # competition's exact official range and angle so 950-1000 m near misses
    # cannot become profitable.
    blue_angle_score = 1.0 - _sigmoid(
        theta_bar, 1.0e5, official_half_angle_deg / 180.0
    )
    blue_in_official_wez = (
        r_min <= distance_m <= r_max and own_track_deg <= official_half_angle_deg
    )
    r_gunsnap_blue = (
        _gamma_blue(distance_m, blue_beta) * blue_angle_score
        if blue_in_official_wez
        else 0.0
    )

    # Eq. (13): red gun-snap threat.  Table I uses adverse angle >=178 deg,
    # equivalent to the opponent's nose lying within a two-degree threat cone.
    # The paper prints beta_R as negative while also placing a leading minus in
    # Eq. (13), which conflicts with its prose.  We store a positive magnitude
    # and apply the negative sign exactly once, so enemy gun solutions are never
    # rewarded.
    red_angle_threat = _sigmoid(phi_bar, 800.0, 178.0 / 180.0)
    r_gunsnap_red = -_gamma_red(distance_m, red_beta) * red_angle_threat

    # Eq. (15), behavior-consistent form.  The printed table uses S(d,...), but
    # the prose says it penalizes *below* the 800-ft minimum distance.  1-S is
    # therefore used to implement the stated anti-overshoot behavior.
    r_too_close = -2.0 * (1.0 - s_phi_half) * (
        1.0 - _sigmoid(distance_m, 1.0 / _ft(50.0), _ft(800.0))
    )

    # Eq. (14), translated relative to the challenge hard deck: paper hard deck
    # 1000 ft, midpoint 1300 ft, transition 20 ft.
    altitude_m = _altitude_m(ownship_state)
    hard_deck_m = _cfg(reward_config, "hard_deck_m")
    deck_midpoint_m = hard_deck_m + _cfg(reward_config, "deck_warning_offset_m")
    deck_transition_m = max(_cfg(reward_config, "deck_transition_m"), 1.0)
    r_deck = -4.0 * (
        1.0 - _sigmoid(altitude_m, 1.0 / deck_transition_m, deck_midpoint_m)
    )

    # ----------------------------- Role gates ------------------------------
    if role == "control_zone":
        cz_gate, shooter_gate, too_close_gate = 1.0, 0.0, 1.0
        r_relative_position = r_relative_position_raw  # exact Eq. (10)
    elif role in {"aggressive_shooter", "conservative_shooter", "policy_selector"}:
        cz_gate, shooter_gate, too_close_gate = 0.0, 1.0, 0.0
        r_relative_position = 0.0
    else:
        shooter_entry_m = _cfg(reward_config, "shooter_entry_range_factor") * r_max
        shooter_width_m = max(
            _cfg(reward_config, "shooter_transition_range_factor") * r_max,
            25.0,
        )
        shooter_gate = _smooth_decreasing(distance_m, shooter_entry_m, shooter_width_m)
        cz_gate = 1.0 - shooter_gate
        too_close_gate = 1.0
        r_relative_position = (
            r_relative_position_raw
            - _cfg(reward_config, "soft_cz_relative_position_offset")
        )

    # --------------------- Competition precision bridge -------------------
    precision_active_role = role in {
        "aggressive_shooter", "conservative_shooter", "soft_hierarchy"
    }
    precision_gate = 0.0
    if precision_active_role and _cfg(reward_config, "enable_precision_bridge") > 0.0:
        if r_min <= distance_m <= r_max:
            precision_gate = 1.0
        elif r_max < distance_m <= r_max + _cfg(reward_config, "pre_wez_bridge_m"):
            bridge_m = max(_cfg(reward_config, "pre_wez_bridge_m"), 1.0)
            distance_fraction = _clip01((r_max + bridge_m - distance_m) / bridge_m)
            closing_fraction = _clip01(
                closure_mps / max(_cfg(reward_config, "pre_wez_closure_ref_mps"), 1.0)
            )
            precision_gate = (
                _cfg(reward_config, "pre_wez_bridge_fraction")
                * distance_fraction
                * closing_fraction
            )

    precision_angle_deg = max(_cfg(reward_config, "precision_bridge_angle_deg"), 0.1)
    precision_error = _clip01(own_track_deg / precision_angle_deg) ** 2
    r_precision_aim = -_cfg(reward_config, "precision_aim_penalty_per_s") * precision_gate * precision_error * dt_s

    # ----------------------- Negative-only safety -------------------------
    safety_start_m = max(_cfg(reward_config, "safety_envelope_start_m"), hard_deck_m + 1.0)
    altitude_risk = _clip01(
        (safety_start_m - altitude_m) / (safety_start_m - hard_deck_m)
    )
    pitch_deg = _finite(ownship_state[StateIndex.PITCH])
    roll_deg = abs(_finite(ownship_state[StateIndex.ROLL]))
    downward_mps = max(0.0, geometry["own_down_mps"])

    nose_risk = _clip01(
        (max(0.0, -pitch_deg) - _cfg(reward_config, "safe_nose_down_deg"))
        / max(_cfg(reward_config, "nose_down_span_deg"), 1.0)
    )
    roll_risk = _clip01(
        (roll_deg - _cfg(reward_config, "safe_roll_deg"))
        / max(_cfg(reward_config, "roll_span_deg"), 1.0)
    )
    descent_risk = _clip01(
        (downward_mps - _cfg(reward_config, "safe_descent_mps"))
        / max(_cfg(reward_config, "descent_span_mps"), 1.0)
    )
    dynamic_risk = max(nose_risk, 0.5 * roll_risk, descent_risk)
    r_safety_envelope = (
        -_cfg(reward_config, "safety_envelope_rate_per_s")
        * altitude_risk
        * dynamic_risk * dynamic_risk
        * dt_s
    )

    # --------------------------- Final components --------------------------
    table_scale = _cfg(reward_config, "table1_step_scale") * step_time_scale
    components: dict[str, float] = {
        "time": -_cfg(reward_config, "time_penalty_per_s") * dt_s,
        "cz_relative_position": table_scale * cz_gate * r_relative_position,
        "cz_closure": table_scale * cz_gate * r_closure,
        "shooter_track": table_scale * shooter_gate * r_track,
        "precision_aim": r_precision_aim,
        "gunsnap_blue": table_scale * r_gunsnap_blue,
        "gunsnap_red": table_scale * r_gunsnap_red,
        "too_close": table_scale * too_close_gate * r_too_close,
        "deck": table_scale * r_deck,
        "safety_envelope": r_safety_envelope,
        "damage": _cfg(reward_config, "damage_scale") * (
            _finite(target_damage) - _finite(ownship_damage)
        ),
        "terminal": _terminal_reward(
            ownship_state,
            target_state,
            terminated,
            truncated,
            end_condition,
            reward_config,
        ),
    }

    total = float(sum(components.values()))
    if not math.isfinite(total):
        fallback = _cfg(reward_config, "loss_reward")
        components = {name: 0.0 for name in components}
        components["terminal"] = fallback
        total = fallback

    return total, components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
