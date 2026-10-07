# -*- coding: utf-8 -*-
"""Final single-policy AlphaDogfight reward for the competition simulation.

This module deliberately uses one continuous policy and one inference pass.  It
never selects among Control-Zone, Aggressive-Shooter, or Conservative-Shooter
networks.  Instead, Lockheed Martin Table-I geometry is rewritten as a symmetric
blue-versus-red combat advantage:

* blue relative-position advantage is rewarded and the mirrored red advantage
  is penalized;
* closing is rewarded when blue is behind red and penalized when red is behind
  blue, so the same term supports pursuit and escape;
* own track quality is compared directly with opponent track quality;
* official blue gun snap is positive, official red gun snap is negative;
* actual health differential, deck safety, the 15+15 s evasion notice, and
  competition terminal outcomes are included.

Positive scoring terms are tied to the official WEZ or real closing/damage.  A
stationary 1000-m near miss and a distant tail position cannot farm reward.
External position-capture/progress rewards must be disabled in final combat.
The 16.67-ms response rule is not an RL state variable; this reward therefore
maps server-applied latency damage and response-failure end conditions, while a
single asynchronous actor should be used at deployment.

Drop-in contract:
    MY_REWARD_CONFIG: dict
    compute_reward(...) -> (float, dict[str, float])
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from pathlib import Path
import sys
import weakref
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC, Path.cwd(), Path.cwd() / "src"):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from dogfight.sim.state_schema import StateIndex
except Exception:  # standalone validation fallback
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
G = 9.80665
BODY_U = 6
BODY_V = 7
BODY_W = 8


# Internal values are authoritative by default.  This is deliberate: replacing
# only this file must not inherit stale reward scalars from an older YAML.
MY_REWARD_CONFIG: dict[str, float | str] = {
    "mode": "unified_alpha_competition_v7",
    "allow_external_override": 0.0,

    # Original Release: 60-Hz FDM and step_ratio=6 -> about 0.1 s per RL step.
    # dt is measured from StateIndex.SIM_TIME after the first transition.
    "fallback_dt_s": 0.10,
    "reference_dt_s": 0.10,

    # One policy, one forward pass.  No CZ/AS/CS selector is used.  The terms
    # below are active simultaneously and express a blue-vs-red combat advantage.
    "table1_step_scale": 0.10,
    # Broad absolute position rewards are disabled in the final task.  The same
    # geometry is used only as a potential difference, so a tail position cannot
    # be farmed by simply remaining there.
    "relative_position_weight": 0.0,
    "track_advantage_weight": 0.0,

    # Context-conditioned closure controller.  It rewards neither raw closing
    # nor raw opening.  Instead it penalizes deviation from the closure that is
    # appropriate for the current offense/defense geometry.
    "closure_control_penalty_per_s": 1.25,
    "closure_reference_mps": 80.0,
    "desired_closure_gain_per_s": 0.08,
    "desired_closure_max_mps": 80.0,
    "desired_opening_max_mps": 45.0,
    "hold_range_fraction": 0.50,
    "too_close_penalty_per_s": 1.50,
    "too_close_margin_m": 100.0,
    "too_close_width_m": 50.0,

    # At long range the original competition commonly starts near a symmetric
    # head-on state.  Absolute geometry would either be zero (no learning
    # signal) or become a farmable tail-position bonus.  Instead, the single
    # actor receives a potential-difference signal until the fight reaches the
    # near-WEZ region.  The same potential improves when blue gains the tail or
    # when blue escapes from red's tail.
    "maneuver_progress_scale": 8.0,
    "maneuver_position_weight": 0.65,
    "maneuver_track_weight": 0.35,
    "shooter_track_lambda": 4.0 / 9.0,
    "blue_beta": 3.0,
    "red_beta_magnitude": 3.0,

    # Absolute geometry advantage is useful only near a tactically relevant
    # engagement.  At long range, closing/escaping supplies the dense signal and
    # prevents farming a distant tail position.
    "combat_gate_center_m": 2200.0,
    "combat_gate_width_m": 500.0,

    # Exact competition WEZ alignment.  Positive gun-snap reward is paid only
    # inside the official range and half-angle, and it is multiplied by the same
    # linear distance potential used by the Release damage model.
    "align_gunsnap_with_damage": 1.0,

    # Narrow near-WEZ guidance.  Own aim error and opponent lock quality are
    # negative-only; a 1000-m near miss can never become a profitable endpoint.
    # Precision shaping is bounded but non-saturating across the 1--30 deg
    # recovery range.  The official solution bonus below remains concentrated
    # at the competition 1-deg half cone.
    "precision_aim_penalty_per_s": 0.80,
    "precision_log_reference_deg": 1.0,
    "precision_log_max_deg": 60.0,
    "enemy_lock_penalty_per_s": 1.00,
    "precision_bridge_angle_deg": 3.0,
    "enemy_threat_sigma_deg": 3.0,
    "pre_wez_bridge_m": 220.0,
    "pre_wez_bridge_fraction": 0.30,
    "pre_wez_closure_ref_mps": 80.0,

    # Dense official-range solution quality.  It does not pay outside the
    # official range and therefore cannot create the previous 1000-m near-miss
    # local optimum.
    "wez_solution_reward_per_s": 4.0,
    "wez_solution_sigma_deg": 1.5,

    # Escape is rewarded only while red is an actual tracking threat.  Once
    # that threat is broken, continued long-range opening is penalized so the
    # same actor must reverse and re-engage instead of farming a timeout.
    "defensive_threat_sigma_deg": 20.0,
    "defensive_threat_range_m": 1800.0,
    "defensive_threat_range_width_m": 450.0,
    "post_threat_opening_penalty_per_s": 1.00,
    "post_threat_opening_start_m": 2200.0,
    "post_threat_opening_width_m": 500.0,

    # Mutual head-on risk: break neutral symmetry without choosing a turn side.
    "mutual_headon_penalty_per_s": 0.75,
    "mutual_headon_sigma_deg": 20.0,
    "mutual_headon_range_m": 2500.0,
    "mutual_headon_range_width_m": 500.0,

    # Actual health exchange.  Full health differential contributes about +/-40
    # before terminal reward.  Larger values previously encouraged close fly-through.
    "damage_scale": 50.0,
    "time_penalty_per_s": 0.03,

    # Paper deck term translated relative to the Release 300-m hard floor.
    "hard_deck_m": 300.0,
    "deck_warning_offset_m": 300.0 * FEET_TO_METER,
    "deck_transition_m": 20.0 * FEET_TO_METER,

    # Negative-only dynamic safety envelope.  Normal tactical maneuvering at
    # safe altitude is neutral; no survival or level-flight bonus exists.
    "safety_envelope_start_m": 5000.0,
    "safety_envelope_rate_per_s": 3.0,
    "safe_descent_mps": 25.0,
    "descent_span_mps": 75.0,
    "safe_nose_down_deg": 10.0,
    "nose_down_span_deg": 40.0,
    "safe_roll_deg": 65.0,
    "roll_span_deg": 45.0,

    # Competition outcomes.
    "win_reward": 250.0,
    "target_evasion_reward": 250.0,
    "loss_reward": -250.0,
    "draw_reward": -100.0,
    "own_crash_penalty": -400.0,
    "target_deck_reward": 100.0,
    "guard_fail_penalty": -180.0,
    "range_discipline_penalty": -180.0,
    "curriculum_success_reward": 0.0,

    # Updated infinite-evasion notice: 15 s observation + 15 s warning.
    # The tracker is deliberately conservative and recognises confirmed turns,
    # energy recovery, altitude gain, stall recovery, and active pursuit.
    "evasion_detection_s": 15.0,
    "evasion_warning_s": 15.0,
    "evasion_warning_penalty_per_s": 2.0,
    "evasion_away_cone_deg": 30.0,
    "evasion_pursuit_cone_deg": 60.0,
    "straight_track_rate_deg_s": 2.0,
    "track_rate_filter_s": 0.25,
    "maneuver_confirmation_s": 1.0,
    "evasion_evidence_decay_rate": 2.0,
    "engagement_evidence_decay_rate": 5.0,
    "recovery_grace_s": 5.0,
    "recovery_recharge_s": 15.0,
    "recovery_specific_energy_rate_mps": 5.0,
    "stall_recovery_speed_mps": 180.0,
    "stall_recovery_accel_mps2": 2.0,

    # Curriculum-only auxiliaries.  They default to zero, so the final V5
    # objective above is unchanged.  Early stages may enable them explicitly
    # with allow_external_override=1 while the actor learns basic airmanship.
    "survival_reward_per_s": 0.0,
    "stable_flight_reward_per_s": 0.0,
    "stable_roll_ref_deg": 35.0,
    "stable_pitch_ref_deg": 15.0,
    "stable_vertical_rate_ref_mps": 20.0,
    "altitude_hold_target_m": 0.0,
    "altitude_hold_band_m": 750.0,
    "altitude_hold_reward_per_s": 0.0,
    "altitude_error_penalty_per_s": 0.0,
    "attitude_delta_penalty_scale": 0.0,
    "attitude_delta_roll_ref_deg": 8.0,
    "attitude_delta_pitch_ref_deg": 4.0,
    "control_magnitude_penalty_scale": 0.0,
    "control_delta_penalty_scale": 0.0,
}



def _external_override_enabled(config: dict | None) -> bool:
    if not config:
        return False
    try:
        return float(config.get("allow_external_override", 0.0)) >= 0.5
    except (TypeError, ValueError):
        return False


def _cfg(config: dict | None, key: str) -> float:
    default = float(MY_REWARD_CONFIG[key])
    if _external_override_enabled(config) and key in config:
        try:
            value = float(config[key])
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default
    return default


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _clip(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _clip01(value: float) -> float:
    return _clip(value, 0.0, 1.0)


def _safe_sigmoid_z(z: float) -> float:
    if z >= 60.0:
        return 1.0
    if z <= -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _sigmoid(x: float, alpha: float, x0: float) -> float:
    return _safe_sigmoid_z(float(alpha) * (float(x) - float(x0)))


def _smooth_decreasing(x: float, center: float, width: float) -> float:
    return _safe_sigmoid_z((float(center) - float(x)) / max(abs(float(width)), 1.0e-6))


def _bounded_log_angle_error(angle_deg: float, reference_deg: float, max_deg: float) -> float:
    """Smooth [0, 1] error that keeps distinguishing 3, 10 and 30 degrees.

    The previous clipped square became exactly one above 3 degrees, so most of
    a failed engagement received an almost constant penalty.  log1p keeps the
    signal bounded without deleting recovery gradients at large ATA.
    """
    reference = max(abs(float(reference_deg)), 1.0e-3)
    maximum = max(abs(float(max_deg)), reference)
    angle = min(abs(float(angle_deg)), maximum)
    denominator = math.log1p((maximum / reference) ** 2)
    if denominator <= EPS:
        return 0.0
    return _clip01(math.log1p((angle / reference) ** 2) / denominator)


def _ft(value: float) -> float:
    return float(value) * FEET_TO_METER


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _neg(v: tuple[float, float, float]) -> tuple[float, float, float]:
    return (-v[0], -v[1], -v[2])


def _dot(a: tuple[float, float, float] | None, b: tuple[float, float, float] | None) -> float:
    """Clipped dot product for unit-direction comparisons."""
    if a is None or b is None:
        return -1.0
    return _clip(a[0] * b[0] + a[1] * b[1] + a[2] * b[2], -1.0, 1.0)


def _dot_raw(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Unclipped dot product for metric quantities such as closure speed."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(max(v[0] * v[0] + v[1] * v[1] + v[2] * v[2], 0.0))


def _unit(v: tuple[float, float, float]) -> tuple[float, float, float] | None:
    magnitude = _norm(v)
    if magnitude <= EPS:
        return None
    return (v[0] / magnitude, v[1] / magnitude, v[2] / magnitude)


def _angle_deg(a: tuple[float, float, float] | None, b: tuple[float, float, float] | None) -> float:
    if a is None or b is None:
        return 0.0
    return math.degrees(math.acos(_dot(a, b)))


def _position(state: Any) -> tuple[float, float, float]:
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


def _body_velocity_ned(state: Any) -> tuple[float, float, float]:
    try:
        u = _finite(state[BODY_U], float("nan"))
        v = _finite(state[BODY_V], float("nan"))
        w = _finite(state[BODY_W], float("nan"))
    except (IndexError, TypeError):
        u = v = w = float("nan")

    if all(math.isfinite(x) for x in (u, v, w)) and abs(u) + abs(v) + abs(w) > EPS:
        roll = math.radians(_finite(state[StateIndex.ROLL]))
        pitch = math.radians(_finite(state[StateIndex.PITCH]))
        yaw = math.radians(_finite(state[StateIndex.YAW]))
        cphi, sphi = math.cos(roll), math.sin(roll)
        cth, sth = math.cos(pitch), math.sin(pitch)
        cpsi, spsi = math.cos(yaw), math.sin(yaw)
        return (
            cth * cpsi * u + (sphi * sth * cpsi - cphi * spsi) * v + (cphi * sth * cpsi + sphi * spsi) * w,
            cth * spsi * u + (sphi * sth * spsi + cphi * cpsi) * v + (cphi * sth * spsi - sphi * cpsi) * w,
            -sth * u + sphi * cth * v + cphi * cth * w,
        )

    speed = max(0.0, _finite(state[StateIndex.KCAS]))
    forward = _forward_ned(state)
    return (speed * forward[0], speed * forward[1], speed * forward[2])


def _altitude_m(state: Any) -> float:
    altitude = _finite(state[StateIndex.ALT])
    return altitude if altitude > 1.0 else max(0.0, -_finite(state[StateIndex.D]))


def _speed_mps(state: Any) -> float:
    # Body u/v/w are already converted to m/s by FighterSim and are a better
    # energy proxy than calibrated airspeed.  KCAS remains the fallback for
    # standalone/unit-test states without a valid body-velocity vector.
    velocity_speed = _norm(_body_velocity_ned(state))
    if velocity_speed > EPS:
        return velocity_speed
    return max(0.0, _finite(state[StateIndex.KCAS]))


def _sim_time_s(state: Any) -> float:
    try:
        return max(0.0, _finite(state[StateIndex.SIM_TIME]))
    except (IndexError, TypeError):
        return 0.0


def _geometry(ownship_state: Any, target_state: Any) -> dict[str, float]:
    own_pos = _position(ownship_state)
    target_pos = _position(target_state)
    los = _sub(target_pos, own_pos)
    distance = _norm(los)
    los_hat = _unit(los) or (1.0, 0.0, 0.0)

    own_nose = _forward_ned(ownship_state)
    target_nose = _forward_ned(target_state)
    target_to_own = _neg(los_hat)
    own_tail = _neg(own_nose)
    target_tail = _neg(target_nose)

    own_track_deg = _angle_deg(own_nose, los_hat)
    # Blue adverse angle: blue position relative to red's tail.
    adverse_deg = _angle_deg(target_tail, target_to_own)
    red_track_deg = _angle_deg(target_nose, target_to_own)
    # Mirrored red adverse angle: red position relative to blue's tail.
    red_adverse_deg = _angle_deg(own_tail, los_hat)

    own_vel = _body_velocity_ned(ownship_state)
    target_vel = _body_velocity_ned(target_state)
    # Positive closure means decreasing range.
    closure_mps = _dot_raw(_sub(own_vel, target_vel), los_hat)

    return {
        "distance_m": distance,
        "own_track_deg": _clip(own_track_deg, 0.0, 180.0),
        "adverse_deg": _clip(adverse_deg, 0.0, 180.0),
        "red_track_deg": _clip(red_track_deg, 0.0, 180.0),
        "red_adverse_deg": _clip(red_adverse_deg, 0.0, 180.0),
        "closure_mps": closure_mps,
        "own_down_mps": max(0.0, own_vel[2]),
    }


def _gamma_blue(distance_m: float, beta: float) -> float:
    if distance_m < _ft(1950.0):
        return beta * _sigmoid(distance_m, 1.0 / _ft(50.0), _ft(1000.0))
    return beta * (1.0 - _sigmoid(distance_m, 1.0 / _ft(50.0), _ft(2900.0)))


def _gamma_red(distance_m: float, beta_magnitude: float) -> float:
    if distance_m < _ft(2250.0):
        return beta_magnitude * _sigmoid(distance_m, 1.0 / _ft(35.0), _ft(400.0))
    return beta_magnitude * (1.0 - _sigmoid(distance_m, 1.0 / _ft(200.0), _ft(4100.0)))


@dataclass
class _RuleState:
    initialized: bool = False
    previous_time_s: float = 0.0
    previous_distance_m: float = 0.0
    previous_own_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    previous_target_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    previous_own_track: tuple[float, float, float] | None = None
    previous_target_track: tuple[float, float, float] | None = None
    own_track_rate_ema: float = 0.0
    target_track_rate_ema: float = 0.0
    own_turn_run_s: float = 0.0
    target_turn_run_s: float = 0.0
    previous_own_speed: float = 0.0
    previous_target_speed: float = 0.0
    previous_own_energy: float = 0.0
    previous_target_energy: float = 0.0
    own_recovery_used_s: float = 0.0
    target_recovery_used_s: float = 0.0
    own_non_evasion_s: float = 0.0
    target_non_evasion_s: float = 0.0
    own_evasion_s: float = 0.0
    target_evasion_s: float = 0.0
    mutual_evasion_s: float = 0.0
    own_forfeit_latched: bool = False
    target_forfeit_latched: bool = False
    mutual_draw_latched: bool = False
    ended: bool = False
    previous_maneuver_potential: float = float("nan")

    def reset(self, ownship_state: Any, target_state: Any, distance_m: float) -> None:
        self.__dict__.update(_RuleState().__dict__)
        self.initialized = True
        self.previous_time_s = _sim_time_s(ownship_state)
        self.previous_distance_m = float(distance_m)
        self.previous_own_pos = _position(ownship_state)
        self.previous_target_pos = _position(target_state)
        self.previous_own_track = _unit(_body_velocity_ned(ownship_state))
        self.previous_target_track = _unit(_body_velocity_ned(target_state))
        self.previous_own_speed = _speed_mps(ownship_state)
        self.previous_target_speed = _speed_mps(target_state)
        self.previous_own_energy = _altitude_m(ownship_state) + self.previous_own_speed ** 2 / (2.0 * G)
        self.previous_target_energy = _altitude_m(target_state) + self.previous_target_speed ** 2 / (2.0 * G)


@dataclass(frozen=True)
class _RuleOutput:
    dt_s: float
    warning_penalty: float
    adjudication_reward: float
    adjudication: str


_WEAK_TRACKERS: "weakref.WeakKeyDictionary[Any, _RuleState]" = weakref.WeakKeyDictionary()
_FALLBACK_TRACKERS: dict[int, _RuleState] = {}


def _tracker_for(geo_info: Any) -> _RuleState:
    try:
        tracker = _WEAK_TRACKERS.get(geo_info)
        if tracker is None:
            tracker = _RuleState()
            _WEAK_TRACKERS[geo_info] = tracker
        return tracker
    except TypeError:
        key = id(geo_info)
        tracker = _FALLBACK_TRACKERS.get(key)
        if tracker is None:
            tracker = _RuleState()
            _FALLBACK_TRACKERS[key] = tracker
        return tracker


def _low_pass(previous: float, current: float, dt_s: float, tau_s: float) -> float:
    if tau_s <= EPS:
        return float(current)
    alpha = _clip01(dt_s / (tau_s + dt_s))
    return float(previous) + alpha * (float(current) - float(previous))


def _warning_fraction(timer_s: float, detection_s: float, warning_s: float) -> float:
    if warning_s <= EPS:
        return float(timer_s >= detection_s)
    return _clip01((timer_s - detection_s) / warning_s)


def _advance_evidence(
    value_s: float,
    condition: bool,
    hold: bool,
    damage_event: bool,
    dt_s: float,
    config: dict | None,
    maximum_s: float,
) -> float:
    if condition:
        return min(maximum_s, value_s + dt_s)
    if hold and not damage_event:
        return value_s
    rate = (
        _cfg(config, "engagement_evidence_decay_rate")
        if damage_event
        else _cfg(config, "evasion_evidence_decay_rate")
    )
    return max(0.0, value_s - rate * dt_s)


def _rule_step(
    tracker: _RuleState,
    ownship_state: Any,
    target_state: Any,
    ownship_damage: float,
    target_damage: float,
    distance_m: float,
    reward_config: dict | None,
    external_end_condition: str,
) -> _RuleOutput:
    fallback_dt = max(_cfg(reward_config, "fallback_dt_s"), 1.0e-3)
    now = _sim_time_s(ownship_state)

    # Same GeometryInfo object is reused across episodes.  A lower simulation
    # time, or the first call after a terminal call, starts a fresh tracker.
    if (
        not tracker.initialized
        or tracker.ended
        or now + 1.0e-6 < tracker.previous_time_s
    ):
        tracker.reset(ownship_state, target_state, distance_m)
        return _RuleOutput(fallback_dt, 0.0, 0.0, "")

    dt_s = now - tracker.previous_time_s
    if not math.isfinite(dt_s) or dt_s <= 1.0e-5 or dt_s > 1.0:
        dt_s = fallback_dt

    own_pos = _position(ownship_state)
    target_pos = _position(target_state)
    own_track = _unit(_sub(own_pos, tracker.previous_own_pos))
    target_track = _unit(_sub(target_pos, tracker.previous_target_pos))
    if own_track is None:
        own_track = _unit(_body_velocity_ned(ownship_state))
    if target_track is None:
        target_track = _unit(_body_velocity_ned(target_state))

    own_rate = _angle_deg(tracker.previous_own_track, own_track) / max(dt_s, EPS)
    target_rate = _angle_deg(tracker.previous_target_track, target_track) / max(dt_s, EPS)
    tau = _cfg(reward_config, "track_rate_filter_s")
    tracker.own_track_rate_ema = _low_pass(tracker.own_track_rate_ema, own_rate, dt_s, tau)
    tracker.target_track_rate_ema = _low_pass(tracker.target_track_rate_ema, target_rate, dt_s, tau)

    straight_limit = _cfg(reward_config, "straight_track_rate_deg_s")
    own_straight = tracker.own_track_rate_ema <= straight_limit
    target_straight = tracker.target_track_rate_ema <= straight_limit
    tracker.own_turn_run_s = 0.0 if own_straight else tracker.own_turn_run_s + dt_s
    tracker.target_turn_run_s = 0.0 if target_straight else tracker.target_turn_run_s + dt_s

    los_target_to_own = _unit(_sub(own_pos, target_pos))
    los_own_to_target = _neg(los_target_to_own) if los_target_to_own is not None else None
    own_away = _dot(own_track, los_target_to_own)
    own_pursuit = _dot(own_track, los_own_to_target)
    target_away = _dot(target_track, los_own_to_target)
    target_pursuit = _dot(target_track, los_target_to_own)

    away_cos = math.cos(math.radians(_cfg(reward_config, "evasion_away_cone_deg")))
    pursuit_cos = math.cos(math.radians(_cfg(reward_config, "evasion_pursuit_cone_deg")))
    damage_event = _finite(ownship_damage) > EPS or _finite(target_damage) > EPS

    own_speed = _speed_mps(ownship_state)
    target_speed = _speed_mps(target_state)
    own_energy = _altitude_m(ownship_state) + own_speed ** 2 / (2.0 * G)
    target_energy = _altitude_m(target_state) + target_speed ** 2 / (2.0 * G)
    own_energy_rate = (own_energy - tracker.previous_own_energy) / max(dt_s, EPS)
    target_energy_rate = (target_energy - tracker.previous_target_energy) / max(dt_s, EPS)
    own_accel = (own_speed - tracker.previous_own_speed) / max(dt_s, EPS)
    target_accel = (target_speed - tracker.previous_target_speed) / max(dt_s, EPS)

    own_recovery_candidate = (
        own_energy_rate >= _cfg(reward_config, "recovery_specific_energy_rate_mps")
        or (
            own_speed <= _cfg(reward_config, "stall_recovery_speed_mps")
            and own_accel >= _cfg(reward_config, "stall_recovery_accel_mps2")
        )
    )
    target_recovery_candidate = (
        target_energy_rate >= _cfg(reward_config, "recovery_specific_energy_rate_mps")
        or (
            target_speed <= _cfg(reward_config, "stall_recovery_speed_mps")
            and target_accel >= _cfg(reward_config, "stall_recovery_accel_mps2")
        )
    )

    grace = _cfg(reward_config, "recovery_grace_s")
    own_recovery = own_recovery_candidate and tracker.own_recovery_used_s < grace
    target_recovery = target_recovery_candidate and tracker.target_recovery_used_s < grace
    if own_recovery:
        tracker.own_recovery_used_s = min(grace, tracker.own_recovery_used_s + dt_s)
    if target_recovery:
        tracker.target_recovery_used_s = min(grace, tracker.target_recovery_used_s + dt_s)

    mutual_candidate = (
        own_straight
        and target_straight
        and own_away >= away_cos
        and target_away >= away_cos
        and not damage_event
    )
    own_candidate = (
        not mutual_candidate
        and own_straight
        and own_away >= away_cos
        and target_pursuit >= pursuit_cos
        and not damage_event
    )
    target_candidate = (
        not mutual_candidate
        and target_straight
        and target_away >= away_cos
        and own_pursuit >= pursuit_cos
        and not damage_event
    )

    # Genuine positioning/approach and confirmed reattack turns recharge the
    # bounded recovery budget.  A one-frame token turn only holds evidence.
    confirm_s = _cfg(reward_config, "maneuver_confirmation_s")
    own_confirmed_turn = tracker.own_turn_run_s >= confirm_s
    target_confirmed_turn = tracker.target_turn_run_s >= confirm_s
    own_holding_turn = (not own_straight) and not own_confirmed_turn
    target_holding_turn = (not target_straight) and not target_confirmed_turn

    own_clear = own_candidate and not own_recovery
    target_clear = target_candidate and not target_recovery
    mutual_clear = mutual_candidate and not own_recovery and not target_recovery

    maximum_s = _cfg(reward_config, "evasion_detection_s") + _cfg(reward_config, "evasion_warning_s")
    tracker.own_evasion_s = _advance_evidence(
        tracker.own_evasion_s,
        own_clear,
        own_recovery or own_holding_turn,
        damage_event,
        dt_s,
        reward_config,
        maximum_s,
    )
    tracker.target_evasion_s = _advance_evidence(
        tracker.target_evasion_s,
        target_clear,
        target_recovery or target_holding_turn,
        damage_event,
        dt_s,
        reward_config,
        maximum_s,
    )
    tracker.mutual_evasion_s = _advance_evidence(
        tracker.mutual_evasion_s,
        mutual_clear,
        own_recovery or target_recovery or own_holding_turn or target_holding_turn,
        damage_event,
        dt_s,
        reward_config,
        maximum_s,
    )

    own_real_non_evasion = damage_event or own_confirmed_turn or own_pursuit >= pursuit_cos
    target_real_non_evasion = damage_event or target_confirmed_turn or target_pursuit >= pursuit_cos
    tracker.own_non_evasion_s = tracker.own_non_evasion_s + dt_s if own_real_non_evasion else 0.0
    tracker.target_non_evasion_s = tracker.target_non_evasion_s + dt_s if target_real_non_evasion else 0.0
    recharge_s = _cfg(reward_config, "recovery_recharge_s")
    if tracker.own_non_evasion_s >= recharge_s:
        tracker.own_recovery_used_s = 0.0
    if tracker.target_non_evasion_s >= recharge_s:
        tracker.target_recovery_used_s = 0.0

    detect = _cfg(reward_config, "evasion_detection_s")
    warning = _cfg(reward_config, "evasion_warning_s")
    own_warning = _warning_fraction(tracker.own_evasion_s, detect, warning)
    mutual_warning = _warning_fraction(tracker.mutual_evasion_s, detect, warning)
    warning_penalty = (
        -_cfg(reward_config, "evasion_warning_penalty_per_s")
        * max(own_warning, mutual_warning)
        * dt_s
    )

    adjudication = ""
    adjudication_reward = 0.0
    external_end = str(external_end_condition or "").lower()
    external_evasion = "evasion" in external_end
    if maximum_s > EPS and not external_evasion:
        if tracker.mutual_evasion_s >= maximum_s and not tracker.mutual_draw_latched:
            tracker.mutual_draw_latched = True
            adjudication = "mutual infinite evasion"
            adjudication_reward = _cfg(reward_config, "draw_reward")
        elif tracker.own_evasion_s >= maximum_s and not tracker.own_forfeit_latched:
            tracker.own_forfeit_latched = True
            adjudication = "ownship infinite evasion"
            adjudication_reward = _cfg(reward_config, "loss_reward")
        elif tracker.target_evasion_s >= maximum_s and not tracker.target_forfeit_latched:
            tracker.target_forfeit_latched = True
            adjudication = "target infinite evasion"
            adjudication_reward = _cfg(reward_config, "target_evasion_reward")

    tracker.previous_time_s = now
    tracker.previous_distance_m = distance_m
    tracker.previous_own_pos = own_pos
    tracker.previous_target_pos = target_pos
    tracker.previous_own_track = own_track
    tracker.previous_target_track = target_track
    tracker.previous_own_speed = own_speed
    tracker.previous_target_speed = target_speed
    tracker.previous_own_energy = own_energy
    tracker.previous_target_energy = target_energy

    return _RuleOutput(dt_s, warning_penalty, adjudication_reward, adjudication)


def _terminal_reward(
    ownship_state: Any,
    target_state: Any,
    terminated: bool,
    truncated: bool,
    end_condition: str,
    reward_config: dict | None,
) -> float:
    if not (terminated or truncated):
        return 0.0

    own_health = _finite(ownship_state[StateIndex.HEALTH])
    target_health = _finite(target_state[StateIndex.HEALTH])
    end = str(end_condition or "").strip().lower()

    # Server-side response/no-response adjudication.  Wall-clock latency itself
    # is not available here, but explicit server outcomes are mapped exactly.
    response_tokens = ("no response", "no-response", "response fail", "response failure", "unresponsive")
    if any(token in end for token in response_tokens):
        if "mutual" in end or "both" in end:
            return _cfg(reward_config, "draw_reward")
        if "target" in end or "opponent" in end or "red" in end:
            return _cfg(reward_config, "win_reward")
        if "ownship" in end or "blue" in end or "client 1" in end:
            return _cfg(reward_config, "loss_reward")

    if "target" in end and "evasion" in end:
        return _cfg(reward_config, "target_evasion_reward")
    if "ownship" in end and "evasion" in end:
        return _cfg(reward_config, "loss_reward")
    if "mutual" in end and "evasion" in end:
        return _cfg(reward_config, "draw_reward")
    if end == "range discipline":
        return _cfg(reward_config, "range_discipline_penalty")
    if end in {"position acquired", "position capture", "curriculum success"}:
        return _cfg(reward_config, "curriculum_success_reward")
    if "two circle" in end and "guard" in end:
        return _cfg(reward_config, "guard_fail_penalty")

    if target_health <= 0.0 < own_health:
        return _cfg(reward_config, "win_reward")
    if own_health <= 0.0 < target_health:
        return _cfg(reward_config, "loss_reward")

    if "ownship altitude" in end or "ownship fdm" in end or end == "fdm update fail":
        return _cfg(reward_config, "own_crash_penalty")
    if "target altitude" in end or "target fdm" in end:
        return _cfg(reward_config, "target_deck_reward")
    if "fuel fail" in end:
        return _cfg(reward_config, "loss_reward")

    return _cfg(reward_config, "draw_reward")


def _compute_reward(
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
    tracker: _RuleState,
    rule: _RuleOutput | None = None,
    include_adjudication_reward: bool = True,
) -> tuple[float, dict]:
    """Compute a single-policy offense-and-defense reward.

    Every tactical term is active in the same policy.  There is no role flag,
    policy selector, or neural-network switch.
    """
    geometry = _geometry(ownship_state, target_state)
    distance_m = geometry["distance_m"]
    own_track_deg = geometry["own_track_deg"]
    blue_adverse_deg = geometry["adverse_deg"]
    red_track_deg = geometry["red_track_deg"]
    red_adverse_deg = geometry["red_adverse_deg"]
    closure_mps = geometry["closure_mps"]

    if rule is None:
        rule = _rule_step(
            tracker,
            ownship_state,
            target_state,
            ownship_damage,
            target_damage,
            distance_m,
            reward_config,
            end_condition,
        )
    dt_s = rule.dt_s

    r_min = max(_finite(wez_config.get("min_range_m"), _ft(500.0)), 1.0)
    r_max = max(_finite(wez_config.get("max_range_m"), _ft(3000.0)), r_min + 1.0)
    half_angle_deg = max(_finite(wez_config.get("angle_deg"), 2.0) * 0.5, 0.05)

    own_theta = own_track_deg / 180.0
    blue_phi = blue_adverse_deg / 180.0
    red_theta = red_track_deg / 180.0
    red_phi = red_adverse_deg / 180.0
    blue_phi_sigmoid = _sigmoid(blue_phi, 18.0, 0.5)
    red_phi_sigmoid = _sigmoid(red_phi, 18.0, 0.5)
    blue_tail_gate = 1.0 - blue_phi_sigmoid
    red_tail_gate = 1.0 - red_phi_sigmoid

    # ------------------- Symmetric adaptation of Table I -------------------
    # Eq. (10) evaluated from both aircraft perspectives.  Half the difference
    # is in [-1, 1]: +1 means blue controls red's tail, -1 means red controls
    # blue's tail, and symmetric/head-on geometry is near zero.
    blue_relative = (own_theta - 2.0) * blue_phi_sigmoid - own_theta + 1.0
    red_relative = (red_theta - 2.0) * red_phi_sigmoid - red_theta + 1.0
    relative_advantage = 0.5 * (blue_relative - red_relative)

    # Context-conditioned closure control.  The previous reward paid raw
    # closing whenever blue was roughly behind red, which encouraged one-pass
    # fly-through.  Here the reward specifies a desired closure instead:
    #   * attack geometry -> close rapidly when far, then decelerate near WEZ;
    #   * red tracking threat -> open the range;
    #   * poor geometry with no immediate threat -> do not charge blindly.
    blue_pursuit_quality = blue_tail_gate * math.exp(
        -0.5 * (own_track_deg / 30.0) ** 2
    )
    red_threat_quality = (
        red_tail_gate
        * math.exp(
            -0.5
            * (
                red_track_deg
                / max(_cfg(reward_config, "defensive_threat_sigma_deg"), 0.1)
            ) ** 2
        )
        * _smooth_decreasing(
            distance_m,
            _cfg(reward_config, "defensive_threat_range_m"),
            _cfg(reward_config, "defensive_threat_range_width_m"),
        )
    )

    hold_fraction = _clip01(_cfg(reward_config, "hold_range_fraction"))
    hold_range_m = r_min + hold_fraction * (r_max - r_min)
    attack_desired_closure = _clip(
        _cfg(reward_config, "desired_closure_gain_per_s")
        * (distance_m - hold_range_m),
        -_cfg(reward_config, "desired_opening_max_mps"),
        _cfg(reward_config, "desired_closure_max_mps"),
    ) * blue_pursuit_quality
    defensive_desired_closure = (
        -_cfg(reward_config, "desired_opening_max_mps")
        * red_threat_quality
    )
    # A genuine red threat takes precedence.  Otherwise blue closes only in
    # proportion to its own tail/nose quality.
    threat_mix = _clip01(red_threat_quality / max(red_threat_quality + blue_pursuit_quality, EPS))
    desired_closure_mps = (
        (1.0 - threat_mix) * attack_desired_closure
        + threat_mix * defensive_desired_closure
    )
    closure_reference = max(_cfg(reward_config, "closure_reference_mps"), 1.0)
    closure_error = _clip(
        (closure_mps - desired_closure_mps) / closure_reference,
        -1.0,
        1.0,
    )
    closure_relevance = _clip01(
        max(blue_pursuit_quality, red_threat_quality)
        + 0.25 * _smooth_decreasing(distance_m, 2500.0, 600.0)
    )
    closure_control = (
        -_cfg(reward_config, "closure_control_penalty_per_s")
        * closure_relevance
        * closure_error * closure_error
        * dt_s
    )

    # Mirrored Eq. (9): positive when blue's nose alignment is better than red's.
    track_lambda = max(_cfg(reward_config, "shooter_track_lambda"), 0.05)
    blue_track_reward = -(own_theta ** track_lambda)
    red_track_reward = -(red_theta ** track_lambda)
    track_advantage = blue_track_reward - red_track_reward

    # Long-range single-policy maneuver potential.  This is not a selector and
    # does not switch networks: it is one symmetric scalar evaluated from both
    # aircraft perspectives.  Blue gains potential by improving its own tail/
    # nose geometry or by removing red's corresponding advantage.
    maneuver_raw = _clip(
        _cfg(reward_config, "maneuver_position_weight") * relative_advantage
        + _cfg(reward_config, "maneuver_track_weight") * track_advantage,
        -1.0,
        1.0,
    )
    maneuver_potential = 0.5 * (maneuver_raw + 1.0)

    # Absolute positive geometry should matter only in a real engagement.  At
    # long range, it is reduced and closure supplies the approach/escape signal.
    combat_gate = _smooth_decreasing(
        distance_m,
        _cfg(reward_config, "combat_gate_center_m"),
        _cfg(reward_config, "combat_gate_width_m"),
    )

    # Far/mid-range acquisition and defensive recovery use only improvement,
    # not absolute position.  Hence a distant tail position held stationary
    # yields zero maneuver reward, while turning from head-on toward advantage
    # is positive and allowing red onto blue's tail is negative.  The term fades
    # near the official WEZ, where gun-snap, aim, threat and damage take over.
    previous_potential = tracker.previous_maneuver_potential
    if math.isfinite(previous_potential):
        maneuver_progress = (
            _cfg(reward_config, "maneuver_progress_scale")
            * (1.0 - combat_gate)
            * (maneuver_potential - previous_potential)
        )
    else:
        maneuver_progress = 0.0
    tracker.previous_maneuver_potential = maneuver_potential

    in_official_range = r_min <= distance_m <= r_max
    damage_potential = _clip01((r_max - distance_m) / max(r_max - r_min, 1.0))

    # A positive advantage outside the official WEZ is paid only while actually
    # entering the short pre-WEZ bridge.  A negative advantage is always kept,
    # so being pursued remains costly and creates an evasion signal.
    positive_advantage_gate = 1.0 if in_official_range else 0.0
    bridge_gate = 0.0
    bridge_m = max(_cfg(reward_config, "pre_wez_bridge_m"), 1.0)
    if r_max < distance_m <= r_max + bridge_m:
        range_fraction = _clip01((r_max + bridge_m - distance_m) / bridge_m)
        closing_fraction = _clip01(
            closure_mps / max(_cfg(reward_config, "pre_wez_closure_ref_mps"), 1.0)
        )
        bridge_gate = (
            _cfg(reward_config, "pre_wez_bridge_fraction")
            * range_fraction
            * closing_fraction
        )
        positive_advantage_gate = bridge_gate

    def _gate_signed_advantage(value: float) -> float:
        return min(value, 0.0) + positive_advantage_gate * max(value, 0.0)

    relative_shaped = combat_gate * _gate_signed_advantage(relative_advantage)
    track_shaped = combat_gate * _gate_signed_advantage(track_advantage)

    # -------------------------- Official gun snaps -------------------------
    blue_angle_score = 1.0 - _sigmoid(own_theta, 1.0e5, half_angle_deg / 180.0)
    red_angle_score = 1.0 - _sigmoid(red_theta, 1.0e5, half_angle_deg / 180.0)
    snap_alignment = (
        damage_potential
        if _cfg(reward_config, "align_gunsnap_with_damage") >= 0.5
        else 1.0
    )
    blue_snap = (
        _gamma_blue(distance_m, _cfg(reward_config, "blue_beta"))
        * blue_angle_score
        * snap_alignment
        if in_official_range and own_track_deg <= half_angle_deg
        else 0.0
    )
    red_snap = (
        -_gamma_red(distance_m, _cfg(reward_config, "red_beta_magnitude"))
        * red_angle_score
        * snap_alignment
        if in_official_range and red_track_deg <= half_angle_deg
        else 0.0
    )

    # Anti-overshoot guard tied to the official minimum range rather than to
    # a generic absolute-distance reward.  It is active only when blue is in a
    # plausible pursuit geometry and becomes strong before crossing r_min.
    too_close_center_m = r_min + _cfg(reward_config, "too_close_margin_m")
    too_close_gate = _safe_sigmoid_z(
        (too_close_center_m - distance_m)
        / max(_cfg(reward_config, "too_close_width_m"), 1.0)
    )
    too_close = (
        -_cfg(reward_config, "too_close_penalty_per_s")
        * blue_pursuit_quality
        * too_close_gate
        * dt_s
    )

    # ------------------- Precision offense + dense defense -----------------
    # The previous clipped-square precision error saturated above 3 degrees.
    # This bounded logarithmic error still distinguishes 3, 10 and 30 degrees,
    # while the positive solution quality below is paid only in official range.
    precision_gate = 1.0 if in_official_range else bridge_gate
    precision_error = _bounded_log_angle_error(
        own_track_deg,
        _cfg(reward_config, "precision_log_reference_deg"),
        _cfg(reward_config, "precision_log_max_deg"),
    )
    enemy_lock_quality = math.exp(
        -0.5
        * (
            red_track_deg
            / max(_cfg(reward_config, "enemy_threat_sigma_deg"), 0.1)
        ) ** 2
    )
    precision_aim = (
        -_cfg(reward_config, "precision_aim_penalty_per_s")
        * precision_gate
        * precision_error
        * dt_s
    )
    enemy_lock = (
        -_cfg(reward_config, "enemy_lock_penalty_per_s")
        * red_threat_quality
        * dt_s
    )

    solution_sigma = max(_cfg(reward_config, "wez_solution_sigma_deg"), 0.1)
    blue_solution_quality = (
        damage_potential
        * blue_tail_gate
        * math.exp(-0.5 * (own_track_deg / solution_sigma) ** 2)
        if in_official_range
        else 0.0
    )
    red_solution_quality = (
        damage_potential
        * red_tail_gate
        * math.exp(-0.5 * (red_track_deg / solution_sigma) ** 2)
        if in_official_range
        else 0.0
    )
    wez_solution = (
        _cfg(reward_config, "wez_solution_reward_per_s")
        * (blue_solution_quality - red_solution_quality)
        * dt_s
    )

    # A limited extension is allowed while red is dangerous.  Once red's lock
    # is broken, continuing to open beyond the re-engagement radius is costly.
    opening_norm = _clip01(
        -closure_mps / max(_cfg(reward_config, "pre_wez_closure_ref_mps"), 1.0)
    )
    post_threat_distance_gate = _safe_sigmoid_z(
        (
            distance_m - _cfg(reward_config, "post_threat_opening_start_m")
        )
        / max(_cfg(reward_config, "post_threat_opening_width_m"), 1.0)
    )
    post_threat_opening = (
        -_cfg(reward_config, "post_threat_opening_penalty_per_s")
        * (1.0 - red_threat_quality)
        * post_threat_distance_gate
        * opening_norm
        * dt_s
    )

    headon_sigma = max(_cfg(reward_config, "mutual_headon_sigma_deg"), 0.1)
    mutual_headon_quality = (
        math.exp(-0.5 * (own_track_deg / headon_sigma) ** 2)
        * math.exp(-0.5 * (red_track_deg / headon_sigma) ** 2)
        * _smooth_decreasing(
            distance_m,
            _cfg(reward_config, "mutual_headon_range_m"),
            _cfg(reward_config, "mutual_headon_range_width_m"),
        )
    )
    mutual_headon = (
        -_cfg(reward_config, "mutual_headon_penalty_per_s")
        * mutual_headon_quality
        * dt_s
    )

    # ------------------------------- Safety --------------------------------
    altitude_m = _altitude_m(ownship_state)
    deck_midpoint_m = (
        _cfg(reward_config, "hard_deck_m")
        + _cfg(reward_config, "deck_warning_offset_m")
    )
    deck = -4.0 * (
        1.0
        - _sigmoid(
            altitude_m,
            1.0 / max(_cfg(reward_config, "deck_transition_m"), 1.0),
            deck_midpoint_m,
        )
    )

    hard_deck = _cfg(reward_config, "hard_deck_m")
    safety_start = max(_cfg(reward_config, "safety_envelope_start_m"), hard_deck + 1.0)
    altitude_risk = _clip01((safety_start - altitude_m) / (safety_start - hard_deck))
    pitch_deg = _finite(ownship_state[StateIndex.PITCH])
    roll_deg = abs(_finite(ownship_state[StateIndex.ROLL]))
    nose_risk = _clip01(
        (max(0.0, -pitch_deg) - _cfg(reward_config, "safe_nose_down_deg"))
        / max(_cfg(reward_config, "nose_down_span_deg"), 1.0)
    )
    roll_risk = _clip01(
        (roll_deg - _cfg(reward_config, "safe_roll_deg"))
        / max(_cfg(reward_config, "roll_span_deg"), 1.0)
    )
    descent_risk = _clip01(
        (geometry["own_down_mps"] - _cfg(reward_config, "safe_descent_mps"))
        / max(_cfg(reward_config, "descent_span_mps"), 1.0)
    )
    dynamic_risk = max(nose_risk, 0.5 * roll_risk, descent_risk)
    dynamic_safety = (
        -_cfg(reward_config, "safety_envelope_rate_per_s")
        * altitude_risk
        * dynamic_risk * dynamic_risk
        * dt_s
    )

    reference_dt = max(_cfg(reward_config, "reference_dt_s"), 1.0e-6)
    table_scale = _cfg(reward_config, "table1_step_scale") * dt_s / reference_dt
    terminal = _terminal_reward(
        ownship_state,
        target_state,
        terminated,
        truncated,
        end_condition,
        reward_config,
    )

    components: dict[str, float] = {
        "time": -_cfg(reward_config, "time_penalty_per_s") * dt_s,
        "maneuver_progress": maneuver_progress,
        "relative_advantage": (
            table_scale
            * _cfg(reward_config, "relative_position_weight")
            * relative_shaped
        ),
        "closure_control": closure_control,
        "track_advantage": (
            table_scale
            * _cfg(reward_config, "track_advantage_weight")
            * track_shaped
        ),
        "precision_aim": precision_aim,
        "enemy_lock": enemy_lock,
        "wez_solution": wez_solution,
        "post_threat_opening": post_threat_opening,
        "mutual_headon": mutual_headon,
        "gunsnap_blue": table_scale * blue_snap,
        "gunsnap_red": table_scale * red_snap,
        "too_close": too_close,
        "safety": table_scale * deck + dynamic_safety,
        "damage": _cfg(reward_config, "damage_scale") * (
            _finite(target_damage) - _finite(ownship_damage)
        ),
        "evasion_warning": rule.warning_penalty,
        "evasion_adjudication": (
            rule.adjudication_reward if include_adjudication_reward else 0.0
        ),
        "terminal": terminal,
    }

    total = float(sum(components.values()))
    if not math.isfinite(total):
        fallback = _cfg(reward_config, "loss_reward")
        components = {name: 0.0 for name in components}
        components["terminal"] = fallback
        total = fallback

    if terminated or truncated:
        tracker.ended = True

    return total, components


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
    """Legacy stateless entry point retained for standalone validation."""
    return _compute_reward(
        ownship_state,
        target_state,
        ownship_damage,
        target_damage,
        geo_info,
        wez_config,
        reward_config,
        terminated,
        truncated,
        end_condition,
        tracker=_tracker_for(geo_info),
    )


class UnifiedCompetitionReward:
    """Single-policy competition reward, rule adjudicator, and telemetry."""

    def __init__(self) -> None:
        self._tracker = _RuleState()
        self._pending_time_s = float("nan")
        self._pending_rule: _RuleOutput | None = None
        self._previous_action: tuple[float, ...] | None = None
        self._previous_roll_deg = 0.0
        self._previous_pitch_deg = 0.0
        self._min_altitude_m = math.inf
        self._max_descent_rate_mps = 0.0
        self._max_abs_roll_deg = 0.0
        self._max_abs_pitch_deg = 0.0
        self._max_own_evasion_s = 0.0
        self._max_target_evasion_s = 0.0
        self._max_mutual_evasion_s = 0.0
        self._last_metrics: dict[str, Any] = {}

    def reset(
        self,
        ownship_state=None,
        target_state=None,
        geo_info=None,
        wez_config: dict | None = None,
        reward_config: dict | None = None,
    ) -> None:
        self.__init__()
        if ownship_state is None or target_state is None:
            return
        distance_m = _geometry(ownship_state, target_state)["distance_m"]
        self._tracker.reset(ownship_state, target_state, distance_m)
        altitude_m = _altitude_m(ownship_state)
        self._min_altitude_m = altitude_m
        self._previous_roll_deg = _finite(ownship_state[StateIndex.ROLL])
        self._previous_pitch_deg = _finite(ownship_state[StateIndex.PITCH])

    def _ensure_rule(
        self,
        ownship_state,
        target_state,
        ownship_damage: float,
        target_damage: float,
        reward_config: dict,
        end_condition: str = "",
    ) -> _RuleOutput:
        now = _sim_time_s(ownship_state)
        if self._pending_rule is not None and abs(now - self._pending_time_s) <= 1.0e-9:
            return self._pending_rule
        distance_m = _geometry(ownship_state, target_state)["distance_m"]
        rule = _rule_step(
            self._tracker,
            ownship_state,
            target_state,
            ownship_damage,
            target_damage,
            distance_m,
            reward_config,
            end_condition,
        )
        self._pending_time_s = now
        self._pending_rule = rule
        self._max_own_evasion_s = max(
            self._max_own_evasion_s, self._tracker.own_evasion_s
        )
        self._max_target_evasion_s = max(
            self._max_target_evasion_s, self._tracker.target_evasion_s
        )
        self._max_mutual_evasion_s = max(
            self._max_mutual_evasion_s, self._tracker.mutual_evasion_s
        )
        return rule

    def adjudicate(
        self,
        ownship_state,
        target_state,
        ownship_damage: float,
        target_damage: float,
        geo_info,
        wez_config: dict,
        reward_config: dict,
    ) -> dict[str, Any]:
        rule = self._ensure_rule(
            ownship_state,
            target_state,
            ownship_damage,
            target_damage,
            reward_config,
        )
        return {
            "terminated": bool(rule.adjudication),
            "truncated": False,
            "end_condition": rule.adjudication,
        }

    def _curriculum_components(
        self,
        ownship_state,
        reward_config: dict,
        dt_s: float,
        action: Any,
    ) -> dict[str, float]:
        vertical_rate_mps = _body_velocity_ned(ownship_state)[2]
        geometry_down_mps = max(0.0, vertical_rate_mps)
        roll_deg = _finite(ownship_state[StateIndex.ROLL])
        pitch_deg = _finite(ownship_state[StateIndex.PITCH])
        altitude_m = _altitude_m(ownship_state)

        stable_roll = 1.0 - _clip01(
            abs(roll_deg) / max(_cfg(reward_config, "stable_roll_ref_deg"), 1.0)
        )
        stable_pitch = 1.0 - _clip01(
            abs(pitch_deg) / max(_cfg(reward_config, "stable_pitch_ref_deg"), 1.0)
        )
        stable_vertical = 1.0 - _clip01(
            abs(vertical_rate_mps)
            / max(_cfg(reward_config, "stable_vertical_rate_ref_mps"), 1.0)
        )
        stable_score = stable_roll * stable_pitch * stable_vertical

        altitude_hold = 0.0
        altitude_target = _cfg(reward_config, "altitude_hold_target_m")
        if altitude_target > 0.0:
            band = max(_cfg(reward_config, "altitude_hold_band_m"), 1.0)
            error = abs(altitude_m - altitude_target)
            hold_score = 1.0 - _clip01(error / band)
            error_excess = _clip01((error - band) / band)
            altitude_hold = (
                _cfg(reward_config, "altitude_hold_reward_per_s") * hold_score
                - _cfg(reward_config, "altitude_error_penalty_per_s") * error_excess
            ) * dt_s

        roll_delta = abs(roll_deg - self._previous_roll_deg)
        pitch_delta = abs(pitch_deg - self._previous_pitch_deg)
        attitude_delta = -_cfg(reward_config, "attitude_delta_penalty_scale") * (
            roll_delta
            / max(_cfg(reward_config, "attitude_delta_roll_ref_deg"), 1.0)
            + pitch_delta
            / max(_cfg(reward_config, "attitude_delta_pitch_ref_deg"), 1.0)
        )

        action_values: tuple[float, ...] = ()
        if action is not None:
            try:
                action_values = tuple(_finite(value) for value in action)
            except TypeError:
                action_values = ()
        control_axes = action_values[:3]
        control_magnitude = -_cfg(
            reward_config, "control_magnitude_penalty_scale"
        ) * sum(value * value for value in control_axes)
        control_delta = 0.0
        if self._previous_action is not None and action_values:
            count = min(3, len(action_values), len(self._previous_action))
            control_delta = -_cfg(
                reward_config, "control_delta_penalty_scale"
            ) * sum(
                (action_values[index] - self._previous_action[index]) ** 2
                for index in range(count)
            )

        self._previous_action = action_values or self._previous_action
        self._previous_roll_deg = roll_deg
        self._previous_pitch_deg = pitch_deg
        self._min_altitude_m = min(self._min_altitude_m, altitude_m)
        self._max_descent_rate_mps = max(
            self._max_descent_rate_mps, geometry_down_mps
        )
        self._max_abs_roll_deg = max(self._max_abs_roll_deg, abs(roll_deg))
        self._max_abs_pitch_deg = max(self._max_abs_pitch_deg, abs(pitch_deg))
        self._last_metrics = {
            "own_altitude_m": altitude_m,
            "own_descent_rate_mps": geometry_down_mps,
            "stable_flight_score": stable_score,
            "own_roll_deg": roll_deg,
            "own_pitch_deg": pitch_deg,
        }
        return {
            "curriculum_survival": _cfg(
                reward_config, "survival_reward_per_s"
            ) * dt_s,
            "curriculum_stable_flight": _cfg(
                reward_config, "stable_flight_reward_per_s"
            ) * stable_score * dt_s,
            "curriculum_altitude_hold": altitude_hold,
            "curriculum_attitude_delta": attitude_delta,
            "curriculum_control_magnitude": control_magnitude,
            "curriculum_control_delta": control_delta,
        }

    def __call__(
        self,
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
    ) -> tuple[float, dict[str, float]]:
        rule = self._ensure_rule(
            ownship_state,
            target_state,
            ownship_damage,
            target_damage,
            reward_config,
            end_condition,
        )
        total, components = _compute_reward(
            ownship_state,
            target_state,
            ownship_damage,
            target_damage,
            geo_info,
            wez_config,
            reward_config,
            terminated,
            truncated,
            end_condition,
            tracker=self._tracker,
            rule=rule,
            include_adjudication_reward=False,
        )
        curriculum = self._curriculum_components(
            ownship_state, reward_config, rule.dt_s, action
        )
        components.update(curriculum)
        total = float(total + sum(curriculum.values()))
        return total, components

    def metrics(self) -> dict[str, Any]:
        metrics = {
            "own_evasion_s": self._tracker.own_evasion_s,
            "target_evasion_s": self._tracker.target_evasion_s,
            "mutual_evasion_s": self._tracker.mutual_evasion_s,
            "max_own_evasion_s": self._max_own_evasion_s,
            "max_target_evasion_s": self._max_target_evasion_s,
            "max_mutual_evasion_s": self._max_mutual_evasion_s,
            "evasion_warning": bool(
                self._pending_rule and self._pending_rule.warning_penalty < 0.0
            ),
            "evasion_judgement": (
                self._pending_rule.adjudication if self._pending_rule else ""
            ),
            "min_own_altitude_m": (
                0.0 if math.isinf(self._min_altitude_m) else self._min_altitude_m
            ),
            "max_own_descent_rate_mps": self._max_descent_rate_mps,
            "max_abs_own_roll_deg": self._max_abs_roll_deg,
            "max_abs_own_pitch_deg": self._max_abs_pitch_deg,
        }
        metrics.update(self._last_metrics)
        return metrics


def build_reward() -> UnifiedCompetitionReward:
    return UnifiedCompetitionReward()


__all__ = ["MY_REWARD_CONFIG", "UnifiedCompetitionReward", "build_reward", "compute_reward"]
