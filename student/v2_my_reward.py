# -*- coding: utf-8 -*-
"""Competition-rule-aware WVR reward for PPO and SAC.

This module has two responsibilities:

1. Potential-based dense shaping for WVR positional improvement.
2. Per-environment adjudication of the competition's clear-cut infinite-evasion
   cases (15 s observation + 15 s warning).

The evasion detector is temporal.  It therefore MUST be instantiated once per
Gym environment through ``build_reward()``.  A module-level singleton is not
used because Ray/vector environments would otherwise share episode history.

The separate 16.67 ms response-deadline rule is intentionally not part of this
reward.  Wall-clock inference latency is a deployment constraint, not a
simulator-state reward signal; the supplied realtime client patch handles it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.sim.state_schema import StateIndex


# Values are initial baselines for this Release.  The rule-derived values are
# 15 s + 15 s.  The remaining scales are normalized against this simulator's
# health=1, 60 Hz damage integration, 300 m hard altitude floor, and ~5 km
# nominal initial separation.
MY_REWARD_CONFIG: dict[str, float] = {
    # Base task ---------------------------------------------------------------
    # Equivalent to the old -0.01 per 0.1 s RL step, but invariant to step_ratio.
    "time_penalty_per_s": 0.10,
    # Full target-health removal contributes approximately +20; full own-health
    # loss contributes approximately -20 before terminal reward.
    "damage_scale": 20.0,

    # Potential-based shaping -------------------------------------------------
    # MUST equal PPO/SAC gamma.  PBRS uses beta*(gamma*Phi(s') - Phi(s)).
    "shaping_gamma": 0.9976921765,
    # Phi is normalized to [0, 1], so beta=1 preserves the natural scale.
    "potential_scale": 5.0,
    "threat_weight": 1.0,
    # Covers the nominal ~5 km initial separation while becoming zero beyond it.
    "shaping_range_m": 6000.0,

    # Safety ------------------------------------------------------------------
    "low_altitude_floor_m": 300.0,
    "low_altitude_warning_m": 600.0,
    # Maximum -1 reward/s at the hard-floor boundary; linearly zero at 600 m.
    "low_altitude_penalty_per_s": 1.0,

    # Outcome -----------------------------------------------------------------
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "draw_reward": -30.0,
    "guard_fail_penalty": -50.0,

    # Infinite-evasion rule ---------------------------------------------------
    "evasion_detection_s": 15.0,
    "evasion_warning_s": 15.0,
    # During the warning phase, ramps from 0 to -2 reward/s.  At 10 RL steps/s
    # this is 0 to -0.2 per step and totals -15 over the full linear ramp.
    "evasion_warning_penalty_per_s": 2.0,

    # Clear-cut straight-line proxy.  The evader must fly within 30 deg of the
    # direct-away ray, while the pursuer need only be within 60 deg of pursuit.
    "evasion_away_cone_deg": 30.0,
    "evasion_pursuit_cone_deg": 60.0,
    # <=2 deg/s means <=30 deg track change during the initial 15 s window.
    "straight_track_rate_deg_s": 2.0,
    # A short low-pass suppresses frame jitter.  A turn must remain above the
    # threshold for 1 s before it actively erases accumulated evasion evidence.
    # During that confirmation second, evidence is held rather than increased,
    # preventing both one-frame reset exploits and last-moment false forfeits.
    "track_rate_filter_s": 0.25,
    "maneuver_confirmation_s": 1.0,

    # Evidence is leaky rather than instant-resetting, so a one-frame token turn
    # or one tiny damage tick cannot erase a long evasion history.  Normal
    # maneuvering removes 2 s of evidence per 1 s; sustained damage exchange
    # removes it faster, at 5 s per 1 s.
    "evasion_evidence_decay_rate": 2.0,
    "engagement_evidence_decay_rate": 5.0,

    # Bounded exemption for measurable energy/stall recovery.  It does not make
    # indefinite straight escape legal; it only pauses detection for up to 5 s.
    "recovery_grace_s": 5.0,
    # The grace budget is replenished only after 15 s of genuine non-evasion
    # (or an actual damage event), preventing 5 s on/off energy-gain pulses
    # from resetting the exemption indefinitely.
    "recovery_recharge_s": 15.0,
    "recovery_specific_energy_rate_mps": 5.0,
    "stall_recovery_speed_mps": 180.0,
    "stall_recovery_accel_mps2": 2.0,
}

_G = 9.80665
_EPS = 1e-9


@dataclass(frozen=True)
class _StepContext:
    sim_time_s: float
    dt_s: float
    distance_m: float
    range_rate_mps: float
    own_ata_deg: float
    target_ata_deg: float
    own_aa_deg: float
    own_track_rate_deg_s: float
    target_track_rate_deg_s: float
    own_descent_rate_mps: float
    target_descent_rate_mps: float
    own_away_alignment: float
    target_away_alignment: float
    own_pursuit_alignment: float
    target_pursuit_alignment: float
    own_recovery_exempt: bool
    target_recovery_exempt: bool
    own_recovery_budget_used_s: float
    target_recovery_budget_used_s: float
    own_evasion_s: float
    target_evasion_s: float
    mutual_evasion_s: float
    own_warning_fraction: float
    target_warning_fraction: float
    mutual_warning_fraction: float
    potential_before: float
    potential_now: float
    adjudication_end_condition: str


def _cfg(config: dict[str, Any], key: str) -> float:
    return float(config.get(key, MY_REWARD_CONFIG[key]))


def _clip(x: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(x)))


def _clip01(x: float) -> float:
    return _clip(x, 0.0, 1.0)


def _forward_score(angle_deg: float) -> float:
    """Nose-to-LOS alignment: 1 at 0 deg and 0 at 90 deg or worse."""
    return _clip01(math.cos(math.radians(abs(float(angle_deg)))))


def _tail_score(aspect_angle_deg: float) -> float:
    """Target-tail position: 1 at projected AA=0 deg and 0 at 180 deg."""
    return _clip01(
        0.5 * (1.0 + math.cos(math.radians(abs(float(aspect_angle_deg)))))
    )


def _range_potential(
    distance_m: float,
    min_range_m: float,
    shaping_range_m: float,
) -> float:
    """Continuous, O(1) range potential with no WEZ plateau.

    It is zero at collision distance, peaks at the Release's WEZ minimum range,
    and decreases linearly to zero at ``shaping_range_m``.  Because the reward
    uses a potential difference, merely holding any fixed range does not create
    a positive dense-reward stream.
    """
    d = max(0.0, float(distance_m))
    r_min = max(1.0, float(min_range_m))
    r_shape = max(r_min + 1.0, float(shaping_range_m))
    if d < r_min:
        return _clip01(d / r_min)
    if d >= r_shape:
        return 0.0
    return _clip01((r_shape - d) / (r_shape - r_min))


def _position(state) -> tuple[float, float, float]:
    return (
        float(state[StateIndex.N]),
        float(state[StateIndex.E]),
        float(state[StateIndex.D]),
    )


def _sub(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _neg(v: tuple[float, float, float]) -> tuple[float, float, float]:
    return (-v[0], -v[1], -v[2])


def _unit(v: tuple[float, float, float]) -> tuple[float, float, float] | None:
    norm = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if norm <= _EPS:
        return None
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def _dot(
    a: tuple[float, float, float] | None,
    b: tuple[float, float, float] | None,
) -> float:
    if a is None or b is None:
        return -1.0
    return _clip(a[0] * b[0] + a[1] * b[1] + a[2] * b[2], -1.0, 1.0)


def _instant_track_rate_deg_s(
    previous: tuple[float, float, float] | None,
    current: tuple[float, float, float] | None,
    dt_s: float,
) -> float:
    if previous is None or current is None or dt_s <= _EPS:
        return 0.0
    return math.degrees(math.acos(_dot(previous, current))) / dt_s


def _low_pass(previous: float, current: float, dt_s: float, tau_s: float) -> float:
    if tau_s <= _EPS:
        return float(current)
    alpha = _clip01(dt_s / (tau_s + dt_s))
    return float(previous) + alpha * (float(current) - float(previous))


def _warning_fraction(timer_s: float, detection_s: float, warning_s: float) -> float:
    if warning_s <= _EPS:
        return float(timer_s >= detection_s)
    return _clip01((float(timer_s) - detection_s) / warning_s)


def _advance_evidence(
    value_s: float,
    condition: bool,
    hold: bool,
    damage_event: bool,
    dt_s: float,
    normal_decay_rate: float,
    engagement_decay_rate: float,
    maximum_s: float,
) -> float:
    """Leaky temporal evidence that is resistant to token maneuvers.

    A clear straight-escape frame adds real elapsed time.  A bounded recovery
    exemption or an unconfirmed turn holds evidence.  A confirmed normal
    maneuver removes evidence, and actual damage exchange removes it faster.
    No single frame hard-resets the timer.
    """
    if condition:
        return min(maximum_s, value_s + dt_s)
    if hold and not damage_event:
        return value_s
    decay = engagement_decay_rate if damage_event else normal_decay_rate
    return max(0.0, value_s - decay * dt_s)


class RuleAwareReward:
    """One stateful reward/adjudicator instance per environment."""

    def __init__(self) -> None:
        self._initialized = False
        self._prev_time_s = 0.0
        self._prev_distance_m = 0.0
        self._prev_own_pos = (0.0, 0.0, 0.0)
        self._prev_target_pos = (0.0, 0.0, 0.0)
        self._prev_own_track: tuple[float, float, float] | None = None
        self._prev_target_track: tuple[float, float, float] | None = None
        self._own_track_rate_ema = 0.0
        self._target_track_rate_ema = 0.0
        self._own_turn_run_s = 0.0
        self._target_turn_run_s = 0.0
        self._prev_own_speed = 0.0
        self._prev_target_speed = 0.0
        self._prev_own_specific_energy = 0.0
        self._prev_target_specific_energy = 0.0
        self._own_recovery_budget_used_s = 0.0
        self._target_recovery_budget_used_s = 0.0
        self._own_non_evasion_run_s = 0.0
        self._target_non_evasion_run_s = 0.0
        self._own_evasion_s = 0.0
        self._target_evasion_s = 0.0
        self._mutual_evasion_s = 0.0
        self._max_own_evasion_s = 0.0
        self._max_target_evasion_s = 0.0
        self._max_mutual_evasion_s = 0.0
        self._previous_potential = 0.0
        self._last_context: _StepContext | None = None

    def reset(
        self,
        ownship_state=None,
        target_state=None,
        geo_info=None,
        wez_config: dict | None = None,
        reward_config: dict | None = None,
    ) -> None:
        """Clear all episode history and initialize the PBRS potential."""
        self.__init__()
        if ownship_state is None or target_state is None or geo_info is None:
            return

        cfg = reward_config or {}
        wez = wez_config or {"min_range_m": 152.4}
        self._prev_time_s = float(ownship_state[StateIndex.SIM_TIME])
        self._prev_distance_m = float(
            geo_info._get_distance(ownship_state, target_state)
        )
        self._prev_own_pos = _position(ownship_state)
        self._prev_target_pos = _position(target_state)
        self._prev_own_speed = float(ownship_state[StateIndex.KCAS])
        self._prev_target_speed = float(target_state[StateIndex.KCAS])
        self._prev_own_specific_energy = self._specific_energy(ownship_state)
        self._prev_target_specific_energy = self._specific_energy(target_state)
        self._previous_potential = self._potential(
            ownship_state, target_state, geo_info, wez, cfg
        )
        self._initialized = True

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
        """Advance rule evidence and return a terminal override if warranted."""
        context = self._ensure_context(
            ownship_state,
            target_state,
            ownship_damage,
            target_damage,
            geo_info,
            wez_config,
            reward_config,
        )
        if context.adjudication_end_condition:
            return {
                "terminated": True,
                "truncated": False,
                "end_condition": context.adjudication_end_condition,
            }
        return {"terminated": False, "truncated": False, "end_condition": ""}

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
    ) -> tuple[float, dict[str, float]]:
        context = self._ensure_context(
            ownship_state,
            target_state,
            ownship_damage,
            target_damage,
            geo_info,
            wez_config,
            reward_config,
        )

        gamma = _clip(_cfg(reward_config, "shaping_gamma"), 0.0, 1.0)
        beta = max(0.0, _cfg(reward_config, "potential_scale"))
        if terminated or truncated:
            # The competition outcome is terminal; use Phi(absorbing)=0.
            shaping = beta * (0.0 - context.potential_before)
        else:
            shaping = beta * (
                gamma * context.potential_now - context.potential_before
            )

        dt_s = max(0.0, context.dt_s)
        warning_fraction = max(
            context.own_warning_fraction,
            context.mutual_warning_fraction,
        )
        evasion_penalty = (
            -_cfg(reward_config, "evasion_warning_penalty_per_s")
            * warning_fraction
            * dt_s
        )

        altitude_floor = _cfg(reward_config, "low_altitude_floor_m")
        altitude_warning = max(
            altitude_floor + 1.0,
            _cfg(reward_config, "low_altitude_warning_m"),
        )
        altitude = float(ownship_state[StateIndex.ALT])
        altitude_risk = _clip01(
            (altitude_warning - altitude)
            / (altitude_warning - altitude_floor)
        )

        components: dict[str, float] = {
            "time": -_cfg(reward_config, "time_penalty_per_s") * dt_s,
            "shaping": float(shaping),
            "damage": _cfg(reward_config, "damage_scale")
            * (float(target_damage) - float(ownship_damage)),
            "evasion": float(evasion_penalty),
            "safety": -_cfg(
                reward_config, "low_altitude_penalty_per_s"
            )
            * altitude_risk
            * dt_s,
            "terminal": 0.0,
        }

        if terminated or truncated:
            components["terminal"] = self._terminal_reward(
                ownship_state,
                target_state,
                reward_config,
                end_condition,
            )

        total = float(sum(components.values()))
        if not math.isfinite(total):
            fallback = _cfg(reward_config, "loss_reward")
            components = {
                "time": 0.0,
                "shaping": 0.0,
                "damage": 0.0,
                "evasion": 0.0,
                "safety": 0.0,
                "terminal": fallback,
            }
            total = fallback
        return total, components

    def metrics(self) -> dict[str, Any]:
        context = self._last_context
        if context is None:
            return {
                "own_evasion_s": 0.0,
                "target_evasion_s": 0.0,
                "mutual_evasion_s": 0.0,
                "max_own_evasion_s": self._max_own_evasion_s,
                "max_target_evasion_s": self._max_target_evasion_s,
                "max_mutual_evasion_s": self._max_mutual_evasion_s,
                "own_recovery_budget_used_s": self._own_recovery_budget_used_s,
                "target_recovery_budget_used_s": self._target_recovery_budget_used_s,
                "evasion_warning": False,
                "evasion_judgement": "",
            }
        return {
            "own_evasion_s": context.own_evasion_s,
            "target_evasion_s": context.target_evasion_s,
            "mutual_evasion_s": context.mutual_evasion_s,
            "max_own_evasion_s": self._max_own_evasion_s,
            "max_target_evasion_s": self._max_target_evasion_s,
            "max_mutual_evasion_s": self._max_mutual_evasion_s,
            "own_warning_fraction": context.own_warning_fraction,
            "target_warning_fraction": context.target_warning_fraction,
            "mutual_warning_fraction": context.mutual_warning_fraction,
            "own_track_rate_deg_s": context.own_track_rate_deg_s,
            "target_track_rate_deg_s": context.target_track_rate_deg_s,
            "own_descent_rate_mps": context.own_descent_rate_mps,
            "target_descent_rate_mps": context.target_descent_rate_mps,
            "own_away_alignment": context.own_away_alignment,
            "target_away_alignment": context.target_away_alignment,
            "own_pursuit_alignment": context.own_pursuit_alignment,
            "target_pursuit_alignment": context.target_pursuit_alignment,
            "range_rate_mps": context.range_rate_mps,
            "own_recovery_exempt": context.own_recovery_exempt,
            "target_recovery_exempt": context.target_recovery_exempt,
            "own_recovery_budget_used_s": context.own_recovery_budget_used_s,
            "target_recovery_budget_used_s": context.target_recovery_budget_used_s,
            "evasion_warning": bool(
                context.own_warning_fraction > 0.0
                or context.target_warning_fraction > 0.0
                or context.mutual_warning_fraction > 0.0
            ),
            "evasion_judgement": context.adjudication_end_condition,
        }

    def _ensure_context(
        self,
        ownship_state,
        target_state,
        ownship_damage: float,
        target_damage: float,
        geo_info,
        wez_config: dict,
        reward_config: dict,
    ) -> _StepContext:
        sim_time_s = float(ownship_state[StateIndex.SIM_TIME])
        if (
            self._last_context is not None
            and abs(self._last_context.sim_time_s - sim_time_s) <= 1e-9
        ):
            return self._last_context

        if (not self._initialized) or sim_time_s < self._prev_time_s - 1e-6:
            self.reset(
                ownship_state,
                target_state,
                geo_info,
                wez_config,
                reward_config,
            )
            context = self._initial_context(
                ownship_state,
                target_state,
                geo_info,
                wez_config,
                reward_config,
            )
            self._last_context = context
            return context

        dt_s = max(sim_time_s - self._prev_time_s, 1e-6)
        distance_m = float(geo_info._get_distance(ownship_state, target_state))
        range_rate_mps = (distance_m - self._prev_distance_m) / dt_s

        own_pos = _position(ownship_state)
        target_pos = _position(target_state)
        own_descent_rate_mps = (own_pos[2] - self._prev_own_pos[2]) / dt_s
        target_descent_rate_mps = (target_pos[2] - self._prev_target_pos[2]) / dt_s
        own_track = _unit(_sub(own_pos, self._prev_own_pos))
        target_track = _unit(_sub(target_pos, self._prev_target_pos))

        own_rate_raw = _instant_track_rate_deg_s(
            self._prev_own_track, own_track, dt_s
        )
        target_rate_raw = _instant_track_rate_deg_s(
            self._prev_target_track, target_track, dt_s
        )
        filter_tau = max(0.0, _cfg(reward_config, "track_rate_filter_s"))
        self._own_track_rate_ema = _low_pass(
            self._own_track_rate_ema, own_rate_raw, dt_s, filter_tau
        )
        self._target_track_rate_ema = _low_pass(
            self._target_track_rate_ema, target_rate_raw, dt_s, filter_tau
        )

        target_to_own = _unit(_sub(own_pos, target_pos))
        own_to_target = _neg(target_to_own) if target_to_own is not None else None
        own_away_alignment = _dot(own_track, target_to_own)
        target_pursuit_alignment = _dot(target_track, target_to_own)
        target_away_alignment = _dot(target_track, own_to_target)
        own_pursuit_alignment = _dot(own_track, own_to_target)

        own_ata = abs(float(
            geo_info._get_antenna_train_angle(
                ownship_state, target_state, False
            )
        ))
        target_ata = abs(float(
            geo_info._get_antenna_train_angle(
                target_state, ownship_state, False
            )
        ))
        own_aa = abs(float(
            geo_info._get_aspect_angle(ownship_state, target_state, True)
        ))

        own_speed = float(ownship_state[StateIndex.KCAS])
        target_speed = float(target_state[StateIndex.KCAS])
        own_accel = (own_speed - self._prev_own_speed) / dt_s
        target_accel = (target_speed - self._prev_target_speed) / dt_s
        own_specific_energy = self._specific_energy(ownship_state)
        target_specific_energy = self._specific_energy(target_state)
        own_energy_rate = (
            own_specific_energy - self._prev_own_specific_energy
        ) / dt_s
        target_energy_rate = (
            target_specific_energy - self._prev_target_specific_energy
        ) / dt_s

        own_recovery_candidate = self._is_recovery_candidate(
            own_speed, own_accel, own_energy_rate, reward_config
        )
        target_recovery_candidate = self._is_recovery_candidate(
            target_speed, target_accel, target_energy_rate, reward_config
        )

        away_threshold = math.cos(math.radians(_clip(
            _cfg(reward_config, "evasion_away_cone_deg"), 0.0, 89.0
        )))
        pursuit_threshold = math.cos(math.radians(_clip(
            _cfg(reward_config, "evasion_pursuit_cone_deg"), 0.0, 89.0
        )))
        straight_threshold = max(
            0.1, _cfg(reward_config, "straight_track_rate_deg_s")
        )
        own_track_valid = own_track is not None
        target_track_valid = target_track is not None
        self._own_turn_run_s = (
            self._own_turn_run_s + dt_s
            if self._own_track_rate_ema > straight_threshold
            else 0.0
        )
        self._target_turn_run_s = (
            self._target_turn_run_s + dt_s
            if self._target_track_rate_ema > straight_threshold
            else 0.0
        )
        maneuver_confirmation_s = max(
            0.0, _cfg(reward_config, "maneuver_confirmation_s")
        )
        # Stop adding evidence as soon as a measurable turn starts.  Only a turn
        # sustained for confirmation_s actively removes prior evidence.  This
        # distinguishes a real reattack turn from a one-frame token heading kick.
        own_straight = (
            own_track_valid
            and self._own_track_rate_ema <= straight_threshold
        )
        target_straight = (
            target_track_valid
            and self._target_track_rate_ema <= straight_threshold
        )
        own_turn_unconfirmed = (
            own_track_valid
            and not own_straight
            and self._own_turn_run_s < maneuver_confirmation_s
        )
        target_turn_unconfirmed = (
            target_track_valid
            and not target_straight
            and self._target_turn_run_s < maneuver_confirmation_s
        )

        own_away = own_away_alignment >= away_threshold
        target_away = target_away_alignment >= away_threshold
        own_pursuit = own_pursuit_alignment >= pursuit_threshold
        target_pursuit = target_pursuit_alignment >= pursuit_threshold
        damage_event = (
            float(ownship_damage) > _EPS or float(target_damage) > _EPS
        )

        # Separate geometry from straightness.  This is needed so an incipient
        # genuine turn can hold the timer immediately, while only a sustained
        # turn is allowed to erase accumulated evidence.
        own_one_sided_geometry = (
            own_away and target_pursuit and not damage_event
        )
        target_one_sided_geometry = (
            target_away and own_pursuit and not damage_event
        )
        mutual_geometry = (
            not own_one_sided_geometry
            and not target_one_sided_geometry
            and own_away
            and target_away
            and not damage_event
        )
        own_one_sided_raw = own_one_sided_geometry and own_straight
        target_one_sided_raw = target_one_sided_geometry and target_straight
        mutual_raw = mutual_geometry and own_straight and target_straight

        recovery_grace = max(0.0, _cfg(reward_config, "recovery_grace_s"))
        recovery_recharge = max(0.0, _cfg(reward_config, "recovery_recharge_s"))

        own_recovery_relevant = own_one_sided_raw or mutual_raw
        target_recovery_relevant = target_one_sided_raw or mutual_raw
        # Recovery grace is replenished only by a sustained non-evasion period.
        # A one-frame damage tick is therefore not a free 5 s grace reset.
        self._own_non_evasion_run_s = (
            0.0
            if own_recovery_relevant
            else self._own_non_evasion_run_s + dt_s
        )
        self._target_non_evasion_run_s = (
            0.0
            if target_recovery_relevant
            else self._target_non_evasion_run_s + dt_s
        )
        if (
            recovery_recharge > _EPS
            and self._own_non_evasion_run_s >= recovery_recharge
        ):
            self._own_recovery_budget_used_s = 0.0
        if (
            recovery_recharge > _EPS
            and self._target_non_evasion_run_s >= recovery_recharge
        ):
            self._target_recovery_budget_used_s = 0.0

        own_recovery_exempt = (
            own_recovery_relevant
            and own_recovery_candidate
            and self._own_recovery_budget_used_s < recovery_grace - _EPS
        )
        target_recovery_exempt = (
            target_recovery_relevant
            and target_recovery_candidate
            and self._target_recovery_budget_used_s < recovery_grace - _EPS
        )
        if own_recovery_exempt:
            self._own_recovery_budget_used_s = min(
                recovery_grace, self._own_recovery_budget_used_s + dt_s
            )
        if target_recovery_exempt:
            self._target_recovery_budget_used_s = min(
                recovery_grace, self._target_recovery_budget_used_s + dt_s
            )

        own_one_sided = own_one_sided_raw and not own_recovery_exempt
        target_one_sided = target_one_sided_raw and not target_recovery_exempt
        mutual = (
            mutual_raw
            and not own_recovery_exempt
            and not target_recovery_exempt
        )

        detection_s = max(0.0, _cfg(reward_config, "evasion_detection_s"))
        warning_s = max(0.0, _cfg(reward_config, "evasion_warning_s"))
        total_s = detection_s + warning_s
        decay_rate = max(
            0.0, _cfg(reward_config, "evasion_evidence_decay_rate")
        )
        engagement_decay_rate = max(
            decay_rate,
            _cfg(reward_config, "engagement_evidence_decay_rate"),
        )

        own_hold = (
            own_one_sided_geometry
            and (own_turn_unconfirmed or own_recovery_exempt)
        )
        target_hold = (
            target_one_sided_geometry
            and (target_turn_unconfirmed or target_recovery_exempt)
        )
        mutual_hold = (
            mutual_geometry
            and (
                own_turn_unconfirmed
                or target_turn_unconfirmed
                or own_recovery_exempt
                or target_recovery_exempt
            )
        )

        self._own_evasion_s = _advance_evidence(
            self._own_evasion_s,
            own_one_sided,
            own_hold,
            damage_event,
            dt_s,
            decay_rate,
            engagement_decay_rate,
            total_s,
        )
        self._target_evasion_s = _advance_evidence(
            self._target_evasion_s,
            target_one_sided,
            target_hold,
            damage_event,
            dt_s,
            decay_rate,
            engagement_decay_rate,
            total_s,
        )
        self._mutual_evasion_s = _advance_evidence(
            self._mutual_evasion_s,
            mutual,
            mutual_hold,
            damage_event,
            dt_s,
            decay_rate,
            engagement_decay_rate,
            total_s,
        )

        self._max_own_evasion_s = max(
            self._max_own_evasion_s, self._own_evasion_s
        )
        self._max_target_evasion_s = max(
            self._max_target_evasion_s, self._target_evasion_s
        )
        self._max_mutual_evasion_s = max(
            self._max_mutual_evasion_s, self._mutual_evasion_s
        )

        adjudication = ""
        if total_s > 0.0:
            if self._own_evasion_s >= total_s - 1e-9:
                adjudication = "ownship infinite evasion"
            elif self._target_evasion_s >= total_s - 1e-9:
                adjudication = "target infinite evasion"
            elif self._mutual_evasion_s >= total_s - 1e-9:
                adjudication = "mutual infinite evasion"

        potential_before = self._previous_potential
        potential_now = self._potential(
            ownship_state,
            target_state,
            geo_info,
            wez_config,
            reward_config,
        )
        self._previous_potential = potential_now

        context = _StepContext(
            sim_time_s=sim_time_s,
            dt_s=dt_s,
            distance_m=distance_m,
            range_rate_mps=range_rate_mps,
            own_ata_deg=own_ata,
            target_ata_deg=target_ata,
            own_aa_deg=own_aa,
            own_track_rate_deg_s=self._own_track_rate_ema,
            target_track_rate_deg_s=self._target_track_rate_ema,
            own_descent_rate_mps=own_descent_rate_mps,
            target_descent_rate_mps=target_descent_rate_mps,
            own_away_alignment=own_away_alignment,
            target_away_alignment=target_away_alignment,
            own_pursuit_alignment=own_pursuit_alignment,
            target_pursuit_alignment=target_pursuit_alignment,
            own_recovery_exempt=own_recovery_exempt,
            target_recovery_exempt=target_recovery_exempt,
            own_recovery_budget_used_s=self._own_recovery_budget_used_s,
            target_recovery_budget_used_s=self._target_recovery_budget_used_s,
            own_evasion_s=self._own_evasion_s,
            target_evasion_s=self._target_evasion_s,
            mutual_evasion_s=self._mutual_evasion_s,
            own_warning_fraction=_warning_fraction(
                self._own_evasion_s, detection_s, warning_s
            ),
            target_warning_fraction=_warning_fraction(
                self._target_evasion_s, detection_s, warning_s
            ),
            mutual_warning_fraction=_warning_fraction(
                self._mutual_evasion_s, detection_s, warning_s
            ),
            potential_before=potential_before,
            potential_now=potential_now,
            adjudication_end_condition=adjudication,
        )
        self._last_context = context

        self._prev_time_s = sim_time_s
        self._prev_distance_m = distance_m
        self._prev_own_pos = own_pos
        self._prev_target_pos = target_pos
        if own_track is not None:
            self._prev_own_track = own_track
        if target_track is not None:
            self._prev_target_track = target_track
        self._prev_own_speed = own_speed
        self._prev_target_speed = target_speed
        self._prev_own_specific_energy = own_specific_energy
        self._prev_target_specific_energy = target_specific_energy
        return context

    def _initial_context(
        self,
        ownship_state,
        target_state,
        geo_info,
        wez_config: dict,
        reward_config: dict,
    ) -> _StepContext:
        distance = float(geo_info._get_distance(ownship_state, target_state))
        own_ata = abs(float(
            geo_info._get_antenna_train_angle(ownship_state, target_state, False)
        ))
        target_ata = abs(float(
            geo_info._get_antenna_train_angle(target_state, ownship_state, False)
        ))
        own_aa = abs(float(
            geo_info._get_aspect_angle(ownship_state, target_state, True)
        ))
        return _StepContext(
            sim_time_s=float(ownship_state[StateIndex.SIM_TIME]),
            dt_s=0.0,
            distance_m=distance,
            range_rate_mps=0.0,
            own_ata_deg=own_ata,
            target_ata_deg=target_ata,
            own_aa_deg=own_aa,
            own_track_rate_deg_s=0.0,
            target_track_rate_deg_s=0.0,
            own_descent_rate_mps=0.0,
            target_descent_rate_mps=0.0,
            own_away_alignment=-1.0,
            target_away_alignment=-1.0,
            own_pursuit_alignment=-1.0,
            target_pursuit_alignment=-1.0,
            own_recovery_exempt=False,
            target_recovery_exempt=False,
            own_recovery_budget_used_s=0.0,
            target_recovery_budget_used_s=0.0,
            own_evasion_s=0.0,
            target_evasion_s=0.0,
            mutual_evasion_s=0.0,
            own_warning_fraction=0.0,
            target_warning_fraction=0.0,
            mutual_warning_fraction=0.0,
            potential_before=self._previous_potential,
            potential_now=self._previous_potential,
            adjudication_end_condition="",
        )

    @staticmethod
    def _specific_energy(state) -> float:
        # True airspeed is the physically appropriate kinetic-energy term.
        # Fall back to KCAS for old/synthetic states that do not populate KTAS.
        ktas_index = getattr(StateIndex, "KTAS", 27)
        true_speed = float(state[int(ktas_index)])
        if not math.isfinite(true_speed) or true_speed <= 0.0:
            true_speed = float(state[StateIndex.KCAS])
        speed = max(0.0, true_speed)
        altitude = float(state[StateIndex.ALT])
        return altitude + speed * speed / (2.0 * _G)

    @staticmethod
    def _is_recovery_candidate(
        speed_mps: float,
        accel_mps2: float,
        specific_energy_rate_mps: float,
        reward_config: dict,
    ) -> bool:
        energy_recovery = specific_energy_rate_mps >= _cfg(
            reward_config, "recovery_specific_energy_rate_mps"
        )
        stall_recovery = (
            speed_mps <= _cfg(reward_config, "stall_recovery_speed_mps")
            and accel_mps2 >= _cfg(
                reward_config, "stall_recovery_accel_mps2"
            )
        )
        return bool(energy_recovery or stall_recovery)

    @staticmethod
    def _potential(
        ownship_state,
        target_state,
        geo_info,
        wez_config: dict,
        reward_config: dict,
    ) -> float:
        distance = float(geo_info._get_distance(ownship_state, target_state))
        own_ata = float(
            geo_info._get_antenna_train_angle(
                ownship_state, target_state, False
            )
        )
        target_ata = float(
            geo_info._get_antenna_train_angle(
                target_state, ownship_state, False
            )
        )
        # Projected AA avoids this Release's exact-centerline 3-D sign collapse.
        own_aa = float(
            geo_info._get_aspect_angle(ownship_state, target_state, True)
        )
        range_score = _range_potential(
            distance,
            float(wez_config["min_range_m"]),
            _cfg(reward_config, "shaping_range_m"),
        )
        offensive = _forward_score(own_ata) * _tail_score(own_aa)
        # The simulator's incoming-damage test uses target ATA + range, not AA.
        threat = _forward_score(target_ata)
        # Map the relative combat advantage to [0, 1] before multiplying by
        # range.  This keeps Phi non-negative: holding a fixed bad state cannot
        # generate the small positive per-step term that a signed potential can
        # produce when gamma < 1.  PBRS policy invariance is retained because any
        # bounded state potential is valid in gamma*Phi(s') - Phi(s).
        advantage = _clip(
            offensive - _cfg(reward_config, "threat_weight") * threat,
            -1.0,
            1.0,
        )
        combat_quality = 0.5 * (advantage + 1.0)
        return _clip01(range_score * combat_quality)

    @staticmethod
    def _terminal_reward(
        ownship_state,
        target_state,
        reward_config: dict,
        end_condition: str,
    ) -> float:
        if end_condition == "target infinite evasion":
            return _cfg(reward_config, "win_reward")
        if end_condition == "ownship infinite evasion":
            return _cfg(reward_config, "loss_reward")
        if end_condition == "mutual infinite evasion":
            return _cfg(reward_config, "draw_reward")
        if end_condition == "two circle headon guard fail":
            return _cfg(reward_config, "guard_fail_penalty")
        if end_condition == "ownship altitude below min":
            return _cfg(reward_config, "loss_reward")
        if end_condition == "target altitude below min":
            return _cfg(reward_config, "win_reward")

        own_health = float(ownship_state[StateIndex.HEALTH])
        target_health = float(target_state[StateIndex.HEALTH])
        if target_health <= 0.0 < own_health:
            return _cfg(reward_config, "win_reward")
        if own_health <= 0.0 < target_health:
            return _cfg(reward_config, "loss_reward")
        return _cfg(reward_config, "draw_reward")


def build_reward() -> RuleAwareReward:
    """Factory called once for each Ray/Gym environment instance."""
    return RuleAwareReward()


def compute_reward(*args, **kwargs):
    """Prevent unsafe use through the old shared stateless hook contract."""
    raise RuntimeError(
        "Rule-aware reward requires the supplied stateful hook/environment patch; "
        "load build_reward() once per environment."
    )


__all__ = [
    "MY_REWARD_CONFIG",
    "RuleAwareReward",
    "build_reward",
    "compute_reward",
]
