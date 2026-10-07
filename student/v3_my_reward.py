# -*- coding: utf-8 -*-
"""Result-driven WVR reward revision for the Siwon SAC curriculum.

This module reuses the temporal infinite-evasion adjudicator from
``student.v2_my_reward`` and changes only the learning objective that was shown
to be problematic in SIWON_ARCHIVE_20260726.

Main changes from v2
--------------------
1. Remove positive stable-flight / altitude-hold accumulation from S1+.
   Safety is penalty-only, so a 60 s timeout cannot be profitable merely by
   flying level.
2. Replace the absolute capture-geometry occupancy reward with finite-horizon
   progress shaping: scale * (Phi(s_t) - Phi(s_{t-1})).  Holding a pre-WEZ
   boundary no longer earns reward every frame.
3. Align Phi with the actual curriculum gate using smooth range, own-ATA, and
   target-ATA scores.  The scores are added (not multiplied), so a near miss in
   one variable still produces a useful learning signal.
4. Add one evidence-driven term for range-rate control.  The archived failures
   repeatedly entered the capture range and then overshot/extended; this term
   teaches the aircraft to reduce closure as it approaches the capture band.
5. Keep only direct combat effect, rule-compliance, safety, and terminal terms.

The separate 16.67 ms response deadline remains a deployment/runtime concern,
not an RL reward signal.
"""
from __future__ import annotations

import math
from typing import Any

from dogfight.sim.state_schema import StateIndex
from student.v2_my_reward import (
    MY_REWARD_CONFIG as _V2_CONFIG,
    RuleAwareReward as _V2RuleAwareReward,
)


# Start from v2 so the inherited evasion detector always receives every key it
# expects.  Stage YAML values still override these defaults.
MY_REWARD_CONFIG: dict[str, Any] = dict(_V2_CONFIG)
MY_REWARD_CONFIG.update(
    {
        # Core task ----------------------------------------------------------
        "time_penalty_per_s": 0.05,
        "damage_scale": 60.0,

        # Finite-horizon geometry progress ---------------------------------
        # Use 1.0 for curriculum progress: constant geometry gives exactly 0.
        # Set this equal to the RL gamma only when deliberately running PBRS.
        "progress_gamma": 1.0,
        "progress_scale": 12.0,

        # These four values must match the current position-capture curriculum.
        "capture_range_inner_m": 500.0,
        "capture_range_outer_m": 1500.0,
        "capture_own_ata_gate_deg": 25.0,
        "capture_target_ata_gate_deg": 125.0,

        # Smoothness of each score outside its gate.
        "capture_range_approach_scale_m": 1800.0,
        "capture_range_too_close_scale_m": 200.0,
        "capture_own_ata_softness_deg": 8.0,
        "capture_target_ata_softness_deg": 12.0,

        # Archive evidence: range was generally reached; own ATA was the main
        # final-gate failure.  Range still receives the largest weight because
        # the initial rear geometry was already nearly perfect.
        "capture_range_weight": 0.45,
        "capture_own_ata_weight": 0.40,
        "capture_rear_weight": 0.15,

        # Closure-profile control -------------------------------------------
        # closure = -range_rate, so positive values mean getting closer.
        "range_rate_penalty_per_s": 1.0,
        "desired_closure_gain_per_s": 0.04,
        "desired_closure_max_mps": 80.0,
        "desired_opening_max_mps": 30.0,
        "range_rate_error_reference_mps": 80.0,

        # Safety is penalty-only in S1+. ------------------------------------
        "stable_flight_reward_per_s": 0.0,
        "altitude_hold_reward_per_s": 0.0,
        "low_altitude_floor_m": 300.0,
        "low_altitude_warning_m": 4500.0,
        "low_altitude_penalty_per_s": 4.0,
        "descent_guard_start_m": 6500.0,
        "descent_rate_safe_mps": 15.0,
        "descent_rate_reference_mps": 60.0,
        "descent_rate_penalty_per_s": 2.0,
        "attitude_roll_safe_deg": 50.0,
        "attitude_nose_down_safe_deg": 12.0,
        "attitude_penalty_per_s": 1.0,
        "altitude_hold_target_m": 7000.0,
        "altitude_hold_band_m": 1500.0,
        "altitude_error_penalty_per_s": 2.0,

        # Outcome scale ------------------------------------------------------
        "win_reward": 250.0,
        "loss_reward": -250.0,
        "draw_reward": -80.0,
        "guard_fail_penalty": -150.0,
        "crash_extra_penalty": -150.0,
        # A target ground event is logged separately and is not a health kill.
        "target_ground_reward": 0.0,

        # Legacy v2 direct-positive terms remain disabled. ------------------
        "potential_scale": 0.0,
        "pursuit_reward_per_s": 0.0,
        "capture_geometry_reward_per_s": 0.0,
    }
)

_EPS = 1.0e-9


def _get(config: dict[str, Any], key: str) -> float:
    return float(config.get(key, MY_REWARD_CONFIG[key]))


def _clip(x: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(x)))


def _clip01(x: float) -> float:
    return _clip(x, 0.0, 1.0)


def _sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    x = float(x)
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _low_is_good_score(value: float, gate: float, softness: float) -> float:
    """Smooth score for conditions of the form value <= gate.

    The score is normalized so value=0 maps to 1.  At the gate it is close to
    0.5, and it remains nonzero just outside the gate, which is important for
    the archived 25-35 deg own-ATA near misses.
    """
    softness = max(float(softness), 1.0e-6)
    gate = max(float(gate), 0.0)
    maximum = _sigmoid(gate / softness)
    return _clip01(_sigmoid((gate - float(value)) / softness) / max(maximum, _EPS))


def _high_is_good_score(value: float, gate: float, softness: float) -> float:
    """Smooth score for conditions of the form value >= gate."""
    softness = max(float(softness), 1.0e-6)
    gate = _clip(float(gate), 0.0, 180.0)
    maximum = _sigmoid((180.0 - gate) / softness)
    return _clip01(_sigmoid((float(value) - gate) / softness) / max(maximum, _EPS))


def _soft_range_band_score(
    distance_m: float,
    inner_m: float,
    outer_m: float,
    approach_scale_m: float,
    too_close_scale_m: float,
) -> float:
    """1 inside the desired band, with exponential tails outside it.

    This is intentionally a band rather than a point target.  It gives a useful
    approach gradient from 2.8-3.2 km while not rewarding collision distance.
    Because the learning term uses a potential difference, remaining anywhere
    inside the band does not create a positive reward stream.
    """
    d = max(0.0, float(distance_m))
    inner = max(1.0, float(inner_m))
    outer = max(inner + 1.0, float(outer_m))
    if d < inner:
        scale = max(1.0, float(too_close_scale_m))
        return _clip01(math.exp(-(inner - d) / scale))
    if d <= outer:
        return 1.0
    scale = max(1.0, float(approach_scale_m))
    return _clip01(math.exp(-(d - outer) / scale))


class ResultDrivenReward(_V2RuleAwareReward):
    """Stateful reward with v2 rule adjudication and revised S1+/combat terms."""

    def __init__(self) -> None:
        super().__init__()
        self._last_quality: dict[str, float] = {
            "range_score": 0.0,
            "own_ata_score": 0.0,
            "rear_score": 0.0,
            "progress_potential": 0.0,
            "closure_mps": 0.0,
            "desired_closure_mps": 0.0,
        }

    def _potential(
        self,
        ownship_state,
        target_state,
        geo_info,
        wez_config: dict,
        reward_config: dict,
    ) -> float:
        distance = float(geo_info._get_distance(ownship_state, target_state))
        own_ata = abs(
            float(
                geo_info._get_antenna_train_angle(
                    ownship_state, target_state, False
                )
            )
        )
        target_ata = abs(
            float(
                geo_info._get_antenna_train_angle(
                    target_state, ownship_state, False
                )
            )
        )

        range_score = _soft_range_band_score(
            distance,
            _get(reward_config, "capture_range_inner_m"),
            _get(reward_config, "capture_range_outer_m"),
            _get(reward_config, "capture_range_approach_scale_m"),
            _get(reward_config, "capture_range_too_close_scale_m"),
        )
        own_ata_score = _low_is_good_score(
            own_ata,
            _get(reward_config, "capture_own_ata_gate_deg"),
            _get(reward_config, "capture_own_ata_softness_deg"),
        )
        rear_score = _high_is_good_score(
            target_ata,
            _get(reward_config, "capture_target_ata_gate_deg"),
            _get(reward_config, "capture_target_ata_softness_deg"),
        )

        w_range = max(0.0, _get(reward_config, "capture_range_weight"))
        w_ata = max(0.0, _get(reward_config, "capture_own_ata_weight"))
        w_rear = max(0.0, _get(reward_config, "capture_rear_weight"))
        weight_sum = max(w_range + w_ata + w_rear, _EPS)
        potential = _clip01(
            (w_range * range_score + w_ata * own_ata_score + w_rear * rear_score)
            / weight_sum
        )

        self._last_quality.update(
            {
                "range_score": range_score,
                "own_ata_score": own_ata_score,
                "rear_score": rear_score,
                "progress_potential": potential,
            }
        )
        return potential

    @staticmethod
    def _desired_closure(distance_m: float, reward_config: dict) -> float:
        """Desired closure profile in m/s; positive means closing."""
        d = max(0.0, float(distance_m))
        inner = max(1.0, _get(reward_config, "capture_range_inner_m"))
        outer = max(inner + 1.0, _get(reward_config, "capture_range_outer_m"))
        gain = max(0.0, _get(reward_config, "desired_closure_gain_per_s"))
        max_close = max(0.0, _get(reward_config, "desired_closure_max_mps"))
        max_open = max(0.0, _get(reward_config, "desired_opening_max_mps"))
        if d > outer:
            return min(max_close, gain * (d - outer))
        if d < inner:
            return -min(max_open, gain * (inner - d))
        return 0.0

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
        dt_s = max(0.0, context.dt_s)

        # Finite-horizon progress shaping.  gamma=1 means no positive reward for
        # holding a state and no terminal cancellation of already-earned progress.
        progress_gamma = _clip(_get(reward_config, "progress_gamma"), 0.0, 1.0)
        progress = max(0.0, _get(reward_config, "progress_scale")) * (
            progress_gamma * context.potential_now - context.potential_before
        )

        # Dense speed-profile supervision for the observed one-pass overshoot.
        closure_mps = -float(context.range_rate_mps)
        desired_closure_mps = self._desired_closure(
            context.distance_m, reward_config
        )
        range_rate_ref = max(
            1.0, _get(reward_config, "range_rate_error_reference_mps")
        )
        range_rate_error = _clip(
            (closure_mps - desired_closure_mps) / range_rate_ref,
            -1.0,
            1.0,
        )
        range_rate_penalty = (
            -max(0.0, _get(reward_config, "range_rate_penalty_per_s"))
            * range_rate_error
            * range_rate_error
            * dt_s
        )
        self._last_quality.update(
            {
                "closure_mps": closure_mps,
                "desired_closure_mps": desired_closure_mps,
            }
        )

        # Competition infinite-evasion warning inherited from v2.
        evasion_total_s = max(0.0, _get(reward_config, "evasion_detection_s")) + max(
            0.0, _get(reward_config, "evasion_warning_s")
        )
        warning_fraction = (
            max(context.own_warning_fraction, context.mutual_warning_fraction)
            if evasion_total_s > _EPS
            else 0.0
        )
        evasion_penalty = (
            -max(0.0, _get(reward_config, "evasion_warning_penalty_per_s"))
            * warning_fraction
            * dt_s
        )

        # Penalty-only safety envelope.  There is deliberately no reward for
        # merely flying level or holding 7 km altitude.
        altitude_floor = _get(reward_config, "low_altitude_floor_m")
        altitude_warning = max(
            altitude_floor + 1.0,
            _get(reward_config, "low_altitude_warning_m"),
        )
        altitude = float(ownship_state[StateIndex.ALT])
        altitude_risk = _clip01(
            (altitude_warning - altitude)
            / (altitude_warning - altitude_floor)
        )

        descent_guard_start = max(
            altitude_floor + 1.0,
            _get(reward_config, "descent_guard_start_m"),
        )
        descent_altitude_risk = _clip01(
            (descent_guard_start - altitude)
            / (descent_guard_start - altitude_floor)
        )
        descent_excess = _clip01(
            (
                context.own_descent_rate_mps
                - _get(reward_config, "descent_rate_safe_mps")
            )
            / max(1.0, _get(reward_config, "descent_rate_reference_mps"))
        )

        roll_deg = abs(float(ownship_state[StateIndex.ROLL]))
        pitch_deg = float(ownship_state[StateIndex.PITCH])
        roll_safe = _clip(
            _get(reward_config, "attitude_roll_safe_deg"), 0.0, 179.0
        )
        nose_down_safe = _clip(
            _get(reward_config, "attitude_nose_down_safe_deg"), 0.0, 89.0
        )
        roll_excess = _clip01((roll_deg - roll_safe) / (180.0 - roll_safe))
        nose_down_excess = _clip01(
            (-pitch_deg - nose_down_safe) / (90.0 - nose_down_safe)
        )
        attitude_risk = max(roll_excess, nose_down_excess)

        altitude_error_risk = 0.0
        altitude_target = _get(reward_config, "altitude_hold_target_m")
        if altitude_target > 0.0:
            altitude_band = max(
                1.0, _get(reward_config, "altitude_hold_band_m")
            )
            altitude_error = abs(altitude - altitude_target)
            altitude_error_risk = _clip01(
                (altitude_error - altitude_band) / altitude_band
            )

        components: dict[str, float] = {
            "time": -max(0.0, _get(reward_config, "time_penalty_per_s")) * dt_s,
            "progress": float(progress),
            "range_rate": float(range_rate_penalty),
            "damage": _get(reward_config, "damage_scale")
            * (float(target_damage) - float(ownship_damage)),
            "evasion": float(evasion_penalty),
            "safety_low_altitude": -max(
                0.0, _get(reward_config, "low_altitude_penalty_per_s")
            )
            * altitude_risk
            * dt_s,
            "safety_descent": -max(
                0.0, _get(reward_config, "descent_rate_penalty_per_s")
            )
            * descent_altitude_risk
            * descent_excess
            * dt_s,
            "safety_attitude": -max(
                0.0, _get(reward_config, "attitude_penalty_per_s")
            )
            * attitude_risk
            * dt_s,
            "safety_altitude_error": -max(
                0.0, _get(reward_config, "altitude_error_penalty_per_s")
            )
            * altitude_error_risk
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
            fallback = _get(reward_config, "loss_reward")
            components = {key: 0.0 for key in components}
            components["terminal"] = fallback
            total = fallback
        return total, components

    def _terminal_reward(
        self,
        ownship_state,
        target_state,
        reward_config: dict,
        end_condition: str,
    ) -> float:
        if end_condition == "ownship altitude below min":
            return _get(reward_config, "loss_reward") + _get(
                reward_config, "crash_extra_penalty"
            )
        if end_condition == "target altitude below min":
            return _get(reward_config, "target_ground_reward")
        return super()._terminal_reward(
            ownship_state,
            target_state,
            reward_config,
            end_condition,
        )

    def metrics(self) -> dict[str, Any]:
        metrics = dict(super().metrics())
        metrics.update(self._last_quality)
        return metrics


def build_reward() -> ResultDrivenReward:
    """Factory called once for each Ray/Gym environment instance."""
    return ResultDrivenReward()


def describe_reward(reward_config: dict | None = None, wez_config: dict | None = None) -> dict:
    cfg = dict(MY_REWARD_CONFIG)
    if reward_config:
        cfg.update(reward_config)
    return {
        "description": (
            "Stateful v3 WVR reward: time cost, finite-horizon capture progress, "
            "range-rate discipline, damage differential, infinite-evasion warning, "
            "penalty-only safety envelope, and terminal outcome."
        ),
        "components": [
            "time",
            "progress",
            "range_rate",
            "damage",
            "evasion",
            "safety_low_altitude",
            "safety_descent",
            "safety_attitude",
            "safety_altitude_error",
            "terminal",
        ],
        "time_penalty_per_s": cfg.get("time_penalty_per_s"),
        "progress_scale": cfg.get("progress_scale"),
        "damage_scale": cfg.get("damage_scale"),
        "capture_range_inner_m": cfg.get("capture_range_inner_m"),
        "capture_range_outer_m": cfg.get("capture_range_outer_m"),
        "capture_own_ata_gate_deg": cfg.get("capture_own_ata_gate_deg"),
        "capture_target_ata_gate_deg": cfg.get("capture_target_ata_gate_deg"),
        "low_altitude_warning_m": cfg.get("low_altitude_warning_m"),
        "win_reward": cfg.get("win_reward"),
        "loss_reward": cfg.get("loss_reward"),
        "draw_reward": cfg.get("draw_reward"),
    }


def compute_reward(*args, **kwargs):
    raise RuntimeError(
        "v3_my_reward is stateful; load build_reward() once per environment."
    )


__all__ = [
    "MY_REWARD_CONFIG",
    "ResultDrivenReward",
    "build_reward",
    "compute_reward",
    "describe_reward",
]
