# -*- coding: utf-8 -*-
"""Dense reward for the tail-acquisition curriculum.

This phase deliberately does not teach shooting. It rewards improvement toward
one objective: point at the opponent, move behind it, and enter a useful range.
The wrapper separately requires that position to be held for two seconds.
"""
from __future__ import annotations

import math
from typing import Any

from dogfight.sim.state_schema import StateIndex
from student import v6_ppo_my_reward as base_reward


MY_REWARD_CONFIG = dict(base_reward.MY_REWARD_CONFIG)
MY_REWARD_CONFIG.update(
    {
        "tail_progress_scale": 12.0,
        "tail_progress_gamma": 0.997,
        "tail_nose_sigma_deg": 40.0,
        "tail_range_center_m": 1150.0,
        "tail_range_sigma_m": 900.0,
        "tail_rear_start_deg": 80.0,
        "tail_nose_weight": 0.50,
        "tail_rear_weight": 0.35,
        "tail_range_weight": 0.15,
        "tail_hold_reward_per_s": 2.0,
        "tail_hold_min_range_m": 500.0,
        "tail_hold_max_range_m": 1800.0,
        "tail_hold_own_ata_deg": 25.0,
        "tail_hold_target_ata_deg": 120.0,
    }
)


def _cfg(config: dict | None, key: str) -> float:
    raw = config.get(key, MY_REWARD_CONFIG[key]) if config else MY_REWARD_CONFIG[key]
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float(MY_REWARD_CONFIG[key])
    return value if math.isfinite(value) else float(MY_REWARD_CONFIG[key])


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class TailCaptureReward:
    def __init__(self) -> None:
        self._base = base_reward.UnifiedCompetitionReward()
        self._previous_time_s = float("nan")
        self._previous_potential = float("nan")
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
        self._base.reset(
            ownship_state, target_state, geo_info, wez_config, reward_config
        )
        if ownship_state is not None:
            self._previous_time_s = float(ownship_state[StateIndex.SIM_TIME])

    def adjudicate(self, *args, **kwargs):
        return self._base.adjudicate(*args, **kwargs)

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
        total, components = self._base(
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
            action=action,
        )

        now_s = float(ownship_state[StateIndex.SIM_TIME])
        dt_s = (
            max(0.0, min(0.5, now_s - self._previous_time_s))
            if math.isfinite(self._previous_time_s)
            else 0.0
        )
        distance_m = float(geo_info._get_distance(ownship_state, target_state))
        own_ata_deg = abs(float(geo_info._get_antenna_train_angle(
            ownship_state, target_state, False
        )))
        target_ata_deg = abs(float(geo_info._get_antenna_train_angle(
            target_state, ownship_state, False
        )))

        nose_sigma = max(_cfg(reward_config, "tail_nose_sigma_deg"), 0.1)
        range_sigma = max(_cfg(reward_config, "tail_range_sigma_m"), 1.0)
        rear_start = _cfg(reward_config, "tail_rear_start_deg")
        nose_score = math.exp(-0.5 * (own_ata_deg / nose_sigma) ** 2)
        range_score = math.exp(-0.5 * (
            (distance_m - _cfg(reward_config, "tail_range_center_m")) / range_sigma
        ) ** 2)
        rear_score = _clip01(
            (target_ata_deg - rear_start) / max(180.0 - rear_start, 1.0)
        )
        potential = (
            _cfg(reward_config, "tail_nose_weight") * nose_score
            + _cfg(reward_config, "tail_rear_weight") * rear_score
            + _cfg(reward_config, "tail_range_weight") * range_score
        )
        progress = 0.0
        if math.isfinite(self._previous_potential):
            progress = _cfg(reward_config, "tail_progress_scale") * (
                _cfg(reward_config, "tail_progress_gamma") * potential
                - self._previous_potential
            )

        in_capture = (
            _cfg(reward_config, "tail_hold_min_range_m") <= distance_m
            <= _cfg(reward_config, "tail_hold_max_range_m")
            and own_ata_deg <= _cfg(reward_config, "tail_hold_own_ata_deg")
            and target_ata_deg >= _cfg(reward_config, "tail_hold_target_ata_deg")
        )
        hold = (
            _cfg(reward_config, "tail_hold_reward_per_s") * dt_s
            if in_capture
            else 0.0
        )
        components.update({"tail_progress": progress, "tail_hold": hold})
        total = float(total + progress + hold)
        self._previous_time_s = now_s
        self._previous_potential = potential
        self._last_metrics = {
            "tail_potential": potential,
            "tail_nose_score": nose_score,
            "tail_rear_score": rear_score,
            "tail_range_score": range_score,
            "tail_in_capture": float(in_capture),
        }
        return total, components

    def metrics(self) -> dict[str, Any]:
        metrics = self._base.metrics()
        metrics.update(self._last_metrics)
        return metrics


def build_reward() -> TailCaptureReward:
    return TailCaptureReward()


__all__ = ["MY_REWARD_CONFIG", "TailCaptureReward", "build_reward"]
