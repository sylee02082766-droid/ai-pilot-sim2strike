# -*- coding: utf-8 -*-
"""Post-merge reacquisition curriculum layered on the competition V6 reward.

The extra terms activate only after a close head-on pass has put the target far
off the ownship nose. They teach one missing skill: turn back, reacquire, and
close again. Final competition evaluation still uses ``v6_ppo_my_reward``.
"""
from __future__ import annotations

import math
from typing import Any

from dogfight.sim.state_schema import StateIndex
from student import v6_ppo_my_reward as base_reward


MY_REWARD_CONFIG = dict(base_reward.MY_REWARD_CONFIG)
MY_REWARD_CONFIG.update(
    {
        "reacquire_merge_distance_m": 500.0,
        "reacquire_activation_ata_deg": 60.0,
        "reacquire_success_ata_deg": 20.0,
        "reacquire_potential_scale": 60.0,
        "reacquire_aim_weight": 0.65,
        "reacquire_tail_weight": 0.15,
        "reacquire_range_weight": 0.20,
        "reacquire_range_scale_m": 5000.0,
        "reacquire_opening_penalty_per_s": 2.5,
        "reacquire_opening_ref_mps": 250.0,
        "reacquire_closure_reward_per_s": 1.5,
        "reacquire_closure_ref_mps": 150.0,
        "reacquire_success_bonus": 40.0,
        "reacquire_step_penalty_per_s": 0.05,
    }
)


def _value(config: dict, key: str) -> float:
    raw = config.get(key, MY_REWARD_CONFIG[key]) if config else MY_REWARD_CONFIG[key]
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float(MY_REWARD_CONFIG[key])
    return value if math.isfinite(value) else float(MY_REWARD_CONFIG[key])


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class ReacquisitionCurriculumReward:
    def __init__(self) -> None:
        self._base = base_reward.UnifiedCompetitionReward()
        self._previous_time_s = float("nan")
        self._previous_distance_m = float("nan")
        self._previous_potential = float("nan")
        self._close_pass_seen = False
        self._active = False
        self._success_latched = False
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
        if ownship_state is not None and target_state is not None and geo_info is not None:
            self._previous_time_s = float(ownship_state[StateIndex.SIM_TIME])
            self._previous_distance_m = float(
                geo_info._get_distance(ownship_state, target_state)
            )

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

        now = float(ownship_state[StateIndex.SIM_TIME])
        dt_s = (
            max(0.0, min(0.5, now - self._previous_time_s))
            if math.isfinite(self._previous_time_s)
            else 0.0
        )
        distance_m = float(geo_info._get_distance(ownship_state, target_state))
        ata_deg = abs(float(
            geo_info._get_antenna_train_angle(ownship_state, target_state, False)
        ))
        aa_deg = abs(float(
            geo_info._get_aspect_angle(ownship_state, target_state, False)
        ))

        if distance_m <= _value(reward_config, "reacquire_merge_distance_m"):
            self._close_pass_seen = True
        if (
            self._close_pass_seen
            and ata_deg >= _value(reward_config, "reacquire_activation_ata_deg")
        ):
            self._active = True

        progress = 0.0
        opening = 0.0
        closure_reward = 0.0
        success = 0.0
        step_cost = 0.0
        closure_mps = 0.0
        potential = 0.0
        if self._active and dt_s > 0.0:
            closure_mps = -(
                distance_m - self._previous_distance_m
            ) / dt_s if math.isfinite(self._previous_distance_m) else 0.0
            aim_score = 1.0 - _clip01(ata_deg / 180.0)
            tail_score = 1.0 - _clip01(aa_deg / 180.0)
            range_score = math.exp(
                -distance_m
                / max(_value(reward_config, "reacquire_range_scale_m"), 1.0)
            )
            potential = (
                _value(reward_config, "reacquire_aim_weight") * aim_score
                + _value(reward_config, "reacquire_tail_weight") * tail_score
                + _value(reward_config, "reacquire_range_weight") * range_score
            )
            if math.isfinite(self._previous_potential):
                progress = _value(
                    reward_config, "reacquire_potential_scale"
                ) * (potential - self._previous_potential)
            self._previous_potential = potential

            opening = -_value(
                reward_config, "reacquire_opening_penalty_per_s"
            ) * _clip01(
                -closure_mps
                / max(_value(reward_config, "reacquire_opening_ref_mps"), 1.0)
            ) * dt_s
            if ata_deg <= _value(reward_config, "reacquire_success_ata_deg"):
                closure_reward = _value(
                    reward_config, "reacquire_closure_reward_per_s"
                ) * _clip01(
                    closure_mps
                    / max(_value(reward_config, "reacquire_closure_ref_mps"), 1.0)
                ) * dt_s
                if not self._success_latched:
                    success = _value(reward_config, "reacquire_success_bonus")
                    self._success_latched = True
            step_cost = -_value(
                reward_config, "reacquire_step_penalty_per_s"
            ) * dt_s

        extra = {
            "reacquire_progress": progress,
            "reacquire_opening": opening,
            "reacquire_closure": closure_reward,
            "reacquire_success": success,
            "reacquire_time": step_cost,
        }
        components.update(extra)
        total = float(total + sum(extra.values()))
        self._previous_time_s = now
        self._previous_distance_m = distance_m
        self._last_metrics = {
            "reacquire_active": float(self._active),
            "reacquire_success": float(self._success_latched),
            "reacquire_potential": potential,
            "reacquire_closure_mps": closure_mps,
        }
        return total, components

    def metrics(self) -> dict[str, Any]:
        metrics = self._base.metrics()
        metrics.update(self._last_metrics)
        return metrics


def build_reward() -> ReacquisitionCurriculumReward:
    return ReacquisitionCurriculumReward()


__all__ = ["MY_REWARD_CONFIG", "ReacquisitionCurriculumReward", "build_reward"]
