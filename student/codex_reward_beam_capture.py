# -*- coding: utf-8 -*-
"""Stage-1 reward for acquiring an opponent from the HABFM beam start.

This is a curriculum wrapper around the team's V6 competition reward.  It
adds a potential-difference signal that is positive only when blue points its
nose toward red while moving out of red's nose cone.  The added signal is for
training; the final mixed-combat stage can return to ``v6_ppo_my_reward``.
"""
from __future__ import annotations

import math
from typing import Any

from dogfight.sim.state_schema import StateIndex
from student import v6_ppo_my_reward as base_reward


MY_REWARD_CONFIG = dict(base_reward.MY_REWARD_CONFIG)
MY_REWARD_CONFIG.update(
    {
        "beam_progress_scale": 140.0,
        "beam_own_aim_weight": 0.65,
        "beam_target_tail_weight": 0.35,
        "beam_hold_reward_per_s": 1.5,
        "beam_success_own_ata_deg": 25.0,
        "beam_success_target_ata_deg": 120.0,
        "beam_success_max_range_m": 2500.0,
        "beam_success_bonus": 60.0,
        "beam_wrong_way_penalty_per_s": 2.0,
        "beam_wrong_way_start_deg": 100.0,
        "beam_runaway_start_m": 1800.0,
        "beam_runaway_width_m": 2200.0,
        "beam_runaway_penalty_per_s": 3.0,
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


def _angle_score_toward(angle_deg: float) -> float:
    """1 at 0 deg, 0.5 at 90 deg, and 0 at 180 deg."""
    angle = math.radians(max(0.0, min(180.0, angle_deg)))
    return 0.5 * (1.0 + math.cos(angle))


def _angle_score_behind(angle_deg: float) -> float:
    """0 at the opponent's nose and 1 directly behind its tail."""
    return 1.0 - _angle_score_toward(angle_deg)


class BeamCaptureReward:
    def __init__(self) -> None:
        self._base = base_reward.UnifiedCompetitionReward()
        self._previous_time_s = float("nan")
        self._previous_potential = float("nan")
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

        now = float(ownship_state[StateIndex.SIM_TIME])
        dt_s = (
            max(0.0, min(0.5, now - self._previous_time_s))
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

        own_score = _angle_score_toward(own_ata_deg)
        target_tail_score = _angle_score_behind(target_ata_deg)
        potential = (
            _cfg(reward_config, "beam_own_aim_weight") * own_score
            + _cfg(reward_config, "beam_target_tail_weight") * target_tail_score
        )
        progress = 0.0
        if math.isfinite(self._previous_potential):
            progress = _cfg(reward_config, "beam_progress_scale") * (
                potential - self._previous_potential
            )
        self._previous_potential = potential

        success_now = (
            own_ata_deg <= _cfg(reward_config, "beam_success_own_ata_deg")
            and target_ata_deg >= _cfg(reward_config, "beam_success_target_ata_deg")
            and distance_m <= _cfg(reward_config, "beam_success_max_range_m")
        )
        hold = (
            _cfg(reward_config, "beam_hold_reward_per_s") * dt_s
            if success_now
            else 0.0
        )
        success = 0.0
        if success_now and not self._success_latched:
            success = _cfg(reward_config, "beam_success_bonus")
            self._success_latched = True

        wrong_start = _cfg(reward_config, "beam_wrong_way_start_deg")
        wrong_way = -_cfg(reward_config, "beam_wrong_way_penalty_per_s") * _clip01(
            (own_ata_deg - wrong_start) / max(180.0 - wrong_start, 1.0)
        ) * dt_s
        runaway = -_cfg(reward_config, "beam_runaway_penalty_per_s") * _clip01(
            (distance_m - _cfg(reward_config, "beam_runaway_start_m"))
            / max(_cfg(reward_config, "beam_runaway_width_m"), 1.0)
        ) * dt_s

        extra = {
            "beam_progress": progress,
            "beam_hold": hold,
            "beam_success": success,
            "beam_wrong_way": wrong_way,
            "beam_runaway": runaway,
        }
        components.update(extra)
        total = float(total + sum(extra.values()))
        self._previous_time_s = now
        self._last_metrics = {
            "beam_own_ata_deg": own_ata_deg,
            "beam_target_ata_deg": target_ata_deg,
            "beam_potential": potential,
            "beam_success_latched": float(self._success_latched),
        }
        return total, components

    def metrics(self) -> dict[str, Any]:
        metrics = self._base.metrics()
        metrics.update(self._last_metrics)
        return metrics


def build_reward() -> BeamCaptureReward:
    return BeamCaptureReward()


__all__ = ["MY_REWARD_CONFIG", "BeamCaptureReward", "build_reward"]
