from __future__ import annotations

import math

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.sim.state_schema import StateIndex


def _nose(state: np.ndarray) -> np.ndarray:
    pitch = math.radians(float(state[StateIndex.PITCH]))
    yaw = math.radians(float(state[StateIndex.YAW]))
    return np.asarray([
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        -math.sin(pitch),
    ])


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1.0e-6)
    return math.degrees(math.acos(float(np.clip(
        np.dot(first, second) / denominator,
        -1.0,
        1.0,
    ))))


class CompetitionHybridProvider(ActionProvider):
    """Acquire with the reference BT and use PPO only inside an attack window."""

    def __init__(
        self,
        primary: ActionProvider,
        attack: ActionProvider,
        min_range_m: float = 500.0,
        max_range_m: float = 2200.0,
        max_own_ata_deg: float = 50.0,
        min_target_ata_deg: float = 90.0,
        capture_hold_s: float = 0.2,
        attack_min_hold_s: float = 2.0,
        release_range_m: float = 2800.0,
        release_own_ata_deg: float = 80.0,
        release_target_ata_deg: float = 50.0,
        debug: bool = False,
    ) -> None:
        self.primary = primary
        self.attack = attack
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)
        self.max_own_ata_deg = float(max_own_ata_deg)
        self.min_target_ata_deg = float(min_target_ata_deg)
        self.capture_hold_s = float(capture_hold_s)
        self.attack_min_hold_s = float(attack_min_hold_s)
        self.release_range_m = float(release_range_m)
        self.release_own_ata_deg = float(release_own_ata_deg)
        self.release_target_ata_deg = float(release_target_ata_deg)
        self.debug = bool(debug)
        self.reset()

    def reset(self, context: ActionContext | None = None) -> None:
        self._capture_start_s: float | None = None
        self._attack_start_s: float | None = None
        self._attacking = False
        self._last_stage: str | None = None
        self.primary.reset(context)
        self.attack.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        own = np.asarray(context.ownship_state, dtype=np.float64)
        target = np.asarray(context.target_state, dtype=np.float64)
        relative = target[:3] - own[:3]
        distance_m = float(np.linalg.norm(relative))
        own_ata_deg = _angle(_nose(own), relative)
        target_ata_deg = _angle(_nose(target), -relative)
        sim_time_s = float(own[StateIndex.SIM_TIME])

        primary_result = self.primary.compute_action(context)
        primary_stage = str(primary_result.info.get("bt32_stage", "reference_bt"))
        in_window = (
            self.min_range_m <= distance_m <= self.max_range_m
            and own_ata_deg <= self.max_own_ata_deg
            and target_ata_deg >= self.min_target_ata_deg
        )
        if in_window:
            if self._capture_start_s is None:
                self._capture_start_s = sim_time_s
        else:
            self._capture_start_s = None
        capture_duration_s = (
            sim_time_s - self._capture_start_s
            if self._capture_start_s is not None
            else 0.0
        )

        if primary_stage == "deck_recovery":
            self._attacking = False
            self._attack_start_s = None
        elif not self._attacking and capture_duration_s >= self.capture_hold_s:
            self._attacking = True
            self._attack_start_s = sim_time_s
        elif self._attacking:
            attack_elapsed_s = sim_time_s - (self._attack_start_s or sim_time_s)
            if attack_elapsed_s >= self.attack_min_hold_s and (
                distance_m >= self.release_range_m
                or own_ata_deg >= self.release_own_ata_deg
                or target_ata_deg <= self.release_target_ata_deg
            ):
                self._attacking = False
                self._attack_start_s = None

        if self._attacking:
            result = self.attack.compute_action(context)
            stage = "ppo_attack"
        else:
            result = primary_result
            stage = primary_stage

        if self.debug and stage != self._last_stage:
            print(
                "[competition-hybrid] "
                f"stage={stage} t={sim_time_s:.1f}s distance={distance_m:.0f}m "
                f"own_ata={own_ata_deg:.1f} target_ata={target_ata_deg:.1f}",
                flush=True,
            )
            self._last_stage = stage
        result.info = dict(result.info)
        result.info.update({
            "competition_stage": stage,
            "competition_distance_m": distance_m,
            "competition_own_ata_deg": own_ata_deg,
            "competition_target_ata_deg": target_ata_deg,
        })
        return result

    def close(self) -> None:
        self.primary.close()
        self.attack.close()


__all__ = ["CompetitionHybridProvider"]
