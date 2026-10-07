from __future__ import annotations

import math

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.sim.state_schema import StateIndex


def _nose_vector_ned(state: np.ndarray) -> np.ndarray:
    yaw = math.radians(float(state[StateIndex.YAW]))
    pitch = math.radians(float(state[StateIndex.PITCH]))
    return np.asarray(
        [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            -math.sin(pitch),
        ],
        dtype=np.float64,
    )


def _target_ata_deg(ownship_state: np.ndarray, target_state: np.ndarray) -> float:
    target_to_own = np.asarray(ownship_state[:3] - target_state[:3], dtype=np.float64)
    distance_m = float(np.linalg.norm(target_to_own))
    if distance_m <= 1.0e-6:
        return 0.0
    cosine = float(np.dot(_nose_vector_ned(target_state), target_to_own) / distance_m)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


class PolicyHandoffActionProvider(ActionProvider):
    """Use an acquisition policy until geometry is stable, then switch once."""

    def __init__(
        self,
        primary: ActionProvider,
        delegate: ActionProvider,
        handoff_ata_deg: float = 20.0,
        handoff_distance_m: float = 15000.0,
        handoff_frames: int = 12,
        handoff_min_target_ata_deg: float = 0.0,
        delegate_observation_size: int | None = None,
        action_repeat: int = 1,
    ):
        self.primary = primary
        self.delegate = delegate
        self.handoff_ata_deg = float(handoff_ata_deg)
        self.handoff_distance_m = float(handoff_distance_m)
        self.handoff_frames = max(1, int(handoff_frames))
        self.handoff_min_target_ata_deg = float(handoff_min_target_ata_deg)
        self.delegate_observation_size = (
            max(1, int(delegate_observation_size))
            if delegate_observation_size is not None
            else None
        )
        self.action_repeat = max(1, int(action_repeat))
        self._handoff_count = 0
        self._delegating = False
        self._repeat_remaining = 0
        self._last_result: ActionResult | None = None

    def reset(self, context: ActionContext | None = None) -> None:
        self._handoff_count = 0
        self._delegating = False
        self._repeat_remaining = 0
        self._last_result = None
        self.primary.reset(context)
        self.delegate.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        if self._repeat_remaining > 0 and self._last_result is not None:
            self._repeat_remaining -= 1
            return self._last_result

        observation = np.asarray(context.observation, dtype=np.float32).reshape(-1)
        if observation.size < 10:
            raise ValueError("PolicyHandoffActionProvider requires tactical16-compatible observation")

        ata_deg = abs(float(observation[9]) * 180.0)
        ownship_state = np.asarray(context.ownship_state, dtype=np.float64)
        target_state = np.asarray(context.target_state, dtype=np.float64)
        target_ata_deg = _target_ata_deg(ownship_state, target_state)
        delta = np.asarray(target_state[:3] - ownship_state[:3], dtype=np.float64)
        distance_m = float(np.linalg.norm(delta))
        if (
            ata_deg <= self.handoff_ata_deg
            and target_ata_deg >= self.handoff_min_target_ata_deg
            and distance_m <= self.handoff_distance_m
        ):
            self._handoff_count += 1
        else:
            self._handoff_count = 0
        if self._handoff_count >= self.handoff_frames:
            self._delegating = True

        provider = self.delegate if self._delegating else self.primary
        provider_context = context
        if self._delegating and self.delegate_observation_size is not None:
            provider_context = ActionContext(
                sim=context.sim,
                opponent_sim=context.opponent_sim,
                ownship_state=context.ownship_state,
                target_state=context.target_state,
                observation=observation[: self.delegate_observation_size],
                info=dict(context.info),
            )
        result = provider.compute_action(provider_context)
        result.info = dict(result.info)
        result.info.update(
            {
                "policy_handoff": self._delegating,
                "handoff_count": self._handoff_count,
                "handoff_ata_deg": ata_deg,
                "handoff_target_ata_deg": target_ata_deg,
                "handoff_distance_m": distance_m,
            }
        )
        self._last_result = result
        self._repeat_remaining = self.action_repeat - 1
        return result

    def close(self) -> None:
        self.primary.close()
        self.delegate.close()


__all__ = ["PolicyHandoffActionProvider"]
