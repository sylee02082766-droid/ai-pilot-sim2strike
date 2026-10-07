from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.sim.state_schema import StateIndex


class TeacherKNNActionProvider(ActionProvider):
    """Imitate captured reference actions with a role-specific nearest-neighbor policy."""

    def __init__(
        self,
        dataset_path: str | Path,
        neighbors: int = 6,
        distance_power: float = 2.0,
        debug: bool = False,
    ) -> None:
        payload = np.load(Path(dataset_path), allow_pickle=False)
        observations = np.asarray(payload["observations"], dtype=np.float32)
        actions = np.asarray(payload["actions"], dtype=np.float32)
        plane_ids = np.asarray(payload["plane_ids"], dtype=np.int8)
        if observations.ndim != 2 or observations.shape[1] != 26:
            raise ValueError(f"expected (N, 26) observations, got {observations.shape}")
        if actions.shape != (len(observations), 4):
            raise ValueError(f"expected (N, 4) actions, got {actions.shape}")

        self.neighbors = max(1, int(neighbors))
        self.distance_power = max(0.25, float(distance_power))
        self.debug = bool(debug)
        self._roles: dict[int, tuple[np.ndarray, np.ndarray, cKDTree, np.ndarray]] = {}
        for plane_id in (0, 1):
            mask = plane_ids == plane_id
            role_observations = observations[mask]
            role_actions = actions[mask]
            if not len(role_observations):
                raise ValueError(f"dataset has no samples for plane_id={plane_id}")
            mean = role_observations.mean(axis=0, dtype=np.float64).astype(np.float32)
            std = role_observations.std(axis=0, dtype=np.float64).astype(np.float32)
            std = np.maximum(std, 0.03).astype(np.float32)
            standardized = (role_observations - mean) / std
            self._roles[plane_id] = (mean, std, cKDTree(standardized), role_actions)
        self.reset()

    def reset(self, context: ActionContext | None = None) -> None:
        self._next_status_s = 0.0

    def compute_action(self, context: ActionContext) -> ActionResult:
        observation = np.asarray(context.observation, dtype=np.float32).reshape(-1)
        if observation.shape != (26,) or not np.all(np.isfinite(observation)):
            raise ValueError(f"teacher KNN requires one finite 26-D observation, got {observation.shape}")
        plane_id = int(context.info.get("my_plane_id", 0))
        if plane_id not in self._roles:
            plane_id = 0
        mean, std, tree, actions = self._roles[plane_id]
        query = (observation - mean) / std
        started = time.perf_counter()
        distances, indices = tree.query(query, k=self.neighbors, workers=1)
        query_ms = (time.perf_counter() - started) * 1000.0
        distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))
        indices = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        weights = 1.0 / np.power(distances + 1.0e-3, self.distance_power)
        action = np.average(actions[indices], axis=0, weights=weights).astype(np.float32)
        # The behavior-cloning dataset stores RL throttle in [-1, 1]. Direct
        # Unreal providers send server throttle in [0, 1].
        action[3] = 0.5 * (float(action[3]) + 1.0)
        action = clip_action(action)

        sim_time_s = 0.0
        if context.ownship_state is not None:
            sim_time_s = float(context.ownship_state[StateIndex.SIM_TIME])
        if self.debug and sim_time_s >= self._next_status_s:
            print(
                f"[teacher-knn] t={sim_time_s:.1f}s plane={plane_id} "
                f"d1={distances[0]:.3f} dk={distances[-1]:.3f} "
                f"query={query_ms:.3f}ms action={action.tolist()}",
                flush=True,
            )
            self._next_status_s = sim_time_s + 1.0
        return ActionResult(
            action=action,
            source="teacher_knn",
            confidence=float(1.0 / (1.0 + distances[0])),
            info={"nearest_distance": float(distances[0]), "query_ms": query_ms},
        )


__all__ = ["TeacherKNNActionProvider"]
