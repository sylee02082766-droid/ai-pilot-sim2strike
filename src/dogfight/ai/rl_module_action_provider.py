from __future__ import annotations

from pathlib import Path

import numpy as np

from dogfight.ai.action_provider import (
    ActionContext,
    ActionProvider,
    ActionResult,
    rl_action_to_sim_action,
)
from dogfight.ai.checkpoint_io import load_lightweight_policy_bundle
from dogfight.ai.inference_env import RLLibInferenceEnv
from dogfight.ai.rllib_utils import (
    _apply_bundle_model_config,
    _sanitize_inference_config,
)


class RLModuleActionProvider(ActionProvider):
    """Inference-only provider for lightweight RLlib bundles.

    Unlike RLActionProvider, this loads the saved RLModule directly and does not
    start Ray or build an Algorithm. That makes it safe to use as a frozen
    opponent inside RLlib training workers.
    """

    def __init__(
        self,
        bundle_dir: str,
        policy_id: str = "default_policy",
        explore: bool = False,
        confidence: float = 0.9,
    ):
        self.policy_id = policy_id
        self.explore = explore
        self.confidence = confidence
        self._module_state = None
        self.metadata, weights = load_lightweight_policy_bundle(Path(bundle_dir))
        self._action_postprocess_config = (
            self.metadata.get("algorithm_config", {})
            .get("env_config", {})
            .get("action_postprocess")
        )
        self.module = _build_module_from_metadata(self.metadata, weights)

    def reset(self, context: ActionContext | None = None) -> None:
        self._module_state = None

    def compute_action(self, context: ActionContext) -> ActionResult:
        if context.observation is None:
            raise ValueError("observation is required for RLModuleActionProvider")

        action = self._compute_module_action(np.asarray(context.observation, dtype=np.float32))
        return ActionResult(
            action=rl_action_to_sim_action(
                action,
                self._action_postprocess_config,
                observation=context.observation,
            ),
            source="rl_module",
            confidence=self.confidence,
            info={"policy_id": self.policy_id, "explore": self.explore},
        )

    def close(self) -> None:
        pass

    def _compute_module_action(self, observation: np.ndarray) -> np.ndarray:
        import torch
        from ray.rllib.core.columns import Columns

        self.module.eval()
        self._ensure_module_state(torch)

        if self._module_state:
            obs_batch = observation[None, None, :]
        else:
            obs_batch = observation[None, :]
        batch = {Columns.OBS: torch.as_tensor(obs_batch, dtype=torch.float32)}
        if self._module_state:
            batch[Columns.STATE_IN] = self._module_state
            batch[Columns.SEQ_LENS] = torch.as_tensor([1], dtype=torch.int32)

        with torch.no_grad():
            output = self.module.forward_inference(batch)
            if Columns.STATE_OUT in output:
                self._module_state = _detach_state(output[Columns.STATE_OUT])

            if Columns.ACTIONS in output:
                action = output[Columns.ACTIONS]
            else:
                logits = output[Columns.ACTION_DIST_INPUTS]
                if self.explore:
                    dist_class = self.module.get_exploration_action_dist_cls()
                else:
                    dist_class = self.module.get_inference_action_dist_cls()
                action_dist = dist_class.from_logits(logits)
                if not self.explore:
                    action_dist = action_dist.to_deterministic()
                action = action_dist.sample()

        return _to_numpy_action(action)

    def _ensure_module_state(self, torch_module) -> None:
        if self._module_state is not None:
            return
        if not hasattr(self.module, "get_initial_state"):
            return
        initial_state = self.module.get_initial_state()
        if not initial_state:
            return
        self._module_state = _batch_state(initial_state, torch_module)


def _build_module_from_metadata(metadata: dict, weights: dict):
    algorithm_class = metadata.get("algorithm_class", "").strip()
    config_dict = _sanitize_inference_config(metadata.get("algorithm_config", {}))
    env_config = dict(config_dict.get("env_config") or {})
    bundle_meta = metadata.get("metadata", {})
    env_config.setdefault("observation_mode", bundle_meta.get("obs_mode", "tactical16"))

    if algorithm_class == "PPO":
        from ray.rllib.algorithms.ppo import PPOConfig

        config = PPOConfig.from_dict(config_dict)
    elif algorithm_class == "SAC":
        from ray.rllib.algorithms.sac import SACConfig

        config = SACConfig.from_dict(config_dict)
    else:
        raise ValueError(f"Unsupported lightweight bundle algorithm: {algorithm_class!r}")

    config, restored_model_config = _apply_bundle_model_config(config, bundle_meta)
    if not restored_model_config:
        raise RuntimeError("Lightweight bundle is missing restorable model_config metadata")

    inference_env = RLLibInferenceEnv(env_config)
    spec = config.get_rl_module_spec(env=inference_env)
    module = spec.build()
    module.set_state(weights)
    return module


def _to_numpy_action(action) -> np.ndarray:
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    action_array = np.asarray(action, dtype=np.float32)
    while action_array.ndim > 1 and action_array.shape[0] == 1:
        action_array = action_array[0]
    return action_array


def _batch_state(state, torch_module):
    if isinstance(state, dict):
        return {key: _batch_state(value, torch_module) for key, value in state.items()}
    tensor = state if hasattr(state, "detach") else torch_module.as_tensor(state)
    return tensor.detach().clone().float().unsqueeze(0)


def _detach_state(state):
    if isinstance(state, dict):
        return {key: _detach_state(value) for key, value in state.items()}
    return state.detach().clone() if hasattr(state, "detach") else state


__all__ = ["RLModuleActionProvider"]
