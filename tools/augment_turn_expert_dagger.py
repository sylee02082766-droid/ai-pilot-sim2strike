# -*- coding: utf-8 -*-
"""Add expert corrections on states visited by a cloned turn policy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from DogFightEnvWrapper import DogFightWrapper
from build_turn_expert_dataset import TurnExpert, _load_env
from dogfight.ai.rl_module_action_provider import RLModuleActionProvider
from dogfight.sim.state_schema import StateIndex


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-bundle", required=True)
    parser.add_argument(
        "--source-dataset", default="artifacts/expert/habfm_turn_expert_v1.npz"
    )
    parser.add_argument(
        "--output", default="artifacts/expert/habfm_turn_expert_dagger_v1.npz"
    )
    parser.add_argument(
        "--experiment-yaml",
        default="experiments/codex_eval_habfm_beam_fixed_30s.yaml",
    )
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=260924)
    args = parser.parse_args()

    source = np.load(ROOT / args.source_dataset)
    observations = [row.copy() for row in np.asarray(source["observations"], dtype=np.float32)]
    actions = [row.copy() for row in np.asarray(source["actions"], dtype=np.float32)]
    config, reward_fn, observation = _load_env(ROOT / args.experiment_yaml)
    rows = []

    for side in (-1.0, 1.0):
        episode_config = dict(config)
        episode_config["initial_scenario"] = dict(config["initial_scenario"])
        episode_config["initial_scenario"]["side_choices"] = [side]
        episode_config["max_engage_time"] = args.steps * 0.1 + 0.1
        episode_config["episode_step_limit"] = args.steps + 1
        episode_config["action_postprocess"] = {}
        policy = RLModuleActionProvider(args.policy_bundle, explore=False)
        env = DogFightWrapper(
            env_config=episode_config,
            reward_fn=reward_fn,
            observation_fn=observation["build_observation"],
            observation_size=observation["size"],
            observation_low=observation["low"],
            observation_high=observation["high"],
        )
        try:
            obs, _ = env.reset(seed=args.seed)
            env.config["target_mode"] = "loiter"
            env.config["target_loiter"] = {
                "enabled": True,
                "bank": 0.0,
                "pitch": 0.0,
            }
            expert = TurnExpert(
                reference_altitude_m=float(env._ownship_state[StateIndex.ALT]),
                pitch_control="pull",
                max_bank_deg=88.0,
                pull_gain=1.3,
                pitch_gain=0.04,
                max_lead_time_s=0.0,
                turn_throttle=-1.0,
                rudder_gain=0.0,
            )
            collected = 0
            terminated = truncated = False
            for _ in range(args.steps):
                expert_action = expert.action(env._ownship_state, env._target_state)
                policy_action = policy._compute_module_action(np.asarray(obs, dtype=np.float32))
                observations.append(np.asarray(obs, dtype=np.float32).copy())
                actions.append(expert_action.copy())
                collected += 1
                obs, _, terminated, truncated, _ = env.step(policy_action)
                if terminated or truncated:
                    break
            row = {
                "side": side,
                "collected": collected,
                "terminated": terminated,
                "truncated": truncated,
            }
            rows.append(row)
            print(json.dumps(row))
        finally:
            env.close()
            policy.close()

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        report=np.asarray(json.dumps({
            "source_dataset": str((ROOT / args.source_dataset).resolve()),
            "policy_bundle": str((ROOT / args.policy_bundle).resolve()),
            "dagger_rows": rows,
        })),
    )
    print(f"saved={output} samples={len(observations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
