# -*- coding: utf-8 -*-
"""Replay saved HABFM turn-expert actions without running the expert."""
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
from build_turn_expert_dataset import _load_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="artifacts/expert/habfm_turn_expert_v1.npz"
    )
    parser.add_argument(
        "--experiment-yaml",
        default="experiments/codex_eval_habfm_beam_fixed_30s.yaml",
    )
    args = parser.parse_args()

    payload = np.load(ROOT / args.dataset)
    actions = np.asarray(payload["actions"], dtype=np.float32)
    source_rows = json.loads(str(payload["report"].item()))
    config, reward_fn, observation = _load_env(ROOT / args.experiment_yaml)

    offset = 0
    replay_rows = []
    for source in source_rows:
        side = float(source["side"])
        count = int(source["steps"])
        episode_actions = actions[offset : offset + count]
        offset += count

        episode_config = dict(config)
        episode_config["initial_scenario"] = dict(config["initial_scenario"])
        episode_config["initial_scenario"]["side_choices"] = [side]
        episode_config["max_engage_time"] = count * 0.1 + 1.0
        episode_config["episode_step_limit"] = count + 10
        env = DogFightWrapper(
            env_config=episode_config,
            reward_fn=reward_fn,
            observation_fn=observation["build_observation"],
            observation_size=observation["size"],
            observation_low=observation["low"],
            observation_high=observation["high"],
        )
        try:
            env.reset(seed=int(source["seed"]))
            env.config["target_mode"] = "loiter"
            env.config["target_loiter"] = {
                "enabled": True,
                "bank": 0.0,
                "pitch": 0.0,
            }
            capture_steps = 0
            max_capture_steps = 0
            terminated = truncated = False
            for action in episode_actions:
                _, _, terminated, truncated, _ = env.step(action)
                own_ata = abs(float(env._geo_info._get_antenna_train_angle(
                    env._ownship_state, env._target_state, False
                )))
                target_ata = abs(float(env._geo_info._get_antenna_train_angle(
                    env._target_state, env._ownship_state, False
                )))
                distance = float(env._geo_info._get_distance(
                    env._ownship_state, env._target_state
                ))
                captured = own_ata <= 25.0 and target_ata >= 120.0 and distance <= 2500.0
                capture_steps = capture_steps + 1 if captured else 0
                max_capture_steps = max(max_capture_steps, capture_steps)
                if terminated or truncated:
                    break

            row = {
                "side": side,
                "steps": len(episode_actions),
                "terminated": terminated,
                "truncated": truncated,
                "final_ata_deg": own_ata,
                "final_target_ata_deg": target_ata,
                "final_distance_m": distance,
                "max_capture_hold_s": max_capture_steps * 0.1,
                "passed": (
                    not terminated
                    and not truncated
                    and max_capture_steps >= 10
                ),
            }
            replay_rows.append(row)
            print(json.dumps(row))
        finally:
            env.close()

    if offset != len(actions):
        raise RuntimeError(f"dataset split mismatch: used {offset}, total {len(actions)}")
    all_passed = all(row["passed"] for row in replay_rows)
    print(f"all_passed={all_passed}")
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
