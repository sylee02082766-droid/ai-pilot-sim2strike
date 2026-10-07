# -*- coding: utf-8 -*-
"""Evaluate a PPO bundle at 10 Hz on both official-style beam directions."""
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
from dogfight.ai.rl_module_action_provider import RLModuleActionProvider
from dogfight.sim.state_schema import StateIndex


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument(
        "--experiment-yaml",
        default="experiments/codex_eval_habfm_beam_fixed_30s.yaml",
    )
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=260923)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    config, reward_fn, observation = _load_env(ROOT / args.experiment_yaml)
    rows = []
    for side in (-1.0, 1.0):
        episode_config = dict(config)
        episode_config["initial_scenario"] = dict(config["initial_scenario"])
        episode_config["initial_scenario"]["side_choices"] = [side]
        episode_config["max_engage_time"] = args.steps * 0.1 + 0.1
        episode_config["episode_step_limit"] = args.steps + 1
        policy = RLModuleActionProvider(args.bundle, explore=False)
        episode_config["action_postprocess"] = dict(
            policy._action_postprocess_config or {}
        )
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
            min_ata_10s = 180.0
            max_distance_10s = 0.0
            min_ata = 180.0
            min_altitude = float("inf")
            first_ata45_s = None
            capture_steps = 0
            max_capture_steps = 0
            terminated = truncated = False
            end_condition = ""
            action_sum = np.zeros(4, dtype=np.float64)
            for step in range(args.steps):
                action = policy._compute_module_action(np.asarray(obs, dtype=np.float32))
                action_sum += action
                obs, _, terminated, truncated, info = env.step(action)
                end_condition = str(info.get("end_condition", ""))
                own_ata = abs(float(env._geo_info._get_antenna_train_angle(
                    env._ownship_state, env._target_state, False
                )))
                target_ata = abs(float(env._geo_info._get_antenna_train_angle(
                    env._target_state, env._ownship_state, False
                )))
                distance = float(env._geo_info._get_distance(
                    env._ownship_state, env._target_state
                ))
                altitude = float(env._ownship_state[StateIndex.ALT])
                time_s = (step + 1) * 0.1
                min_ata = min(min_ata, own_ata)
                min_altitude = min(min_altitude, altitude)
                if time_s <= 10.0 + 1.0e-6:
                    min_ata_10s = min(min_ata_10s, own_ata)
                    max_distance_10s = max(max_distance_10s, distance)
                if first_ata45_s is None and own_ata <= 45.0:
                    first_ata45_s = time_s
                captured = own_ata <= 25.0 and target_ata >= 120.0 and distance <= 2500.0
                capture_steps = capture_steps + 1 if captured else 0
                max_capture_steps = max(max_capture_steps, capture_steps)
                if terminated or truncated:
                    break

            row = {
                "side": side,
                "steps": step + 1,
                "terminated": terminated,
                "truncated": truncated,
                "end_condition": end_condition,
                "ownship_health": float(info.get("ownship_health", 1.0)),
                "target_health": float(info.get("target_health", 1.0)),
                "ep_wez_steps": int(info.get("ep_wez_steps", 0)),
                "first_ata45_s": first_ata45_s,
                "min_ata_10s": min_ata_10s,
                "min_ata_deg": min_ata,
                "max_distance_10s": max_distance_10s,
                "final_ata_deg": own_ata,
                "final_target_ata_deg": target_ata,
                "final_distance_m": distance,
                "min_altitude_m": min_altitude,
                "max_capture_hold_s": max_capture_steps * 0.1,
                "mean_action": (action_sum / (step + 1)).tolist(),
                "action_postprocess": episode_config["action_postprocess"],
            }
            row["passed"] = (
                not terminated
                and min_ata_10s <= 80.0
                and max_distance_10s <= 4000.0
                and first_ata45_s is not None
                and first_ata45_s <= 25.0
                and min_altitude >= 300.0
                and max_capture_steps >= 10
            )
            rows.append(row)
            print(json.dumps(row))
        finally:
            env.close()
            policy.close()

    report = Path(args.report) if args.report else (
        ROOT / "artifacts" / "evals" / f"{Path(args.bundle).name}_turn_eval.json"
    )
    if not report.is_absolute():
        report = ROOT / report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    all_passed = all(row["passed"] for row in rows)
    print(f"all_passed={all_passed} report={report}")
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
