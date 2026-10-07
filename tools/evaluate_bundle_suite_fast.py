# -*- coding: utf-8 -*-
"""Fast multi-bundle, multi-scenario evaluator.

Unlike evaluate_bundle_scenario.py, this keeps one environment and one loaded
RL bundle alive for all episodes of the same candidate/scenario pair.  This is
for local experiment triage only; it does not change official training code.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
SRC = ROOT / "src"
for path in (ROOT, SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from DogFightEnvWrapper import DogFightWrapper
from dogfight.ai.bt_rule_manager import activate_rule_xml
from dogfight.ai.rllib_utils import build_algorithm_from_bundle
from dogfight.ai.rl_action_provider import RLActionProvider
from evaluate_bundle_scenario import build_env_from_experiment, load_yaml


def parse_named_path(text: str) -> tuple[str, str]:
    if "=" not in text:
        path = text
        name = Path(text).name
    else:
        name, path = text.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError(f"Expected name=path, got {text!r}")
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True, type=parse_named_path)
    parser.add_argument("--scenario", action="append", required=True, type=parse_named_path)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=10001)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated seed list; overrides --episodes/--seed-start.",
    )
    parser.add_argument("--policy-id", default="default_policy")
    parser.add_argument("--bt-rule-xml", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def make_row(
    candidate_name: str,
    bundle_dir: str,
    scenario_name: str,
    scenario_yaml: str,
    episode: int,
    seed: int,
    steps: int,
    terminated: bool,
    truncated: bool,
    total_reward: float,
    info: dict[str, Any],
    initial: dict[str, float],
) -> dict[str, Any]:
    return {
        "candidate": candidate_name,
        "bundle_dir": str(Path(bundle_dir)),
        "scenario": scenario_name,
        "scenario_yaml": str(resolve(scenario_yaml)),
        "episode": episode,
        "seed": seed,
        "initial_altitude_m": initial.get("altitude_m", ""),
        "initial_own_speed_mps": initial.get("own_speed_mps", ""),
        "initial_target_speed_mps": initial.get("target_speed_mps", ""),
        "initial_distance_m": initial.get("distance_m", ""),
        "steps": steps,
        "terminated": terminated,
        "truncated": truncated,
        "end_condition": info.get("end_condition", ""),
        "outcome": info.get("outcome", ""),
        "total_reward": round(float(total_reward), 6),
        "ownship_health": info.get("ownship_health", ""),
        "target_health": info.get("target_health", ""),
        "ep_wez_steps": info.get("ep_wez_steps", ""),
        "ep_mean_distance": info.get("ep_mean_distance", ""),
        "ep_min_distance": info.get("ep_min_distance", ""),
        "final_ata_deg": info.get("final_ata_deg", ""),
        "final_aa_deg": info.get("final_aa_deg", ""),
    }


def evaluate_pair(
    candidate_name: str,
    bundle_dir: str,
    scenario_name: str,
    scenario_yaml: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    experiment = load_yaml(scenario_yaml)
    env_config, reward_fn, observation_hook = build_env_from_experiment(experiment)
    provider = RLActionProvider(
        bundle_dir=bundle_dir,
        algorithm_factory=build_algorithm_from_bundle,
        policy_id=args.policy_id,
        explore=False,
    )
    env = DogFightWrapper(
        env_config=dict(env_config),
        reward_fn=reward_fn,
        observation_fn=observation_hook["build_observation"] if observation_hook else None,
        observation_size=observation_hook["size"] if observation_hook else None,
        observation_low=observation_hook["low"] if observation_hook else None,
        observation_high=observation_hook["high"] if observation_hook else None,
        ownship_action_provider=provider,
    )
    rows: list[dict[str, Any]] = []
    try:
        seeds = (
            [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
            if args.seeds
            else [
                args.seed_start + index * args.seed_step
                for index in range(args.episodes)
            ]
        )
        for episode, seed in enumerate(seeds, start=1):
            _, info = env.reset(seed=seed)
            initial = {
                "altitude_m": -float(env._sim._init_pos_d),
                "own_speed_mps": float(env._sim._init_speed),
                "target_speed_mps": float(env._target_sim._init_speed),
                "distance_m": float(info.get("initial_distance_m", 0.0)),
            }
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
            while not (terminated or truncated):
                _, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
                total_reward += float(reward)
                steps += 1
            row = make_row(
                candidate_name,
                bundle_dir,
                scenario_name,
                scenario_yaml,
                episode,
                seed,
                steps,
                terminated,
                truncated,
                total_reward,
                info,
                initial,
            )
            rows.append(row)
            print(
                f"{candidate_name}/{scenario_name} ep={episode} "
                f"end={row['end_condition']} outcome={row['outcome']} "
                f"wez={row['ep_wez_steps']} target_health={row['target_health']} "
                f"reward={row['total_reward']}"
            )
    finally:
        env.close()
    return rows


def main() -> None:
    args = parse_args()
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with activate_rule_xml(args.bt_rule_xml or None, ROOT):
        for candidate_name, bundle_dir in args.candidate:
            for scenario_name, scenario_yaml in args.scenario:
                rows.extend(
                    evaluate_pair(
                        candidate_name,
                        bundle_dir,
                        scenario_name,
                        scenario_yaml,
                        args,
                    )
                )

    fields = list(rows[0].keys()) if rows else []
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"csv saved: {output}")


if __name__ == "__main__":
    main()
