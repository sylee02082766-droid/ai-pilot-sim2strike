# -*- coding: utf-8 -*-
"""Evaluate one RL bundle against fixed, loiter, and BT targets.

This is a small project utility, not an official competition file. It exists so
that each experiment can be compared with the same target matrix and written to
CSV/JSON for the team log.
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
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from DogFightEnvWrapper import DogFightWrapper
from dogfight.ai.bt_rule_manager import activate_rule_xml
from dogfight.ai.rllib_utils import build_algorithm_from_bundle
from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.ai.student_hooks import load_observation_hook


TARGET_PRESETS: dict[str, dict[str, Any]] = {
    "fixed": {"target_mode": "fixed"},
    "loiter10": {
        "target_mode": "loiter",
        "target_loiter": {"enabled": True, "bank": 10.0, "pitch": 0.0},
    },
    "loiter20": {
        "target_mode": "loiter",
        "target_loiter": {"enabled": True, "bank": 20.0, "pitch": 0.0},
    },
    "loiter30": {
        "target_mode": "loiter",
        "target_loiter": {"enabled": True, "bank": 30.0, "pitch": 0.0},
    },
    "bt": {
        "target_mode": "behavior_tree",
        "target_behavior_dll": "AIP_BASE_target.dll",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--policy-id", default="default_policy")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["fixed", "loiter10", "loiter20", "loiter30", "bt"],
        choices=sorted(TARGET_PRESETS),
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--observation-mode", default="tactical16")
    parser.add_argument("--observation-module", default="")
    parser.add_argument("--max-engage-time", type=float, default=60.0)
    parser.add_argument("--episode-step-limit", type=int, default=3600)
    parser.add_argument("--min-altitude", type=float, default=300.0)
    parser.add_argument("--step-ratio", type=int, default=6)
    parser.add_argument("--bt-rule-xml", default="")
    parser.add_argument("--save-log", action="store_true")
    parser.add_argument(
        "--output",
        default="",
        help="CSV output path. Defaults to artifacts/evals/<bundle-name>_matrix.csv",
    )
    return parser.parse_args()


def build_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.output:
        csv_path = Path(args.output)
        if not csv_path.is_absolute():
            csv_path = ROOT / csv_path
    else:
        bundle_name = Path(args.bundle_dir).name
        parent_name = Path(args.bundle_dir).parent.name
        csv_path = ROOT / "artifacts" / "evals" / f"{parent_name}_{bundle_name}_matrix.csv"
    json_path = csv_path.with_suffix(".json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    return csv_path, json_path


def run_one(
    args: argparse.Namespace,
    target_name: str,
    episode_index: int,
    observation_hook: dict[str, Any] | None,
) -> dict[str, Any]:
    preset = dict(TARGET_PRESETS[target_name])
    env_config = {
        "observation_mode": observation_hook["mode"] if observation_hook else args.observation_mode,
        "observation_module": args.observation_module,
        "ownship_control_mode": "rl",
        "target_mode": preset.pop("target_mode"),
        "target_behavior_dll": preset.pop("target_behavior_dll", "AIP_BASE_target.dll"),
        "max_engage_time": args.max_engage_time,
        "episode_step_limit": args.episode_step_limit,
        "min_altitude": args.min_altitude,
        "step_ratio": args.step_ratio,
    }
    env_config.update(preset)

    ownship_provider = RLActionProvider(
        bundle_dir=args.bundle_dir,
        algorithm_factory=build_algorithm_from_bundle,
        policy_id=args.policy_id,
        explore=False,
    )
    env = DogFightWrapper(
        env_config=env_config,
        observation_fn=observation_hook["build_observation"] if observation_hook else None,
        observation_size=observation_hook["size"] if observation_hook else None,
        observation_low=observation_hook["low"] if observation_hook else None,
        observation_high=observation_hook["high"] if observation_hook else None,
        ownship_action_provider=ownship_provider,
    )

    try:
        _, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        while not (terminated or truncated):
            _, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            total_reward += float(reward)
            steps += 1
        if args.save_log:
            env.make_tacviewLog()
    finally:
        env.close()

    return {
        "bundle_dir": str(Path(args.bundle_dir)),
        "target": target_name,
        "episode": episode_index,
        "steps": steps,
        "terminated": terminated,
        "truncated": truncated,
        "end_condition": info.get("end_condition", ""),
        "total_reward": round(total_reward, 6),
        "ownship_health": info.get("ownship_health", ""),
        "target_health": info.get("target_health", ""),
        "ep_wez_steps": info.get("ep_wez_steps", ""),
        "ep_mean_distance": info.get("ep_mean_distance", ""),
        "ep_min_distance": info.get("ep_min_distance", ""),
        "final_ata_deg": info.get("final_ata_deg", ""),
        "final_aa_deg": info.get("final_aa_deg", ""),
        "outcome": info.get("outcome", ""),
    }


def main() -> None:
    args = parse_args()
    observation_hook = (
        load_observation_hook(args.observation_module) if args.observation_module else None
    )
    csv_path, json_path = build_output_paths(args)

    rows: list[dict[str, Any]] = []
    with activate_rule_xml(args.bt_rule_xml or None, ROOT):
        for target_name in args.targets:
            for episode_index in range(1, args.episodes + 1):
                row = run_one(args, target_name, episode_index, observation_hook)
                rows.append(row)
                print(
                    f"{target_name} ep={episode_index} steps={row['steps']} "
                    f"end={row['end_condition']} reward={row['total_reward']}"
                )

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"csv saved: {csv_path}")
    print(f"json saved: {json_path}")


if __name__ == "__main__":
    main()
