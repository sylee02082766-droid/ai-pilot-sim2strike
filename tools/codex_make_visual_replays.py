# -*- coding: utf-8 -*-
"""Create Tacview CSV replays for a lightweight bundle and experiment YAML.

This is a small Codex helper for visual inspection. It does not train or modify
the bundle; it only runs seeded inference episodes and asks DogFightWrapper to
write the same Blue/Red CSV pair that the dashboard Replay tab consumes.
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
TOOLS = ROOT / "tools"
for path in (ROOT, SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from DogFightEnvWrapper import DogFightWrapper  # noqa: E402
from codex_eval_seeded_scenario import (  # noqa: E402
    build_env_from_experiment,
    load_yaml,
    outcome_from_row,
    resolve,
)
from dogfight.ai.bt_action_provider import BTActionProvider  # noqa: E402
from dogfight.ai.bt_rule_manager import activate_rule_xml  # noqa: E402
from dogfight.ai.rllib_utils import build_algorithm_from_bundle  # noqa: E402
from dogfight.ai.rl_action_provider import RLActionProvider  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--experiment-yaml", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="visual")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=8200)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--policy-id", default="default_policy")
    parser.add_argument("--bt-rule-xml", default="")
    parser.add_argument("--max-steps", type=int, default=450)
    return parser.parse_args()


def first_match(root: Path, pattern: str) -> Path | None:
    for path in sorted(root.glob(pattern)):
        if path.is_file():
            return path
    return None


def make_env(
    env_config: dict[str, Any],
    reward_fn,
    observation_hook,
    ownship_provider,
    target_provider,
    replay_dir: Path,
) -> DogFightWrapper:
    cfg = dict(env_config)
    cfg["artifacts_dir"] = str(replay_dir)
    cfg["_runner_index"] = "visual_replay"
    cfg["_env_index"] = 0
    replay_dir.mkdir(parents=True, exist_ok=True)
    return DogFightWrapper(
        env_config=cfg,
        reward_fn=reward_fn,
        observation_fn=observation_hook["build_observation"] if observation_hook else None,
        observation_size=observation_hook["size"] if observation_hook else None,
        observation_low=observation_hook["low"] if observation_hook else None,
        observation_high=observation_hook["high"] if observation_hook else None,
        ownship_action_provider=ownship_provider,
        target_action_provider=target_provider,
    )


def main() -> None:
    args = parse_args()
    experiment = load_yaml(args.experiment_yaml)
    env_config, reward_fn, observation_hook = build_env_from_experiment(
        experiment,
        radius=None,
        max_engage_time=None,
        episode_step_limit=None,
    )

    ownship_provider = RLActionProvider(
        bundle_dir=str(resolve(args.bundle_dir)),
        algorithm_factory=build_algorithm_from_bundle,
        policy_id=args.policy_id,
        explore=False,
    )
    target_provider = None
    if bool(env_config.get("target_behavior_provider", False)):
        target_provider = BTActionProvider(
            dll_name=str(env_config.get("target_behavior_dll", "AIP_BASE_target.dll"))
        )

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    try:
        with activate_rule_xml(args.bt_rule_xml or None, ROOT):
            for episode in range(args.episodes):
                seed = args.seed_start + episode * args.seed_step
                replay_dir = output_dir / f"{args.label}_seed_{seed}"
                env = make_env(
                    env_config,
                    reward_fn,
                    observation_hook,
                    ownship_provider,
                    target_provider,
                    replay_dir,
                )
                try:
                    _, _ = env.reset(seed=seed)
                    total_reward = 0.0
                    terminated = False
                    truncated = False
                    info: dict[str, Any] = {}
                    steps = 0
                    while not (terminated or truncated) and steps < args.max_steps:
                        _, reward, terminated, truncated, info = env.step(
                            np.zeros(4, dtype=np.float32)
                        )
                        total_reward += float(reward)
                        steps += 1
                    if not (terminated or truncated):
                        info = dict(info)
                        info.setdefault("end_condition", "visual replay step limit")
                        info.setdefault("outcome", "preview")
                    env.make_tacviewLog()
                    ownship_log = first_match(replay_dir, "*_ownship_(F-16)[Blue].csv")
                    target_log = first_match(replay_dir, "*_target_(F-16)[Red].csv")
                    summary_json = first_match(replay_dir, "*_summary.json")
                    row = {
                        "label": args.label,
                        "seed": seed,
                        "steps": steps,
                        "terminated": terminated,
                        "truncated": truncated,
                        "outcome": info.get("outcome", ""),
                        "end_condition": info.get("end_condition", ""),
                        "total_reward": total_reward,
                        "ownship_health": info.get("ownship_health", ""),
                        "target_health": info.get("target_health", ""),
                        "ep_wez_steps": info.get("ep_wez_steps", ""),
                        "ep_min_distance": info.get("ep_min_distance", ""),
                        "replay_dir": str(replay_dir),
                        "ownship_log": str(ownship_log or ""),
                        "target_log": str(target_log or ""),
                        "summary_json": str(summary_json or ""),
                    }
                    rows.append(row)
                    print(
                        f"{args.label} seed={seed} outcome={outcome_from_row(row)} "
                        f"steps={steps} target_health={row['target_health']} "
                        f"wez={row['ep_wez_steps']}",
                        flush=True,
                    )
                finally:
                    env.close()
    finally:
        ownship_provider.close()
        if target_provider is not None:
            target_provider.close()

    index_path = output_dir / f"{args.label}_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / f"{args.label}_index.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"index saved: {index_path}", flush=True)


if __name__ == "__main__":
    main()
