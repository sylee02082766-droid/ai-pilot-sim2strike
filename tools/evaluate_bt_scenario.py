# -*- coding: utf-8 -*-
"""Evaluate a native Behavior Tree controller in an experiment scenario."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from DogFightEnvWrapper import DogFightWrapper
from dogfight.ai.bt_action_provider import BTActionProvider
from dogfight.ai.bt_rule_manager import activate_rule_xml
from evaluate_bundle_scenario import build_env_from_experiment, load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", default="AIP_BASE.dll")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=27001)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    experiment = load_yaml(args.scenario)
    env_config, reward_fn, observation_hook = build_env_from_experiment(experiment)
    provider = BTActionProvider(dll_name=args.dll)
    env = DogFightWrapper(
        env_config=dict(env_config),
        reward_fn=reward_fn,
        observation_fn=observation_hook["build_observation"] if observation_hook else None,
        observation_size=observation_hook["size"] if observation_hook else None,
        observation_low=observation_hook["low"] if observation_hook else None,
        observation_high=observation_hook["high"] if observation_hook else None,
        ownship_action_provider=provider,
    )

    rows = []
    with activate_rule_xml(None, ROOT):
        try:
            for episode in range(1, args.episodes + 1):
                seed = args.seed_start + episode - 1
                env.reset(seed=seed)
                terminated = truncated = False
                info = {}
                while not (terminated or truncated):
                    _, _, terminated, truncated, info = env.step(
                        np.zeros(4, dtype=np.float32)
                    )
                rows.append({
                    "episode": episode,
                    "seed": seed,
                    "end_condition": info.get("end_condition", ""),
                    "ownship_health": info.get("ownship_health", ""),
                    "target_health": info.get("target_health", ""),
                    "ep_wez_steps": info.get("ep_wez_steps", ""),
                    "ep_mean_distance": info.get("ep_mean_distance", ""),
                    "ep_min_distance": info.get("ep_min_distance", ""),
                    "final_ata_deg": info.get("final_ata_deg", ""),
                    "final_aa_deg": info.get("final_aa_deg", ""),
                })
                print(rows[-1])
        finally:
            env.close()

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
