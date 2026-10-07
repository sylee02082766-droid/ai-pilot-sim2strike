# -*- coding: utf-8 -*-
"""Evaluate the classical high-ATA acquisition controller."""
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
from dogfight.ai.acquisition_action_provider import AcquisitionActionProvider
from dogfight.ai.policy_handoff_action_provider import PolicyHandoffActionProvider
from dogfight.ai.rl_module_action_provider import RLModuleActionProvider
from dogfight.ai.tail_capture_action_provider import TailCaptureActionProvider
from evaluate_bundle_scenario import build_env_from_experiment, load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=28001)
    parser.add_argument("--output", required=True)
    parser.add_argument("--delegate-bundle", default="")
    parser.add_argument("--primary-bundle", default="")
    parser.add_argument("--handoff-ata-deg", type=float, default=18.0)
    parser.add_argument("--handoff-distance-m", type=float, default=2500.0)
    parser.add_argument("--handoff-frames", type=int, default=30)
    parser.add_argument("--handoff-min-target-ata-deg", type=float, default=0.0)
    parser.add_argument("--delegate-observation-size", type=int, default=0)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--tail-capture", action="store_true")
    parser.add_argument("--tail-capture-max-range-m", type=float, default=2200.0)
    parser.add_argument("--tail-capture-hold-s", type=float, default=1.0)
    args = parser.parse_args()

    experiment = load_yaml(args.scenario)
    env_config, reward_fn, observation_hook = build_env_from_experiment(experiment)
    if args.max_seconds > 0.0:
        env_config["max_engage_time"] = args.max_seconds
        env_config["episode_step_limit"] = int(round(args.max_seconds * 10.0)) + 1
    delegate = (
        RLModuleActionProvider(args.delegate_bundle, explore=False)
        if args.delegate_bundle
        else None
    )
    if args.primary_bundle:
        if delegate is None:
            raise ValueError("--primary-bundle requires --delegate-bundle")
        provider = PolicyHandoffActionProvider(
            primary=RLModuleActionProvider(args.primary_bundle, explore=False),
            delegate=delegate,
            handoff_ata_deg=args.handoff_ata_deg,
            handoff_distance_m=args.handoff_distance_m,
            handoff_frames=args.handoff_frames,
            handoff_min_target_ata_deg=args.handoff_min_target_ata_deg,
            delegate_observation_size=(
                args.delegate_observation_size or None
            ),
            action_repeat=args.action_repeat,
        )
    elif args.tail_capture:
        provider = TailCaptureActionProvider(
            delegate=delegate,
            capture_max_range_m=args.tail_capture_max_range_m,
            capture_hold_s=args.tail_capture_hold_s,
        )
    else:
        provider = AcquisitionActionProvider(
            delegate=delegate,
            handoff_ata_deg=args.handoff_ata_deg,
            handoff_distance_m=args.handoff_distance_m,
            handoff_frames=args.handoff_frames,
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
    rows = []
    try:
        for episode in range(1, args.episodes + 1):
            seed = args.seed_start + episode - 1
            env.reset(seed=seed)
            terminated = truncated = False
            info = {}
            min_ata_deg = 180.0
            min_tail_distance_m = float("inf")
            min_tail_range_ata_deg = 180.0
            step_count = 0
            first_handoff_s = None
            handoff_distance_m = None
            handoff_ata_deg = None
            handoff_target_ata_deg = None
            while not (terminated or truncated):
                _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
                step_count += 1
                ata_deg = abs(float(env._geo_info._get_antenna_train_angle(
                    env._ownship_state, env._target_state, False
                )))
                target_ata_deg = abs(float(env._geo_info._get_antenna_train_angle(
                    env._target_state, env._ownship_state, False
                )))
                distance_m = float(env._geo_info._get_distance(
                    env._ownship_state, env._target_state
                ))
                min_ata_deg = min(min_ata_deg, ata_deg)
                if target_ata_deg >= 120.0:
                    min_tail_distance_m = min(min_tail_distance_m, distance_m)
                    if distance_m <= 2500.0:
                        min_tail_range_ata_deg = min(min_tail_range_ata_deg, ata_deg)
                if provider._delegating and first_handoff_s is None:
                    first_handoff_s = step_count * 0.1
                    handoff_distance_m = distance_m
                    handoff_ata_deg = ata_deg
                    handoff_target_ata_deg = target_ata_deg
            rows.append({
                "episode": episode,
                "seed": seed,
                "end_condition": info.get("end_condition", ""),
                "ownship_health": info.get("ownship_health", ""),
                "target_health": info.get("target_health", ""),
                "ep_wez_steps": info.get("ep_wez_steps", ""),
                "ep_mean_distance": info.get("ep_mean_distance", ""),
                "ep_min_distance": info.get("ep_min_distance", ""),
                "min_ata_deg": min_ata_deg,
                "min_tail_distance_m": (
                    min_tail_distance_m if np.isfinite(min_tail_distance_m) else ""
                ),
                "min_tail_range_ata_deg": (
                    min_tail_range_ata_deg if min_tail_range_ata_deg < 180.0 else ""
                ),
                "first_handoff_s": first_handoff_s,
                "handoff_distance_m": handoff_distance_m,
                "handoff_ata_deg": handoff_ata_deg,
                "handoff_target_ata_deg": handoff_target_ata_deg,
                "final_ata_deg": info.get("final_ata_deg", ""),
                "final_aa_deg": info.get("final_aa_deg", ""),
                "final_direct_ata_deg": abs(float(
                    env._geo_info._get_antenna_train_angle(
                        env._ownship_state, env._target_state, False
                    )
                )),
                "final_target_ata_deg": abs(float(
                    env._geo_info._get_antenna_train_angle(
                        env._target_state, env._ownship_state, False
                    )
                )),
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
