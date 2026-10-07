# -*- coding: utf-8 -*-
"""Evaluate one RL bundle in the env/reward scenario defined by an experiment YAML.

This is a project utility, not an official competition file. It complements
evaluate_bundle_matrix.py when a run uses a custom scenario such as tail-chase
WEZ training and we need reproducible apples-to-apples checks.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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
from dogfight.ai.student_hooks import load_observation_hook, load_reward_hook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--experiment-yaml", required=True)
    parser.add_argument("--policy-id", default="default_policy")
    parser.add_argument(
        "--target-mode",
        choices=("fixed", "loiter", "autopilot", "behavior_tree"),
        default=None,
        help="Override the YAML target mode for controlled opponent A/B tests.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="Optional first reset seed. When set, episode N uses seed-start + (N-1)*seed-step.",
    )
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument(
        "--wez-angle-deg",
        type=float,
        default=None,
        help="Override env_config.wez.angle_deg for a held-out cone check.",
    )
    parser.add_argument(
        "--target-bt-prob",
        type=float,
        default=None,
        help="Override env_config.target_bt.prob (0 = straight autopilot, 1 = all BT).",
    )
    parser.add_argument("--bt-rule-xml", default="")
    parser.add_argument("--save-log", action="store_true")
    parser.add_argument(
        "--postprocess-from-yaml",
        action="store_true",
        help=(
            "Use env_config.action_postprocess from the experiment YAML instead "
            "of the bundle metadata. Intended for controlled controller A/B tests."
        ),
    )
    parser.add_argument(
        "--output",
        default="",
        help="CSV output path. Defaults to artifacts/evals/<run>_<bundle>_scenario.csv",
    )
    return parser.parse_args()


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path_text: str) -> dict[str, Any]:
    path = _resolve(path_text)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Experiment YAML must be a mapping: {path}")
    return data


def build_env_from_experiment(experiment: dict[str, Any]) -> tuple[dict[str, Any], Any, dict[str, Any] | None]:
    env = dict(experiment.get("env") or {})
    env_config = dict(experiment.get("env_config") or {})

    for yaml_key, config_key in (
        ("observation_mode", "observation_mode"),
        ("target_mode", "target_mode"),
        ("target_behavior_dll", "target_behavior_dll"),
        ("max_engage_time", "max_engage_time"),
        ("episode_step_limit", "episode_step_limit"),
        ("min_altitude", "min_altitude"),
    ):
        if yaml_key in env:
            env_config[config_key] = env[yaml_key]

    reward_fn = None
    reward_module = env.get("reward_module") or env_config.get("reward_module")
    if reward_module:
        reward_fn, reward_defaults = load_reward_hook(str(reward_module))
        reward_overrides = dict(env_config.get("reward") or {})
        env_config["reward"] = _deep_update(reward_defaults, reward_overrides)
        env_config["reward_module"] = str(reward_module)

    observation_hook = None
    observation_module = env.get("observation_module") or env_config.get("observation_module")
    if observation_module:
        observation_hook = load_observation_hook(str(observation_module))
        env_config["observation_mode"] = observation_hook["mode"]
        env_config["observation_module"] = str(observation_module)

    return env_config, reward_fn, observation_hook


def output_paths(args: argparse.Namespace, experiment: dict[str, Any]) -> tuple[Path, Path]:
    if args.output:
        csv_path = _resolve(args.output)
    else:
        run_name = str(experiment.get("name") or Path(args.experiment_yaml).stem)
        bundle_name = Path(args.bundle_dir).name
        parent_name = Path(args.bundle_dir).parent.name
        csv_path = ROOT / "artifacts" / "evals" / f"{run_name}_{parent_name}_{bundle_name}_scenario.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    return csv_path, csv_path.with_suffix(".json")


def run_episode(
    args: argparse.Namespace,
    env_config: dict[str, Any],
    reward_fn: Any,
    observation_hook: dict[str, Any] | None,
    episode_index: int,
) -> dict[str, Any]:
    ownship_provider = RLActionProvider(
        bundle_dir=args.bundle_dir,
        algorithm_factory=build_algorithm_from_bundle,
        policy_id=args.policy_id,
        explore=False,
        action_postprocess_config=(
            env_config.get("action_postprocess")
            if args.postprocess_from_yaml
            else None
        ),
    )
    env = DogFightWrapper(
        env_config=dict(env_config),
        reward_fn=reward_fn,
        observation_fn=observation_hook["build_observation"] if observation_hook else None,
        observation_size=observation_hook["size"] if observation_hook else None,
        observation_low=observation_hook["low"] if observation_hook else None,
        observation_high=observation_hook["high"] if observation_hook else None,
        ownship_action_provider=ownship_provider,
    )

    info: dict[str, Any] = {}
    try:
        reset_seed = (
            args.seed_start + (episode_index - 1) * args.seed_step
            if args.seed_start is not None
            else None
        )
        _, info = env.reset(seed=reset_seed)
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        min_direct_ata_deg = 180.0
        while not (terminated or truncated):
            _, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            total_reward += float(reward)
            steps += 1
            min_direct_ata_deg = min(
                min_direct_ata_deg,
                abs(float(env._geo_info._get_antenna_train_angle(
                    env._ownship_state, env._target_state, False
                ))),
            )
        final_direct_ata_deg = abs(float(env._geo_info._get_antenna_train_angle(
            env._ownship_state, env._target_state, False
        )))
        final_target_ata_deg = abs(float(env._geo_info._get_antenna_train_angle(
            env._target_state, env._ownship_state, False
        )))
        if args.save_log:
            env.make_tacviewLog()
    finally:
        env.close()

    return {
        "bundle_dir": str(Path(args.bundle_dir)),
        "experiment_yaml": str(_resolve(args.experiment_yaml)),
        "episode": episode_index,
        "seed": (
            args.seed_start + (episode_index - 1) * args.seed_step
            if args.seed_start is not None
            else ""
        ),
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
        "min_direct_ata_deg": min_direct_ata_deg,
        "final_direct_ata_deg": final_direct_ata_deg,
        "final_target_ata_deg": final_target_ata_deg,
        "outcome": info.get("outcome", ""),
    }


def run_episodes_reusing_model(
    args: argparse.Namespace,
    env_config: dict[str, Any],
    reward_fn: Any,
    observation_hook: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Load the RLlib model once and reset only the simulator per episode."""
    ownship_provider = RLActionProvider(
        bundle_dir=args.bundle_dir,
        algorithm_factory=build_algorithm_from_bundle,
        policy_id=args.policy_id,
        explore=False,
        action_postprocess_config=(
            env_config.get("action_postprocess")
            if args.postprocess_from_yaml
            else None
        ),
    )
    env = DogFightWrapper(
        env_config=dict(env_config),
        reward_fn=reward_fn,
        observation_fn=observation_hook["build_observation"] if observation_hook else None,
        observation_size=observation_hook["size"] if observation_hook else None,
        observation_low=observation_hook["low"] if observation_hook else None,
        observation_high=observation_hook["high"] if observation_hook else None,
        ownship_action_provider=ownship_provider,
    )

    rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(1, args.episodes + 1):
            reset_seed = (
                args.seed_start + (episode_index - 1) * args.seed_step
                if args.seed_start is not None
                else None
            )
            _, info = env.reset(seed=reset_seed)
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
            min_direct_ata_deg = 180.0
            while not (terminated or truncated):
                _, reward, terminated, truncated, info = env.step(
                    np.zeros(4, dtype=np.float32)
                )
                total_reward += float(reward)
                steps += 1
                min_direct_ata_deg = min(
                    min_direct_ata_deg,
                    abs(float(env._geo_info._get_antenna_train_angle(
                        env._ownship_state, env._target_state, False
                    ))),
                )
            final_direct_ata_deg = abs(float(env._geo_info._get_antenna_train_angle(
                env._ownship_state, env._target_state, False
            )))
            final_target_ata_deg = abs(float(env._geo_info._get_antenna_train_angle(
                env._target_state, env._ownship_state, False
            )))
            if args.save_log:
                env.make_tacviewLog()

            row = {
                "bundle_dir": str(Path(args.bundle_dir)),
                "experiment_yaml": str(_resolve(args.experiment_yaml)),
                "episode": episode_index,
                "seed": reset_seed if reset_seed is not None else "",
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
                "min_direct_ata_deg": min_direct_ata_deg,
                "final_direct_ata_deg": final_direct_ata_deg,
                "final_target_ata_deg": final_target_ata_deg,
                "outcome": info.get("outcome", ""),
            }
            rows.append(row)
            print(
                f"episode={episode_index} steps={row['steps']} "
                f"end={row['end_condition']} reward={row['total_reward']} "
                f"wez={row['ep_wez_steps']}"
            )
    finally:
        env.close()
    return rows


def main() -> None:
    args = parse_args()
    experiment = load_yaml(args.experiment_yaml)
    env_config, reward_fn, observation_hook = build_env_from_experiment(experiment)
    if args.target_mode is not None:
        env_config["target_mode"] = args.target_mode
    if args.wez_angle_deg is not None:
        wez = dict(env_config.get("wez") or {})
        wez["angle_deg"] = float(args.wez_angle_deg)
        env_config["wez"] = wez
    if args.target_bt_prob is not None:
        target_bt = dict(env_config.get("target_bt") or {})
        target_bt["prob"] = float(np.clip(args.target_bt_prob, 0.0, 1.0))
        env_config["target_bt"] = target_bt
    csv_path, json_path = output_paths(args, experiment)

    with activate_rule_xml(args.bt_rule_xml or None, ROOT):
        rows = run_episodes_reusing_model(args, env_config, reward_fn, observation_hook)

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
