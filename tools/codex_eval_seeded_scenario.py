# -*- coding: utf-8 -*-
"""Fast seeded scenario evaluation for Codex experiment sweeps.

This is not an official competition entrypoint. It loads one lightweight RL
bundle once, then runs a fixed seed set in the scenario defined by an experiment
YAML. Before each reset it restores the simulator initial-condition fields so
per-episode randomization does not accumulate across seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
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
from dogfight.ai.bt_action_provider import BTActionProvider
from dogfight.ai.bt_rule_manager import activate_rule_xml
from dogfight.ai.rllib_utils import build_algorithm_from_bundle
from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.ai.student_hooks import load_observation_hook, load_reward_hook


INIT_ATTRS = (
    "_init_pos_n",
    "_init_pos_e",
    "_init_pos_d",
    "_init_pos_lat",
    "_init_pos_lon",
    "_init_pos_alt",
    "_init_roll",
    "_init_pitch",
    "_init_heading",
    "_init_speed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--experiment-yaml", required=True)
    parser.add_argument("--policy-id", default="default_policy")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=7000)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--max-engage-time", type=float, default=None)
    parser.add_argument("--episode-step-limit", type=int, default=None)
    parser.add_argument("--bt-rule-xml", default="")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path_text: str) -> dict[str, Any]:
    path = resolve(path_text)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Experiment YAML must be a mapping: {path}")
    return data


def build_env_from_experiment(
    experiment: dict[str, Any],
    radius: float | None,
    max_engage_time: float | None,
    episode_step_limit: int | None,
):
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

    if radius is not None:
        rand = dict(env_config.get("ownship_randomization") or {})
        rand["enabled"] = True
        rand["radius"] = float(radius)
        env_config["ownship_randomization"] = rand

    if max_engage_time is not None:
        env_config["max_engage_time"] = float(max_engage_time)
    if episode_step_limit is not None:
        env_config["episode_step_limit"] = int(episode_step_limit)

    reward_fn = None
    reward_module = env.get("reward_module") or env_config.get("reward_module")
    if reward_module:
        reward_fn, reward_defaults = load_reward_hook(str(reward_module))
        reward_overrides = dict(env_config.get("reward") or {})
        env_config["reward"] = deep_update(reward_defaults, reward_overrides)
        env_config["reward_module"] = str(reward_module)

    observation_hook = None
    observation_module = env.get("observation_module") or env_config.get("observation_module")
    if observation_module:
        observation_hook = load_observation_hook(str(observation_module))
        env_config["observation_mode"] = observation_hook["mode"]
        env_config["observation_module"] = str(observation_module)

    return env_config, reward_fn, observation_hook


def snapshot_initials(env: DogFightWrapper) -> dict[str, dict[str, Any]]:
    snapshots = {}
    for name, fighter in (("ownship", env._sim), ("target", env._target_sim)):
        snapshots[name] = {
            attr: getattr(fighter, attr)
            for attr in INIT_ATTRS
            if hasattr(fighter, attr)
        }
    return snapshots


def restore_initials(env: DogFightWrapper, snapshots: dict[str, dict[str, Any]]) -> None:
    for name, fighter in (("ownship", env._sim), ("target", env._target_sim)):
        for attr, value in snapshots[name].items():
            setattr(fighter, attr, value)


def outcome_from_row(row: dict[str, Any]) -> str:
    outcome = str(row.get("outcome") or "")
    if outcome:
        return outcome
    end_condition = str(row.get("end_condition") or "")
    if "target destroyed" in end_condition:
        return "win"
    if "ownship destroyed" in end_condition or "ownship crash" in end_condition:
        return "loss"
    return "timeout" if row.get("truncated") else "other"


def numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value == "" or value is None:
            continue
        values.append(float(value))
    return values


def _step_geometry(env: DogFightWrapper) -> dict[str, float]:
    distance = float(env._geo_info._get_distance(env._ownship_state, env._target_state))
    ata = abs(float(env._geo_info._get_antenna_train_angle(
        env._ownship_state,
        env._target_state,
        False,
    )))
    aa = abs(float(env._geo_info._get_aspect_angle(
        env._ownship_state,
        env._target_state,
        False,
    )))
    return {"distance": distance, "ata": ata, "aa": aa}


def _wez_score(distance_m: float, ata_deg: float, wez: dict[str, Any]) -> float:
    min_range = float(wez.get("min_range_m", 152.4))
    max_range = float(wez.get("max_range_m", 914.4))
    angle = max(float(wez.get("angle_deg", 2.0)), 1.0e-6)
    if distance_m < min_range:
        range_score = max(0.0, distance_m / max(min_range, 1.0))
    elif distance_m > max_range:
        range_score = max(0.0, 1.0 - (distance_m - max_range) / max(max_range, 1.0))
    else:
        range_score = 1.0
    angle_score = max(0.0, 1.0 - ata_deg / angle)
    return float(range_score * angle_score)


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    outcomes = [outcome_from_row(row) for row in rows]
    target_health = numeric(rows, "target_health")
    ownship_health = numeric(rows, "ownship_health")
    wez_steps = numeric(rows, "ep_wez_steps")
    steps = numeric(rows, "steps")
    min_distance = numeric(rows, "ep_min_distance")
    min_ata = numeric(rows, "min_ata_deg")
    min_ata_in_range = numeric(rows, "min_ata_in_wez_range_deg")
    best_wez_score = numeric(rows, "best_wez_score")
    return {
        "bundle_dir": str(resolve(args.bundle_dir)),
        "experiment_yaml": str(resolve(args.experiment_yaml)),
        "episodes": len(rows),
        "seed_start": args.seed_start,
        "seed_step": args.seed_step,
        "radius": args.radius,
        "max_engage_time": args.max_engage_time,
        "episode_step_limit": args.episode_step_limit,
        "wins": outcomes.count("win"),
        "timeouts": outcomes.count("timeout"),
        "losses": outcomes.count("loss"),
        "others": len(rows) - outcomes.count("win") - outcomes.count("timeout") - outcomes.count("loss"),
        "win_rate": outcomes.count("win") / max(len(rows), 1),
        "avg_target_health": statistics.fmean(target_health) if target_health else None,
        "median_target_health": statistics.median(target_health) if target_health else None,
        "avg_ownship_health": statistics.fmean(ownship_health) if ownship_health else None,
        "avg_wez_steps": statistics.fmean(wez_steps) if wez_steps else None,
        "avg_steps": statistics.fmean(steps) if steps else None,
        "avg_min_distance": statistics.fmean(min_distance) if min_distance else None,
        "avg_min_ata_deg": statistics.fmean(min_ata) if min_ata else None,
        "avg_min_ata_in_wez_range_deg": (
            statistics.fmean(min_ata_in_range) if min_ata_in_range else None
        ),
        "avg_best_wez_score": statistics.fmean(best_wez_score) if best_wez_score else None,
    }


def main() -> None:
    args = parse_args()
    experiment = load_yaml(args.experiment_yaml)
    env_config, reward_fn, observation_hook = build_env_from_experiment(
        experiment,
        args.radius,
        args.max_engage_time,
        args.episode_step_limit,
    )

    provider = RLActionProvider(
        bundle_dir=str(resolve(args.bundle_dir)),
        algorithm_factory=build_algorithm_from_bundle,
        policy_id=args.policy_id,
        explore=False,
    )
    target_action_provider = None
    if bool(env_config.get("target_behavior_provider", False)):
        target_action_provider = BTActionProvider(
            dll_name=str(env_config.get("target_behavior_dll", "AIP_BASE_target.dll"))
        )

    env = DogFightWrapper(
        env_config=dict(env_config),
        reward_fn=reward_fn,
        observation_fn=observation_hook["build_observation"] if observation_hook else None,
        observation_size=observation_hook["size"] if observation_hook else None,
        observation_low=observation_hook["low"] if observation_hook else None,
        observation_high=observation_hook["high"] if observation_hook else None,
        ownship_action_provider=provider,
        target_action_provider=target_action_provider,
    )
    initial_snapshot = snapshot_initials(env)

    rows: list[dict[str, Any]] = []
    try:
        with activate_rule_xml(args.bt_rule_xml or None, ROOT):
            for episode_index in range(1, args.episodes + 1):
                seed = args.seed_start + (episode_index - 1) * args.seed_step
                restore_initials(env, initial_snapshot)
                _, info = env.reset(seed=seed)
                terminated = False
                truncated = False
                total_reward = 0.0
                steps = 0
                min_distance_m = float("inf")
                min_ata_deg = float("inf")
                min_aa_deg = float("inf")
                min_ata_in_range_deg = float("inf")
                min_distance_at_low_ata_m = float("inf")
                best_wez_score = 0.0
                while not (terminated or truncated):
                    _, reward, terminated, truncated, info = env.step(
                        np.zeros(4, dtype=np.float32)
                    )
                    total_reward += float(reward)
                    steps += 1
                    geom = _step_geometry(env)
                    distance_m = geom["distance"]
                    ata_deg = geom["ata"]
                    aa_deg = geom["aa"]
                    min_distance_m = min(min_distance_m, distance_m)
                    min_ata_deg = min(min_ata_deg, ata_deg)
                    min_aa_deg = min(min_aa_deg, aa_deg)
                    wez = env.config.get("wez", {})
                    if (
                        float(wez.get("min_range_m", 152.4))
                        <= distance_m
                        <= float(wez.get("max_range_m", 914.4))
                    ):
                        min_ata_in_range_deg = min(min_ata_in_range_deg, ata_deg)
                    if ata_deg <= float(wez.get("angle_deg", 2.0)):
                        min_distance_at_low_ata_m = min(
                            min_distance_at_low_ata_m,
                            distance_m,
                        )
                    best_wez_score = max(
                        best_wez_score,
                        _wez_score(distance_m, ata_deg, wez),
                    )
                if not np.isfinite(min_ata_in_range_deg):
                    min_ata_in_range_deg = float("nan")
                if not np.isfinite(min_distance_at_low_ata_m):
                    min_distance_at_low_ata_m = float("nan")
                row = {
                    "bundle_dir": str(resolve(args.bundle_dir)),
                    "experiment_yaml": str(resolve(args.experiment_yaml)),
                    "episode": episode_index,
                    "seed": seed,
                    "radius": args.radius,
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
                    "min_distance_m": round(min_distance_m, 6),
                    "min_ata_deg": round(min_ata_deg, 6),
                    "min_aa_deg": round(min_aa_deg, 6),
                    "min_ata_in_wez_range_deg": round(min_ata_in_range_deg, 6),
                    "min_distance_at_low_ata_m": round(min_distance_at_low_ata_m, 6),
                    "best_wez_score": round(best_wez_score, 6),
                    "outcome": info.get("outcome", ""),
                }
                rows.append(row)
                print(
                    f"episode={episode_index} seed={seed} "
                    f"outcome={outcome_from_row(row)} steps={steps} "
                    f"target_health={row['target_health']} wez={row['ep_wez_steps']}",
                    flush=True,
                )
    finally:
        env.close()

    csv_path = resolve(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = csv_path.with_suffix(".json")
    summary_path = csv_path.with_name(csv_path.stem + "_summary.json")
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(
        json.dumps(summarize(rows, args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"csv saved: {csv_path}", flush=True)
    print(f"json saved: {json_path}", flush=True)
    print(f"summary saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
