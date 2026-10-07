# -*- coding: utf-8 -*-
"""Probe JSBSim AutoRun controls from the official-style beam start."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from DogFightEnvWrapper import DogFightWrapper
from dogfight.ai.student_hooks import load_observation_hook, load_reward_hook


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_env(path: Path) -> tuple[dict, object, dict]:
    experiment = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    env = dict(experiment.get("env") or {})
    config = dict(experiment.get("env_config") or {})
    for key in (
        "observation_mode",
        "target_mode",
        "target_behavior_dll",
        "max_engage_time",
        "episode_step_limit",
        "min_altitude",
    ):
        if key in env:
            config[key] = env[key]

    reward_fn = None
    if env.get("reward_module"):
        reward_fn, defaults = load_reward_hook(env["reward_module"])
        config["reward"] = _deep_update(defaults, dict(config.get("reward") or {}))

    observation = load_observation_hook(env["observation_module"])
    config["observation_mode"] = observation["mode"]
    return config, reward_fn, observation


def _bearing_deg(own: np.ndarray, target: np.ndarray) -> float:
    delta = target[:3] - own[:3]
    bearing = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    current = float(own[5])
    error = (bearing - current + 180.0) % 360.0 - 180.0
    return current + error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-yaml",
        default="experiments/codex_eval_habfm_beam_fixed_30s.yaml",
    )
    parser.add_argument("--side", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--speed", type=float, default=300.0)
    parser.add_argument("--bank", type=float, default=None)
    args = parser.parse_args()

    config, reward_fn, observation = _load_env(ROOT / args.experiment_yaml)
    config["initial_scenario"]["side_choices"] = [args.side]
    env = DogFightWrapper(
        env_config=config,
        reward_fn=reward_fn,
        observation_fn=observation["build_observation"],
        observation_size=observation["size"],
        observation_low=observation["low"],
        observation_high=observation["high"],
    )
    try:
        env.reset(seed=260822)
        reference_altitude = float(env._ownship_state[44])
        target_heading = float(env._target_state[5])
        target_altitude = float(env._target_state[44])
        target_speed = float(env._target_state[27] or 250.0)
        print("frame,heading_cmd,roll,pitch,yaw,roll_cmd,pitch_cmd,rudder_cmd,throttle_cmd,ata,distance,alt")
        for frame in range(args.frames):
            own = env._sim.get_state()
            target = env._target_sim.get_state()
            heading = _bearing_deg(own, target)
            if args.bank is None:
                env._sim.step_autopilot(heading, reference_altitude, args.speed)
            else:
                env._sim.step_loiter(True, args.bank, 0.0)
            env._target_sim.step_autopilot(target_heading, target_altitude, target_speed)
            env._ownship_state = env._sim.get_state()
            env._target_state = env._target_sim.get_state()
            own = env._ownship_state
            ata = abs(float(env._geo_info._get_antenna_train_angle(own, env._target_state, False)))
            distance = float(env._geo_info._get_distance(own, env._target_state))
            print(
                f"{frame},{heading:.3f},{own[3]:.3f},{own[4]:.3f},{own[5]:.3f},"
                f"{own[15]:.5f},{own[17]:.5f},{own[19]:.5f},{own[21]:.5f},"
                f"{ata:.3f},{distance:.3f},{own[44]:.3f}"
            )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
