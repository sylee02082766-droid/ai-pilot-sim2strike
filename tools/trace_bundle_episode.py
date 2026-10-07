# -*- coding: utf-8 -*-
"""Save one deterministic bundle episode as a step-by-step diagnostic CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from DogFightEnvWrapper import DogFightWrapper
from dogfight.ai.rllib_utils import build_algorithm_from_bundle
from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.sim.state_schema import StateIndex
from evaluate_bundle_scenario import build_env_from_experiment, load_yaml


def _resolve(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    experiment = load_yaml(args.scenario)
    env_config, reward_fn, observation_hook = build_env_from_experiment(experiment)
    provider = RLActionProvider(
        bundle_dir=str(_resolve(args.bundle)),
        algorithm_factory=build_algorithm_from_bundle,
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

    rows = []
    try:
        _, info = env.reset(seed=args.seed)
        terminated = truncated = False
        step = 0
        previous_distance = None
        previous_time = None
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            step += 1
            obs = np.asarray(env.pre_obs, dtype=np.float32).reshape(-1)
            distance = float(env._geo_info._get_distance(env._ownship_state, env._target_state))
            ata = abs(float(env._geo_info._get_antenna_train_angle(
                env._ownship_state, env._target_state, False
            )))
            aa = abs(float(env._geo_info._get_aspect_angle(
                env._ownship_state, env._target_state, False
            )))
            sim_time = float(env._ownship_state[41])
            measured_closure = 0.0
            if previous_time is not None and sim_time > previous_time:
                measured_closure = -(distance - previous_distance) / (sim_time - previous_time)
            previous_distance = distance
            previous_time = sim_time
            raw = provider.last_raw_action if provider.last_raw_action is not None else np.zeros(4)
            sim = provider.last_sim_action if provider.last_sim_action is not None else np.zeros(4)
            wez = env.config.get("wez", {})
            half_angle = float(wez.get("angle_deg", 2.0)) * 0.5
            in_wez = (
                float(wez.get("min_range_m", 152.4)) <= distance <= float(wez.get("max_range_m", 914.4))
                and ata <= half_angle
            )
            rows.append({
                "step": step,
                "sim_time_s": sim_time,
                "own_roll_deg": float(env._ownship_state[StateIndex.ROLL]),
                "own_pitch_deg": float(env._ownship_state[StateIndex.PITCH]),
                "own_yaw_deg": float(env._ownship_state[StateIndex.YAW]),
                "own_speed_mps": float(env._ownship_state[StateIndex.KCAS]),
                "own_altitude_m": float(env._ownship_state[StateIndex.ALT]),
                "target_roll_deg": float(env._target_state[StateIndex.ROLL]),
                "target_pitch_deg": float(env._target_state[StateIndex.PITCH]),
                "target_yaw_deg": float(env._target_state[StateIndex.YAW]),
                "target_speed_mps": float(env._target_state[StateIndex.KCAS]),
                "target_altitude_m": float(env._target_state[StateIndex.ALT]),
                "distance_m": distance,
                "closure_measured_mps": measured_closure,
                "closure_obs_mps": float(obs[17]) * 300.0 if obs.size >= 18 else "",
                "ata_deg": ata,
                "aa_deg": aa,
                "los_az_deg": float(obs[11]) * 180.0 if obs.size >= 13 else "",
                "los_el_deg": float(obs[12]) * 90.0 if obs.size >= 13 else "",
                "in_wez": int(in_wez),
                "ownship_health": info.get("ownship_health", ""),
                "target_health": info.get("target_health", ""),
                **{f"raw_{name}": float(raw[i]) for i, name in enumerate(("roll", "pitch", "rudder", "throttle"))},
                **{f"sim_{name}": float(sim[i]) for i, name in enumerate(("roll", "pitch", "rudder", "throttle"))},
            })
    finally:
        env.close()

    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {output}")
    print(f"steps: {len(rows)} end={info.get('end_condition', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
