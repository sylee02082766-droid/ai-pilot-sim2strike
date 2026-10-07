# -*- coding: utf-8 -*-
"""Evaluate and record a 10 Hz initial-turn expert for the HABFM beam start.

The expert emits the same normalized four-axis action used by PPO. Each action
is held by DogFightEnv for six 60 Hz FDM frames, so passing trajectories are
directly suitable for behavior cloning.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from DogFightEnvWrapper import DogFightWrapper
from dogfight.ai.acquisition_action_provider import (
    _body_velocity_ned,
    _positive_intercept_time,
)
from dogfight.ai.student_hooks import load_observation_hook, load_reward_hook
from dogfight.sim.state_schema import StateIndex


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

    observation_module = env.get("observation_module")
    if observation_module:
        observation = load_observation_hook(observation_module)
        config["observation_mode"] = observation["mode"]
        config["observation_module"] = observation_module
    else:
        # Built-in modes (for example tactical16) are constructed by DogFightEnv.
        mode = str(env.get("observation_mode", config.get("observation_mode", "classic12")))
        observation = {
            "mode": mode,
            "build_observation": None,
            "size": None,
            "low": None,
            "high": None,
        }
        config["observation_mode"] = mode
        config.pop("observation_module", None)
    return config, reward_fn, observation


def _wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


@dataclass
class TurnExpert:
    reference_altitude_m: float
    pitch_control: str = "nz"
    max_bank_deg: float = 68.0
    bank_per_heading_error: float = 1.05
    roll_gain: float = 0.040
    pull_gain: float = 0.60
    target_nz_g: float = 4.5
    nz_gain: float = 0.18
    pitch_gain: float = 0.035
    altitude_gain: float = 0.0025
    max_lead_time_s: float = 4.0
    lead_enable_error_deg: float = 60.0
    rudder_gain: float = 0.0
    max_rudder: float = 0.85
    turn_throttle: float = -0.5
    trail_distance_m: float = 0.0
    hard_turn_until_ata_deg: float = 0.0
    hard_turn_rudder: float = 0.0
    _hard_turn_complete: bool = False
    _hard_turn_sign: float = 0.0

    def action(self, own: np.ndarray, target: np.ndarray) -> np.ndarray:
        target_velocity = _body_velocity_ned(target)
        own_velocity = _body_velocity_ned(own)
        target_speed = float(np.linalg.norm(target_velocity))
        target_forward = (
            target_velocity / target_speed
            if target_speed > 1.0
            else np.zeros(3, dtype=np.float64)
        )
        # Aim behind the target to acquire its rear quarter instead of flying
        # pure pursuit through the target's current position.
        aim_point = np.asarray(target[:3], dtype=np.float64) - (
            self.trail_distance_m * target_forward
        )
        relative = np.asarray(aim_point - own[:3], dtype=np.float64)
        target_relative = np.asarray(target[:3] - own[:3], dtype=np.float64)
        own_speed = max(float(np.linalg.norm(own_velocity)), 285.0)
        pure_heading = math.degrees(math.atan2(float(relative[1]), float(relative[0])))
        pure_error = _wrap_deg(pure_heading - float(own[StateIndex.YAW]))
        lead_time = 0.0
        if abs(pure_error) <= self.lead_enable_error_deg:
            lead_time = _positive_intercept_time(
                relative,
                target_velocity,
                own_speed,
                self.max_lead_time_s,
            )
        intercept = relative + lead_time * target_velocity
        desired_heading = math.degrees(math.atan2(float(intercept[1]), float(intercept[0])))
        heading_error = _wrap_deg(desired_heading - float(own[StateIndex.YAW]))

        target_distance = max(float(np.linalg.norm(target_relative)), 1.0)
        own_ata_deg = math.degrees(math.acos(float(np.clip(
            np.dot(own_velocity, target_relative)
            / max(float(np.linalg.norm(own_velocity)) * target_distance, 1.0e-6),
            -1.0,
            1.0,
        ))))
        if self._hard_turn_sign == 0.0 and abs(heading_error) > 1.0e-6:
            self._hard_turn_sign = math.copysign(1.0, heading_error)
        if (
            self.hard_turn_until_ata_deg > 0.0
            and not self._hard_turn_complete
            and own_ata_deg <= self.hard_turn_until_ata_deg
        ):
            self._hard_turn_complete = True
        if self.hard_turn_until_ata_deg > 0.0 and not self._hard_turn_complete:
            desired_bank = self._hard_turn_sign * self.max_bank_deg
            bank_error = _wrap_deg(desired_bank - float(own[StateIndex.ROLL]))
            roll = float(np.clip(self.roll_gain * bank_error, -1.0, 1.0))
            rudder = float(np.clip(
                -self._hard_turn_sign * self.hard_turn_rudder,
                -self.max_rudder,
                self.max_rudder,
            ))
            return np.asarray([roll, -1.0, rudder, 1.0], dtype=np.float32)

        desired_bank = float(np.clip(
            self.bank_per_heading_error * heading_error,
            -self.max_bank_deg,
            self.max_bank_deg,
        ))
        bank_error = _wrap_deg(desired_bank - float(own[StateIndex.ROLL]))
        roll = float(np.clip(self.roll_gain * bank_error, -1.0, 1.0))

        horizontal = max(float(np.linalg.norm(intercept[:2])), 1.0)
        path_pitch = math.degrees(math.atan2(-float(intercept[2]), horizontal))
        altitude_error = self.reference_altitude_m - float(own[StateIndex.ALT])
        desired_pitch = float(np.clip(
            path_pitch + self.altitude_gain * altitude_error,
            -8.0,
            12.0,
        ))
        bank_fraction = min(abs(desired_bank) / max(self.max_bank_deg, 1.0), 1.0)
        pitch = self.pitch_gain * (float(own[StateIndex.PITCH]) - desired_pitch)
        if self.pitch_control == "pull":
            pitch -= self.pull_gain * bank_fraction
        else:
            commanded_nz = 1.0 + (self.target_nz_g - 1.0) * bank_fraction
            pitch -= self.nz_gain * (commanded_nz - float(own[31]))
        pitch = float(np.clip(pitch, -1.0, 0.45))

        distance = float(np.linalg.norm(target_relative))
        closure = 0.0
        if distance > 1.0:
            closure = -float(
                np.dot(target_velocity - own_velocity, target_relative / distance)
            )
        if abs(heading_error) > 45.0:
            throttle = self.turn_throttle
        elif distance < 450.0:
            throttle = -0.60
        elif distance < 700.0 and closure > 45.0:
            throttle = -0.10
        elif distance < 1100.0 and closure > 90.0:
            throttle = 0.25
        else:
            throttle = 1.0
        rudder = float(np.clip(
            self.rudder_gain * heading_error,
            -self.max_rudder,
            self.max_rudder,
        ))
        return np.asarray([roll, pitch, rudder, throttle], dtype=np.float32)


def _geometry(env: DogFightWrapper) -> tuple[float, float, float]:
    own = env._ownship_state
    target = env._target_state
    own_ata = abs(float(env._geo_info._get_antenna_train_angle(own, target, False)))
    target_ata = abs(float(env._geo_info._get_antenna_train_angle(target, own, False)))
    distance = float(env._geo_info._get_distance(own, target))
    return own_ata, target_ata, distance


def _episode(
    config: dict,
    reward_fn: object,
    observation: dict,
    side: float,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict, list[np.ndarray], list[np.ndarray]]:
    episode_config = dict(config)
    episode_config["initial_scenario"] = dict(config["initial_scenario"])
    episode_config["initial_scenario"]["side_choices"] = [side]
    episode_config["max_engage_time"] = max(
        float(config.get("max_engage_time", 0.0)), args.steps * 0.1 + 0.1
    )
    episode_config["episode_step_limit"] = max(
        int(config.get("episode_step_limit", 0) or 0), args.steps + 1
    )
    env = DogFightWrapper(
        env_config=episode_config,
        reward_fn=reward_fn,
        observation_fn=observation["build_observation"],
        observation_size=observation["size"],
        observation_low=observation["low"],
        observation_high=observation["high"],
    )
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    try:
        obs, _ = env.reset(seed=seed)
        expert = TurnExpert(
            reference_altitude_m=float(env._ownship_state[StateIndex.ALT]),
            pitch_control=args.pitch_control,
            max_bank_deg=args.max_bank,
            bank_per_heading_error=args.bank_per_heading,
            roll_gain=args.roll_gain,
            pull_gain=args.pull_gain,
            target_nz_g=args.target_nz,
            nz_gain=args.nz_gain,
            pitch_gain=args.pitch_gain,
            altitude_gain=args.altitude_gain,
            max_lead_time_s=args.max_lead_time,
            lead_enable_error_deg=args.lead_enable_error,
            rudder_gain=args.rudder_gain,
            max_rudder=args.max_rudder,
            turn_throttle=args.turn_throttle,
            trail_distance_m=args.trail_distance,
            hard_turn_until_ata_deg=args.hard_turn_until_ata,
            hard_turn_rudder=args.hard_turn_rudder,
        )
        if args.target == "autopilot":
            # The distributed heading autopilot steadily dives even for a
            # level altitude command. Bank-hold AutoRun with bank=0 provides
            # the intended stable straight target through the same FDM.
            env.config["target_mode"] = "loiter"
            env.config["target_loiter"] = {
                "enabled": True,
                "bank": 0.0,
                "pitch": 0.0,
            }
        else:
            env.config["target_mode"] = "fixed"

        min_ata = 180.0
        min_ata_10s = 180.0
        max_distance_10s = 0.0
        min_altitude = float("inf")
        capture_steps = 0
        max_capture_steps = 0
        first_ata45_s = None
        trace: list[dict] = []
        terminated = truncated = False
        info: dict = {}
        step = 0
        while not (terminated or truncated) and step < args.steps:
            action = expert.action(env._ownship_state, env._target_state)
            observations.append(np.asarray(obs, dtype=np.float32).copy())
            actions.append(action.copy())
            obs, _, terminated, truncated, info = env.step(action)
            own_ata, target_ata, distance = _geometry(env)
            time_s = (step + 1) * 0.1
            altitude = float(env._ownship_state[StateIndex.ALT])
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
            if step % 10 == 0:
                trace.append({
                    "time_s": round(time_s, 1),
                    "own_ata_deg": round(own_ata, 3),
                    "target_ata_deg": round(target_ata, 3),
                    "distance_m": round(distance, 3),
                    "altitude_m": round(altitude, 3),
                    "target_altitude_m": round(
                        float(env._target_state[StateIndex.ALT]), 3
                    ),
                    "roll_deg": round(float(env._ownship_state[StateIndex.ROLL]), 3),
                    "pitch_deg": round(float(env._ownship_state[StateIndex.PITCH]), 3),
                    "yaw_deg": round(float(env._ownship_state[StateIndex.YAW]), 3),
                    "kcas_mps": round(float(env._ownship_state[StateIndex.KCAS]), 3),
                    "target_kcas_mps": round(
                        float(env._target_state[StateIndex.KCAS]), 3
                    ),
                    "nz_g": round(float(env._ownship_state[31]), 3),
                })
            step += 1
            if capture_steps >= int(round(args.stop_after_capture_s * 10.0)):
                break

        final_ata, final_target_ata, final_distance = _geometry(env)
        passed = (
            not terminated
            and min_ata_10s <= 80.0
            and max_distance_10s <= 4000.0
            and first_ata45_s is not None
            and first_ata45_s <= 25.0
            and min_altitude >= 300.0
            and max_capture_steps >= 10
        )
        row = {
            "side": side,
            "seed": seed,
            "target": args.target,
            "passed": passed,
            "steps": step,
            "terminated": terminated,
            "truncated": truncated,
            "end_condition": info.get("end_condition", ""),
            "first_ata45_s": first_ata45_s,
            "min_ata_10s": min_ata_10s,
            "min_ata_deg": min_ata,
            "max_distance_10s": max_distance_10s,
            "final_ata_deg": final_ata,
            "final_target_ata_deg": final_target_ata,
            "final_distance_m": final_distance,
            "min_altitude_m": min_altitude,
            "max_capture_hold_s": max_capture_steps * 0.1,
            "trace_1hz": trace,
        }
        return row, observations, actions
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-yaml",
        default="experiments/codex_eval_habfm_beam_fixed_30s.yaml",
    )
    parser.add_argument("--target", choices=("fixed", "autopilot"), default="autopilot")
    parser.add_argument("--target-speed", type=float, default=250.0)
    parser.add_argument("--episodes-per-side", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=260822)
    parser.add_argument("--output", default="artifacts/expert/habfm_turn_expert_v1.npz")
    parser.add_argument("--report", default="artifacts/expert/habfm_turn_expert_v1.json")
    parser.add_argument("--save-dataset", action="store_true")
    parser.add_argument("--stop-after-capture-s", type=float, default=2.0)
    parser.add_argument("--max-bank", type=float, default=68.0)
    parser.add_argument("--bank-per-heading", type=float, default=1.05)
    parser.add_argument("--roll-gain", type=float, default=0.040)
    parser.add_argument("--pull-gain", type=float, default=0.60)
    parser.add_argument("--pitch-control", choices=("nz", "pull"), default="nz")
    parser.add_argument("--target-nz", type=float, default=4.5)
    parser.add_argument("--nz-gain", type=float, default=0.18)
    parser.add_argument("--pitch-gain", type=float, default=0.035)
    parser.add_argument("--altitude-gain", type=float, default=0.0025)
    parser.add_argument("--max-lead-time", type=float, default=4.0)
    parser.add_argument("--lead-enable-error", type=float, default=60.0)
    parser.add_argument("--rudder-gain", type=float, default=0.0)
    parser.add_argument("--max-rudder", type=float, default=0.85)
    parser.add_argument("--turn-throttle", type=float, default=-0.5)
    parser.add_argument("--trail-distance", type=float, default=0.0)
    parser.add_argument("--hard-turn-until-ata", type=float, default=0.0)
    parser.add_argument("--hard-turn-rudder", type=float, default=0.0)
    args = parser.parse_args()

    config, reward_fn, observation = _load_env(ROOT / args.experiment_yaml)
    rows: list[dict] = []
    all_observations: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    for side in (-1.0, 1.0):
        for episode_index in range(args.episodes_per_side):
            row, observations, actions = _episode(
                config,
                reward_fn,
                observation,
                side,
                args.seed + episode_index,
                args,
            )
            rows.append(row)
            all_observations.extend(observations)
            all_actions.extend(actions)
            print(json.dumps({key: value for key, value in row.items() if key != "trace_1hz"}))

    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    all_passed = all(bool(row["passed"]) for row in rows)
    if args.save_dataset and all_passed:
        output_path = ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            observations=np.asarray(all_observations, dtype=np.float32),
            actions=np.asarray(all_actions, dtype=np.float32),
            report=np.asarray(json.dumps(rows)),
        )
        print(f"dataset={output_path}")
    print(f"all_passed={all_passed} report={report_path}")
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
