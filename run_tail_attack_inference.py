from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.ai.rl_module_action_provider import RLModuleActionProvider
from dogfight.ai.action_provider import ActionContext, ActionResult
from dogfight.ai.student_hooks import load_observation_hook
from dogfight.ai.tail_capture_action_provider import TailCaptureActionProvider
from dogfight.unreal.async_policy import AsyncProviderCommandPolicy
from dogfight.unreal import AIType, ProviderCommandPolicy, UnrealAIPilotUDPClient
from dogfight.sim.state_schema import StateIndex


class _LegacyMergeBrakeWrapper:
    def __init__(
        self,
        inner,
        duration_s: float,
        distance_m: float,
        defense_brake_s: float = 0.0,
        post_merge_snap_s: float = 0.0,
        post_merge_snap_distance_m: float = 600.0,
    ):
        self.inner = inner
        self.duration_s = max(0.0, float(duration_s))
        self.distance_m = max(0.0, float(distance_m))
        self.defense_brake_s = max(0.0, float(defense_brake_s))
        self.post_merge_snap_s = max(0.0, float(post_merge_snap_s))
        self.post_merge_snap_distance_m = max(0.0, float(post_merge_snap_distance_m))
        self.reset()

    def reset(self, context: ActionContext | None = None) -> None:
        self._used = False
        self._until_s = -1.0
        self._defense_used = False
        self._defense_until_s = -1.0
        self._snap_until_s = -1.0
        self._snap_sign = 1.0
        self._start_s = None
        self._next_debug_s = 0.0
        self.inner.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        result = self.inner.compute_action(context)
        sim_time_s = float(context.ownship_state[StateIndex.SIM_TIME])
        distance_m = float(np.linalg.norm(
            context.target_state[:3] - context.ownship_state[:3]
        ))
        if self._start_s is None or sim_time_s < self._start_s:
            self._start_s = sim_time_s
            self._next_debug_s = 0.0
        in_pursuit = "pursuit" in str(result.source)
        if not self._used and in_pursuit and distance_m <= self.distance_m:
            self._used = True
            self._until_s = sim_time_s + self.duration_s
        info = getattr(self.inner, "last_info", {})
        own_ata_deg = float(info.get("own_ata_deg", -1.0))
        target_ata_deg = float(info.get("target_ata_deg", -1.0))
        threatened = (
            self.defense_brake_s > 0.0
            and not self._defense_used
            and distance_m <= 2200.0
            and own_ata_deg >= 60.0
            and 0.0 <= target_ata_deg <= 30.0
        )
        if threatened:
            self._defense_used = True
            self._defense_until_s = sim_time_s + self.defense_brake_s
        snap_trigger = (
            self.post_merge_snap_s > 0.0
            and sim_time_s >= self._snap_until_s
            and distance_m <= self.post_merge_snap_distance_m
            and own_ata_deg >= 60.0
            and target_ata_deg >= 90.0
        )
        if snap_trigger:
            relative = context.target_state[:3] - context.ownship_state[:3]
            target_heading_deg = np.degrees(np.arctan2(relative[1], relative[0]))
            heading_error_deg = (
                target_heading_deg - float(context.ownship_state[StateIndex.YAW]) + 180.0
            ) % 360.0 - 180.0
            # The server roll command turns opposite to the NED heading-error sign.
            self._snap_sign = 1.0 if heading_error_deg < 0.0 else -1.0
            self._snap_until_s = sim_time_s + self.post_merge_snap_s
        snap_active = sim_time_s < self._snap_until_s and own_ata_deg > 25.0
        braking = sim_time_s < max(self._until_s, self._defense_until_s)
        if snap_active:
            current_roll_deg = float(context.ownship_state[StateIndex.ROLL])
            desired_bank_deg = 85.0 * self._snap_sign
            bank_error_deg = (desired_bank_deg - current_roll_deg + 180.0) % 360.0 - 180.0
            result.action = np.asarray([
                np.clip(0.05 * bank_error_deg, -1.0, 1.0),
                -1.0,
                -0.30 * self._snap_sign,
                1.0,
            ], dtype=np.float32)
            result.source = "tail_capture_post_merge_snap"
            result.info = dict(result.info)
            result.info["post_merge_snap"] = True
        elif braking:
            action = np.asarray(result.action, dtype=np.float32).copy()
            action[3] = -1.0
            result.action = action
            result.source = "tail_capture_pursuit_brake"
            result.info = dict(result.info)
            result.info["legacy_merge_brake"] = True
        elapsed_s = sim_time_s - self._start_s
        if elapsed_s + 1.0e-6 >= self._next_debug_s:
            self._next_debug_s = elapsed_s + 5.0
            speed_mps = float(np.linalg.norm(context.ownship_state[6:9]))
            print(
                "[legacy-brake-status] "
                f"t={elapsed_s:.1f}s source={result.source} distance={distance_m:.0f}m "
                f"own_ata={float(info.get('own_ata_deg', -1.0)):.1f} "
                f"target_ata={float(info.get('target_ata_deg', -1.0)):.1f} "
                f"speed={speed_mps:.1f}mps "
                f"alt={float(context.ownship_state[StateIndex.ALT]):.0f}m",
                flush=True,
            )
        return result

    def close(self) -> None:
        self.inner.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline-style tail acquisition followed by the attack PPO."
    )
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=9999)
    parser.add_argument("--team-name", default="FDSA")
    parser.add_argument(
        "--bundle-dir",
        default="artifacts/models/codex/candidates/power80_precision_v5_earlybrake",
    )
    parser.add_argument("--observation-module", default="student.codex_tactical26")
    parser.add_argument("--action-repeat", type=int, default=6)
    parser.add_argument("--initial-turn-sign", type=float, default=1.0)
    parser.add_argument("--hard-turn-until-ata-deg", type=float, default=60.0)
    parser.add_argument("--hard-turn-max-s", type=float, default=18.0)
    parser.add_argument("--pursuit-max-bank-deg", type=float, default=88.0)
    parser.add_argument("--max-bank-deg", type=float, default=88.0)
    parser.add_argument("--merge-brake-s", type=float, default=0.0)
    parser.add_argument("--merge-brake-distance-m", type=float, default=1700.0)
    parser.add_argument("--defense-brake-s", type=float, default=0.0)
    parser.add_argument("--post-merge-snap-s", type=float, default=0.0)
    parser.add_argument("--post-merge-snap-distance-m", type=float, default=600.0)
    parser.add_argument("--capture-max-range-m", type=float, default=2200.0)
    parser.add_argument("--capture-own-ata-deg", type=float, default=25.0)
    parser.add_argument("--capture-target-ata-deg", type=float, default=120.0)
    parser.add_argument("--capture-hold-s", type=float, default=1.0)
    parser.add_argument("--max-acquisition-s", type=float, default=205.0)
    parser.add_argument("--recovery-enter-altitude-m", type=float, default=500.0)
    parser.add_argument("--recovery-exit-altitude-m", type=float, default=1000.0)
    parser.add_argument("--recovery-bank-deg", type=float, default=45.0)
    parser.add_argument("--emergency-altitude-m", type=float, default=500.0)
    parser.add_argument("--recovery-reengage-s", type=float, default=6.0)
    parser.add_argument("--packet-monitor", action="store_true")
    parser.add_argument("--legacy-server-z", action="store_true")
    parser.add_argument(
        "--legacy-provider",
        action="store_true",
        help="Use the preserved pre-energy-limiter tail-capture controller.",
    )
    return parser.parse_args()


def _load_legacy_provider_class():
    source = (
        ROOT
        / "artifacts"
        / "candidates"
        / "tail_capture"
        / "direction_adaptive_v1"
        / "tail_capture_action_provider.py"
    )
    spec = importlib.util.spec_from_file_location("legacy_tail_capture_provider", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load legacy provider: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TailCaptureActionProvider


def main():
    args = parse_args()
    bundle = Path(args.bundle_dir)
    if not bundle.is_absolute():
        bundle = ROOT / bundle
    observation = load_observation_hook(args.observation_module)
    attack_policy = RLModuleActionProvider(str(bundle), explore=False)
    provider_class = (
        _load_legacy_provider_class()
        if args.legacy_provider
        else TailCaptureActionProvider
    )
    provider_kwargs = dict(
        delegate=attack_policy,
        initial_turn_sign=args.initial_turn_sign,
        hard_turn_until_ata_deg=args.hard_turn_until_ata_deg,
        max_bank_deg=args.max_bank_deg,
        pursuit_max_bank_deg=args.pursuit_max_bank_deg,
        capture_max_range_m=args.capture_max_range_m,
        capture_own_ata_deg=args.capture_own_ata_deg,
        capture_target_ata_deg=args.capture_target_ata_deg,
        capture_hold_s=args.capture_hold_s,
        max_acquisition_s=args.max_acquisition_s,
        recovery_enter_altitude_m=args.recovery_enter_altitude_m,
        recovery_exit_altitude_m=args.recovery_exit_altitude_m,
        recovery_bank_deg=args.recovery_bank_deg,
        emergency_altitude_m=args.emergency_altitude_m,
        debug=True,
    )
    if not args.legacy_provider:
        provider_kwargs.update(
            hard_turn_max_s=args.hard_turn_max_s,
            recovery_reengage_s=args.recovery_reengage_s,
            merge_brake_s=args.merge_brake_s,
            merge_brake_distance_m=args.merge_brake_distance_m,
        )
    provider = provider_class(**provider_kwargs)
    if args.legacy_provider:
        provider = _LegacyMergeBrakeWrapper(
            provider,
            duration_s=args.merge_brake_s,
            distance_m=args.merge_brake_distance_m,
            defense_brake_s=args.defense_brake_s,
            post_merge_snap_s=args.post_merge_snap_s,
            post_merge_snap_distance_m=args.post_merge_snap_distance_m,
        )
    command_policy = AsyncProviderCommandPolicy(
        action_provider=provider,
        observation_mode=observation["mode"],
        observation_fn=observation["build_observation"],
        ownship_force_side=1,
        target_force_side=2,
        action_repeat=args.action_repeat,
        server_z_up=not args.legacy_server_z,
    )
    client = UnrealAIPilotUDPClient(
        command_policy=command_policy,
        server_ip=args.server_ip,
        server_port=args.server_port,
        team_name=args.team_name,
        ai_type=AIType.Fusion,
        heartbeat_interval_sec=1.0,
        command_delay_sec=0.0,
        recv_timeout_sec=0.2,
        enable_terminal_monitor=args.packet_monitor,
    )
    print(f"[tail-attack] server={args.server_ip}:{args.server_port}")
    print(f"[tail-attack] attack_bundle={bundle}")
    print(f"[tail-attack] provider={'legacy' if args.legacy_provider else 'current'}")
    try:
        client.run()
    finally:
        if hasattr(command_policy, "close"):
            command_policy.close()
        provider.close()


if __name__ == "__main__":
    main()
