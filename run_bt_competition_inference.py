from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.ai.bt32_action_provider import BT32ActionProvider
from dogfight.ai.bt_competition_provider import BTCompetitionProvider
from dogfight.ai.student_hooks import load_observation_hook
from dogfight.unreal import AIType, UnrealAIPilotUDPClient
from dogfight.unreal.async_policy import AsyncProviderCommandPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=9999)
    parser.add_argument("--team-name", default="FDSA")
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--reference-rudder-scale", type=float, default=0.45)
    parser.add_argument("--max-attack-closure-mps", type=float, default=180.0)
    parser.add_argument("--reattack-duration-s", type=float, default=3.8)
    parser.add_argument("--energy-dive-enter-speed-mps", type=float, default=200.0)
    parser.add_argument("--energy-dive-exit-speed-mps", type=float, default=235.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    candidate = ROOT / "artifacts/candidates/reference_bt_reconstruction"
    primary = BT32ActionProvider(
        candidate / "bt32_host.exe",
        candidate / "AIP_DCS.dll",
        candidate,
        candidate / "debug_crt_x86",
        debug=args.debug,
        recovery_enter_altitude_m=500.0,
        recovery_exit_altitude_m=1000.0,
    )
    provider = BTCompetitionProvider(
        primary,
        debug=args.debug,
        reference_rudder_scale=args.reference_rudder_scale,
        max_attack_closure_mps=args.max_attack_closure_mps,
        reattack_duration_s=args.reattack_duration_s,
        energy_dive_enter_speed_mps=args.energy_dive_enter_speed_mps,
        energy_dive_exit_speed_mps=args.energy_dive_exit_speed_mps,
    )
    observation = load_observation_hook("student.codex_tactical26")
    policy = AsyncProviderCommandPolicy(
        action_provider=provider,
        observation_mode=observation["mode"],
        observation_fn=observation["build_observation"],
        ownship_force_side=1,
        target_force_side=2,
        action_repeat=args.action_repeat,
        server_z_up=True,
    )
    client = UnrealAIPilotUDPClient(
        command_policy=policy,
        server_ip=args.server_ip,
        server_port=args.server_port,
        team_name=args.team_name,
        ai_type=AIType.Fusion,
        heartbeat_interval_sec=1.0,
        command_delay_sec=0.0,
        recv_timeout_sec=0.2,
        enable_terminal_monitor=False,
    )
    print(f"[bt-competition] server={args.server_ip}:{args.server_port}", flush=True)
    try:
        client.run()
    finally:
        if hasattr(policy, "close"):
            policy.close()
        provider.close()


if __name__ == "__main__":
    main()
