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
from dogfight.ai.competition_hybrid_provider import CompetitionHybridProvider
from dogfight.ai.rl_module_action_provider import RLModuleActionProvider
from dogfight.ai.student_hooks import load_observation_hook
from dogfight.unreal.async_policy import AsyncProviderCommandPolicy
from dogfight.unreal import AIType, ProviderCommandPolicy, UnrealAIPilotUDPClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=9999)
    parser.add_argument("--team-name", default="FDSA")
    parser.add_argument("--action-repeat", type=int, default=6)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    candidate = ROOT / "artifacts/candidates/reference_bt_reconstruction"
    primary = BT32ActionProvider(
        candidate / "bt32_host.exe",
        candidate / "AIP_DCS.dll",
        candidate,
        candidate / "debug_crt_x86",
        debug=args.debug,
        recovery_enter_altitude_m=900.0,
        recovery_exit_altitude_m=1500.0,
    )
    attack = RLModuleActionProvider(
        str(ROOT / "artifacts/models/codex/candidates/power80_precision_v5_earlybrake"),
        explore=False,
    )
    provider = CompetitionHybridProvider(primary, attack, debug=args.debug)
    observation = load_observation_hook("student.codex_tactical26")
    policy = AsyncProviderCommandPolicy(
        action_provider=provider,
        observation_mode=observation["mode"],
        observation_fn=observation["build_observation"],
        ownship_force_side=1,
        target_force_side=2,
        action_repeat=args.action_repeat,
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
    print(f"[competition-hybrid] server={args.server_ip}:{args.server_port}", flush=True)
    try:
        client.run()
    finally:
        policy.close()
        provider.close()


if __name__ == "__main__":
    main()
