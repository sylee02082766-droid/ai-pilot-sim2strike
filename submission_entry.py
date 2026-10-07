from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _runtime_root() -> Path:
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return Path(bundled).resolve()
    return Path(__file__).resolve().parent


ROOT = _runtime_root()
SOURCE_ROOT = Path(__file__).resolve().parent if not getattr(sys, "frozen", False) else ROOT
for path in (SOURCE_ROOT, SOURCE_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.ai.bt32_action_provider import BT32ActionProvider
from dogfight.ai.bt_gun_opportunity_provider import BTGunOpportunityProvider
from dogfight.unreal.client import UnrealAIPilotUDPClient
from dogfight.unreal.policies import ProviderCommandPolicy
from dogfight.unreal.protocol import AIType
from student.codex_tactical26 import OBSERVATION_MODE, build_observation


def main() -> None:
    parser = argparse.ArgumentParser(description="FDSA AI Pilot competition client")
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=9999)
    parser.add_argument("--team-name", default="FDSA")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    candidate = ROOT / "reference_bt_reconstruction"
    host_exe = candidate / "bt32_host_forceaware.exe"
    dll_path = candidate / "AIP_DCS.dll"
    if not host_exe.is_file() or not dll_path.is_file():
        raise FileNotFoundError(
            "Submission assets are missing. Keep the executable and _internal folder together."
        )

    primary = BT32ActionProvider(
        host_exe=host_exe,
        dll_path=dll_path,
        working_directory=candidate,
        runtime_directory=candidate / "debug_crt_x86",
        debug=args.debug,
        recovery_enter_altitude_m=500.0,
        recovery_exit_altitude_m=1000.0,
    )
    provider = BTGunOpportunityProvider(
        primary=primary,
        debug=args.debug,
        acquire_max_ata_deg=25.0,
        acquire_max_distance_m=1500.0,
        attack_min_target_ata_deg=0.0,
        attack_max_closure_mps=10000.0,
        red_attack_min_target_ata_deg=90.0,
        red_attack_max_closure_mps=180.0,
        energy_defense=False,
        mirror_red_opening_s=0.0,
        mirror_red_commands=False,
        canonicalize_red_input=False,
        reference_rudder_scale=0.55,
        red_reference_rudder_scale=1.0,
        red_firing_line_rudder_only=True,
        red_firing_line_az_ref_deg=0.8,
        blue_pre_firing_rudder_only=True,
        blue_pre_firing_enter_ata_deg=25.0,
        blue_pre_firing_max_distance_m=1500.0,
        blue_pre_firing_az_ref_deg=4.0,
        blue_firing_line_micro=True,
        blue_firing_line_enter_ata_deg=5.0,
        blue_firing_line_az_ref_deg=0.8,
        blue_firing_line_az_rate_gain=0.05,
        blue_firing_line_pitch_gain=0.0,
        blue_firing_line_pitch_rate_gain=0.02,
        blue_firing_line_rate_gate_ata_deg=2.5,
        blue_firing_line_pitch_lead_enter_ata_deg=8.0,
        blue_firing_line_pitch_lead_cmd=1.0,
        blue_threat_rudder_jink=True,
        blue_threat_jink_max_distance_m=2400.0,
        blue_threat_jink_max_target_ata_deg=25.0,
        blue_threat_jink_min_own_ata_deg=5.0,
        blue_threat_jink_direction=-1.0,
        blue_threat_jink_bank_deg=78.0,
        blue_threat_jink_pitch_cmd=-1.0,
        blue_threat_jink_rudder_scale=0.15,
        blue_threat_jink_pulse_s=4.0,
        blue_threat_jink_cooldown_s=0.0,
        blue_threat_jink_alternate_pulses=True,
        blue_threat_jink_throttle_cut_speed_mps=220.0,
        energy_recovery_enter_speed_mps=165.0,
        energy_recovery_exit_speed_mps=205.0,
        force_energy_recovery_under_threat=False,
        tactical_energy_recovery=True,
        tactical_energy_recovery_blue_only=True,
    )
    policy = ProviderCommandPolicy(
        action_provider=provider,
        observation_mode=OBSERVATION_MODE,
        observation_fn=build_observation,
        ownship_force_side=1,
        target_force_side=2,
        dynamic_force_sides=True,
        action_repeat=1,
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

    os.chdir(candidate)
    print(
        f"[FDSA] connecting to {args.server_ip}:{args.server_port} "
        "(60 Hz synchronous BT, safe gun opportunity)",
        flush=True,
    )
    try:
        client.run()
    finally:
        if hasattr(policy, "close"):
            policy.close()
        provider.close()


if __name__ == "__main__":
    main()
