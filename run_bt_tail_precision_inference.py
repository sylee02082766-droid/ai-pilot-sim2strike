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
from dogfight.ai.bt_gun_opportunity_provider import BTGunOpportunityProvider
from dogfight.ai.bt_tail_precision_provider import BTTailPrecisionProvider
from dogfight.ai.lag_track_action_provider import LAGTrackActionProvider
from dogfight.ai.student_hooks import load_observation_hook
from dogfight.ai.teacher_knn_action_provider import TeacherKNNActionProvider
from dogfight.unreal.async_policy import AsyncProviderCommandPolicy
from dogfight.unreal import AIType, ProviderCommandPolicy, UnrealAIPilotUDPClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=9999)
    parser.add_argument("--team-name", default="FDSA")
    parser.add_argument("--action-repeat", type=int, default=6)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sync-policy", action="store_true")
    parser.add_argument("--dynamic-force-sides", action="store_true")
    parser.add_argument("--swap-blue-force-sides", action="store_true")
    parser.add_argument("--force-aware-host", action="store_true")
    parser.add_argument("--pure-bt", action="store_true")
    parser.add_argument("--legacy-server-z", action="store_true")
    parser.add_argument("--precision-enter-ata-deg", type=float, default=45.0)
    parser.add_argument("--merge-yoyo", action="store_true")
    parser.add_argument("--merge-yoyo-duration-s", type=float, default=5.0)
    parser.add_argument("--merge-brake-s", type=float, default=0.0)
    parser.add_argument("--opportunity-gun", action="store_true")
    parser.add_argument("--opportunity-gun-brake", action="store_true")
    parser.add_argument("--direct-los-assist", action="store_true")
    parser.add_argument("--minimal-gun-opportunity", action="store_true")
    parser.add_argument("--gun-acquire-ata-deg", type=float, default=25.0)
    parser.add_argument("--gun-acquire-max-distance-m", type=float, default=1500.0)
    parser.add_argument("--energy-defense", action="store_true")
    parser.add_argument("--terrain-drag-defense", action="store_true")
    parser.add_argument("--mirror-red-opening-s", type=float, default=0.0)
    parser.add_argument("--mirror-red-commands", action="store_true")
    parser.add_argument("--canonicalize-red-input", action="store_true")
    parser.add_argument("--reference-rudder-scale", type=float, default=1.0)
    parser.add_argument("--red-reference-rudder-scale", type=float)
    parser.add_argument("--blue-post-attack-reference-rudder-scale", type=float)
    parser.add_argument("--gun-hold-enter-ata-deg", type=float, default=16.0)
    parser.add_argument("--gun-hold-exit-ata-deg", type=float, default=25.0)
    parser.add_argument("--gun-hold-max-distance-m", type=float, default=1050.0)
    parser.add_argument("--gun-hold-rate-damping", type=float, default=0.0)
    parser.add_argument("--attack-min-target-ata-deg", type=float, default=0.0)
    parser.add_argument("--attack-max-closure-mps", type=float, default=10000.0)
    parser.add_argument("--diverse-energy-recovery", action="store_true")
    parser.add_argument("--energy-recovery-bank-deg", type=float, default=25.0)
    parser.add_argument("--energy-recovery-enter-speed-mps", type=float, default=165.0)
    parser.add_argument("--energy-recovery-exit-speed-mps", type=float, default=205.0)
    parser.add_argument("--force-energy-recovery-under-threat", action="store_true")
    parser.add_argument("--early-threat-unload", action="store_true")
    parser.add_argument("--tactical-defense", action="store_true")
    parser.add_argument("--tactical-defense-duration-s", type=float, default=3.2)
    parser.add_argument("--tactical-energy-recovery", action="store_true")
    parser.add_argument("--firing-line-micro", action="store_true")
    parser.add_argument("--firing-line-enter-ata-deg", type=float, default=3.5)
    parser.add_argument("--firing-line-max-distance-m", type=float, default=950.0)
    parser.add_argument("--firing-line-rudder-gain", type=float, default=0.20)
    parser.add_argument("--firing-line-rudder-limit", type=float, default=0.30)
    parser.add_argument("--pre-firing-assist", action="store_true")
    parser.add_argument("--pre-firing-pitch-gain", type=float, default=0.25)
    parser.add_argument("--pre-firing-rudder-gain", type=float, default=0.50)
    parser.add_argument("--red-firing-line-rudder-only", action="store_true")
    parser.add_argument("--red-firing-line-az-ref-deg", type=float, default=0.8)
    parser.add_argument("--blue-pre-firing-rudder-only", action="store_true")
    parser.add_argument("--blue-pre-firing-enter-ata-deg", type=float, default=25.0)
    parser.add_argument("--blue-pre-firing-max-distance-m", type=float, default=1500.0)
    parser.add_argument("--blue-pre-firing-az-ref-deg", type=float, default=4.0)
    parser.add_argument("--blue-pre-firing-az-rate-gain", type=float, default=0.0)
    parser.add_argument("--blue-late-pre-firing-start-s", type=float, default=0.0)
    parser.add_argument("--blue-late-pre-firing-enter-ata-deg", type=float, default=60.0)
    parser.add_argument("--blue-late-pre-firing-max-distance-m", type=float, default=2500.0)
    parser.add_argument("--blue-firing-line-micro", action="store_true")
    parser.add_argument("--blue-firing-line-enter-ata-deg", type=float, default=5.0)
    parser.add_argument("--blue-firing-line-az-ref-deg", type=float, default=2.0)
    parser.add_argument("--blue-firing-line-az-rate-gain", type=float, default=0.05)
    parser.add_argument("--blue-firing-line-pitch-gain", type=float, default=0.30)
    parser.add_argument("--blue-firing-line-pitch-rate-gain", type=float, default=0.04)
    parser.add_argument("--blue-firing-line-rate-gate-ata-deg", type=float, default=0.0)
    parser.add_argument("--blue-firing-line-roll-arrest", action="store_true")
    parser.add_argument("--blue-firing-line-roll-arrest-trigger-ata-deg", type=float, default=1.2)
    parser.add_argument("--blue-firing-line-roll-arrest-duration-s", type=float, default=0.35)
    parser.add_argument("--blue-firing-line-roll-arrest-gain", type=float, default=0.02)
    parser.add_argument("--blue-firing-line-roll-cmd", type=float)
    parser.add_argument("--blue-firing-line-pitch-cmd", type=float)
    parser.add_argument("--blue-firing-line-pitch-lead-enter-ata-deg", type=float, default=0.0)
    parser.add_argument("--blue-post-defense-pitch-lead-enter-ata-deg", type=float, default=0.0)
    parser.add_argument("--blue-firing-line-pitch-lead-cmd", type=float)
    parser.add_argument("--blue-attack-throttle-cut-speed-mps", type=float, default=0.0)
    parser.add_argument("--blue-threat-rudder-jink", action="store_true")
    parser.add_argument("--blue-threat-jink-max-distance-m", type=float, default=1300.0)
    parser.add_argument("--blue-threat-jink-max-target-ata-deg", type=float, default=3.0)
    parser.add_argument("--blue-threat-jink-min-own-ata-deg", type=float, default=20.0)
    parser.add_argument("--blue-threat-jink-direction", type=float, default=-1.0)
    parser.add_argument("--blue-threat-jink-roll-cmd", type=float, default=None)
    parser.add_argument("--blue-threat-jink-bank-deg", type=float, default=0.0)
    parser.add_argument("--blue-threat-jink-pitch-cmd", type=float)
    parser.add_argument("--blue-threat-jink-rudder-scale", type=float, default=1.0)
    parser.add_argument("--blue-threat-jink-pulse-s", type=float, default=0.0)
    parser.add_argument("--blue-threat-jink-cooldown-s", type=float, default=12.0)
    parser.add_argument("--blue-threat-jink-alternate-pulses", action="store_true")
    parser.add_argument("--blue-threat-jink-throttle-cut-speed-mps", type=float, default=0.0)
    parser.add_argument("--blue-tail-pursuit-brake", action="store_true")
    parser.add_argument("--blue-tail-pursuit-max-distance-m", type=float, default=3000.0)
    parser.add_argument("--blue-tail-pursuit-max-own-ata-deg", type=float, default=90.0)
    parser.add_argument("--blue-tail-pursuit-min-target-ata-deg", type=float, default=80.0)
    parser.add_argument("--blue-tail-pursuit-cut-speed-mps", type=float, default=220.0)
    parser.add_argument("--lag-track", action="store_true")
    parser.add_argument("--teacher-knn-dataset")
    parser.add_argument("--teacher-knn-neighbors", type=int, default=6)
    parser.add_argument("--teacher-knn-distance-power", type=float, default=2.0)
    parser.add_argument(
        "--lag-actor-module",
        default=r"external/LAG/envs/JSBSim/model/baseline_actor.py",
    )
    parser.add_argument(
        "--lag-model-path",
        default=r"external/LAG/envs/JSBSim/model/baseline_model.pt",
    )
    args = parser.parse_args()

    if args.teacher_knn_dataset:
        primary = TeacherKNNActionProvider(
            args.teacher_knn_dataset,
            neighbors=args.teacher_knn_neighbors,
            distance_power=args.teacher_knn_distance_power,
            debug=args.debug,
        )
    elif args.lag_track:
        primary = LAGTrackActionProvider(
            actor_module_path=args.lag_actor_module,
            model_path=args.lag_model_path,
            debug=args.debug,
        )
    else:
        candidate = ROOT / "artifacts/candidates/reference_bt_reconstruction"
        host_name = "bt32_host_forceaware.exe" if args.force_aware_host else "bt32_host.exe"
        primary = BT32ActionProvider(
            candidate / host_name,
            candidate / "AIP_DCS.dll",
            candidate,
            candidate / "debug_crt_x86",
            debug=args.debug,
            recovery_enter_altitude_m=0.0 if args.pure_bt else 500.0,
            recovery_exit_altitude_m=0.0 if args.pure_bt else 1000.0,
        )
    if args.pure_bt and not args.lag_track:
        provider = primary
    elif args.minimal_gun_opportunity or args.lag_track or args.teacher_knn_dataset:
        provider = BTGunOpportunityProvider(
            primary,
            debug=args.debug,
            acquire_max_ata_deg=args.gun_acquire_ata_deg,
            acquire_max_distance_m=args.gun_acquire_max_distance_m,
            energy_defense=args.energy_defense,
            terrain_drag_defense=args.terrain_drag_defense,
            mirror_red_opening_s=args.mirror_red_opening_s,
            mirror_red_commands=args.mirror_red_commands,
            canonicalize_red_input=args.canonicalize_red_input,
            reference_rudder_scale=args.reference_rudder_scale,
            red_reference_rudder_scale=args.red_reference_rudder_scale,
            blue_post_attack_reference_rudder_scale=args.blue_post_attack_reference_rudder_scale,
            gun_hold_enter_ata_deg=args.gun_hold_enter_ata_deg,
            gun_hold_exit_ata_deg=args.gun_hold_exit_ata_deg,
            gun_hold_max_distance_m=args.gun_hold_max_distance_m,
            gun_hold_rate_damping=args.gun_hold_rate_damping,
            attack_min_target_ata_deg=args.attack_min_target_ata_deg,
            attack_max_closure_mps=args.attack_max_closure_mps,
            diverse_energy_recovery=args.diverse_energy_recovery,
            energy_recovery_bank_deg=args.energy_recovery_bank_deg,
            energy_recovery_enter_speed_mps=args.energy_recovery_enter_speed_mps,
            energy_recovery_exit_speed_mps=args.energy_recovery_exit_speed_mps,
            force_energy_recovery_under_threat=args.force_energy_recovery_under_threat,
            early_threat_unload=args.early_threat_unload,
            tactical_defense=args.tactical_defense,
            tactical_defense_duration_s=args.tactical_defense_duration_s,
            tactical_energy_recovery=args.tactical_energy_recovery,
            firing_line_micro=args.firing_line_micro,
            firing_line_enter_ata_deg=args.firing_line_enter_ata_deg,
            firing_line_max_distance_m=args.firing_line_max_distance_m,
            firing_line_rudder_gain=args.firing_line_rudder_gain,
            firing_line_rudder_limit=args.firing_line_rudder_limit,
            pre_firing_assist=args.pre_firing_assist,
            pre_firing_pitch_gain=args.pre_firing_pitch_gain,
            pre_firing_rudder_gain=args.pre_firing_rudder_gain,
            red_firing_line_rudder_only=args.red_firing_line_rudder_only,
            red_firing_line_az_ref_deg=args.red_firing_line_az_ref_deg,
            blue_pre_firing_rudder_only=args.blue_pre_firing_rudder_only,
            blue_pre_firing_enter_ata_deg=args.blue_pre_firing_enter_ata_deg,
            blue_pre_firing_max_distance_m=args.blue_pre_firing_max_distance_m,
            blue_pre_firing_az_ref_deg=args.blue_pre_firing_az_ref_deg,
            blue_pre_firing_az_rate_gain=args.blue_pre_firing_az_rate_gain,
            blue_late_pre_firing_start_s=args.blue_late_pre_firing_start_s,
            blue_late_pre_firing_enter_ata_deg=args.blue_late_pre_firing_enter_ata_deg,
            blue_late_pre_firing_max_distance_m=args.blue_late_pre_firing_max_distance_m,
            blue_firing_line_micro=args.blue_firing_line_micro,
            blue_firing_line_enter_ata_deg=args.blue_firing_line_enter_ata_deg,
            blue_firing_line_az_ref_deg=args.blue_firing_line_az_ref_deg,
            blue_firing_line_az_rate_gain=args.blue_firing_line_az_rate_gain,
            blue_firing_line_pitch_gain=args.blue_firing_line_pitch_gain,
            blue_firing_line_pitch_rate_gain=args.blue_firing_line_pitch_rate_gain,
            blue_firing_line_rate_gate_ata_deg=args.blue_firing_line_rate_gate_ata_deg,
            blue_firing_line_roll_arrest=args.blue_firing_line_roll_arrest,
            blue_firing_line_roll_arrest_trigger_ata_deg=args.blue_firing_line_roll_arrest_trigger_ata_deg,
            blue_firing_line_roll_arrest_duration_s=args.blue_firing_line_roll_arrest_duration_s,
            blue_firing_line_roll_arrest_gain=args.blue_firing_line_roll_arrest_gain,
            blue_firing_line_roll_cmd=args.blue_firing_line_roll_cmd,
            blue_firing_line_pitch_cmd=args.blue_firing_line_pitch_cmd,
            blue_firing_line_pitch_lead_enter_ata_deg=args.blue_firing_line_pitch_lead_enter_ata_deg,
            blue_post_defense_pitch_lead_enter_ata_deg=args.blue_post_defense_pitch_lead_enter_ata_deg,
            blue_firing_line_pitch_lead_cmd=args.blue_firing_line_pitch_lead_cmd,
            blue_attack_throttle_cut_speed_mps=args.blue_attack_throttle_cut_speed_mps,
            blue_threat_rudder_jink=args.blue_threat_rudder_jink,
            blue_threat_jink_max_distance_m=args.blue_threat_jink_max_distance_m,
            blue_threat_jink_max_target_ata_deg=args.blue_threat_jink_max_target_ata_deg,
            blue_threat_jink_min_own_ata_deg=args.blue_threat_jink_min_own_ata_deg,
            blue_threat_jink_direction=args.blue_threat_jink_direction,
            blue_threat_jink_roll_cmd=args.blue_threat_jink_roll_cmd,
            blue_threat_jink_bank_deg=args.blue_threat_jink_bank_deg,
            blue_threat_jink_pitch_cmd=args.blue_threat_jink_pitch_cmd,
            blue_threat_jink_rudder_scale=args.blue_threat_jink_rudder_scale,
            blue_threat_jink_pulse_s=args.blue_threat_jink_pulse_s,
            blue_threat_jink_cooldown_s=args.blue_threat_jink_cooldown_s,
            blue_threat_jink_alternate_pulses=args.blue_threat_jink_alternate_pulses,
            blue_threat_jink_throttle_cut_speed_mps=args.blue_threat_jink_throttle_cut_speed_mps,
            blue_tail_pursuit_brake=args.blue_tail_pursuit_brake,
            blue_tail_pursuit_max_distance_m=args.blue_tail_pursuit_max_distance_m,
            blue_tail_pursuit_max_own_ata_deg=args.blue_tail_pursuit_max_own_ata_deg,
            blue_tail_pursuit_min_target_ata_deg=args.blue_tail_pursuit_min_target_ata_deg,
            blue_tail_pursuit_cut_speed_mps=args.blue_tail_pursuit_cut_speed_mps,
        )
    else:
        provider = BTTailPrecisionProvider(
            primary,
            debug=args.debug,
            precision_enter_ata_deg=args.precision_enter_ata_deg,
            merge_yoyo=args.merge_yoyo,
            merge_yoyo_duration_s=args.merge_yoyo_duration_s,
            merge_brake_s=args.merge_brake_s,
            opportunity_gun=args.opportunity_gun,
            opportunity_gun_brake=args.opportunity_gun_brake,
            direct_los_assist=args.direct_los_assist,
        )
    observation = load_observation_hook("student.codex_tactical26")
    policy_class = ProviderCommandPolicy if args.sync_policy else AsyncProviderCommandPolicy
    policy_kwargs = dict(
        action_provider=provider,
        observation_mode=observation["mode"],
        observation_fn=observation["build_observation"],
        ownship_force_side=1,
        target_force_side=2,
        action_repeat=args.action_repeat,
        server_z_up=not args.legacy_server_z,
    )
    if args.sync_policy:
        policy_kwargs["dynamic_force_sides"] = args.dynamic_force_sides
        policy_kwargs["swap_blue_force_sides"] = args.swap_blue_force_sides
    elif args.dynamic_force_sides:
        parser.error("--dynamic-force-sides requires --sync-policy")
    elif args.swap_blue_force_sides:
        parser.error("--swap-blue-force-sides requires --sync-policy")
    if args.swap_blue_force_sides and not args.dynamic_force_sides:
        parser.error("--swap-blue-force-sides requires --dynamic-force-sides")
    policy = policy_class(**policy_kwargs)
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
    print(f"[bt-tail-precision] server={args.server_ip}:{args.server_port}", flush=True)
    try:
        client.run()
    finally:
        if hasattr(policy, "close"):
            policy.close()
        provider.close()


if __name__ == "__main__":
    main()
