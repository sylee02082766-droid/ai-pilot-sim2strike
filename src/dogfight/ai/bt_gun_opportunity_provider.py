from __future__ import annotations

import math

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.ai.native_bt import AIPilot
from dogfight.sim.state_schema import StateIndex


def _nose(state: np.ndarray) -> np.ndarray:
    pitch = math.radians(float(state[StateIndex.PITCH]))
    yaw = math.radians(float(state[StateIndex.YAW]))
    return np.asarray([
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        -math.sin(pitch),
    ])


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1.0e-6)
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


class BTGunOpportunityProvider(ActionProvider):
    """Preserve the reference BT and stabilize only a real gun opportunity."""

    def __init__(
        self,
        primary: ActionProvider,
        debug: bool = False,
        acquire_max_ata_deg: float = 25.0,
        acquire_max_distance_m: float = 1500.0,
        energy_defense: bool = False,
        terrain_drag_defense: bool = False,
        mirror_red_opening_s: float = 0.0,
        mirror_red_commands: bool = False,
        canonicalize_red_input: bool = False,
        reference_rudder_scale: float = 1.0,
        red_reference_rudder_scale: float | None = None,
        blue_post_attack_reference_rudder_scale: float | None = None,
        gun_hold_enter_ata_deg: float = 16.0,
        gun_hold_exit_ata_deg: float = 25.0,
        gun_hold_max_distance_m: float = 1050.0,
        gun_hold_rate_damping: float = 0.0,
        attack_min_target_ata_deg: float = 0.0,
        attack_max_closure_mps: float = 10000.0,
        red_attack_min_target_ata_deg: float | None = None,
        red_attack_max_closure_mps: float | None = None,
        diverse_energy_recovery: bool = False,
        energy_recovery_bank_deg: float = 25.0,
        energy_recovery_enter_speed_mps: float = 165.0,
        energy_recovery_exit_speed_mps: float = 205.0,
        force_energy_recovery_under_threat: bool = False,
        early_threat_unload: bool = False,
        tactical_defense: bool = False,
        tactical_defense_duration_s: float = 3.2,
        tactical_energy_recovery: bool = False,
        tactical_energy_recovery_blue_only: bool = False,
        firing_line_micro: bool = False,
        firing_line_enter_ata_deg: float = 3.5,
        firing_line_max_distance_m: float = 950.0,
        firing_line_rudder_gain: float = 0.20,
        firing_line_rudder_limit: float = 0.30,
        pre_firing_assist: bool = False,
        pre_firing_pitch_gain: float = 0.25,
        pre_firing_rudder_gain: float = 0.50,
        red_firing_line_rudder_only: bool = False,
        red_firing_line_az_ref_deg: float = 0.8,
        red_firing_line_enter_ata_deg: float = 3.5,
        red_firing_line_exit_ata_deg: float = 3.5,
        red_firing_line_max_distance_m: float = 950.0,
        red_firing_line_rudder_limit: float = 1.0,
        red_firing_line_az_rate_gain: float = 0.0,
        red_firing_line_pitch_gain: float = 0.0,
        red_firing_line_pitch_rate_gain: float = 0.0,
        red_firing_line_roll_rate_gain: float = 0.0,
        red_firing_line_throttle_cut_closure_mps: float = 0.0,
        blue_pre_firing_rudder_only: bool = False,
        blue_pre_firing_enter_ata_deg: float = 25.0,
        blue_pre_firing_max_distance_m: float = 1500.0,
        blue_pre_firing_az_ref_deg: float = 4.0,
        blue_pre_firing_az_rate_gain: float = 0.0,
        blue_late_pre_firing_start_s: float = 0.0,
        blue_late_pre_firing_enter_ata_deg: float = 60.0,
        blue_late_pre_firing_max_distance_m: float = 2500.0,
        blue_firing_line_micro: bool = False,
        blue_firing_line_enter_ata_deg: float = 5.0,
        blue_firing_line_az_ref_deg: float = 2.0,
        blue_firing_line_az_rate_gain: float = 0.05,
        blue_firing_line_pitch_gain: float = 0.30,
        blue_firing_line_pitch_rate_gain: float = 0.04,
        blue_firing_line_rate_gate_ata_deg: float = 0.0,
        blue_firing_line_roll_arrest: bool = False,
        blue_firing_line_roll_arrest_trigger_ata_deg: float = 1.2,
        blue_firing_line_roll_arrest_duration_s: float = 0.35,
        blue_firing_line_roll_arrest_gain: float = 0.02,
        blue_firing_line_roll_cmd: float | None = None,
        blue_firing_line_pitch_cmd: float | None = None,
        blue_firing_line_pitch_lead_enter_ata_deg: float = 0.0,
        blue_post_defense_pitch_lead_enter_ata_deg: float = 0.0,
        blue_firing_line_pitch_lead_cmd: float | None = None,
        blue_attack_throttle_cut_speed_mps: float = 0.0,
        blue_threat_rudder_jink: bool = False,
        blue_threat_jink_max_distance_m: float = 1300.0,
        blue_threat_jink_max_target_ata_deg: float = 3.0,
        blue_threat_jink_min_own_ata_deg: float = 20.0,
        blue_threat_jink_direction: float = -1.0,
        blue_threat_jink_roll_cmd: float | None = None,
        blue_threat_jink_bank_deg: float = 0.0,
        blue_threat_jink_pitch_cmd: float | None = None,
        blue_threat_jink_rudder_scale: float = 1.0,
        blue_threat_jink_pulse_s: float = 0.0,
        blue_threat_jink_cooldown_s: float = 12.0,
        blue_threat_jink_alternate_pulses: bool = False,
        blue_threat_jink_throttle_cut_speed_mps: float = 0.0,
        blue_tail_pursuit_brake: bool = False,
        blue_tail_pursuit_max_distance_m: float = 3000.0,
        blue_tail_pursuit_max_own_ata_deg: float = 90.0,
        blue_tail_pursuit_min_target_ata_deg: float = 80.0,
        blue_tail_pursuit_cut_speed_mps: float = 220.0,
    ) -> None:
        self.primary = primary
        self.debug = bool(debug)
        self.acquire_max_ata_deg = float(acquire_max_ata_deg)
        self.acquire_max_distance_m = float(np.clip(
            acquire_max_distance_m, 914.4, 3000.0
        ))
        self.energy_defense = bool(energy_defense)
        self.terrain_drag_defense = bool(terrain_drag_defense)
        self.mirror_red_opening_s = max(0.0, float(mirror_red_opening_s))
        self.mirror_red_commands = bool(mirror_red_commands)
        self.canonicalize_red_input = bool(canonicalize_red_input)
        self.reference_rudder_scale = float(np.clip(reference_rudder_scale, 0.0, 1.0))
        self.red_reference_rudder_scale = (
            self.reference_rudder_scale
            if red_reference_rudder_scale is None
            else float(np.clip(red_reference_rudder_scale, 0.0, 1.0))
        )
        self.blue_post_attack_reference_rudder_scale = (
            None
            if blue_post_attack_reference_rudder_scale is None
            else float(np.clip(blue_post_attack_reference_rudder_scale, 0.0, 1.0))
        )
        self.gun_hold_enter_ata_deg = max(0.0, float(gun_hold_enter_ata_deg))
        self.gun_hold_exit_ata_deg = max(
            self.gun_hold_enter_ata_deg,
            float(gun_hold_exit_ata_deg),
        )
        self.gun_hold_max_distance_m = float(np.clip(
            gun_hold_max_distance_m, 152.4, 1250.0
        ))
        self.gun_hold_rate_damping = float(np.clip(gun_hold_rate_damping, 0.0, 1.0))
        self.attack_min_target_ata_deg = float(np.clip(
            attack_min_target_ata_deg, 0.0, 180.0
        ))
        self.attack_max_closure_mps = float(attack_max_closure_mps)
        self.red_attack_min_target_ata_deg = (
            None
            if red_attack_min_target_ata_deg is None
            else float(np.clip(red_attack_min_target_ata_deg, 0.0, 180.0))
        )
        self.red_attack_max_closure_mps = (
            None
            if red_attack_max_closure_mps is None
            else float(red_attack_max_closure_mps)
        )
        self.diverse_energy_recovery = bool(diverse_energy_recovery)
        self.energy_recovery_bank_deg = float(np.clip(
            abs(energy_recovery_bank_deg), 0.0, 45.0
        ))
        self.energy_recovery_enter_speed_mps = max(
            100.0, float(energy_recovery_enter_speed_mps)
        )
        self.energy_recovery_exit_speed_mps = max(
            self.energy_recovery_enter_speed_mps + 10.0,
            float(energy_recovery_exit_speed_mps),
        )
        self.force_energy_recovery_under_threat = bool(
            force_energy_recovery_under_threat
        )
        self.early_threat_unload = bool(early_threat_unload)
        self.tactical_defense = bool(tactical_defense)
        self.tactical_defense_duration_s = float(np.clip(
            tactical_defense_duration_s, 1.0, 8.0
        ))
        self.tactical_energy_recovery = bool(tactical_energy_recovery)
        self.tactical_energy_recovery_blue_only = bool(
            tactical_energy_recovery_blue_only
        )
        self.firing_line_micro = bool(firing_line_micro)
        self.firing_line_enter_ata_deg = float(np.clip(
            firing_line_enter_ata_deg, 0.5, 15.0
        ))
        self.firing_line_max_distance_m = float(np.clip(
            firing_line_max_distance_m, 914.4, 1500.0
        ))
        self.firing_line_rudder_gain = float(np.clip(
            firing_line_rudder_gain, 0.0, 1.0
        ))
        self.firing_line_rudder_limit = float(np.clip(
            firing_line_rudder_limit, 0.1, 0.7
        ))
        self.pre_firing_assist = bool(pre_firing_assist)
        self.pre_firing_pitch_gain = float(np.clip(
            pre_firing_pitch_gain, 0.0, 0.6
        ))
        self.pre_firing_rudder_gain = float(np.clip(
            pre_firing_rudder_gain, 0.0, 1.0
        ))
        self.red_firing_line_rudder_only = bool(red_firing_line_rudder_only)
        self.red_firing_line_az_ref_deg = float(np.clip(
            red_firing_line_az_ref_deg, 0.25, 10.0
        ))
        self.red_firing_line_enter_ata_deg = float(np.clip(
            red_firing_line_enter_ata_deg, 0.5, 10.0
        ))
        self.red_firing_line_exit_ata_deg = float(np.clip(
            red_firing_line_exit_ata_deg,
            self.red_firing_line_enter_ata_deg,
            25.0,
        ))
        self.red_firing_line_max_distance_m = float(np.clip(
            red_firing_line_max_distance_m, 914.4, 1250.0
        ))
        self.red_firing_line_rudder_limit = float(np.clip(
            red_firing_line_rudder_limit, 0.1, 1.0
        ))
        self.red_firing_line_az_rate_gain = float(np.clip(
            red_firing_line_az_rate_gain, 0.0, 1.0
        ))
        self.red_firing_line_pitch_gain = float(np.clip(
            red_firing_line_pitch_gain, 0.0, 1.0
        ))
        self.red_firing_line_pitch_rate_gain = float(np.clip(
            red_firing_line_pitch_rate_gain, 0.0, 0.2
        ))
        self.red_firing_line_roll_rate_gain = float(np.clip(
            red_firing_line_roll_rate_gain, 0.0, 0.1
        ))
        self.red_firing_line_throttle_cut_closure_mps = max(
            0.0, float(red_firing_line_throttle_cut_closure_mps)
        )
        self.blue_pre_firing_rudder_only = bool(blue_pre_firing_rudder_only)
        self.blue_pre_firing_enter_ata_deg = float(np.clip(
            blue_pre_firing_enter_ata_deg, 1.0, 45.0
        ))
        self.blue_pre_firing_max_distance_m = float(np.clip(
            blue_pre_firing_max_distance_m, 500.0, 2500.0
        ))
        self.blue_pre_firing_az_ref_deg = float(np.clip(
            blue_pre_firing_az_ref_deg, 0.25, 15.0
        ))
        self.blue_pre_firing_az_rate_gain = float(np.clip(
            blue_pre_firing_az_rate_gain, 0.0, 0.2
        ))
        self.blue_late_pre_firing_start_s = max(
            0.0, float(blue_late_pre_firing_start_s)
        )
        self.blue_late_pre_firing_enter_ata_deg = float(np.clip(
            blue_late_pre_firing_enter_ata_deg, 1.0, 120.0
        ))
        self.blue_late_pre_firing_max_distance_m = float(np.clip(
            blue_late_pre_firing_max_distance_m, 500.0, 3000.0
        ))
        self.blue_firing_line_micro = bool(blue_firing_line_micro)
        self.blue_firing_line_enter_ata_deg = float(np.clip(
            blue_firing_line_enter_ata_deg, 1.0, 10.0
        ))
        self.blue_firing_line_az_ref_deg = float(np.clip(
            blue_firing_line_az_ref_deg, 0.25, 8.0
        ))
        self.blue_firing_line_az_rate_gain = float(np.clip(
            blue_firing_line_az_rate_gain, 0.0, 0.2
        ))
        self.blue_firing_line_pitch_gain = float(np.clip(
            blue_firing_line_pitch_gain, 0.0, 1.0
        ))
        self.blue_firing_line_pitch_rate_gain = float(np.clip(
            blue_firing_line_pitch_rate_gain, 0.0, 0.2
        ))
        self.blue_firing_line_rate_gate_ata_deg = float(np.clip(
            blue_firing_line_rate_gate_ata_deg, 0.0, 10.0
        ))
        self.blue_firing_line_roll_arrest = bool(blue_firing_line_roll_arrest)
        self.blue_firing_line_roll_arrest_trigger_ata_deg = float(np.clip(
            blue_firing_line_roll_arrest_trigger_ata_deg, 0.1, 5.0
        ))
        self.blue_firing_line_roll_arrest_duration_s = float(np.clip(
            blue_firing_line_roll_arrest_duration_s, 0.05, 1.0
        ))
        self.blue_firing_line_roll_arrest_gain = float(np.clip(
            blue_firing_line_roll_arrest_gain, 0.001, 0.1
        ))
        self.blue_firing_line_roll_cmd = (
            None
            if blue_firing_line_roll_cmd is None
            else float(np.clip(blue_firing_line_roll_cmd, -1.0, 1.0))
        )
        self.blue_firing_line_pitch_cmd = (
            None
            if blue_firing_line_pitch_cmd is None
            else float(np.clip(blue_firing_line_pitch_cmd, -1.0, 1.0))
        )
        self.blue_firing_line_pitch_lead_enter_ata_deg = float(np.clip(
            blue_firing_line_pitch_lead_enter_ata_deg, 0.0, 20.0
        ))
        self.blue_post_defense_pitch_lead_enter_ata_deg = float(np.clip(
            blue_post_defense_pitch_lead_enter_ata_deg, 0.0, 20.0
        ))
        self.blue_firing_line_pitch_lead_cmd = (
            None
            if blue_firing_line_pitch_lead_cmd is None
            else float(np.clip(blue_firing_line_pitch_lead_cmd, -1.0, 1.0))
        )
        self.blue_attack_throttle_cut_speed_mps = (
            0.0
            if float(blue_attack_throttle_cut_speed_mps) <= 0.0
            else float(np.clip(blue_attack_throttle_cut_speed_mps, 150.0, 320.0))
        )
        self.blue_threat_rudder_jink = bool(blue_threat_rudder_jink)
        self.blue_threat_jink_max_distance_m = float(np.clip(
            blue_threat_jink_max_distance_m, 500.0, 3000.0
        ))
        self.blue_threat_jink_max_target_ata_deg = float(np.clip(
            blue_threat_jink_max_target_ata_deg, 0.5, 45.0
        ))
        self.blue_threat_jink_min_own_ata_deg = float(np.clip(
            blue_threat_jink_min_own_ata_deg, 0.0, 90.0
        ))
        self.blue_threat_jink_direction = (
            -1.0 if float(blue_threat_jink_direction) < 0.0 else 1.0
        )
        self.blue_threat_jink_roll_cmd = (
            None
            if blue_threat_jink_roll_cmd is None
            else float(np.clip(blue_threat_jink_roll_cmd, -1.0, 1.0))
        )
        self.blue_threat_jink_bank_deg = float(np.clip(
            abs(blue_threat_jink_bank_deg), 0.0, 85.0
        ))
        self.blue_threat_jink_pitch_cmd = (
            None
            if blue_threat_jink_pitch_cmd is None
            else float(np.clip(blue_threat_jink_pitch_cmd, -1.0, 1.0))
        )
        self.blue_threat_jink_rudder_scale = float(np.clip(
            abs(blue_threat_jink_rudder_scale), 0.0, 1.0
        ))
        self.blue_threat_jink_pulse_s = float(np.clip(
            blue_threat_jink_pulse_s, 0.0, 6.0
        ))
        self.blue_threat_jink_cooldown_s = float(np.clip(
            blue_threat_jink_cooldown_s, 0.0, 60.0
        ))
        self.blue_threat_jink_alternate_pulses = bool(
            blue_threat_jink_alternate_pulses
        )
        self.blue_threat_jink_throttle_cut_speed_mps = (
            0.0
            if float(blue_threat_jink_throttle_cut_speed_mps) <= 0.0
            else float(np.clip(blue_threat_jink_throttle_cut_speed_mps, 150.0, 320.0))
        )
        self.blue_tail_pursuit_brake = bool(blue_tail_pursuit_brake)
        self.blue_tail_pursuit_max_distance_m = float(np.clip(
            blue_tail_pursuit_max_distance_m, 500.0, 4000.0
        ))
        self.blue_tail_pursuit_max_own_ata_deg = float(np.clip(
            blue_tail_pursuit_max_own_ata_deg, 5.0, 150.0
        ))
        self.blue_tail_pursuit_min_target_ata_deg = float(np.clip(
            blue_tail_pursuit_min_target_ata_deg, 20.0, 175.0
        ))
        self.blue_tail_pursuit_cut_speed_mps = float(np.clip(
            blue_tail_pursuit_cut_speed_mps, 150.0, 320.0
        ))
        self.reset()

    def reset(self, context: ActionContext | None = None) -> None:
        self._gun_hold = False
        self._red_firing_line_latched = False
        self._previous_roll_deg: float | None = None
        self._previous_time_s: float | None = None
        self._previous_altitude_m: float | None = None
        self._previous_distance_m: float | None = None
        self._filtered_closure_mps = 0.0
        self._start_time_s: float | None = None
        self._next_status_s = 0.0
        self._last_stage: str | None = None
        self._defense_start_s = -1.0
        self._defense_cooldown_until_s = -1.0
        self._defense_sign = 1.0
        self._defense_attempts = 0
        self._energy_recovering = False
        self._energy_turn_sign = -1.0
        self._last_energy_turn_sign = 1.0
        self._threat_unload_start_s = -1.0
        self._threat_unload_turn_sign = 1.0
        self._tactical_defense_start_s = -1.0
        self._tactical_defense_cooldown_until_s = -1.0
        self._tactical_defense_turn_sign = 1.0
        self._tactical_energy_start_s = -1.0
        self._tactical_energy_turn_sign = 1.0
        self._blue_threat_jink_start_s = -1.0
        self._blue_threat_jink_cooldown_until_s = -1.0
        self._blue_threat_jink_los_side = 1.0
        self._blue_threat_jink_last_end_s = -1.0
        self._blue_threat_jink_ever_started = False
        self._blue_firing_line_ever_started = False
        self._blue_firing_line_roll_arrest_start_s = -1.0
        self._blue_firing_line_roll_arrest_armed = True
        self._terrain_overshoot = False
        self._terrain_overshoot_time_s = -1.0
        self._opening_start_s: float | None = None
        self._mirror_opening = False
        self._canonical_center_ne: tuple[float, float] | None = None
        self.primary.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        own = np.asarray(context.ownship_state, dtype=np.float64)
        target = np.asarray(context.target_state, dtype=np.float64)
        primary_context = context
        if (
            self.canonicalize_red_input
            and int(context.info.get("my_plane_id", -1)) == 1
        ):
            primary_context = self._canonicalize_red_context(context, own, target)
        result = self.primary.compute_action(primary_context)
        primary_stage = str(
            result.info.get("bt32_stage", result.info.get("lag_stage", result.source))
        )

        observation = np.asarray(context.observation, dtype=np.float32).reshape(-1)
        relative = target[:3] - own[:3]
        distance_m = float(np.linalg.norm(relative))
        own_ata_deg = _angle(_nose(own), relative)
        target_ata_deg = _angle(_nose(target), -relative)
        sim_time_s = float(own[StateIndex.SIM_TIME])
        roll_deg = float(own[StateIndex.ROLL])
        speed_mps = float(np.linalg.norm(own[6:9]))
        pitch_deg = float(own[StateIndex.PITCH])
        altitude_m = float(own[StateIndex.ALT])
        if self._opening_start_s is None:
            self._opening_start_s = sim_time_s
            self._mirror_opening = float(own[StateIndex.YAW]) < 0.0
        opening_elapsed_s = sim_time_s - self._opening_start_s
        opening_active = (
            self._mirror_opening
            and opening_elapsed_s < self.mirror_red_opening_s
        )

        roll_rate_dps = 0.0
        altitude_rate_mps = 0.0
        closure_mps = self._filtered_closure_mps
        if self._previous_roll_deg is not None and self._previous_time_s is not None:
            dt_s = sim_time_s - self._previous_time_s
            if 1.0e-4 <= dt_s <= 0.25:
                roll_rate_dps = _wrap_deg(roll_deg - self._previous_roll_deg) / dt_s
                if self._previous_altitude_m is not None:
                    altitude_rate_mps = (altitude_m - self._previous_altitude_m) / dt_s
                if self._previous_distance_m is not None:
                    raw_closure_mps = (self._previous_distance_m - distance_m) / dt_s
                    self._filtered_closure_mps = (
                        0.85 * self._filtered_closure_mps + 0.15 * raw_closure_mps
                    )
                    closure_mps = self._filtered_closure_mps
        self._previous_roll_deg = roll_deg
        self._previous_time_s = sim_time_s
        self._previous_altitude_m = altitude_m
        self._previous_distance_m = distance_m

        if primary_stage.startswith("deck_recovery"):
            self._gun_hold = False
            self._red_firing_line_latched = False
            self._defense_start_s = -1.0
            self._threat_unload_start_s = -1.0
            self._tactical_defense_start_s = -1.0
            self._tactical_energy_start_s = -1.0
            return result

        los_az_deg = float(observation[11]) * 180.0 if observation.size >= 13 else 0.0
        tactical_threat = (
            self.tactical_defense
            and 400.0 <= distance_m <= 1800.0
            and closure_mps <= 220.0
            and own_ata_deg >= 70.0
            and target_ata_deg <= 20.0
            and own_ata_deg >= target_ata_deg + 50.0
            and altitude_m >= 950.0
            and speed_mps >= 235.0
        )
        if (
            tactical_threat
            and self._tactical_defense_start_s < 0.0
            and self._tactical_energy_start_s < 0.0
            and sim_time_s >= self._tactical_defense_cooldown_until_s
        ):
            if abs(los_az_deg) >= 15.0:
                self._tactical_defense_turn_sign = 1.0 if los_az_deg > 0.0 else -1.0
            elif abs(roll_deg) >= 20.0:
                self._tactical_defense_turn_sign = -1.0 if roll_deg > 0.0 else 1.0
            else:
                self._tactical_defense_turn_sign *= -1.0
            self._tactical_defense_start_s = sim_time_s

        tactical_defense_elapsed_s = (
            sim_time_s - self._tactical_defense_start_s
            if self._tactical_defense_start_s >= 0.0
            else 0.0
        )
        tactical_defense_active = (
            self._tactical_defense_start_s >= 0.0
            and tactical_defense_elapsed_s < self.tactical_defense_duration_s
            and altitude_m >= 850.0
            and speed_mps >= 215.0
            and (tactical_defense_elapsed_s < 0.8 or target_ata_deg <= 55.0)
        )
        if self._tactical_defense_start_s >= 0.0 and not tactical_defense_active:
            self._tactical_defense_start_s = -1.0
            self._tactical_defense_cooldown_until_s = sim_time_s + 15.0

        low_energy_under_threat = (
            self.tactical_energy_recovery
            and (
                not self.tactical_energy_recovery_blue_only
                or int(context.info.get("my_plane_id", -1)) == 0
            )
            and not tactical_defense_active
            and self._tactical_energy_start_s < 0.0
            and 300.0 <= distance_m <= 1800.0
            and own_ata_deg >= 70.0
            and target_ata_deg <= 30.0
            and speed_mps < 225.0
            and altitude_m >= 1300.0
        )
        if low_energy_under_threat:
            if abs(los_az_deg) >= 12.0:
                self._tactical_energy_turn_sign = 1.0 if los_az_deg > 0.0 else -1.0
            elif abs(roll_deg) >= 20.0:
                self._tactical_energy_turn_sign = 1.0 if roll_deg > 0.0 else -1.0
            else:
                self._tactical_energy_turn_sign *= -1.0
            self._tactical_energy_start_s = sim_time_s

        tactical_energy_elapsed_s = (
            sim_time_s - self._tactical_energy_start_s
            if self._tactical_energy_start_s >= 0.0
            else 0.0
        )
        tactical_energy_active = (
            self._tactical_energy_start_s >= 0.0
            and tactical_energy_elapsed_s < 6.0
            and altitude_m >= 1000.0
            and speed_mps < 250.0
            and (tactical_energy_elapsed_s < 0.8 or target_ata_deg <= 60.0)
        )
        if self._tactical_energy_start_s >= 0.0 and not tactical_energy_active:
            self._tactical_energy_start_s = -1.0

        threatened = (
            (self.energy_defense or self.terrain_drag_defense)
            and 250.0 <= distance_m <= 1800.0
            and (
                own_ata_deg >= 70.0
                or own_ata_deg >= target_ata_deg + 5.0
            )
            and target_ata_deg <= 25.0
            and altitude_m >= 1200.0
            and speed_mps >= 200.0
            and self._defense_attempts < (1 if self.terrain_drag_defense else 3)
        )
        if (
            threatened
            and self._defense_start_s < 0.0
            and sim_time_s >= self._defense_cooldown_until_s
        ):
            current_sign = 1.0 if roll_deg >= 0.0 else -1.0
            self._defense_sign = -current_sign
            self._defense_start_s = sim_time_s
            self._defense_attempts += 1
            self._terrain_overshoot = False
            self._terrain_overshoot_time_s = -1.0

        defense_elapsed_s = (
            sim_time_s - self._defense_start_s
            if self._defense_start_s >= 0.0
            else 0.0
        )
        defense_duration_s = 20.0 if self.terrain_drag_defense else 6.5
        defense_active = self._defense_start_s >= 0.0 and defense_elapsed_s < defense_duration_s
        if (
            self.terrain_drag_defense
            and defense_active
            and defense_elapsed_s >= 2.0
            and target_ata_deg >= 90.0
            and not self._terrain_overshoot
        ):
            self._terrain_overshoot = True
            self._terrain_overshoot_time_s = sim_time_s
        if (
            self.terrain_drag_defense
            and defense_active
            and self._terrain_overshoot
            and altitude_m >= 900.0
            and own_ata_deg <= 45.0
            and target_ata_deg >= 55.0
        ):
            defense_active = False
        if (
            not self.terrain_drag_defense
            and defense_active
            and defense_elapsed_s >= 2.0
            and distance_m >= 2400.0
        ):
            defense_active = False
        if self._defense_start_s >= 0.0 and not defense_active:
            self._defense_start_s = -1.0
            self._defense_cooldown_until_s = sim_time_s + (
                30.0 if self.terrain_drag_defense else 2.0
            )

        threat_unload_requested = (
            self.early_threat_unload
            and not defense_active
            and 700.0 <= distance_m <= 1800.0
            and own_ata_deg >= 90.0
            and target_ata_deg <= 45.0
            and pitch_deg >= 30.0
            and speed_mps <= 250.0
            and altitude_m >= 1200.0
        )
        if threat_unload_requested and self._threat_unload_start_s < 0.0:
            self._threat_unload_start_s = sim_time_s
            if abs(roll_deg) >= 25.0:
                self._threat_unload_turn_sign = 1.0 if roll_deg > 0.0 else -1.0
            else:
                los_az_deg = (
                    float(observation[11]) * 180.0 if observation.size >= 13 else 0.0
                )
                self._threat_unload_turn_sign = 1.0 if los_az_deg >= 0.0 else -1.0
        threat_unload_elapsed_s = (
            sim_time_s - self._threat_unload_start_s
            if self._threat_unload_start_s >= 0.0
            else 0.0
        )
        threat_unload_active = (
            self._threat_unload_start_s >= 0.0
            and threat_unload_elapsed_s < 4.0
            and altitude_m >= 800.0
            and distance_m <= 2200.0
        )
        if self._threat_unload_start_s >= 0.0 and not threat_unload_active:
            self._threat_unload_start_s = -1.0

        attack_min_target_ata_deg = self.attack_min_target_ata_deg
        attack_max_closure_mps = self.attack_max_closure_mps
        if int(context.info.get("my_plane_id", -1)) == 1:
            if self.red_attack_min_target_ata_deg is not None:
                attack_min_target_ata_deg = self.red_attack_min_target_ata_deg
            if self.red_attack_max_closure_mps is not None:
                attack_max_closure_mps = self.red_attack_max_closure_mps
        attack_geometry_ok = (
            target_ata_deg >= attack_min_target_ata_deg
            and closure_mps <= attack_max_closure_mps
        )
        firing_line_active = (
            self.firing_line_micro
            and 152.4 <= distance_m <= self.firing_line_max_distance_m
            and own_ata_deg <= self.firing_line_enter_ata_deg
        )
        pre_firing_active = (
            self.pre_firing_assist
            and self.firing_line_max_distance_m < distance_m <= 1250.0
            and own_ata_deg <= 8.0
        )
        red_firing_line_requested = (
            self.red_firing_line_rudder_only
            and int(context.info.get("my_plane_id", -1)) == 1
            and 152.4 <= distance_m <= self.red_firing_line_max_distance_m
            and own_ata_deg <= self.red_firing_line_enter_ata_deg
            and attack_geometry_ok
        )
        if red_firing_line_requested:
            self._red_firing_line_latched = True
        elif self._red_firing_line_latched and (
            int(context.info.get("my_plane_id", -1)) != 1
            or not 152.4 <= distance_m <= self.red_firing_line_max_distance_m
            or own_ata_deg >= self.red_firing_line_exit_ata_deg
            or not attack_geometry_ok
        ):
            self._red_firing_line_latched = False
        red_rudder_only_active = self._red_firing_line_latched
        blue_late_pre_firing_active = (
            self.blue_late_pre_firing_start_s > 0.0
            and sim_time_s >= self.blue_late_pre_firing_start_s
        )
        blue_pre_firing_enter_ata_deg = (
            self.blue_late_pre_firing_enter_ata_deg
            if blue_late_pre_firing_active
            else self.blue_pre_firing_enter_ata_deg
        )
        blue_pre_firing_max_distance_m = (
            self.blue_late_pre_firing_max_distance_m
            if blue_late_pre_firing_active
            else self.blue_pre_firing_max_distance_m
        )
        blue_rudder_only_active = (
            self.blue_pre_firing_rudder_only
            and int(context.info.get("my_plane_id", -1)) == 0
            and 152.4 <= distance_m <= blue_pre_firing_max_distance_m
            and own_ata_deg <= blue_pre_firing_enter_ata_deg
        )
        blue_firing_line_active = (
            self.blue_firing_line_micro
            and int(context.info.get("my_plane_id", -1)) == 0
            and 152.4 <= distance_m <= 950.0
            and own_ata_deg <= self.blue_firing_line_enter_ata_deg
        )
        if blue_firing_line_active:
            self._blue_firing_line_ever_started = True
        blue_firing_line_roll_arrest_requested = (
            self.blue_firing_line_roll_arrest
            and int(context.info.get("my_plane_id", -1)) == 0
            and 152.4 <= distance_m <= 950.0
            and own_ata_deg <= self.blue_firing_line_roll_arrest_trigger_ata_deg
        )
        blue_firing_line_roll_arrest_active = (
            self._blue_firing_line_roll_arrest_start_s >= 0.0
            and sim_time_s - self._blue_firing_line_roll_arrest_start_s
            <= self.blue_firing_line_roll_arrest_duration_s
        )
        if not blue_firing_line_roll_arrest_active:
            if self._blue_firing_line_roll_arrest_start_s >= 0.0:
                self._blue_firing_line_roll_arrest_start_s = -1.0
            rearm_ata_deg = max(
                self.blue_firing_line_enter_ata_deg,
                self.blue_firing_line_roll_arrest_trigger_ata_deg + 1.0,
            )
            if (
                not blue_firing_line_roll_arrest_requested
                and (
                    own_ata_deg >= rearm_ata_deg
                    or not 152.4 <= distance_m <= 950.0
                )
            ):
                self._blue_firing_line_roll_arrest_armed = True
            if (
                blue_firing_line_roll_arrest_requested
                and self._blue_firing_line_roll_arrest_armed
            ):
                self._blue_firing_line_roll_arrest_start_s = sim_time_s
                self._blue_firing_line_roll_arrest_armed = False
                blue_firing_line_roll_arrest_active = True
        blue_pitch_lead_enter_ata_deg = self.blue_firing_line_pitch_lead_enter_ata_deg
        if self._blue_threat_jink_ever_started:
            blue_pitch_lead_enter_ata_deg = max(
                blue_pitch_lead_enter_ata_deg,
                self.blue_post_defense_pitch_lead_enter_ata_deg,
            )
        blue_firing_line_pitch_lead_active = (
            self.blue_firing_line_micro
            and (
                self.blue_firing_line_pitch_lead_cmd is not None
                or self.blue_firing_line_pitch_cmd is not None
            )
            and blue_pitch_lead_enter_ata_deg > 0.0
            and int(context.info.get("my_plane_id", -1)) == 0
            and 152.4 <= distance_m <= 1250.0
            and own_ata_deg <= blue_pitch_lead_enter_ata_deg
        )
        blue_threat_jink_requested = (
            self.blue_threat_rudder_jink
            and int(context.info.get("my_plane_id", -1)) == 0
            and 300.0 <= distance_m <= self.blue_threat_jink_max_distance_m
            and target_ata_deg <= self.blue_threat_jink_max_target_ata_deg
            and own_ata_deg >= self.blue_threat_jink_min_own_ata_deg
            and altitude_m >= 700.0
            and speed_mps >= 220.0
        )
        if self.blue_threat_jink_pulse_s > 0.0:
            if (
                blue_threat_jink_requested
                and self._blue_threat_jink_start_s < 0.0
                and sim_time_s >= self._blue_threat_jink_cooldown_until_s
            ):
                self._blue_threat_jink_start_s = sim_time_s
                self._blue_threat_jink_ever_started = True
                requested_side = 1.0 if los_az_deg >= 0.0 else -1.0
                if (
                    self.blue_threat_jink_alternate_pulses
                    and self._blue_threat_jink_last_end_s >= 0.0
                    and sim_time_s - self._blue_threat_jink_last_end_s <= 0.5
                ):
                    requested_side = -self._blue_threat_jink_los_side
                self._blue_threat_jink_los_side = requested_side
            blue_threat_jink_active = (
                self._blue_threat_jink_start_s >= 0.0
                and sim_time_s - self._blue_threat_jink_start_s
                < self.blue_threat_jink_pulse_s
                and altitude_m >= 700.0
                and speed_mps >= 190.0
            )
            if (
                self._blue_threat_jink_start_s >= 0.0
                and not blue_threat_jink_active
            ):
                self._blue_threat_jink_start_s = -1.0
                self._blue_threat_jink_last_end_s = sim_time_s
                self._blue_threat_jink_cooldown_until_s = (
                    sim_time_s + self.blue_threat_jink_cooldown_s
                )
        else:
            blue_threat_jink_active = blue_threat_jink_requested
            if blue_threat_jink_active:
                self._blue_threat_jink_ever_started = True
        blue_tail_pursuit_brake_active = (
            self.blue_tail_pursuit_brake
            and self._blue_threat_jink_ever_started
            and int(context.info.get("my_plane_id", -1)) == 0
            and 300.0 <= distance_m <= self.blue_tail_pursuit_max_distance_m
            and own_ata_deg <= self.blue_tail_pursuit_max_own_ata_deg
            and target_ata_deg >= self.blue_tail_pursuit_min_target_ata_deg
            and speed_mps >= self.blue_tail_pursuit_cut_speed_mps
            and altitude_m >= 700.0
        )
        if self.acquire_max_ata_deg <= 0.0 or not attack_geometry_ok:
            self._gun_hold = False
        elif opening_active:
            self._gun_hold = False
        elif defense_active or tactical_defense_active or tactical_energy_active:
            self._gun_hold = False
        elif (
            152.4 <= distance_m <= self.gun_hold_max_distance_m
            and own_ata_deg <= self.gun_hold_enter_ata_deg
        ):
            self._gun_hold = True
        elif (
            distance_m < 145.0
            or distance_m > 1250.0
            or own_ata_deg >= self.gun_hold_exit_ata_deg
        ):
            self._gun_hold = False

        was_energy_recovering = self._energy_recovering
        proactive_high_climb = (
            self.diverse_energy_recovery
            and altitude_m >= 5000.0
            and pitch_deg >= 15.0
            and speed_mps < 230.0
        )
        initialized_low_speed = (
            speed_mps >= 50.0
            and speed_mps < self.energy_recovery_enter_speed_mps
        )
        if (initialized_low_speed or proactive_high_climb) and altitude_m >= 700.0:
            self._energy_recovering = True
        elif speed_mps >= self.energy_recovery_exit_speed_mps or altitude_m < 700.0:
            self._energy_recovering = False
        if self._energy_recovering and not was_energy_recovering:
            los_az_deg = float(observation[11]) * 180.0 if observation.size >= 13 else 0.0
            if abs(los_az_deg) >= 8.0:
                self._energy_turn_sign = 1.0 if los_az_deg > 0.0 else -1.0
            else:
                self._energy_turn_sign = -self._last_energy_turn_sign
            self._last_energy_turn_sign = self._energy_turn_sign

        action = np.asarray(result.action, dtype=np.float32).copy()
        if int(context.info.get("my_plane_id", -1)) == 1:
            rudder_scale = self.red_reference_rudder_scale
        elif (
            self._blue_firing_line_ever_started
            and self.blue_post_attack_reference_rudder_scale is not None
        ):
            rudder_scale = self.blue_post_attack_reference_rudder_scale
        else:
            rudder_scale = self.reference_rudder_scale
        action[2] *= rudder_scale
        mirror_commands = self.mirror_red_commands and self._mirror_opening
        if mirror_commands:
            action[0] *= -1.0
            action[2] *= -1.0
        stage = "reference_bt_mirrored" if mirror_commands else "reference_bt"
        if opening_active:
            desired_bank_deg = -80.0
            bank_error_deg = _wrap_deg(desired_bank_deg - roll_deg)
            action = clip_action([
                0.060 * bank_error_deg,
                -1.0,
                1.0,
                1.0,
            ])
            stage = "mirror_red_opening"
        elif defense_active:
            if self.terrain_drag_defense:
                current_turn_sign = (
                    -self._defense_sign
                    if self._defense_attempts % 2 == 1
                    else self._defense_sign
                )
                pull_up = altitude_m <= 900.0 or defense_elapsed_s >= 7.5
                reattack_elapsed_s = (
                    sim_time_s - self._terrain_overshoot_time_s
                    if self._terrain_overshoot_time_s >= 0.0
                    else 0.0
                )
                if self._terrain_overshoot:
                    maneuver_sign = -current_turn_sign
                    desired_bank_deg = 78.0 * maneuver_sign
                else:
                    maneuver_sign = current_turn_sign
                    desired_bank_deg = (55.0 if pull_up else 70.0) * maneuver_sign
                bank_error_deg = _wrap_deg(desired_bank_deg - roll_deg)
                roll_cmd = float(np.clip(0.055 * bank_error_deg, -1.0, 1.0))
                if self._terrain_overshoot:
                    pitch_cmd = 0.15 if reattack_elapsed_s < 0.9 else -1.0
                else:
                    pitch_cmd = -1.0 if pull_up else 0.65
                rudder_cmd = -0.15 * maneuver_sign
                throttle_cmd = 1.0
                stage = (
                    "terrain_reattack_roll"
                    if self._terrain_overshoot and reattack_elapsed_s < 0.9
                    else "terrain_reattack_pull"
                ) if self._terrain_overshoot else (
                    "terrain_drag_pull" if pull_up else "terrain_drag_dive"
                )
            else:
                scissors_sign = (
                    self._defense_sign
                    if defense_elapsed_s < 2.8
                    else -self._defense_sign
                )
                desired_bank_deg = 70.0 * scissors_sign
                bank_error_deg = _wrap_deg(desired_bank_deg - roll_deg)
                roll_cmd = float(np.clip(0.055 * bank_error_deg, -1.0, 1.0))
                if defense_elapsed_s < 0.7 or 2.8 <= defense_elapsed_s < 3.5:
                    pitch_cmd = 0.15
                    stage = "defense_roll_reversal"
                else:
                    pitch_cmd = -1.0 if speed_mps >= 155.0 else 0.25
                    stage = "defense_break_pull" if speed_mps >= 155.0 else "defense_unload"
                rudder_cmd = -0.25 * scissors_sign
                throttle_cmd = 1.0 if speed_mps < 190.0 else 0.0
            action = clip_action([
                roll_cmd,
                pitch_cmd,
                rudder_cmd,
                throttle_cmd,
            ])
        elif tactical_defense_active:
            desired_bank_deg = 78.0 * self._tactical_defense_turn_sign
            bank_error_deg = _wrap_deg(desired_bank_deg - roll_deg)
            roll_cmd = float(np.clip(0.055 * bank_error_deg, -1.0, 1.0))
            if abs(bank_error_deg) > 28.0 and tactical_defense_elapsed_s < 1.4:
                pitch_cmd = 0.20
                phase = "roll"
            else:
                pitch_cmd = -1.0 if speed_mps >= 235.0 else -0.55
                phase = "pull"
            rudder_cmd = -0.12 * self._tactical_defense_turn_sign
            action = clip_action([roll_cmd, pitch_cmd, rudder_cmd, 1.0])
            direction = "right" if self._tactical_defense_turn_sign > 0.0 else "left"
            stage = f"tactical_break_{phase}_{direction}"
        elif tactical_energy_active:
            desired_bank_deg = 55.0 * self._tactical_energy_turn_sign
            bank_error_deg = _wrap_deg(desired_bank_deg - roll_deg)
            roll_cmd = float(np.clip(0.045 * bank_error_deg, -0.85, 0.85))
            pitch_cmd = 0.80 if pitch_deg >= 10.0 else 0.45
            rudder_cmd = -0.08 * self._tactical_energy_turn_sign
            action = clip_action([roll_cmd, pitch_cmd, rudder_cmd, 1.0])
            direction = "right" if self._tactical_energy_turn_sign > 0.0 else "left"
            stage = f"tactical_energy_{direction}"
        elif threat_unload_active:
            desired_bank_deg = 55.0 * self._threat_unload_turn_sign
            bank_error_deg = _wrap_deg(desired_bank_deg - roll_deg)
            roll_cmd = float(np.clip(0.050 * bank_error_deg, -0.85, 0.85))
            pitch_cmd = 0.60 if pitch_deg >= 20.0 else 0.30
            action = clip_action([roll_cmd, pitch_cmd, 0.0, 1.0])
            direction = "right" if self._threat_unload_turn_sign > 0.0 else "left"
            stage = f"threat_unload_{direction}"
        elif self._energy_recovering and (
            self.force_energy_recovery_under_threat
            or not (
                target_ata_deg <= 25.0
                and distance_m <= 1800.0
                and own_ata_deg >= 70.0
            )
        ):
            desired_bank_deg = (
                self.energy_recovery_bank_deg * self._energy_turn_sign
                if self.diverse_energy_recovery
                else 0.0
            )
            bank_error_deg = _wrap_deg(desired_bank_deg - roll_deg)
            roll_cmd = float(np.clip(0.055 * bank_error_deg, -1.0, 1.0))
            pitch_cmd = 0.85 if pitch_deg > 20.0 else 0.40
            action = clip_action([roll_cmd, pitch_cmd, 0.0, 1.0])
            if self.diverse_energy_recovery:
                direction = "right" if self._energy_turn_sign > 0.0 else "left"
                stage = f"energy_recovery_{direction}"
            else:
                stage = "energy_recovery"
        elif observation.size >= 13 and red_rudder_only_active:
            los_az_deg = float(observation[11]) * 180.0
            los_el_deg = float(observation[12]) * 90.0
            los_az_rate_dps = (
                float(observation[22]) * 45.0
                if observation.size >= 24
                else 0.0
            )
            los_el_rate_dps = (
                float(observation[23]) * 45.0
                if observation.size >= 24
                else 0.0
            )
            if self.red_firing_line_roll_rate_gain > 0.0:
                action[0] = float(np.clip(
                    -self.red_firing_line_roll_rate_gain * roll_rate_dps,
                    -1.0,
                    1.0,
                ))
            if (
                self.red_firing_line_pitch_gain > 0.0
                or self.red_firing_line_pitch_rate_gain > 0.0
            ):
                action[1] = float(np.clip(
                    float(action[1])
                    - self.red_firing_line_pitch_gain * los_el_deg
                    - self.red_firing_line_pitch_rate_gain * los_el_rate_dps,
                    -1.0,
                    1.0,
                ))
            action[2] = float(np.clip(
                -los_az_deg / self.red_firing_line_az_ref_deg
                - self.red_firing_line_az_rate_gain * los_az_rate_dps,
                -self.red_firing_line_rudder_limit,
                self.red_firing_line_rudder_limit,
            ))
            if (
                self.red_firing_line_throttle_cut_closure_mps > 0.0
                and closure_mps >= self.red_firing_line_throttle_cut_closure_mps
            ):
                action[3] = 0.0
            action = clip_action(action)
            stage = "red_firing_line_precision"
        elif observation.size >= 24 and (
            blue_firing_line_active or blue_firing_line_roll_arrest_active
        ):
            los_az_deg = float(observation[11]) * 180.0
            los_el_deg = float(observation[12]) * 90.0
            los_az_rate_dps = float(observation[22]) * 45.0
            los_el_rate_dps = float(observation[23]) * 45.0
            rate_scale = 1.0
            if self.blue_firing_line_rate_gate_ata_deg > 0.0:
                rate_scale = float(np.clip(
                    (
                        self.blue_firing_line_rate_gate_ata_deg
                        - own_ata_deg
                    ) / max(self.blue_firing_line_rate_gate_ata_deg - 1.0, 0.5),
                    0.0,
                    1.0,
                ))
            if blue_firing_line_roll_arrest_active:
                action[0] = float(np.clip(
                    -self.blue_firing_line_roll_arrest_gain * roll_rate_dps,
                    -1.0,
                    1.0,
                ))
            elif self.blue_firing_line_roll_cmd is not None:
                action[0] = self.blue_firing_line_roll_cmd
            action[1] = float(np.clip(
                float(action[1])
                - self.blue_firing_line_pitch_gain * los_el_deg
                - rate_scale * self.blue_firing_line_pitch_rate_gain * los_el_rate_dps,
                -1.0,
                1.0,
            ))
            if self.blue_firing_line_pitch_cmd is not None:
                action[1] = float(np.clip(
                    self.blue_firing_line_pitch_cmd
                    - self.blue_firing_line_pitch_gain * los_el_deg
                    - rate_scale * self.blue_firing_line_pitch_rate_gain * los_el_rate_dps,
                    -1.0,
                    1.0,
                ))
            action[2] = float(np.clip(
                -los_az_deg / self.blue_firing_line_az_ref_deg
                - rate_scale * self.blue_firing_line_az_rate_gain * los_az_rate_dps,
                -1.0,
                1.0,
            ))
            action[3] = (
                0.0
                if self.blue_attack_throttle_cut_speed_mps > 0.0
                and speed_mps >= self.blue_attack_throttle_cut_speed_mps
                else 1.0
            )
            action = clip_action(action)
            stage = (
                "blue_firing_line_roll_arrest"
                if blue_firing_line_roll_arrest_active
                else "blue_firing_line_micro"
            )
        elif observation.size >= 13 and blue_firing_line_pitch_lead_active:
            action[1] = (
                self.blue_firing_line_pitch_lead_cmd
                if self.blue_firing_line_pitch_lead_cmd is not None
                else self.blue_firing_line_pitch_cmd
            )
            action[3] = (
                0.0
                if self.blue_attack_throttle_cut_speed_mps > 0.0
                and speed_mps >= self.blue_attack_throttle_cut_speed_mps
                else 1.0
            )
            action = clip_action(action)
            stage = "blue_firing_line_pitch_lead"
        elif observation.size >= 13 and blue_threat_jink_active:
            los_az_deg = float(observation[11]) * 180.0
            los_side = (
                self._blue_threat_jink_los_side
                if self.blue_threat_jink_pulse_s > 0.0
                else (1.0 if los_az_deg >= 0.0 else -1.0)
            )
            if self.blue_threat_jink_bank_deg > 0.0:
                desired_bank_deg = self.blue_threat_jink_bank_deg * los_side
                bank_error_deg = _wrap_deg(desired_bank_deg - roll_deg)
                action[0] = float(np.clip(0.055 * bank_error_deg, -1.0, 1.0))
                if self.blue_threat_jink_pitch_cmd is not None:
                    action[1] = (
                        0.15
                        if abs(bank_error_deg) > 28.0
                        and sim_time_s - self._blue_threat_jink_start_s < 1.2
                        else self.blue_threat_jink_pitch_cmd
                    )
            elif self.blue_threat_jink_roll_cmd is not None:
                action[0] = self.blue_threat_jink_roll_cmd
                if self.blue_threat_jink_pitch_cmd is not None:
                    action[1] = self.blue_threat_jink_pitch_cmd
            action[2] = (
                self.blue_threat_jink_direction
                * los_side
                * self.blue_threat_jink_rudder_scale
            )
            action[3] = (
                0.0
                if self.blue_threat_jink_throttle_cut_speed_mps > 0.0
                and speed_mps >= self.blue_threat_jink_throttle_cut_speed_mps
                else 1.0
            )
            action = clip_action(action)
            stage = "blue_threat_rudder_jink"
        elif blue_tail_pursuit_brake_active:
            action[3] = 0.0
            action = clip_action(action)
            stage = "blue_tail_pursuit_brake"
        elif observation.size >= 13 and blue_rudder_only_active:
            los_az_deg = float(observation[11]) * 180.0
            los_az_rate_dps = (
                float(observation[22]) * 45.0
                if observation.size >= 24
                else 0.0
            )
            action[2] = float(np.clip(
                -los_az_deg / self.blue_pre_firing_az_ref_deg
                - self.blue_pre_firing_az_rate_gain * los_az_rate_dps,
                -1.0,
                1.0,
            ))
            action = clip_action(action)
            stage = "blue_pre_firing_rudder"
        elif observation.size >= 13 and firing_line_active:
            az_error = float(np.clip(float(observation[11]) * 180.0 / 2.0, -1.0, 1.0))
            el_error = float(np.clip(float(observation[12]) * 90.0 / 2.0, -1.0, 1.0))
            roll_cmd = float(np.clip(0.30 * float(action[0]) + 0.05 * az_error, -0.25, 0.25))
            pitch_cmd = float(np.clip(0.35 * float(action[1]) - 0.12 * el_error, -0.45, 0.20))
            rudder_cmd = float(np.clip(
                0.15 * float(action[2]) - self.firing_line_rudder_gain * az_error,
                -self.firing_line_rudder_limit,
                self.firing_line_rudder_limit,
            ))
            action = clip_action([roll_cmd, pitch_cmd, rudder_cmd, 1.0])
            stage = "firing_line_micro"
        elif observation.size >= 13 and pre_firing_active:
            az_error = float(np.clip(float(observation[11]) * 180.0 / 4.0, -1.0, 1.0))
            el_error = float(np.clip(float(observation[12]) * 90.0 / 8.0, -1.0, 1.0))
            action[1] += -self.pre_firing_pitch_gain * el_error
            action[2] += -self.pre_firing_rudder_gain * az_error
            action[3] = 1.0
            action = clip_action(action)
            stage = "pre_firing_assist"
        elif observation.size >= 13 and self._gun_hold:
            los_az_deg = float(observation[11]) * 180.0
            los_el_deg = float(observation[12]) * 90.0
            az_error = float(np.clip(los_az_deg / 3.0, -1.0, 1.0))
            el_error = float(np.clip(los_el_deg / 3.0, -1.0, 1.0))
            los_az_rate = float(observation[22]) if observation.size >= 24 else 0.0
            los_el_rate = float(observation[23]) if observation.size >= 24 else 0.0
            alignment_scale = float(np.clip(
                own_ata_deg / max(self.gun_hold_exit_ata_deg, 1.0),
                0.0,
                1.0,
            ))

            # Near the one-degree firing line, keep the established bank and
            # use only small corrections. Leveling the wings here sweeps the
            # nose across the target and destroys an otherwise valid shot.
            roll_cmd = float(np.clip(
                (0.15 + 0.35 * alignment_scale) * float(action[0])
                + 0.08 * az_error
                - 0.001 * roll_rate_dps,
                -0.35,
                0.35,
            ))
            pitch_cmd = float(np.clip(
                -0.25 - 0.50 * alignment_scale - 0.20 * el_error
                - self.gun_hold_rate_damping * los_el_rate,
                -0.85,
                0.10,
            ))
            rudder_gain = 0.45 + 0.35 * alignment_scale
            rudder_cmd = float(np.clip(
                -rudder_gain * az_error
                - self.gun_hold_rate_damping * los_az_rate,
                -0.80,
                0.80,
            ))
            action = clip_action([roll_cmd, pitch_cmd, rudder_cmd, 1.0])
            stage = "gun_hold"
        elif (
            observation.size >= 13
            and attack_geometry_ok
            and 152.4 <= distance_m <= self.acquire_max_distance_m
            and own_ata_deg <= self.acquire_max_ata_deg
        ):
            los_az_deg = float(observation[11]) * 180.0
            los_el_deg = float(observation[12]) * 90.0
            az_error = float(np.clip(los_az_deg / 1.3, -1.0, 1.0))
            el_error = float(np.clip(los_el_deg / 1.1, -1.0, 1.0))
            action[2] += -10.2 * az_error
            action[0] += 0.75 * az_error
            action[1] += -0.52 * el_error
            action[3] = 1.0
            action = clip_action(action)
            stage = "gun_acquire"

        result.action = clip_action(action)
        result.source = stage
        result.info = dict(result.info)
        result.info["bt_gun_opportunity_stage"] = stage

        if self.debug:
            if self._start_time_s is None or sim_time_s < self._start_time_s:
                self._start_time_s = sim_time_s
                self._next_status_s = 0.0
            elapsed_s = sim_time_s - self._start_time_s
            if stage != self._last_stage:
                print(
                    "[bt-gun-opportunity] "
                    f"stage={stage} distance={distance_m:.0f}m "
                    f"own_ata={own_ata_deg:.2f} target_ata={target_ata_deg:.1f}",
                    flush=True,
                )
                self._last_stage = stage
            if elapsed_s + 1.0e-6 >= self._next_status_s:
                self._next_status_s = elapsed_s + 1.0
                print(
                    "[bt-gun-status] "
                    f"t={elapsed_s:.1f}s stage={stage} distance={distance_m:.0f}m "
                    f"own_ata={own_ata_deg:.2f} target_ata={target_ata_deg:.1f} "
                    f"speed={speed_mps:.1f}mps alt={float(own[StateIndex.ALT]):.0f}m "
                    f"roll={roll_deg:.1f}deg roll_rate={roll_rate_dps:.1f}dps "
                    f"los_az={float(observation[11]) * 180.0:.2f}deg "
                    f"los_el={float(observation[12]) * 90.0:.2f}deg "
                    f"los_az_rate={float(observation[22]) * 45.0 if observation.size >= 24 else 0.0:.1f}dps "
                    f"los_el_rate={float(observation[23]) * 45.0 if observation.size >= 24 else 0.0:.1f}dps "
                    f"closure={closure_mps:.1f}mps "
                    f"own_hp={float(own[StateIndex.HEALTH]):.3f} "
                    f"target_hp={float(target[StateIndex.HEALTH]):.3f} "
                    f"action={np.asarray(result.action).round(3).tolist()}",
                    flush=True,
                )
        return result

    def _canonicalize_red_context(
        self,
        context: ActionContext,
        own: np.ndarray,
        target: np.ndarray,
    ) -> ActionContext:
        if self._canonical_center_ne is None:
            self._canonical_center_ne = (
                0.5 * float(own[StateIndex.N] + target[StateIndex.N]),
                0.5 * float(own[StateIndex.E] + target[StateIndex.E]),
            )
        center_n, center_e = self._canonical_center_ne

        def transform(state: np.ndarray) -> np.ndarray:
            transformed = np.asarray(state, dtype=np.float64).copy()
            transformed[StateIndex.N] = 2.0 * center_n - state[StateIndex.N]
            transformed[StateIndex.E] = 2.0 * center_e - state[StateIndex.E]
            transformed[StateIndex.YAW] = _wrap_deg(state[StateIndex.YAW] + 180.0)
            return transformed

        canonical_own = transform(own)
        canonical_target = transform(target)
        info = dict(context.info)
        info["my_plane_data"] = AIPilot.BuildPlaneData(
            [
                canonical_own[StateIndex.N],
                canonical_own[StateIndex.E],
                canonical_own[StateIndex.ALT],
            ],
            [
                canonical_own[StateIndex.ROLL],
                canonical_own[StateIndex.PITCH],
                canonical_own[StateIndex.YAW],
            ],
            float(np.linalg.norm(canonical_own[6:9])),
            int(info.get("my_force_side", 1)),
        )
        info["target_plane_data"] = AIPilot.BuildPlaneData(
            [
                canonical_target[StateIndex.N],
                canonical_target[StateIndex.E],
                canonical_target[StateIndex.ALT],
            ],
            [
                canonical_target[StateIndex.ROLL],
                canonical_target[StateIndex.PITCH],
                canonical_target[StateIndex.YAW],
            ],
            float(np.linalg.norm(canonical_target[6:9])),
            int(info.get("target_force_side", 2)),
        )
        info["canonicalized_red_input"] = True
        return ActionContext(
            sim=context.sim,
            opponent_sim=context.opponent_sim,
            ownship_state=canonical_own,
            target_state=canonical_target,
            observation=context.observation,
            info=info,
        )

    def close(self) -> None:
        self.primary.close()


__all__ = ["BTGunOpportunityProvider"]
