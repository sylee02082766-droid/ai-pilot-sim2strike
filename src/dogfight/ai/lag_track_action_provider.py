from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.sim.state_schema import StateIndex


def _wrap_deg(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _body_velocity_ned(state: np.ndarray) -> np.ndarray:
    roll = math.radians(float(state[StateIndex.ROLL]))
    pitch = math.radians(float(state[StateIndex.PITCH]))
    yaw = math.radians(float(state[StateIndex.YAW]))
    u, v, w = (float(value) for value in state[6:9])
    cphi, sphi = math.cos(roll), math.sin(roll)
    cth, sth = math.cos(pitch), math.sin(pitch)
    cpsi, spsi = math.cos(yaw), math.sin(yaw)
    return np.asarray([
        cth * cpsi * u
        + (sphi * sth * cpsi - cphi * spsi) * v
        + (cphi * sth * cpsi + sphi * spsi) * w,
        cth * spsi * u
        + (sphi * sth * spsi + cphi * cpsi) * v
        + (cphi * sth * spsi - sphi * cpsi) * w,
        -sth * u + sphi * cth * v + cphi * cth * w,
    ], dtype=np.float64)


class LAGTrackActionProvider(ActionProvider):
    """Adapt LAG's released heading-controller weights to the official server."""

    def __init__(
        self,
        actor_module_path: str | Path,
        model_path: str | Path,
        debug: bool = False,
        control_interval_s: float = 0.2,
        recovery_enter_altitude_m: float = 500.0,
        recovery_exit_altitude_m: float = 1000.0,
    ) -> None:
        self.actor_module_path = Path(actor_module_path).resolve()
        self.model_path = Path(model_path).resolve()
        self.debug = bool(debug)
        self.control_interval_s = max(float(control_interval_s), 1.0 / 60.0)
        self.recovery_enter_altitude_m = float(recovery_enter_altitude_m)
        self.recovery_exit_altitude_m = max(
            self.recovery_enter_altitude_m,
            float(recovery_exit_altitude_m),
        )

        if not self.actor_module_path.is_file():
            raise FileNotFoundError(self.actor_module_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)

        import torch

        torch.set_num_threads(1)
        spec = importlib.util.spec_from_file_location(
            "fdsa_lag_baseline_actor", self.actor_module_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load LAG actor module: {self.actor_module_path}")
        actor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(actor_module)
        self._torch = torch
        self.actor = actor_module.BaselineActor()
        self.actor.load_state_dict(
            torch.load(self.model_path, map_location="cpu", weights_only=True)
        )
        self.actor.eval()
        self.reset()

    def reset(self, context: ActionContext | None = None) -> None:
        self._rnn_state = np.zeros((1, 1, 128), dtype=np.float32)
        self._last_update_s: float | None = None
        self._cached_action = np.asarray([0.0, -0.2, 0.0, 0.9], dtype=np.float32)
        self._recovering = False
        self._recovery_turn_sign: float | None = None
        self._last_stage: str | None = None
        self._start_time_s: float | None = None
        self._next_status_s = 0.0

    def _tracking_observation(
        self, own: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, float]:
        own_velocity = _body_velocity_ned(own)
        target_velocity = _body_velocity_ned(target)
        relative = np.asarray(target[:3] - own[:3], dtype=np.float64)

        relative_horizontal = relative[:2]
        desired_yaw_deg = math.degrees(math.atan2(
            float(relative_horizontal[1]), float(relative_horizontal[0])
        ))
        heading_error_rad = math.radians(
            _wrap_deg(desired_yaw_deg - float(own[StateIndex.YAW]))
        )

        own_altitude_m = float(own[StateIndex.ALT])
        target_altitude_m = max(float(target[StateIndex.ALT]), 1000.0)
        roll_rad = math.radians(float(own[StateIndex.ROLL]))
        pitch_rad = math.radians(float(own[StateIndex.PITCH]))

        observation = np.zeros(12, dtype=np.float32)
        observation[0] = np.clip((target_altitude_m - own_altitude_m) / 1000.0, -5.0, 5.0)
        observation[1] = float(np.clip(heading_error_rad, -math.pi, math.pi))
        observation[2] = float(np.clip((target[6] - own[6]) / 340.0, -2.0, 2.0))
        observation[3] = float(np.clip(own_altitude_m / 5000.0, 0.0, 3.0))
        observation[4] = math.sin(roll_rad)
        observation[5] = math.cos(roll_rad)
        observation[6] = math.sin(pitch_rad)
        observation[7] = math.cos(pitch_rad)
        observation[8:11] = np.clip(own[6:9] / 340.0, -2.0, 2.0)
        observation[11] = float(np.clip(own[StateIndex.KCAS] / 340.0, 0.0, 2.0))
        return observation, math.degrees(heading_error_rad)

    def _network_action(self, observation: np.ndarray) -> np.ndarray:
        with self._torch.inference_mode():
            indices, rnn_state = self.actor(
                observation[None, :], self._rnn_state
            )
        self._rnn_state = rnn_state.detach().cpu().numpy()
        values = indices.detach().cpu().numpy().reshape(-1)
        return clip_action([
            float(values[0]) / 20.0 - 1.0,
            float(values[1]) / 20.0 - 1.0,
            float(values[2]) / 20.0 - 1.0,
            1.0,
        ])

    def compute_action(self, context: ActionContext) -> ActionResult:
        own = np.asarray(context.ownship_state, dtype=np.float64)
        target = np.asarray(context.target_state, dtype=np.float64)
        sim_time_s = float(own[StateIndex.SIM_TIME])
        altitude_m = float(own[StateIndex.ALT])
        roll_deg = float(own[StateIndex.ROLL])
        pitch_deg = float(own[StateIndex.PITCH])
        speed_mps = float(np.linalg.norm(own[6:9]))
        distance_m = float(np.linalg.norm(target[:3] - own[:3]))

        if altitude_m <= self.recovery_enter_altitude_m:
            if not self._recovering:
                self._recovery_turn_sign = -1.0 if roll_deg < 0.0 else 1.0
            self._recovering = True
        elif altitude_m >= self.recovery_exit_altitude_m:
            self._recovering = False
            self._recovery_turn_sign = None

        observation, heading_error_deg = self._tracking_observation(own, target)
        if self._recovering:
            turn_sign = self._recovery_turn_sign or 1.0
            desired_bank_deg = 55.0 * turn_sign
            bank_error_deg = _wrap_deg(desired_bank_deg - roll_deg)
            roll_cmd = float(np.clip(0.06 * bank_error_deg, -1.0, 1.0))
            pitch_cmd = float(np.clip(0.060 * (pitch_deg - 30.0), -1.0, 0.20))
            if altitude_m < 450.0:
                pitch_cmd = min(pitch_cmd, -0.75)
            self._cached_action = clip_action([roll_cmd, pitch_cmd, 0.0, 1.0])
            stage = "deck_recovery_lag"
        else:
            due = (
                self._last_update_s is None
                or sim_time_s < self._last_update_s
                or sim_time_s - self._last_update_s >= self.control_interval_s - 1.0e-4
            )
            if due:
                self._cached_action = self._network_action(observation)
                self._last_update_s = sim_time_s
            stage = "lag_track"

        action = np.asarray(self._cached_action, dtype=np.float32)
        if self.debug:
            if self._start_time_s is None or sim_time_s < self._start_time_s:
                self._start_time_s = sim_time_s
                self._next_status_s = 0.0
            elapsed_s = sim_time_s - self._start_time_s
            if stage != self._last_stage:
                print(f"[lag-track] stage={stage}", flush=True)
                self._last_stage = stage
            if elapsed_s + 1.0e-6 >= self._next_status_s:
                self._next_status_s = elapsed_s + 1.0
                print(
                    "[lag-track-status] "
                    f"t={elapsed_s:.1f}s stage={stage} distance={distance_m:.0f}m "
                    f"heading_error={heading_error_deg:.1f}deg speed={speed_mps:.1f}mps "
                    f"alt={altitude_m:.0f}m roll={roll_deg:.1f}deg pitch={pitch_deg:.1f}deg "
                    f"action={action.round(3).tolist()}",
                    flush=True,
                )

        return ActionResult(
            action=action,
            source=stage,
            confidence=0.95,
            info={"lag_stage": stage, "lag_heading_error_deg": heading_error_deg},
        )


__all__ = ["LAGTrackActionProvider"]
