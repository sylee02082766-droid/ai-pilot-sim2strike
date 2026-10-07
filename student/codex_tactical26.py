# -*- coding: utf-8 -*-
"""Tactical16-compatible 26-D observation for single-policy dogfighting.

The first 16 features are byte-for-byte compatible with the Release
``tactical16`` observation.  This lets a trained 16-D PPO actor be migrated by
zero-padding only its first layer.  Ten motion features are appended so an MLP
can observe range rate and target motion without needing recurrent memory.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from dogfight.envs.observation import build_observation as build_builtin_observation
from dogfight.envs.observation import normalize
from dogfight.sim.state_schema import StateIndex


OBSERVATION_MODE = "codex_tactical26"
OBSERVATION_SIZE = 26
OBSERVATION_LOW = -1.0
OBSERVATION_HIGH = 1.0

KNOTS_TO_MPS = 0.514444
BULLET_SPEED_MPS = 700.0
MAX_CLOSURE_MPS = 300.0
MAX_CLIMB_MPS = 150.0
MAX_SPEED_ADVANTAGE_MPS = 200.0
MAX_LOS_RATE_DPS = 45.0
MAX_PLAUSIBLE_VELOCITY_MPS = 500.0
MIN_LOG_RANGE_M = 50.0
MAX_LOG_RANGE_M = 20000.0

# One GeometryInfo object is owned by one environment.  Entries reset when
# simulator time moves backwards and are bounded to avoid retaining stale envs.
_HISTORY: dict[int, dict[str, Any]] = {}
_MAX_HISTORY_ENTRIES = 128


def _clip1(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


def _nose_vector_ned(state) -> np.ndarray:
    yaw = math.radians(float(state[StateIndex.YAW]))
    pitch = math.radians(float(state[StateIndex.PITCH]))
    return np.asarray(
        [
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            -math.sin(pitch),
        ],
        dtype=np.float64,
    )


def _angle_between_deg(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1.0e-6 or second_norm < 1.0e-6:
        return 0.0
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _clamp_velocity(velocity: np.ndarray) -> np.ndarray:
    speed = float(np.linalg.norm(velocity))
    if speed <= MAX_PLAUSIBLE_VELOCITY_MPS:
        return velocity
    return velocity * (MAX_PLAUSIBLE_VELOCITY_MPS / speed)


def _motion_features(ownship_state, target_state, geo_info) -> tuple[float, ...]:
    own_position = np.asarray(ownship_state[:3], dtype=np.float64)
    target_position = np.asarray(target_state[:3], dtype=np.float64)
    sim_time = float(ownship_state[StateIndex.SIM_TIME])
    distance = float(geo_info._get_distance(ownship_state, target_state))
    los_az, los_el = geo_info._get_los_angle(ownship_state, target_state)
    los_az = float(los_az)
    los_el = float(los_el)
    own_altitude = float(ownship_state[StateIndex.ALT])
    target_altitude = float(target_state[StateIndex.ALT])

    if len(_HISTORY) > _MAX_HISTORY_ENTRIES:
        _HISTORY.clear()
    key = id(geo_info)
    previous = _HISTORY.get(key)
    reset = previous is None or sim_time < float(previous["sim_time"]) - 1.0e-6

    if reset:
        closure = 0.0
        own_climb = 0.0
        target_climb = 0.0
        los_az_rate = 0.0
        los_el_rate = 0.0
        own_velocity = np.zeros(3, dtype=np.float64)
        target_velocity = np.zeros(3, dtype=np.float64)
    else:
        dt = sim_time - float(previous["sim_time"])
        if dt <= 1.0e-6:
            return tuple(previous["features"])
        closure = -(distance - float(previous["distance"])) / dt
        own_climb = (own_altitude - float(previous["own_altitude"])) / dt
        target_climb = (target_altitude - float(previous["target_altitude"])) / dt
        az_delta = (los_az - float(previous["los_az"]) + 180.0) % 360.0 - 180.0
        los_az_rate = az_delta / dt
        los_el_rate = (los_el - float(previous["los_el"])) / dt
        own_velocity = _clamp_velocity(
            (own_position - np.asarray(previous["own_position"])) / dt
        )
        target_velocity = _clamp_velocity(
            (target_position - np.asarray(previous["target_position"])) / dt
        )

    target_speed_mps = float(target_state[StateIndex.KCAS]) * KNOTS_TO_MPS
    own_speed_mps = float(ownship_state[StateIndex.KCAS]) * KNOTS_TO_MPS
    speed_advantage = own_speed_mps - target_speed_mps
    time_of_flight = distance / BULLET_SPEED_MPS
    intercept_vector = target_position + target_velocity * time_of_flight - own_position
    lead_error = _angle_between_deg(_nose_vector_ned(ownship_state), intercept_vector)

    log_min = math.log10(MIN_LOG_RANGE_M)
    log_max = math.log10(MAX_LOG_RANGE_M)
    log_distance = math.log10(max(distance, MIN_LOG_RANGE_M))
    features = (
        _clip1(2.0 * (log_distance - log_min) / (log_max - log_min) - 1.0),
        _clip1(closure / MAX_CLOSURE_MPS),
        _clip1(own_climb / MAX_CLIMB_MPS),
        _clip1(target_climb / MAX_CLIMB_MPS),
        normalize(float(target_state[StateIndex.KCAS]), 0.0, 600.0),
        _clip1(speed_advantage / MAX_SPEED_ADVANTAGE_MPS),
        _clip1(los_az_rate / MAX_LOS_RATE_DPS),
        _clip1(los_el_rate / MAX_LOS_RATE_DPS),
        _clip1(2.0 * lead_error / 180.0 - 1.0),
        normalize(target_altitude, 0.0, 15000.0),
    )
    _HISTORY[key] = {
        "sim_time": sim_time,
        "distance": distance,
        "own_altitude": own_altitude,
        "target_altitude": target_altitude,
        "los_az": los_az,
        "los_el": los_el,
        "own_position": own_position,
        "target_position": target_position,
        "features": features,
    }
    return features


def build_observation(ownship_state, target_state, geo_info, wez_config=None):
    tactical16 = build_builtin_observation(
        "tactical16", ownship_state, target_state, geo_info, wez_config
    )
    observation = np.empty(OBSERVATION_SIZE, dtype=np.float32)
    observation[:16] = tactical16
    observation[16:] = np.asarray(
        _motion_features(ownship_state, target_state, geo_info), dtype=np.float32
    )
    return observation


def describe_observation():
    return {
        "mode": OBSERVATION_MODE,
        "size": OBSERVATION_SIZE,
        "features": [
            "tactical16_00_to_15",
            "log_range_norm",
            "closure_rate_norm",
            "own_climb_rate_norm",
            "target_climb_rate_norm",
            "target_kcas_norm",
            "speed_advantage_norm",
            "los_az_rate_norm",
            "los_el_rate_norm",
            "lead_error_norm",
            "target_altitude_norm",
        ],
        "description": (
            "Release tactical16 plus explicit range-rate, target-motion, LOS-rate, "
            "lead-error, and target-altitude features."
        ),
    }


__all__ = [
    "OBSERVATION_MODE",
    "OBSERVATION_SIZE",
    "OBSERVATION_LOW",
    "OBSERVATION_HIGH",
    "build_observation",
    "describe_observation",
]
