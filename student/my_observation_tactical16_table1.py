# -*- coding: utf-8 -*-
"""Student observation: release default tactical16 observation.

The challenge runner already supports the built-in ``tactical16`` observation
mode. This module provides the same 16-feature default as a custom observation
module, so it can be used either by leaving observation_module empty or by
passing ``--observation-mode custom --observation-module student.my_observation``.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np


class StateIndex(IntEnum):
    N = 0
    E = 1
    D = 2
    ROLL = 3
    PITCH = 4
    YAW = 5
    KCAS = 12
    FUEL = 23
    SIM_TIME = 41
    LAT = 42
    LON = 43
    ALT = 44
    HEALTH = 45


OBSERVATION_MODE = "tactical16"
OBSERVATION_SIZE = 16
OBSERVATION_LOW = -1.0
OBSERVATION_HIGH = 1.0


FEATURES = [
    "ownship_roll_norm",
    "ownship_pitch_norm",
    "ownship_yaw_norm",
    "ownship_speed_norm",
    "ownship_alt_norm",
    "ownship_health_norm",
    "delta_n_norm",
    "delta_e_norm",
    "delta_d_norm",
    "ata_norm",
    "aa_norm",
    "az_norm",
    "el_norm",
    "target_health_norm",
    "in_wez",
    "pursuit_score_norm",
]


def _normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    clipped = float(np.clip(value, minimum, maximum))
    midpoint = (maximum + minimum) / 2.0
    half_range = (maximum - minimum) / 2.0
    return (clipped - midpoint) / half_range


def build_observation(
    ownship_state: Any,
    target_state: Any,
    geo_info: Any,
    wez_config: dict | None = None,
):
    """Return the release default 16-D tactical observation vector.

    Index map:
      0-5   ownship: roll, pitch, yaw, speed(KCAS), altitude, health
      6-8   relative position: delta_n, delta_e, delta_d
      9-12  geometry: ATA, AA, LOS_az, LOS_el
      13    target health
      14    in_wez flag  (-1 / +1)
      15    pursuit score (smooth ATA x range gradient, normalized to [-1, 1])
    """
    obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)

    delta = target_state[:3] - ownship_state[:3]
    distance = float(geo_info._get_distance(ownship_state, target_state))
    ata = float(geo_info._get_antenna_train_angle(ownship_state, target_state, False))
    aa = float(geo_info._get_aspect_angle(ownship_state, target_state, False))
    az, el = geo_info._get_los_angle(ownship_state, target_state)
    az = float(az)
    el = float(el)

    # Ownship state
    obs[0] = _normalize(float(ownship_state[StateIndex.ROLL]), -180.0, 180.0)
    obs[1] = _normalize(float(ownship_state[StateIndex.PITCH]), -90.0, 90.0)
    obs[2] = _normalize(float(ownship_state[StateIndex.YAW]), 0.0, 360.0)
    obs[3] = _normalize(float(ownship_state[StateIndex.KCAS]), 0.0, 600.0)
    obs[4] = _normalize(float(ownship_state[StateIndex.ALT]), 0.0, 15000.0)
    obs[5] = _normalize(float(ownship_state[StateIndex.HEALTH]), 0.0, 1.0)

    # Relative position
    obs[6] = _normalize(float(delta[0]), -15000.0, 15000.0)
    obs[7] = _normalize(float(delta[1]), -15000.0, 15000.0)
    obs[8] = _normalize(float(delta[2]), -8000.0, 8000.0)

    # Geometry
    obs[9] = _normalize(ata, -180.0, 180.0)
    obs[10] = _normalize(aa, -180.0, 180.0)
    obs[11] = _normalize(az, -180.0, 180.0)
    obs[12] = _normalize(el, -90.0, 90.0)

    # Target health
    obs[13] = _normalize(float(target_state[StateIndex.HEALTH]), 0.0, 1.0)

    # WEZ flag: +1 if ownship has target inside weapon engagement zone, -1 otherwise.
    if wez_config is not None:
        in_wez = (
            float(wez_config["min_range_m"]) <= distance <= float(wez_config["max_range_m"])
            and abs(ata) <= float(wez_config["angle_deg"]) / 2.0
        )
        obs[14] = 1.0 if in_wez else -1.0
    else:
        obs[14] = -1.0

    # Pursuit score: smooth ATA x range gradient in [-1, 1].
    ata_factor = max(0.0, 1.0 - abs(ata) / 30.0)
    range_factor = max(0.0, 1.0 - distance / 3000.0)
    pursuit_raw = ata_factor * range_factor
    obs[15] = 2.0 * pursuit_raw - 1.0

    return obs


def describe_observation():
    """Return feature metadata for the default tactical16 observation."""
    return {
        "mode": OBSERVATION_MODE,
        "size": OBSERVATION_SIZE,
        "features": FEATURES,
        "description": (
            "Full tactical observation: ownship attitude + speed + altitude + health, "
            "relative geometry (ATA, AA, LOS), target health, WEZ flag, pursuit score. "
            "All features normalized to [-1, 1]. Observation space bounds: [-1, 1]."
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
