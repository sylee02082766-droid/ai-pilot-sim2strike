# -*- coding: utf-8 -*-
"""PHANG-MAN-inspired observation vector for fresh experiments.

The Lockheed Martin paper used a richer state than the public AIP release
exposes. This observation keeps every directly available local/relative state
that maps cleanly to the paper description, then adds WEZ geometry signals.

Do not use this module with old tactical16 bundles; the observation size is
different. Use it for fresh SAC/PPO runs only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _path in (ROOT, SRC, Path.cwd(), Path.cwd() / "src"):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dogfight.sim.state_schema import StateIndex


OBSERVATION_MODE = "lm_phangman27"
OBSERVATION_SIZE = 27
OBSERVATION_LOW = -1.0
OBSERVATION_HIGH = 1.0


def _norm(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    clipped = float(np.clip(float(value), minimum, maximum))
    midpoint = (minimum + maximum) / 2.0
    half = (maximum - minimum) / 2.0
    return float((clipped - midpoint) / half)


def build_observation(ownship_state, target_state, geo_info, wez_config=None) -> np.ndarray:
    obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    delta = np.asarray(target_state[0:3], dtype=np.float64) - np.asarray(
        ownship_state[0:3], dtype=np.float64
    )
    distance = float(geo_info._get_distance(ownship_state, target_state))
    ata = float(geo_info._get_antenna_train_angle(ownship_state, target_state, False))
    aa = float(geo_info._get_aspect_angle(ownship_state, target_state, False))
    red_ata = float(geo_info._get_antenna_train_angle(target_state, ownship_state, False))
    az, el = geo_info._get_los_angle(ownship_state, target_state)

    obs[0] = _norm(ownship_state[StateIndex.N], -20000.0, 20000.0)
    obs[1] = _norm(ownship_state[StateIndex.E], -20000.0, 20000.0)
    obs[2] = _norm(ownship_state[StateIndex.D], -15000.0, 0.0)
    obs[3] = _norm(ownship_state[StateIndex.ROLL], -180.0, 180.0)
    obs[4] = _norm(ownship_state[StateIndex.PITCH], -90.0, 90.0)
    obs[5] = _norm(ownship_state[StateIndex.YAW], 0.0, 360.0)
    obs[6] = _norm(ownship_state[StateIndex.KCAS], 0.0, 700.0)
    obs[7] = _norm(ownship_state[StateIndex.ALT], 0.0, 15000.0)
    obs[8] = _norm(ownship_state[StateIndex.FUEL], 0.0, 10000.0)
    obs[9] = _norm(ownship_state[StateIndex.HEALTH], 0.0, 1.0)

    obs[10] = _norm(delta[0], -20000.0, 20000.0)
    obs[11] = _norm(delta[1], -20000.0, 20000.0)
    obs[12] = _norm(delta[2], -10000.0, 10000.0)
    obs[13] = _norm(target_state[StateIndex.ROLL], -180.0, 180.0)
    obs[14] = _norm(target_state[StateIndex.PITCH], -90.0, 90.0)
    obs[15] = _norm(target_state[StateIndex.YAW], 0.0, 360.0)
    obs[16] = _norm(target_state[StateIndex.KCAS], 0.0, 700.0)
    obs[17] = _norm(target_state[StateIndex.ALT], 0.0, 15000.0)
    obs[18] = _norm(target_state[StateIndex.HEALTH], 0.0, 1.0)

    obs[19] = _norm(distance, 0.0, 20000.0)
    obs[20] = _norm(ata, -180.0, 180.0)
    obs[21] = _norm(aa, -180.0, 180.0)
    obs[22] = _norm(az, -180.0, 180.0)
    obs[23] = _norm(el, -90.0, 90.0)
    obs[24] = _norm(red_ata, -180.0, 180.0)

    if wez_config:
        min_range = float(wez_config["min_range_m"])
        max_range = float(wez_config["max_range_m"])
        half_angle = float(wez_config["angle_deg"]) / 2.0
        in_wez = min_range <= distance <= max_range and abs(ata) <= half_angle
        obs[25] = 1.0 if in_wez else -1.0
        if max_range > min_range and min_range <= distance <= max_range:
            obs[26] = 2.0 * ((max_range - distance) / (max_range - min_range)) - 1.0
        else:
            obs[26] = -1.0
    else:
        obs[25] = -1.0
        obs[26] = -1.0

    return obs


def observation_size() -> int:
    return OBSERVATION_SIZE


def describe_observation() -> dict:
    return {
        "mode": OBSERVATION_MODE,
        "size": OBSERVATION_SIZE,
        "features": [
            "ownship_n",
            "ownship_e",
            "ownship_d",
            "ownship_roll",
            "ownship_pitch",
            "ownship_yaw",
            "ownship_kcas",
            "ownship_alt",
            "ownship_fuel",
            "ownship_health",
            "delta_n",
            "delta_e",
            "delta_d",
            "target_roll",
            "target_pitch",
            "target_yaw",
            "target_kcas",
            "target_alt",
            "target_health",
            "distance",
            "ownship_ata",
            "aspect_angle",
            "los_az",
            "los_el",
            "target_ata",
            "in_wez",
            "wez_damage_profile",
        ],
        "description": (
            "PHANG-MAN-inspired available-state observation. It includes all "
            "AIP-release state fields that map cleanly to the paper, plus "
            "relative geometry and WEZ signals."
        ),
    }


__all__ = [
    "OBSERVATION_MODE",
    "OBSERVATION_SIZE",
    "OBSERVATION_LOW",
    "OBSERVATION_HIGH",
    "build_observation",
    "observation_size",
    "describe_observation",
]
