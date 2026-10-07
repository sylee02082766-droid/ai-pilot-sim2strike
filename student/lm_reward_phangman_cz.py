# -*- coding: utf-8 -*-
"""PHANG-MAN CZ low-level reward wrapper."""
from __future__ import annotations

from student.lm_reward_phangman import MY_REWARD_CONFIG as _BASE_CONFIG
from student.lm_reward_phangman import compute_reward
from student.lm_reward_phangman import describe_reward


MY_REWARD_CONFIG = {
    **_BASE_CONFIG,
    "mode": "lm_phangman_cz",
    "policy_kind": "cz",
    "gunsnap_profile": "cz",
}


__all__ = ["MY_REWARD_CONFIG", "compute_reward", "describe_reward"]
