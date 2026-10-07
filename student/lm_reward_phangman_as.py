# -*- coding: utf-8 -*-
"""PHANG-MAN AS low-level reward wrapper."""
from __future__ import annotations

from student.lm_reward_phangman import MY_REWARD_CONFIG as _BASE_CONFIG
from student.lm_reward_phangman import compute_reward
from student.lm_reward_phangman import describe_reward


MY_REWARD_CONFIG = {
    **_BASE_CONFIG,
    "mode": "lm_phangman_as",
    "policy_kind": "as",
    "gunsnap_profile": "as",
}


__all__ = ["MY_REWARD_CONFIG", "compute_reward", "describe_reward"]
