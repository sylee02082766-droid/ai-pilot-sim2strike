# -*- coding: utf-8 -*-
"""PHANG-MAN selector reward wrapper.

This is only the selector reward signal. A real PHANG-MAN selector also needs
an action space that chooses among frozen CZ/AS/CS policies at 10 Hz; the public
training environment does not provide that switching layer yet.
"""
from __future__ import annotations

from student.lm_reward_phangman import MY_REWARD_CONFIG as _BASE_CONFIG
from student.lm_reward_phangman import compute_reward
from student.lm_reward_phangman import describe_reward


MY_REWARD_CONFIG = {
    **_BASE_CONFIG,
    "mode": "lm_phangman_selector",
    "policy_kind": "selector",
    "gunsnap_profile": "cz",
}


__all__ = ["MY_REWARD_CONFIG", "compute_reward", "describe_reward"]
