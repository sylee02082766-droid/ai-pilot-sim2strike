# -*- coding: utf-8 -*-
"""Benchmark complete deterministic bundle inference including postprocessing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.ai.action_provider import ActionContext
from dogfight.ai.rllib_utils import build_algorithm_from_bundle
from dogfight.ai.rl_action_provider import RLActionProvider


def resolve(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def observations(count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-1.0, 1.0, size=(count, 26)).astype(np.float32)
    obs[:, 3] = rng.uniform(-0.35, 0.35, size=count)  # flight speed
    obs[:, 4] = rng.uniform(-0.90, 0.25, size=count)  # official altitude range
    obs[:, 5] = rng.uniform(0.0, 1.0, size=count)
    obs[:, 13] = rng.uniform(0.0, 1.0, size=count)
    obs[:, 14] = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=count)
    return obs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bundle = resolve(args.bundle)
    started = time.perf_counter()
    provider = RLActionProvider(
        bundle_dir=str(bundle),
        algorithm_factory=build_algorithm_from_bundle,
        explore=False,
    )
    load_s = time.perf_counter() - started
    sample_count = max(args.iterations, args.warmup, 1)
    sample_obs = observations(sample_count, args.seed)
    provider.reset(ActionContext(sim=None, opponent_sim=None, observation=sample_obs[0]))

    try:
        for index in range(args.warmup):
            provider.compute_action(
                ActionContext(sim=None, opponent_sim=None, observation=sample_obs[index])
            )

        durations_ms = np.empty(args.iterations, dtype=np.float64)
        for index in range(args.iterations):
            before = time.perf_counter_ns()
            provider.compute_action(
                ActionContext(sim=None, opponent_sim=None, observation=sample_obs[index])
            )
            durations_ms[index] = (time.perf_counter_ns() - before) / 1_000_000.0
    finally:
        provider.close()

    limit_ms = 1000.0 / 60.0
    result = {
        "bundle": str(bundle.resolve()),
        "load_seconds": load_s,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "mean_ms": float(np.mean(durations_ms)),
        "p50_ms": float(np.percentile(durations_ms, 50)),
        "p95_ms": float(np.percentile(durations_ms, 95)),
        "p99_ms": float(np.percentile(durations_ms, 99)),
        "max_ms": float(np.max(durations_ms)),
        "limit_ms": limit_ms,
        "over_limit_count": int(np.count_nonzero(durations_ms > limit_ms)),
        "over_limit_rate": float(np.mean(durations_ms > limit_ms)),
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
