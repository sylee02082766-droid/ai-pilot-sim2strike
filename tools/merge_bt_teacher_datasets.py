# -*- coding: utf-8 -*-
"""Merge bilateral BT teacher captures and emphasize acquisition states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _resolve(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--head-samples", type=int, default=300)
    parser.add_argument("--head-repeat", type=int, default=4)
    parser.add_argument("--turn-threshold", type=float, default=0.20)
    parser.add_argument("--turn-repeat", type=int, default=3)
    args = parser.parse_args()

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    source_summaries = []
    for input_text in args.inputs:
        path = _resolve(input_text)
        payload = np.load(path)
        obs = np.asarray(payload["observations"], dtype=np.float32)
        act = np.asarray(payload["actions"], dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != 26:
            raise ValueError(f"expected {path} observations [N,26], got {obs.shape}")
        if act.shape != (len(obs), 4):
            raise ValueError(f"expected {path} actions [N,4], got {act.shape}")

        chunks_obs = [obs]
        chunks_act = [act]
        head_count = min(max(args.head_samples, 0), len(obs))
        for _ in range(max(args.head_repeat - 1, 0)):
            chunks_obs.append(obs[:head_count])
            chunks_act.append(act[:head_count])
        turn_indices = np.flatnonzero(np.abs(act[:, 0]) >= args.turn_threshold)
        for _ in range(max(args.turn_repeat - 1, 0)):
            chunks_obs.append(obs[turn_indices])
            chunks_act.append(act[turn_indices])
        observations.extend(chunks_obs)
        actions.extend(chunks_act)
        source_summaries.append(
            {
                "path": str(path.resolve()),
                "raw_samples": int(len(obs)),
                "head_samples": int(head_count),
                "turn_samples": int(len(turn_indices)),
            }
        )

    merged_obs = np.concatenate(observations, axis=0)
    merged_act = np.concatenate(actions, axis=0)
    rng = np.random.default_rng(260822)
    order = rng.permutation(len(merged_obs))
    merged_obs = merged_obs[order]
    merged_act = merged_act[order]

    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        observations=merged_obs,
        actions=merged_act,
        source=np.asarray(["baseline_bt_bilateral_balanced"]),
    )
    summary = {
        "output": str(output.resolve()),
        "sources": source_summaries,
        "samples": int(len(merged_obs)),
        "head_repeat": args.head_repeat,
        "turn_threshold": args.turn_threshold,
        "turn_repeat": args.turn_repeat,
        "action_min": merged_act.min(axis=0).tolist(),
        "action_max": merged_act.max(axis=0).tolist(),
        "action_mean": merged_act.mean(axis=0).tolist(),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
