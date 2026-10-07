# -*- coding: utf-8 -*-
"""Create one PPO bundle by linearly interpolating two compatible bundles."""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.ai.checkpoint_io import load_lightweight_policy_bundle


def resolve(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-a", required=True)
    parser.add_argument("--bundle-b", required=True)
    parser.add_argument("--alpha-b", required=True, type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    alpha = float(args.alpha_b)
    if not 0.0 <= alpha <= 1.0:
        parser.error("--alpha-b must be between 0 and 1")

    bundle_a = resolve(args.bundle_a)
    bundle_b = resolve(args.bundle_b)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    metadata_a, weights_a = load_lightweight_policy_bundle(bundle_a)
    _, weights_b = load_lightweight_policy_bundle(bundle_b)
    if list(weights_a) != list(weights_b):
        raise ValueError("bundle weight keys differ")

    blended = type(weights_a)()
    for key, value_a in weights_a.items():
        value_b = weights_b[key]
        if not isinstance(value_a, np.ndarray) or not isinstance(value_b, np.ndarray):
            raise TypeError(f"unsupported weight type for {key}: {type(value_a)}")
        if value_a.shape != value_b.shape or value_a.dtype != value_b.dtype:
            raise ValueError(f"shape/dtype mismatch for {key}")
        if np.issubdtype(value_a.dtype, np.floating):
            value = (1.0 - alpha) * value_a + alpha * value_b
            blended[key] = value.astype(value_a.dtype, copy=False)
        else:
            blended[key] = np.array(value_a, copy=True)

    payload = dict(metadata_a)
    payload.setdefault("metadata", {})["weight_interpolation"] = {
        "bundle_a": str(bundle_a.resolve()),
        "bundle_b": str(bundle_b.resolve()),
        "alpha_b": alpha,
    }
    output.mkdir(parents=True)
    (output / "metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with gzip.open(output / "policy_weights.pkl.gz", "wb") as file:
        pickle.dump(blended, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"saved: {output}")
    print(f"alpha_b: {alpha:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
