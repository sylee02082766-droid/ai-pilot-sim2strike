# -*- coding: utf-8 -*-
"""Compose diagnostic tactical26 bundles without changing source artifacts."""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_WEIGHT = "encoder.encoder.net.mlp.0.weight"


def _resolve(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def _load(path: Path):
    with gzip.open(path / "policy_weights.pkl.gz", "rb") as file:
        return pickle.load(file)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=("new-columns-only", "old-actor-only", "zero-actor"),
        required=True,
    )
    args = parser.parse_args()

    base = _resolve(args.base)
    variant = _resolve(args.variant)
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    base_weights = _load(base)
    variant_weights = _load(variant)
    result = {key: np.array(value, copy=True) for key, value in base_weights.items()}
    if INPUT_WEIGHT not in result or INPUT_WEIGHT not in variant_weights:
        raise KeyError(f"missing {INPUT_WEIGHT}")
    if np.asarray(result[INPUT_WEIGHT]).shape[1] <= 16:
        raise ValueError("expected a tactical26 input layer")

    if args.mode == "new-columns-only":
        result[INPUT_WEIGHT][:, 16:] = np.asarray(variant_weights[INPUT_WEIGHT])[:, 16:]
    elif args.mode == "old-actor-only":
        for key, value in variant_weights.items():
            if str(key).startswith("vf."):
                continue
            result[key] = np.array(value, copy=True)
        result[INPUT_WEIGHT][:, 16:] = 0.0
    else:
        result["pi.net.mlp.0.weight"][:] = 0.0
        result["pi.net.mlp.0.bias"][:] = 0.0

    payload = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
    metadata = payload.setdefault("metadata", {})
    metadata["composition"] = {
        "mode": args.mode,
        "base": str(base.resolve()),
        "variant": str(variant.resolve()),
        "input_weight": INPUT_WEIGHT,
    }
    output.mkdir(parents=True)
    (output / "metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with gzip.open(output / "policy_weights.pkl.gz", "wb") as file:
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"created: {output}")
    print(f"mode:    {args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
