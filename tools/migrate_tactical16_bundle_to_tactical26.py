# -*- coding: utf-8 -*-
"""Expand a tactical16 PPO actor bundle to student.codex_tactical26.

The original 16 input columns are copied unchanged and the ten new feature
columns are initialized to zero.  Therefore inference is initially identical;
fine-tuning can subsequently learn how to use range-rate and target motion.
"""
from __future__ import annotations

import argparse
import gzip
import importlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--observation-module", default="student.codex_tactical26"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source = _resolve(args.source)
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    metadata_path = source / "metadata.json"
    weights_path = source / "policy_weights.pkl.gz"
    if not metadata_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"source is not a lightweight bundle: {source}")

    observation_module = importlib.import_module(args.observation_module)
    new_size = int(observation_module.OBSERVATION_SIZE)
    if new_size <= 16:
        raise ValueError(f"new observation size must exceed 16, got {new_size}")
    description = observation_module.describe_observation()
    observation_mode = str(observation_module.OBSERVATION_MODE)

    with gzip.open(weights_path, "rb") as file:
        weights = pickle.load(file)

    expanded_keys: list[str] = []
    for key, value in list(weights.items()):
        array = np.asarray(value)
        if array.ndim != 2 or array.shape[1] != 16:
            continue
        expanded = np.zeros((array.shape[0], new_size), dtype=array.dtype)
        expanded[:, :16] = array
        weights[key] = expanded
        expanded_keys.append(str(key))
    if not expanded_keys:
        raise RuntimeError("no [hidden,16] input layer was found in bundle weights")

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    algorithm_config = payload.setdefault("algorithm_config", {})
    env_config = algorithm_config.setdefault("env_config", {})
    env_config["observation_mode"] = observation_mode
    env_config["observation_module"] = args.observation_module
    env_config["observation_size"] = new_size
    env_config["observation_summary"] = description

    bundle_metadata = payload.setdefault("metadata", {})
    bundle_metadata["obs_mode"] = observation_mode
    bundle_metadata["observation_module"] = args.observation_module
    bundle_metadata["observation_size"] = new_size
    bundle_metadata["observation_summary"] = description
    bundle_metadata["migrated_from"] = str(source.resolve())
    bundle_metadata["migration"] = {
        "type": "tactical16_zero_pad",
        "old_size": 16,
        "new_size": new_size,
        "expanded_keys": expanded_keys,
    }

    output.mkdir(parents=True)
    (output / "metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with gzip.open(output / "policy_weights.pkl.gz", "wb") as file:
        pickle.dump(weights, file, protocol=pickle.HIGHEST_PROTOCOL)
    for optional_name in ("README.md",):
        optional_source = source / optional_name
        if optional_source.exists():
            shutil.copy2(optional_source, output / optional_name)

    print(f"source: {source}")
    print(f"output: {output}")
    print(f"observation: 16 -> {new_size}")
    print(f"expanded: {', '.join(expanded_keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
