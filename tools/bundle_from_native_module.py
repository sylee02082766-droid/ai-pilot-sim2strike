# -*- coding: utf-8 -*-
"""Build a full lightweight bundle from an RLlib native module checkpoint.

Older Release exports sometimes saved only the inference actor.  The native
checkpoint still contains the complete RLModule state, including PPO's value
head.  This utility combines that state with metadata from an existing bundle
without restoring Ray or modifying the native checkpoint.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import gzip
import json
import pickle
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MODULE_STATE_RELATIVE = Path(
    "learner_group/learner/rl_module/default_policy/module_state.pkl"
)


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata-source", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    checkpoint = _resolve(args.checkpoint)
    metadata_source = _resolve(args.metadata_source)
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    state_path = checkpoint / MODULE_STATE_RELATIVE
    metadata_path = metadata_source / "metadata.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"native module state not found: {state_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"bundle metadata not found: {metadata_path}")

    with state_path.open("rb") as file:
        weights = pickle.load(file)
    if not isinstance(weights, Mapping) or not weights:
        raise TypeError(f"unexpected module state type: {type(weights)!r}")

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = payload.setdefault("metadata", {})
    metadata["native_checkpoint_source"] = str(checkpoint.resolve())
    metadata["full_module_state"] = True
    metadata["module_state_keys"] = sorted(str(key) for key in weights)
    match = re.fullmatch(r"checkpoint_(\d+)", checkpoint.name)
    if match:
        metadata["native_checkpoint_iteration"] = int(match.group(1))

    output.mkdir(parents=True)
    (output / "metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with gzip.open(output / "policy_weights.pkl.gz", "wb") as file:
        pickle.dump(weights, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"checkpoint: {checkpoint}")
    print(f"metadata:   {metadata_source}")
    print(f"output:     {output}")
    print(f"weights:    {len(weights)} tensors")
    print(
        "value head: "
        + ("present" if any(str(key).startswith("vf.") for key in weights) else "missing")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
