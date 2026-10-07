# -*- coding: utf-8 -*-
"""Clone a policy bundle and replace only its action-postprocess metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _resolve(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = _resolve(args.source)
    experiment = _resolve(args.experiment)
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    payload = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    experiment_data = yaml.safe_load(experiment.read_text(encoding="utf-8")) or {}
    postprocess = experiment_data.get("env_config", {}).get("action_postprocess")
    if not isinstance(postprocess, dict):
        raise ValueError(f"experiment has no env_config.action_postprocess: {experiment}")

    env_config = payload.setdefault("algorithm_config", {}).setdefault("env_config", {})
    env_config["action_postprocess"] = postprocess
    metadata = payload.setdefault("metadata", {})
    metadata["postprocess_clone"] = {
        "source": str(source.resolve()),
        "experiment": str(experiment.resolve()),
    }

    output.mkdir(parents=True)
    (output / "metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    shutil.copy2(source / "policy_weights.pkl.gz", output / "policy_weights.pkl.gz")
    print(f"source:     {source}")
    print(f"experiment: {experiment}")
    print(f"output:     {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
