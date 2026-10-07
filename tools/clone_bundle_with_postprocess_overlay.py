# -*- coding: utf-8 -*-
"""Clone a bundle and deep-merge a small action-postprocess YAML overlay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _resolve(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = _resolve(args.source)
    overlay_path = _resolve(args.overlay)
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    payload = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    overlay_data = yaml.safe_load(overlay_path.read_text(encoding="utf-8")) or {}
    overlay = overlay_data.get("action_postprocess_overlay")
    if not isinstance(overlay, dict):
        raise ValueError(f"missing action_postprocess_overlay: {overlay_path}")

    env_config = payload.setdefault("algorithm_config", {}).setdefault("env_config", {})
    postprocess = env_config.setdefault("action_postprocess", {})
    _deep_update(postprocess, overlay)
    payload.setdefault("metadata", {})["postprocess_overlay_clone"] = {
        "source": str(source.resolve()),
        "overlay": str(overlay_path.resolve()),
    }

    output.mkdir(parents=True)
    (output / "metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    shutil.copy2(source / "policy_weights.pkl.gz", output / "policy_weights.pkl.gz")
    print(f"source:  {source}")
    print(f"overlay: {overlay_path}")
    print(f"output:  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
