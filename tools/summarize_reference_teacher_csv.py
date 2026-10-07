# -*- coding: utf-8 -*-
"""Summarize geometry from a reference-match teacher CSV, including live files."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from GeoMathUtil import GeometryInfo
from dogfight.unreal.policies import plane_info_to_state
from dogfight.sim.state_schema import StateIndex


def _plane(row: dict[str, str], prefix: str, index: int) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        position=SimpleNamespace(**{
            axis: float(row[f"{prefix}_{axis}"]) for axis in ("x", "y", "z")
        }),
        rotation=SimpleNamespace(**{
            axis: float(row[f"{prefix}_{axis}"])
            for axis in ("roll", "pitch", "yaw")
        }),
        velocity=SimpleNamespace(**{
            axis: float(row[f"{prefix}_v{axis}"]) for axis in ("x", "y", "z")
        }),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--stride", type=int, default=6)
    args = parser.parse_args()

    path = Path(args.csv_path)
    if not path.is_absolute():
        path = ROOT / path
    geometry = GeometryInfo()
    samples = []
    last_index: int | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            try:
                index = int(row["index"])
                if last_index is not None and index - last_index < max(1, args.stride):
                    continue
                baseline = plane_info_to_state(_plane(row, "own", index), server_z_up=True)
                candidate = plane_info_to_state(_plane(row, "target", index), server_z_up=True)
            except (KeyError, TypeError, ValueError):
                continue
            relative = candidate[:3] - baseline[:3]
            distance = float(np.linalg.norm(relative))
            baseline_ata = abs(float(
                geometry._get_antenna_train_angle(baseline, candidate, False)
            ))
            candidate_ata = abs(float(
                geometry._get_antenna_train_angle(candidate, baseline, False)
            ))
            samples.append({
                "frame": index,
                "time_s": index / 60.0,
                "distance_m": distance,
                "baseline_ata_deg": baseline_ata,
                "candidate_ata_deg": candidate_ata,
                "baseline_altitude_m": float(baseline[StateIndex.ALT]),
                "candidate_altitude_m": float(candidate[StateIndex.ALT]),
                "baseline_speed_mps": float(np.linalg.norm(baseline[6:9])),
                "candidate_speed_mps": float(np.linalg.norm(candidate[6:9])),
            })
            last_index = index
    if not samples:
        raise RuntimeError(f"no complete data rows in {path}")

    latest = samples[-1]
    first_candidate_25 = next(
        (sample["time_s"] for sample in samples if sample["candidate_ata_deg"] <= 25.0),
        None,
    )
    candidate_attack = [
        sample for sample in samples
        if sample["candidate_ata_deg"] <= 3.0
        and sample["baseline_ata_deg"] >= 90.0
        and 152.4 <= sample["distance_m"] <= 950.0
    ]
    baseline_attack = [
        sample for sample in samples
        if sample["baseline_ata_deg"] <= 3.0
        and sample["candidate_ata_deg"] >= 90.0
        and 152.4 <= sample["distance_m"] <= 950.0
    ]
    summary = {
        "path": str(path.resolve()),
        "samples": len(samples),
        "duration_s": round(latest["time_s"], 2),
        "latest": {key: round(value, 3) for key, value in latest.items()},
        "min_distance_m": round(min(sample["distance_m"] for sample in samples), 3),
        "min_candidate_ata_deg": round(
            min(sample["candidate_ata_deg"] for sample in samples), 3
        ),
        "min_baseline_ata_deg": round(
            min(sample["baseline_ata_deg"] for sample in samples), 3
        ),
        "first_candidate_ata25_s": (
            round(first_candidate_25, 3) if first_candidate_25 is not None else None
        ),
        "candidate_attack_samples": len(candidate_attack),
        "baseline_attack_samples": len(baseline_attack),
        "min_candidate_altitude_m": round(
            min(sample["candidate_altitude_m"] for sample in samples), 3
        ),
        "min_candidate_speed_mps": round(
            min(sample["candidate_speed_mps"] for sample in samples), 3
        ),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
