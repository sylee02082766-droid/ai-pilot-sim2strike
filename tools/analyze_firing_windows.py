from __future__ import annotations

import argparse
import csv
import json
import math
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


def _load(path: Path) -> list[dict[str, float]]:
    geometry = GeometryInfo()
    samples: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                frame = int(row["index"])
                baseline = plane_info_to_state(_plane(row, "own", frame), server_z_up=True)
                candidate = plane_info_to_state(_plane(row, "target", frame), server_z_up=True)
            except (KeyError, TypeError, ValueError):
                continue
            relative = baseline[:3] - candidate[:3]
            distance = float(np.linalg.norm(relative))
            relative_velocity = baseline[6:9] - candidate[6:9]
            candidate_los_az_deg, candidate_los_el_deg = geometry._get_los_angle(
                candidate, baseline
            )
            closure = (
                -float(np.dot(relative, relative_velocity)) / distance
                if distance > 1.0e-6
                else 0.0
            )
            samples.append({
                "frame": frame,
                "time_s": frame / 60.0,
                "distance_m": distance,
                "closure_mps": closure,
                "candidate_ata_deg": abs(float(
                    geometry._get_antenna_train_angle(candidate, baseline, False)
                )),
                "baseline_ata_deg": abs(float(
                    geometry._get_antenna_train_angle(baseline, candidate, False)
                )),
                "candidate_los_az_deg": float(candidate_los_az_deg),
                "candidate_los_el_deg": float(candidate_los_el_deg),
                "candidate_speed_mps": float(np.linalg.norm(candidate[6:9])),
                "baseline_speed_mps": float(np.linalg.norm(baseline[6:9])),
                "candidate_altitude_m": float(candidate[StateIndex.ALT]),
                "baseline_altitude_m": float(baseline[StateIndex.ALT]),
                "candidate_roll_deg": float(candidate[StateIndex.ROLL]),
                "candidate_pitch_deg": float(candidate[StateIndex.PITCH]),
            })
    if not samples:
        raise RuntimeError(f"no complete rows in {path}")
    return samples


def _windows(
    samples: list[dict[str, float]],
    side: str,
    ata_limit: float,
    min_range: float,
    max_range: float,
) -> list[dict[str, float]]:
    ata_key = f"{side}_ata_deg"
    selected = [
        sample for sample in samples
        if min_range <= sample["distance_m"] <= max_range
        and sample[ata_key] <= ata_limit
    ]
    groups: list[list[dict[str, float]]] = []
    for sample in selected:
        if not groups or sample["frame"] != groups[-1][-1]["frame"] + 1:
            groups.append([sample])
        else:
            groups[-1].append(sample)
    result = []
    for group in groups:
        result.append({
            "start_s": group[0]["time_s"],
            "end_s": group[-1]["time_s"],
            "frames": len(group),
            "duration_s": len(group) / 60.0,
            "min_ata_deg": min(sample[ata_key] for sample in group),
            "start_range_m": group[0]["distance_m"],
            "end_range_m": group[-1]["distance_m"],
            "mean_closure_mps": sum(sample["closure_mps"] for sample in group) / len(group),
            "max_closure_mps": max(sample["closure_mps"] for sample in group),
        })
    return sorted(result, key=lambda item: (-item["frames"], item["start_s"]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report frame-accurate gun-line dwell windows from a reference-match CSV."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--min-range", type=float, default=152.4)
    parser.add_argument("--max-range", type=float, default=950.0)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--detail-start", type=float)
    parser.add_argument("--detail-end", type=float)
    args = parser.parse_args()

    path = args.csv_path if args.csv_path.is_absolute() else ROOT / args.csv_path
    samples = _load(path)
    min_candidate = min(samples, key=lambda sample: sample["candidate_ata_deg"])
    min_baseline = min(samples, key=lambda sample: sample["baseline_ata_deg"])
    report = {
        "path": str(path.resolve()),
        "duration_s": round(samples[-1]["time_s"], 3),
        "minimum_candidate_ata": min_candidate,
        "minimum_baseline_ata": min_baseline,
        "candidate_ata_le_1": _windows(
            samples, "candidate", 1.0, args.min_range, args.max_range
        )[: args.top],
        "candidate_ata_le_3": _windows(
            samples, "candidate", 3.0, args.min_range, args.max_range
        )[: args.top],
        "baseline_ata_le_1": _windows(
            samples, "baseline", 1.0, args.min_range, args.max_range
        )[: args.top],
        "baseline_ata_le_3": _windows(
            samples, "baseline", 3.0, args.min_range, args.max_range
        )[: args.top],
    }
    if args.detail_start is not None:
        detail_end = args.detail_end if args.detail_end is not None else args.detail_start
        detail = [
            sample for sample in samples
            if args.detail_start <= sample["time_s"] <= detail_end
        ]
        previous = None
        report["detail"] = []
        for sample in detail:
            row = dict(sample)
            if previous is None:
                row["candidate_los_az_rate_dps"] = 0.0
                row["candidate_los_el_rate_dps"] = 0.0
            else:
                dt = sample["time_s"] - previous["time_s"]
                az_delta = (
                    sample["candidate_los_az_deg"]
                    - previous["candidate_los_az_deg"]
                    + 180.0
                ) % 360.0 - 180.0
                row["candidate_los_az_rate_dps"] = az_delta / dt
                row["candidate_los_el_rate_dps"] = (
                    sample["candidate_los_el_deg"]
                    - previous["candidate_los_el_deg"]
                ) / dt
            report["detail"].append({
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in row.items()
            })
            previous = sample
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
