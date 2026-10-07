from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local dogfight evaluations for multiple RL bundles and summarize Tacview CSV logs."
    )
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--python", dest="python_exe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-engage-time", type=float, default=200.0)
    parser.add_argument("--episode-step-limit", type=int, default=12000)
    parser.add_argument("--ownship-backend", default="rl", choices=["rl", "hybrid"])
    parser.add_argument("--ownship-bt-dll", default="AIP_BASE.dll")
    parser.add_argument("--target-backend", default="bt", choices=["bt", "fixed"])
    parser.add_argument("--target-bt-dll", default="AIP_BASE_target.dll")
    parser.add_argument("--hybrid-mode", default="residual", choices=["residual", "blend", "switch"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate in label=absolute_bundle_path format.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def distance_m(blue: dict[str, str], red: dict[str, str]) -> float:
    radius_m = 6_371_000.0
    lat1 = math.radians(as_float(blue, "Latitude"))
    lat2 = math.radians(as_float(red, "Latitude"))
    dlat = lat2 - lat1
    dlon = math.radians(as_float(red, "Longitude") - as_float(blue, "Longitude"))
    hav = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    ground = 2.0 * radius_m * math.asin(min(1.0, math.sqrt(hav)))
    dz = as_float(red, "Altitude") - as_float(blue, "Altitude")
    return math.hypot(ground, dz)


def parse_stdout(stdout: str) -> dict[str, str | float | bool]:
    parsed: dict[str, str | float | bool] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("end_condition:"):
            parsed["end_condition"] = line.split(":", 1)[1].strip()
        elif line.startswith("terminated:"):
            parsed["terminated"] = line.split(":", 1)[1].strip().lower() == "true"
        elif line.startswith("truncated:"):
            parsed["truncated"] = line.split(":", 1)[1].strip().lower() == "true"
        elif line.startswith("total_reward:"):
            parsed["total_reward"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("ownship_health:"):
            parsed["ownship_health_stdout"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("target_health:"):
            parsed["target_health_stdout"] = float(line.split(":", 1)[1].strip())
    return parsed


def latest_ownship_csv(log_dir: Path, after_timestamp: float) -> Path:
    candidates = [
        path
        for path in log_dir.glob("*_ownship_(F-16)[[]Blue[]].csv")
        if path.stat().st_mtime >= after_timestamp
    ]
    if not candidates:
        raise FileNotFoundError(f"No ownship CSV created after {after_timestamp}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def summarize_csv(label: str, ownship_csv: Path) -> dict[str, object]:
    target_csv = Path(str(ownship_csv).replace("_ownship_(F-16)[Blue].csv", "_target_(F-16)[Red].csv"))
    blue_rows = read_rows(ownship_csv)
    red_rows = read_rows(target_csv)
    paired_rows = zip(blue_rows, red_rows)
    min_distance = min(distance_m(blue, red) for blue, red in paired_rows)
    return {
        "label": label,
        "ownship_csv": ownship_csv.name,
        "target_csv": target_csv.name,
        "duration_s": as_float(blue_rows[-1], "Time"),
        "steps": len(blue_rows),
        "blue_min_alt_m": min(as_float(row, "Altitude") for row in blue_rows),
        "red_min_alt_m": min(as_float(row, "Altitude") for row in red_rows),
        "blue_final_alt_m": as_float(blue_rows[-1], "Altitude"),
        "red_final_alt_m": as_float(red_rows[-1], "Altitude"),
        "blue_health": as_float(blue_rows[-1], "Health"),
        "red_health": as_float(red_rows[-1], "Health"),
        "min_distance_m": min_distance,
    }


def run_candidate(args: argparse.Namespace, label: str, bundle: Path) -> dict[str, object]:
    release_dir = Path(args.release_dir)
    log_dir = release_dir / "artifacts" / "logs"
    started_at = time.time()
    command = [
        args.python_exe,
        "run_local_dogfight.py",
        "--ownship-backend",
        args.ownship_backend,
        "--ownship-bundle-dir",
        str(bundle),
        "--target-backend",
        args.target_backend,
        "--max-engage-time",
        str(args.max_engage_time),
        "--episode-step-limit",
        str(args.episode_step_limit),
        "--save-log",
    ]
    if args.ownship_backend == "hybrid":
        command.extend(
            [
                "--ownship-bt-dll",
                args.ownship_bt_dll,
                "--hybrid-mode",
                args.hybrid_mode,
                "--alpha",
                str(args.alpha),
                "--residual-scale",
                str(args.residual_scale),
            ]
        )
    if args.target_backend == "bt":
        command.extend(["--target-bt-dll", args.target_bt_dll])

    proc = subprocess.run(
        command,
        cwd=release_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    ownship_csv = latest_ownship_csv(log_dir, started_at)
    summary = summarize_csv(label, ownship_csv)
    summary.update(parse_stdout(proc.stdout))
    summary["bundle"] = str(bundle)
    summary["returncode"] = proc.returncode
    return summary


def write_outputs(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    candidates: list[tuple[str, Path]] = []
    for item in args.candidate:
        if "=" not in item:
            raise ValueError(f"Expected label=path candidate, got {item!r}")
        label, path = item.split("=", 1)
        candidates.append((label, Path(path)))

    rows: list[dict[str, object]] = []
    for label, bundle in candidates:
        print(f"[eval] {label}: {bundle}", flush=True)
        rows.append(run_candidate(args, label, bundle))
        last = rows[-1]
        print(
            "[result] "
            f"end={last.get('end_condition')} "
            f"reward={last.get('total_reward')} "
            f"blue_min_alt={float(last['blue_min_alt_m']):.1f} "
            f"min_distance={float(last['min_distance_m']):.1f}",
            flush=True,
        )

    write_outputs(Path(args.output), rows)
    print(f"[saved] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
