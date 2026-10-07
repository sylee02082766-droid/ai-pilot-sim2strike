from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


STATUS_RE = re.compile(
    r"t=(?P<t>[0-9.]+)s.*?own_ata=(?P<own_ata>[0-9.]+).*?"
    r"target_ata=(?P<target_ata>[0-9.]+).*?own_hp=(?P<own_hp>[0-9.]+).*?"
    r"target_hp=(?P<target_hp>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one FDSA self-play match.")
    parser.add_argument("--state-csv", type=Path, required=True)
    parser.add_argument("--blue-out", type=Path, required=True)
    parser.add_argument("--red-out", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def vector_length(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def read_status(path: Path) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {
        "samples": 0,
        "final_time_s": None,
        "final_own_hp": None,
        "final_target_hp": None,
        "min_own_ata_deg": None,
        "gun_geometry_samples": 0,
    }
    if not path.is_file():
        return result
    minimum_ata: float | None = None
    last_match: re.Match[str] | None = None
    gun_samples = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STATUS_RE.search(line)
        if not match:
            continue
        last_match = match
        own_ata = float(match.group("own_ata"))
        target_ata = float(match.group("target_ata"))
        minimum_ata = own_ata if minimum_ata is None else min(minimum_ata, own_ata)
        if own_ata <= 3.0 and target_ata >= 90.0:
            gun_samples += 1
        result["samples"] = int(result["samples"] or 0) + 1
    if last_match:
        result.update(
            final_time_s=float(last_match.group("t")),
            final_own_hp=float(last_match.group("own_hp")),
            final_target_hp=float(last_match.group("target_hp")),
            min_own_ata_deg=minimum_ata,
            gun_geometry_samples=gun_samples,
        )
    return result


def main() -> int:
    args = parse_args()
    rows: list[dict[str, str]] = []
    with args.state_csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"No rows in {args.state_csv}")

    frames: list[int] = []
    distances: list[float] = []
    blue_altitudes: list[float] = []
    red_altitudes: list[float] = []
    blue_speeds: list[float] = []
    red_speeds: list[float] = []
    for row in rows:
        frames.append(int(row["index"]))
        dx = float(row["target_x"]) - float(row["own_x"])
        dy = float(row["target_y"]) - float(row["own_y"])
        dz = float(row["target_z"]) - float(row["own_z"])
        distances.append(vector_length(dx, dy, dz))
        blue_altitudes.append(float(row["own_z"]))
        red_altitudes.append(float(row["target_z"]))
        blue_speeds.append(
            vector_length(float(row["own_vx"]), float(row["own_vy"]), float(row["own_vz"]))
        )
        red_speeds.append(
            vector_length(
                float(row["target_vx"]),
                float(row["target_vy"]),
                float(row["target_vz"]),
            )
        )

    blue_status = read_status(args.blue_out)
    red_status = read_status(args.red_out)
    initial_blue_alt = blue_altitudes[min(1, len(blue_altitudes) - 1)]
    initial_red_alt = red_altitudes[min(1, len(red_altitudes) - 1)]
    summary = {
        "rows": len(rows),
        "final_frame": max(frames),
        "duration_s": max(frames) / 60.0,
        "start_distance_m": distances[min(1, len(distances) - 1)],
        "min_distance_m": min(distances),
        "final_distance_m": distances[-1],
        "blue_initial_altitude_m": initial_blue_alt,
        "blue_min_altitude_m": min(blue_altitudes),
        "blue_max_altitude_m": max(blue_altitudes),
        "blue_max_climb_m": max(blue_altitudes) - initial_blue_alt,
        "blue_min_speed_mps": min(blue_speeds[1:] or blue_speeds),
        "blue_max_speed_mps": max(blue_speeds),
        "red_initial_altitude_m": initial_red_alt,
        "red_min_altitude_m": min(red_altitudes),
        "red_max_altitude_m": max(red_altitudes),
        "red_max_climb_m": max(red_altitudes) - initial_red_alt,
        "red_min_speed_mps": min(red_speeds[1:] or red_speeds),
        "red_max_speed_mps": max(red_speeds),
        "blue_deck_violation": min(blue_altitudes) < 300.0,
        "red_deck_violation": min(red_altitudes) < 300.0,
        "blue_overclimb": max(blue_altitudes) - initial_blue_alt > 3000.0,
        "red_overclimb": max(red_altitudes) - initial_red_alt > 3000.0,
        "blue_status": blue_status,
        "red_status": red_status,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
