"""Summarize a BattleServer V1.1 Blue/Red Tacview CSV pair."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


EARTH_RADIUS_M = 6_378_137.0


def _read(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows: list[dict[str, float]] = []
        for row in csv.DictReader(stream):
            if None in row or any(value in (None, "") for value in row.values()):
                continue
            try:
                rows.append({key: float(value) for key, value in row.items()})
            except (TypeError, ValueError):
                continue
        return rows


def _position_ned(rows: list[dict[str, float]], ref_lat: float, ref_lon: float) -> np.ndarray:
    ref_lat_rad = math.radians(ref_lat)
    result = []
    for row in rows:
        north = math.radians(row["Latitude"] - ref_lat) * EARTH_RADIUS_M
        east = (
            math.radians(row["Longitude"] - ref_lon)
            * EARTH_RADIUS_M
            * math.cos(ref_lat_rad)
        )
        result.append((north, east, -row["Altitude"]))
    return np.asarray(result, dtype=np.float64)


def _nose(rows: list[dict[str, float]]) -> np.ndarray:
    yaw = np.radians([row["Yaw (deg)"] for row in rows])
    pitch = np.radians([row["Pitch (deg)"] for row in rows])
    return np.column_stack(
        (np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch))
    )


def _angle_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    numerator = np.sum(first * second, axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    cosine = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-9)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _first_time(mask: np.ndarray, sample_hz: float) -> float | None:
    indices = np.flatnonzero(mask)
    return None if not len(indices) else float(indices[0] / sample_hz)


def _longest_hold(mask: np.ndarray, sample_hz: float) -> float:
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return float(longest / sample_hz)


def _at(values: np.ndarray, seconds: float, sample_hz: float) -> float:
    index = min(len(values) - 1, int(round(seconds * sample_hz)))
    return float(values[index])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blue_csv", type=Path)
    parser.add_argument("red_csv", type=Path)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    blue_rows = _read(args.blue_csv)
    red_rows = _read(args.red_csv)
    count = min(len(blue_rows), len(red_rows))
    if count < 2:
        raise ValueError("Both logs must contain at least two samples.")
    blue_rows = blue_rows[:count]
    red_rows = red_rows[:count]
    ref_lat = blue_rows[0]["Latitude"]
    ref_lon = blue_rows[0]["Longitude"]
    blue_position = _position_ned(blue_rows, ref_lat, ref_lon)
    red_position = _position_ned(red_rows, ref_lat, ref_lon)
    relative = red_position - blue_position
    distance = np.linalg.norm(relative, axis=1)
    blue_ata = _angle_deg(_nose(blue_rows), relative)
    red_ata = _angle_deg(_nose(red_rows), -relative)
    blue_speed = np.linalg.norm(np.gradient(blue_position, 1.0 / args.sample_hz, axis=0), axis=1)
    red_speed = np.linalg.norm(np.gradient(red_position, 1.0 / args.sample_hz, axis=0), axis=1)
    blue_altitude = np.asarray([row["Altitude"] for row in blue_rows])
    red_altitude = np.asarray([row["Altitude"] for row in red_rows])
    time_s = np.arange(count) / args.sample_hz

    def side_summary(name: str, ata: np.ndarray, enemy_ata: np.ndarray, speed: np.ndarray, altitude: np.ndarray) -> dict:
        capture = (ata <= 25.0) & (enemy_ata >= 120.0) & (distance <= 2500.0)
        wez = (ata <= 1.0) & (distance >= 152.4) & (distance <= 914.4)
        return {
            "side": name,
            "initial_ata_deg": float(ata[0]),
            "min_ata_first_10s_deg": float(np.min(ata[time_s <= 10.0])),
            "min_ata_first_30s_deg": float(np.min(ata[time_s <= 30.0])),
            "min_ata_deg": float(np.min(ata)),
            "first_ata_45_s": _first_time(ata <= 45.0, args.sample_hz),
            "first_ata_25_s": _first_time(ata <= 25.0, args.sample_hz),
            "capture_hold_max_s": _longest_hold(capture, args.sample_hz),
            "wez_hold_max_s": _longest_hold(wez, args.sample_hz),
            "min_speed_mps": float(np.min(speed[2:-2])),
            "speed_10s_mps": _at(speed, 10.0, args.sample_hz),
            "speed_30s_mps": _at(speed, 30.0, args.sample_hz),
            "speed_60s_mps": _at(speed, 60.0, args.sample_hz),
            "min_altitude_m": float(np.min(altitude)),
            "final_altitude_m": float(altitude[-1]),
        }

    report = {
        "blue_csv": str(args.blue_csv.resolve()),
        "red_csv": str(args.red_csv.resolve()),
        "samples": count,
        "sample_hz": args.sample_hz,
        "duration_s": float((count - 1) / args.sample_hz),
        "initial_distance_m": float(distance[0]),
        "min_distance_m": float(np.min(distance)),
        "max_distance_m": float(np.max(distance)),
        "distance_10s_m": _at(distance, 10.0, args.sample_hz),
        "distance_30s_m": _at(distance, 30.0, args.sample_hz),
        "distance_60s_m": _at(distance, 60.0, args.sample_hz),
        "blue": side_summary("blue", blue_ata, red_ata, blue_speed, blue_altitude),
        "red": side_summary("red", red_ata, blue_ata, red_speed, red_altitude),
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
