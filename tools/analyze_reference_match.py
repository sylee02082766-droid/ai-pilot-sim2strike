from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


STATUS_RE = re.compile(
    r"\[bt-gun-status\] t=(?P<time>[0-9.]+)s stage=(?P<stage>\S+) "
    r"distance=(?P<distance>[0-9.]+)m own_ata=(?P<own_ata>[0-9.]+) "
    r"target_ata=(?P<target_ata>[0-9.]+) speed=(?P<speed>[0-9.]+)mps "
    r"alt=(?P<altitude>[0-9.]+)m.*?closure=(?P<closure>-?[0-9.]+)mps.*?"
    r"action=\[(?P<action>[^]]+)\]"
)


@dataclass(frozen=True)
class CandidateSample:
    time_s: float
    stage: str
    distance_m: float
    own_ata_deg: float
    target_ata_deg: float
    speed_mps: float
    altitude_m: float
    closure_mps: float
    action: tuple[float, float, float, float]


@dataclass(frozen=True)
class TeacherSample:
    frame: int
    wall_time_ns: int
    baseline_action: tuple[float, float, float, float]
    distance_m: float
    baseline_ata_deg: float
    candidate_ata_deg: float
    baseline_speed_mps: float
    candidate_speed_mps: float
    baseline_altitude_m: float
    candidate_altitude_m: float


def _vector(row: dict[str, str], prefix: str) -> tuple[float, float, float]:
    return (
        float(row[f"{prefix}_x"]),
        float(row[f"{prefix}_y"]),
        -float(row[f"{prefix}_z"]),
    )


def _velocity(row: dict[str, str], prefix: str) -> tuple[float, float, float]:
    return tuple(float(row[f"{prefix}_v{axis}"]) for axis in ("x", "y", "z"))


def _nose(row: dict[str, str], prefix: str) -> tuple[float, float, float]:
    pitch = math.radians(float(row[f"{prefix}_pitch"]))
    yaw = math.radians(float(row[f"{prefix}_yaw"]))
    return (
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        -math.sin(pitch),
    )


def _subtract(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(left - right for left, right in zip(a, b))


def _norm(vector: tuple[float, ...]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _angle(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    denominator = max(_norm(first) * _norm(second), 1.0e-9)
    cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(first, second)) / denominator))
    return math.degrees(math.acos(cosine))


def load_candidate(path: Path) -> list[CandidateSample]:
    samples: list[CandidateSample] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = STATUS_RE.search(line)
        if not match:
            continue
        action_values = tuple(float(value.strip()) for value in match["action"].split(","))
        if len(action_values) != 4:
            continue
        samples.append(
            CandidateSample(
                time_s=float(match["time"]),
                stage=match["stage"],
                distance_m=float(match["distance"]),
                own_ata_deg=float(match["own_ata"]),
                target_ata_deg=float(match["target_ata"]),
                speed_mps=float(match["speed"]),
                altitude_m=float(match["altitude"]),
                closure_mps=float(match["closure"]),
                action=action_values,
            )
        )
    if not samples:
        raise ValueError(f"No [bt-gun-status] samples found in {path}")
    return samples


def load_teacher(path: Path) -> list[TeacherSample]:
    samples: list[TeacherSample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            own_position = _vector(row, "own")
            target_position = _vector(row, "target")
            relative = _subtract(target_position, own_position)
            samples.append(
                TeacherSample(
                    frame=int(row["index"]),
                    wall_time_ns=int(row["wall_time_ns"]),
                    baseline_action=tuple(
                        float(row[name])
                        for name in ("roll_cmd", "pitch_cmd", "rudder_cmd", "throttle_cmd")
                    ),
                    distance_m=_norm(relative),
                    baseline_ata_deg=_angle(_nose(row, "own"), relative),
                    candidate_ata_deg=_angle(_nose(row, "target"), tuple(-value for value in relative)),
                    baseline_speed_mps=_norm(_velocity(row, "own")),
                    candidate_speed_mps=_norm(_velocity(row, "target")),
                    baseline_altitude_m=float(row["own_z"]),
                    candidate_altitude_m=float(row["target_z"]),
                )
            )
    if not samples:
        raise ValueError(f"No teacher samples found in {path}")
    return samples


def find_frame_offset(
    candidate: list[CandidateSample], teacher: list[TeacherSample]
) -> int:
    by_frame = {sample.frame: sample for sample in teacher}
    probes = [sample for sample in candidate if sample.time_s <= 30.0]
    best_offset = 0
    best_score = float("inf")
    for offset in range(-180, 181):
        score = 0.0
        count = 0
        for sample in probes:
            teacher_sample = by_frame.get(round(sample.time_s * 60.0) + offset)
            if teacher_sample is None:
                continue
            score += abs(sample.distance_m - teacher_sample.distance_m) / 100.0
            score += abs(sample.speed_mps - teacher_sample.candidate_speed_mps) / 10.0
            score += abs(sample.altitude_m - teacher_sample.candidate_altitude_m) / 100.0
            count += 1
        if count and score / count < best_score:
            best_score = score / count
            best_offset = offset
    return best_offset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align candidate debug output with the captured baseline packet stream."
    )
    parser.add_argument("candidate_log", type=Path)
    parser.add_argument("teacher_csv", type=Path)
    parser.add_argument("--start", type=float, default=80.0)
    parser.add_argument("--end", type=float, default=120.0)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    candidate = load_candidate(args.candidate_log)
    teacher = load_teacher(args.teacher_csv)
    by_frame = {sample.frame: sample for sample in teacher}
    offset = find_frame_offset(candidate, teacher)

    aligned: list[tuple[CandidateSample, TeacherSample]] = []
    for sample in candidate:
        teacher_sample = by_frame.get(round(sample.time_s * 60.0) + offset)
        if teacher_sample is not None:
            aligned.append((sample, teacher_sample))

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "time_s", "stage", "distance_m", "candidate_ata_deg",
                    "baseline_ata_deg", "candidate_speed_mps", "baseline_speed_mps",
                    "candidate_altitude_m", "baseline_altitude_m", "candidate_closure_mps",
                    "candidate_roll", "candidate_pitch", "candidate_rudder", "candidate_throttle",
                    "baseline_roll", "baseline_pitch", "baseline_rudder", "baseline_throttle",
                ]
            )
            for own, baseline in aligned:
                writer.writerow(
                    [
                        own.time_s, own.stage, baseline.distance_m,
                        baseline.candidate_ata_deg, baseline.baseline_ata_deg,
                        baseline.candidate_speed_mps, baseline.baseline_speed_mps,
                        baseline.candidate_altitude_m, baseline.baseline_altitude_m,
                        own.closure_mps, *own.action, *baseline.baseline_action,
                    ]
                )

    threat = [
        pair for pair in aligned
        if pair[1].distance_m <= 1500.0
        and pair[1].candidate_ata_deg >= 60.0
        and pair[1].baseline_ata_deg <= 25.0
    ]
    opportunity = [
        pair for pair in aligned
        if pair[1].distance_m <= 1500.0
        and pair[1].candidate_ata_deg <= 25.0
        and pair[1].baseline_ata_deg >= 90.0
        and pair[0].closure_mps <= 180.0
    ]
    moving = [pair for pair in aligned if pair[0].time_s >= 1.0]
    minimum_speed = min(moving, key=lambda pair: pair[1].candidate_speed_mps)
    minimum_altitude = min(aligned, key=lambda pair: pair[1].candidate_altitude_m)

    print(f"frame_offset={offset}")
    print(f"candidate_samples={len(candidate)} aligned_samples={len(aligned)}")
    print(
        "threat_samples=" + str(len(threat))
        + (f" first={threat[0][0].time_s:.1f}s" if threat else "")
    )
    print(
        "opportunity_samples=" + str(len(opportunity))
        + (f" first={opportunity[0][0].time_s:.1f}s" if opportunity else "")
    )
    print(
        f"minimum_speed={minimum_speed[1].candidate_speed_mps:.1f}mps "
        f"at={minimum_speed[0].time_s:.1f}s"
    )
    print(
        f"minimum_altitude={minimum_altitude[1].candidate_altitude_m:.0f}m "
        f"at={minimum_altitude[0].time_s:.1f}s"
    )
    print("time  dist  candATA baseATA candV baseV candAlt baseAlt candAction baseAction")
    in_window = [
        pair for pair in aligned if args.start <= pair[0].time_s <= args.end
    ]
    for own, baseline in in_window[:: max(1, args.stride)]:
        own_action = "/".join(f"{value:+.2f}" for value in own.action)
        baseline_action = "/".join(f"{value:+.2f}" for value in baseline.baseline_action)
        print(
            f"{own.time_s:5.1f} {baseline.distance_m:5.0f} "
            f"{baseline.candidate_ata_deg:7.1f} {baseline.baseline_ata_deg:7.1f} "
            f"{baseline.candidate_speed_mps:5.0f} {baseline.baseline_speed_mps:5.0f} "
            f"{baseline.candidate_altitude_m:7.0f} {baseline.baseline_altitude_m:7.0f} "
            f"{own_action} {baseline_action}"
        )


if __name__ == "__main__":
    main()
