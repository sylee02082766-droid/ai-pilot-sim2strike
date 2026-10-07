# -*- coding: utf-8 -*-
"""Build a lightweight HTML/CSV dashboard from DogFight metrics.jsonl files.

The release environment does not ship with matplotlib, so this script writes a
self-contained HTML file with inline SVG charts. It is intended for experiment
review: loss trends, reward trends, WEZ entries, crash rate, and quick ranking.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DASHBOARD = ROOT / "artifacts" / "dashboard"
DEFAULT_OUTDIR = ROOT / "artifacts" / "reports"


SERIES = [
    ("train/loss/actor", "Actor Loss"),
    ("episode/reward_mean", "Reward Mean"),
    ("dogfight/wez_steps", "WEZ Steps"),
    ("episode/crash_rate", "Crash Rate"),
    ("episode/win_rate", "Win Rate"),
    ("dogfight/distance_min", "Min Distance"),
    ("dogfight/final_ata_deg", "Final ATA"),
    ("action/saturation_rate", "Action Saturation"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard-dir", default=str(DEFAULT_DASHBOARD))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument(
        "--filter",
        default="table1_",
        help="Only include runs whose dashboard folder contains this text. Empty = all.",
    )
    parser.add_argument("--top", type=int, default=40)
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def value_at(row: dict[str, Any], key: str) -> float | None:
    return finite(row.get(key))


def summarize_run(name: str, metrics_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    last = rows[-1] if rows else {}
    best_wez = max((value_at(row, "dogfight/wez_steps") or 0.0 for row in rows), default=0.0)
    best_reward = max(
        (value_at(row, "episode/reward_mean") for row in rows if value_at(row, "episode/reward_mean") is not None),
        default=None,
    )
    min_crash = min(
        (value_at(row, "episode/crash_rate") for row in rows if value_at(row, "episode/crash_rate") is not None),
        default=None,
    )
    best_win = max((value_at(row, "episode/win_rate") or 0.0 for row in rows), default=0.0)
    actor_losses = [
        value_at(row, "train/loss/actor")
        for row in rows
        if value_at(row, "train/loss/actor") is not None
    ]
    return {
        "run": name,
        "metrics_path": str(metrics_path),
        "iterations": len(rows),
        "final_step": value_at(last, "step"),
        "final_actor_loss": value_at(last, "train/loss/actor"),
        "min_actor_loss": min(actor_losses) if actor_losses else None,
        "max_actor_loss": max(actor_losses) if actor_losses else None,
        "final_reward": value_at(last, "episode/reward_mean"),
        "best_reward": best_reward,
        "final_wez_steps": value_at(last, "dogfight/wez_steps"),
        "best_wez_steps": best_wez,
        "final_crash_rate": value_at(last, "episode/crash_rate"),
        "min_crash_rate": min_crash,
        "final_win_rate": value_at(last, "episode/win_rate"),
        "best_win_rate": best_win,
        "final_min_distance": value_at(last, "dogfight/distance_min"),
        "final_ata": value_at(last, "dogfight/final_ata_deg"),
        "final_action_saturation": value_at(last, "action/saturation_rate"),
        "updated": metrics_path.stat().st_mtime,
    }


def discover_runs(dashboard_dir: Path, name_filter: str) -> list[tuple[str, Path, list[dict[str, Any]]]]:
    runs = []
    for metrics_path in sorted(dashboard_dir.rglob("metrics.jsonl")):
        name = metrics_path.parent.name
        if name_filter and name_filter not in name:
            continue
        rows = read_metrics(metrics_path)
        if rows:
            runs.append((name, metrics_path, rows))
    return runs


def score(summary: dict[str, Any]) -> float:
    """Heuristic review score, not a scientific metric."""
    wez = summary.get("best_wez_steps") or 0.0
    reward = summary.get("best_reward") or -10_000.0
    crash = summary.get("final_crash_rate")
    crash_penalty = 0.0 if crash is None else 500.0 * crash
    ata = summary.get("final_ata")
    ata_penalty = 0.0 if ata is None else min(abs(ata), 90.0)
    return 25.0 * wez + 0.01 * reward - crash_penalty - ata_penalty


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "review_score",
        "run",
        "iterations",
        "final_step",
        "final_actor_loss",
        "min_actor_loss",
        "max_actor_loss",
        "final_reward",
        "best_reward",
        "final_wez_steps",
        "best_wez_steps",
        "final_crash_rate",
        "min_crash_rate",
        "final_win_rate",
        "best_win_rate",
        "final_min_distance",
        "final_ata",
        "final_action_saturation",
        "metrics_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(summaries, start=1):
            payload = dict(row)
            payload["rank"] = index
            payload["review_score"] = round(score(row), 6)
            writer.writerow({field: payload.get(field) for field in fields})


def chart_svg(
    rows_by_run: list[tuple[str, list[dict[str, Any]]]],
    key: str,
    title: str,
    width: int = 980,
    height: int = 310,
) -> str:
    margin_l, margin_r, margin_t, margin_b = 62, 20, 34, 44
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    points_by_run = []
    all_y = []
    max_x = 1.0
    for name, rows in rows_by_run:
        points = []
        for idx, row in enumerate(rows, start=1):
            y = value_at(row, key)
            if y is None:
                continue
            x = value_at(row, "step") or float(idx)
            points.append((x, y))
            all_y.append(y)
            max_x = max(max_x, x)
        if points:
            points_by_run.append((name, points))
    if not all_y:
        return f"<h3>{html.escape(title)}</h3><p>No data for {html.escape(key)}.</p>"

    y_min = min(all_y)
    y_max = max(all_y)
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    pad = 0.08 * (y_max - y_min)
    y_min -= pad
    y_max += pad

    def px(x: float) -> float:
        return margin_l + plot_w * (x / max_x)

    def py(y: float) -> float:
        return margin_t + plot_h * (1.0 - (y - y_min) / (y_max - y_min))

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
        "#bcbd22",
        "#7f7f7f",
    ]
    parts = [
        f"<h3>{html.escape(title)}</h3>",
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}' role='img'>",
        "<rect width='100%' height='100%' fill='#101418'/>",
        f"<line x1='{margin_l}' y1='{margin_t}' x2='{margin_l}' y2='{height-margin_b}' stroke='#52616f'/>",
        f"<line x1='{margin_l}' y1='{height-margin_b}' x2='{width-margin_r}' y2='{height-margin_b}' stroke='#52616f'/>",
    ]
    for tick in range(5):
        ratio = tick / 4
        y = y_min + (y_max - y_min) * ratio
        yy = py(y)
        parts.append(
            f"<line x1='{margin_l}' y1='{yy:.1f}' x2='{width-margin_r}' y2='{yy:.1f}' stroke='#26313a'/>"
        )
        parts.append(
            f"<text x='8' y='{yy+4:.1f}' fill='#a9b6c3' font-size='11'>{y:.3g}</text>"
        )
    for idx, (name, points) in enumerate(points_by_run):
        color = colors[idx % len(colors)]
        path_data = " ".join(
            ("M" if i == 0 else "L") + f"{px(x):.1f},{py(y):.1f}"
            for i, (x, y) in enumerate(points)
        )
        parts.append(f"<path d='{path_data}' fill='none' stroke='{color}' stroke-width='2'/>")
        lx = width - margin_r - 300
        ly = 24 + 18 * idx
        if ly < height - margin_b:
            label = html.escape(name[-58:])
            parts.append(f"<rect x='{lx}' y='{ly-9}' width='10' height='10' fill='{color}'/>")
            parts.append(f"<text x='{lx+16}' y='{ly}' fill='#dce5ee' font-size='11'>{label}</text>")
    parts.append(
        f"<text x='{margin_l}' y='22' fill='#f3f6f8' font-size='14' font-weight='700'>{html.escape(title)}</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def fmt(value: Any) -> str:
    number = finite(value)
    if number is None:
        return ""
    if abs(number) >= 1000:
        return f"{number:.0f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.4f}"


def write_html(
    path: Path,
    summaries: list[dict[str, Any]],
    runs: list[tuple[str, Path, list[dict[str, Any]]]],
    top: int,
) -> None:
    selected_names = {row["run"] for row in summaries[:top]}
    selected_runs = [(name, rows) for name, _, rows in runs if name in selected_names]
    selected_runs.sort(key=lambda item: next(i for i, row in enumerate(summaries) if row["run"] == item[0]))

    table_rows = []
    for index, row in enumerate(summaries[:top], start=1):
        table_rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{score(row):.2f}</td>"
            f"<td class='run'>{html.escape(row['run'])}</td>"
            f"<td>{row['iterations']}</td>"
            f"<td>{fmt(row['final_actor_loss'])}</td>"
            f"<td>{fmt(row['best_wez_steps'])}</td>"
            f"<td>{fmt(row['best_reward'])}</td>"
            f"<td>{fmt(row['final_crash_rate'])}</td>"
            f"<td>{fmt(row['final_min_distance'])}</td>"
            f"<td>{fmt(row['final_ata'])}</td>"
            "</tr>"
        )

    charts = "\n".join(chart_svg(selected_runs, key, title) for key, title in SERIES)
    body = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>DogFight Metrics Report</title>
<style>
body {{ margin: 0; background: #0b0f12; color: #e6edf3; font: 14px/1.45 Segoe UI, Arial, sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 26px; }}
h2 {{ margin-top: 34px; }}
h3 {{ margin: 28px 0 8px; color: #e8f1fb; }}
.note {{ color: #a9b6c3; margin-bottom: 22px; }}
table {{ border-collapse: collapse; width: 100%; background: #121820; }}
th, td {{ border-bottom: 1px solid #26313a; padding: 7px 8px; text-align: right; }}
th {{ color: #9ec5ff; background: #17202a; position: sticky; top: 0; }}
td.run, th.run {{ text-align: left; max-width: 520px; overflow-wrap: anywhere; }}
svg {{ border: 1px solid #26313a; border-radius: 6px; background: #101418; }}
code {{ color: #9ec5ff; }}
</style>
</head>
<body>
<main>
<h1>DogFight Metrics Report</h1>
<p class="note">Generated from <code>artifacts/dashboard/**/metrics.jsonl</code>. Review score is a heuristic: WEZ entries and reward help, crash and large final ATA hurt. PPO actor loss is not a supervised-learning convergence metric.</p>
<h2>Top Runs</h2>
<table>
<thead><tr><th>Rank</th><th>Score</th><th class="run">Run</th><th>Iter</th><th>Final Actor Loss</th><th>Best WEZ</th><th>Best Reward</th><th>Final Crash</th><th>Final Min Dist</th><th>Final ATA</th></tr></thead>
<tbody>
{''.join(table_rows)}
</tbody>
</table>
<h2>Trend Charts</h2>
{charts}
</main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dashboard_dir = Path(args.dashboard_dir)
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    runs = discover_runs(dashboard_dir, args.filter)
    summaries = [summarize_run(name, path, rows) for name, path, rows in runs]
    summaries.sort(key=score, reverse=True)
    csv_path = outdir / "codex_metrics_summary.csv"
    html_path = outdir / "codex_metrics_report.html"
    write_csv(csv_path, summaries)
    write_html(html_path, summaries, runs, args.top)
    print(f"runs: {len(summaries)}")
    print(f"csv: {csv_path}")
    print(f"html: {html_path}")
    if summaries:
        print("top:")
        for index, row in enumerate(summaries[:10], start=1):
            print(
                f"{index:02d} score={score(row):8.2f} "
                f"wez={row.get('best_wez_steps')} crash={row.get('final_crash_rate')} "
                f"loss={row.get('final_actor_loss')} run={row['run']}"
            )


if __name__ == "__main__":
    main()
