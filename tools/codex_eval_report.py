#!/usr/bin/env python
"""Build a compact report from seeded evaluation summary JSON files."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


FIELDS = [
    "name",
    "win_rate",
    "wins",
    "episodes",
    "timeouts",
    "losses",
    "avg_steps",
    "avg_target_health",
    "avg_ownship_health",
    "avg_wez_steps",
    "avg_min_distance",
    "avg_min_ata_deg",
    "avg_best_wez_score",
    "bundle_dir",
    "experiment_yaml",
]


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def load_rows(eval_dir: Path, name_filter: str | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(eval_dir.glob("*_summary.json")):
        if name_filter and name_filter not in path.name:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[skip] {path}: {exc}")
            continue
        row = {field: data.get(field) for field in FIELDS}
        row["name"] = path.stem.replace("_summary", "")
        rows.append(row)

    def sort_key(row: dict[str, object]) -> tuple[float, float, float, float]:
        win = float(row.get("win_rate") or 0.0)
        timeouts = float(row.get("timeouts") or 0.0)
        health = float(row.get("avg_target_health") or 999.0)
        ata = float(row.get("avg_min_ata_deg") or 999.0)
        return (-win, timeouts, health, ata)

    rows.sort(key=sort_key)
    return rows


def write_csv(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_html(rows: list[dict[str, object]], out_html: Path) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    trs = []
    for row in rows:
        win_rate = float(row.get("win_rate") or 0.0)
        health = float(row.get("avg_target_health") or 0.0)
        cls = "good" if win_rate >= 1.0 and health <= 0 else "warn" if win_rate > 0 else "bad"
        cells = []
        for field in FIELDS:
            text = html.escape(_fmt(row.get(field)))
            cells.append(f"<td>{text}</td>")
        trs.append(f"<tr class='{cls}'>" + "".join(cells) + "</tr>")

    headers = "".join(f"<th>{html.escape(field)}</th>" for field in FIELDS)
    body = "\n".join(trs)
    out_html.write_text(
        f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Codex Evaluation Report</title>
<style>
body {{ margin: 24px; background: #101418; color: #dce7ef; font-family: Segoe UI, Arial, sans-serif; }}
h1 {{ font-size: 24px; margin: 0 0 8px; }}
p {{ color: #9fb0bd; max-width: 980px; line-height: 1.5; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #2a3540; padding: 8px 10px; text-align: left; vertical-align: top; }}
th {{ position: sticky; top: 0; background: #182029; color: #cfe6f5; }}
tr.good td:first-child {{ border-left: 4px solid #35c46f; }}
tr.warn td:first-child {{ border-left: 4px solid #e2b84a; }}
tr.bad td:first-child {{ border-left: 4px solid #d95757; }}
code {{ color: #9ed1ff; }}
</style>
</head>
<body>
<h1>Codex Evaluation Report</h1>
<p>
Seeded evaluation summary만 모은 표입니다. PPO loss는 정책 업데이트의 진단값일 뿐이고,
실제 모델 선택은 <code>win_rate</code>, <code>avg_target_health</code>,
<code>timeouts</code>, <code>avg_min_ata_deg</code>, replay를 같이 봐야 합니다.
</p>
<table>
<thead><tr>{headers}</tr></thead>
<tbody>
{body}
</tbody>
</table>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="artifacts/evals")
    parser.add_argument("--filter", default="codex_")
    parser.add_argument("--outdir", default="artifacts/reports/codex_only")
    args = parser.parse_args()

    root = Path.cwd()
    eval_dir = (root / args.eval_dir).resolve()
    outdir = (root / args.outdir).resolve()
    rows = load_rows(eval_dir, args.filter or None)
    out_csv = outdir / "codex_eval_summary.csv"
    out_html = outdir / "codex_eval_report.html"
    write_csv(rows, out_csv)
    write_html(rows, out_html)

    print(f"rows: {len(rows)}")
    print(f"csv: {out_csv}")
    print(f"html: {out_html}")
    for idx, row in enumerate(rows[:8], start=1):
        print(
            f"{idx:02d} win={_fmt(row.get('win_rate'))} "
            f"health={_fmt(row.get('avg_target_health'))} "
            f"ata={_fmt(row.get('avg_min_ata_deg'))} "
            f"name={row.get('name')}"
        )


if __name__ == "__main__":
    main()
