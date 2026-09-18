"""Read completed study artifacts without running inference again."""

import csv
from pathlib import Path

from stemscope.errors import StemScopeError


def completed_study(root: Path):
    candidates = sorted(
        root.glob("study-*/report.md"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for report in candidates:
        directory = report.parent
        required = [
            directory / name
            for name in ("summary.csv", "results.csv", "study.json", "quality-distributions.png")
        ]
        if not all(path.is_file() for path in required):
            continue
        with required[0].open() as stream:
            rows = list(csv.DictReader(stream))
        table = [
            [
                (
                    r[key]
                    if key in {"model", "stem", "n"}
                    else f"{float(r[key]):.2f}"
                    if r[key]
                    else "—"
                )
                for key in ("model", "stem", "n", "median", "q25", "q75", "median_improvement_db")
            ]
            for r in rows
        ]
        return (
            f"Completed study · {directory.name}",
            table,
            str(required[3]),
            str(report),
            str(required[1]),
        )
    raise StemScopeError(
        "No completed study report found. Run the study and summary commands in the README first."
    )
