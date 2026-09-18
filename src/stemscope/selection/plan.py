"""Freeze artist-group partitions and selector settings before reading quality labels."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from stemscope.errors import StemScopeError

PREFERENCES = {"Quality": 0.0, "Balanced": 0.5, "Speed": 2.0}
FOREST = {
    "n_estimators": 100,
    "max_depth": 4,
    "min_samples_leaf": 4,
    "random_state": 6,
    "n_jobs": 1,
}


def artist_group(name: str) -> str:
    """Conservative name-based artist grouping; not independently verified artist identity."""
    artist = name.split(" - ", 1)[0].casefold()
    artist = re.split(r"\s+(?:feat\.?|ft\.?)\s+|\s*\(", artist)[0]
    return re.sub(r"[^a-z0-9]", "", artist)


def partition(names: list[str]) -> dict:
    groups = sorted({artist_group(name) for name in names})
    if len(groups) < 15 or len(set(names)) != len(names):
        raise StemScopeError(
            "Selector development needs at least 15 artist groups and unique tracks."
        )
    order = np.random.default_rng(6).permutation(groups).tolist()
    count = max(1, round(len(groups) * 0.2))
    split = {
        g: "test" if i < count else "validation" if i < 2 * count else "train"
        for i, g in enumerate(order)
    }
    return {
        name: {"group": artist_group(name), "split": split[artist_group(name)]}
        for name in sorted(names)
    }


def prepare(dataset: Path, output: Path) -> Path:
    metadata = json.loads((dataset / "provenance.json").read_text())
    if metadata.get("split") != "train":
        raise StemScopeError(
            "Use fresh official training previews, not the explored Phase 4 test set."
        )
    names = metadata["planned_tracks"]
    if sorted(names) != sorted(r["Track Name"] for r in metadata["tracks"]):
        raise StemScopeError("Finish downloading the selector dataset first.")
    plan = {
        "version": 1,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": hashlib.sha256((dataset / "provenance.json").read_bytes()).hexdigest(),
        "partitions": partition(names),
        "preferences": PREFERENCES,
        "forest": FOREST,
        "benchmark": {"repeats": 1, "threads": 4, "device": "cpu"},
        "validation_min_gain": 0.1,
        "target": "mean four-stem SI-SDR difference Demucs minus Open-Unmix; runtime costs are training-only median seconds per 7s excerpt",
        "policy": "Fit on train only. Gate each preference on validation mean utility gain >=0.1 vs training-selected constant baseline. Freeze before test. No tuning, refitting or test-driven promotion.",
        "limitations": "Artist groups inferred from names; official train material may overlap separator pretraining. Holdout independence concerns the selector only. The previously explored official test set is excluded.",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise StemScopeError("A different frozen plan exists; choose a new output directory.")
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/phase6-selector"))
    args = parser.parse_args()
    try:
        print(prepare(args.dataset, args.output))
    except (StemScopeError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Cannot freeze selector plan: {exc}\n")


if __name__ == "__main__":
    main()
