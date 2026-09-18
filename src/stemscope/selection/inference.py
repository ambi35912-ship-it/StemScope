"""Predict from a plain JSON forest; do not deserialize executable model files."""

import json
from pathlib import Path

import numpy as np

from stemscope.errors import StemScopeError
from stemscope.selection.features import FEATURE_NAMES, PROTOCOL, extract

AUTO_CHOICES = ["Auto: Quality", "Auto: Balanced", "Auto: Speed"]


def predict_forest(artifact: dict, features: list[float]) -> float:
    """Evaluate exported regression trees using the training imputation medians."""
    if artifact["feature_names"] != FEATURE_NAMES or artifact["feature_protocol"] != PROTOCOL:
        raise StemScopeError("Selector feature version mismatch. Retrain the selector.")
    x = np.asarray(features, dtype=np.float32)
    if x.shape != (len(FEATURE_NAMES),) or np.isinf(x).any():
        raise StemScopeError("Invalid selector features.")
    x = np.where(np.isnan(x), artifact["imputation"], x).astype(np.float32)
    values = []
    for tree in artifact["trees"]:
        node = 0
        for _ in range(len(tree["left"])):
            left = tree["left"][node]
            if left == -1:
                values.append(tree["value"][node])
                break
            node = (
                left if x[tree["feature"][node]] <= tree["threshold"][node] else tree["right"][node]
            )
        else:
            raise StemScopeError("Invalid selector tree.")
    if not values:
        raise StemScopeError("The selector has no fitted trees.")
    return float(np.mean(values))


def recommend(path: Path, artifact_path: Path, preference: str) -> tuple[str, str]:
    """Apply a validation-selected policy using mixture-only features when required."""
    try:
        artifact = json.loads(artifact_path.read_text())
        policy = artifact["policies"][preference]
        if policy["use_learned"]:
            delta = predict_forest(artifact, extract(path))
            difference = delta - policy["penalty"] * (
                artifact["training_runtime_seconds"][0] - artifact["training_runtime_seconds"][1]
            )
            index = 0 if difference >= 0 else 1
            reason = "Experimental learned selection from the first seven seconds"
        else:
            index = policy["baseline_index"]
            reason = (
                "Validation fallback: the learned selector did not improve on this fixed choice"
            )
        return artifact["models"][
            index
        ], f"{preference}: {reason}. Short-excerpt CPU evidence; full-song quality is unvalidated."
    except (OSError, KeyError, ValueError, IndexError, TypeError) as exc:
        raise StemScopeError(
            "A compatible trained selector is unavailable. Run Phase 6 training or choose a model manually."
        ) from exc


def load_evaluation(directory: Path):
    """Read the saved selector holdout comparison for the app."""
    import csv

    try:
        with (directory / "evaluation.csv").open() as stream:
            rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
        table = [
            [
                row[key]
                for key in (
                    "preference",
                    "policy",
                    "tracks",
                    "mean_utility",
                    "mean_regret",
                    "oracle_agreement",
                )
            ]
            for row in rows
        ]
        return table, str(directory / "report.md"), str(directory / "evaluation.csv")
    except (OSError, ValueError, KeyError) as exc:
        raise StemScopeError("Complete Phase 6 training before loading its evaluation.") from exc
