"""Train a frozen selector experiment and compare policies on untouched selector holdout groups."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from stemscope.benchmarking.datasets import load_prepared_tracks
from stemscope.benchmarking.study import MODELS
from stemscope.benchmarking.summary import aggregate
from stemscope.errors import StemScopeError
from stemscope.selection.features import FEATURE_NAMES, PROTOCOL, extract
from stemscope.selection.inference import predict_forest


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def utility(quality: np.ndarray, times: np.ndarray, penalty: float) -> np.ndarray:
    """Engineering preference: mean four-stem SI-SDR minus seconds times dB/second penalty."""
    return quality - penalty * times


def evaluate(values: np.ndarray, choices: np.ndarray) -> dict:
    realized = values[np.arange(len(values)), choices]
    oracle = values.max(axis=1)
    return {
        "tracks": len(values),
        "mean_utility": float(realized.mean()),
        "mean_regret": float((oracle - realized).mean()),
        "oracle_agreement": float(np.isclose(realized, oracle, atol=1e-8, rtol=0).mean()),
        "demucs_choices": int((choices == 0).sum()),
        "openunmix_choices": int((choices == 1).sum()),
    }


def train(study: Path, output: Path) -> Path:
    plan_path = output / "plan.json"
    plan = json.loads(plan_path.read_text())
    dataset = Path(plan["dataset"])
    if (
        hashlib.sha256((dataset / "provenance.json").read_bytes()).hexdigest()
        != plan["dataset_sha256"]
    ):
        raise StemScopeError("Dataset provenance differs from the frozen selector plan.")
    protocol = json.loads((study / "study.json").read_text())
    if (
        protocol["failures"]
        or protocol["models_in_order"] != list(MODELS)
        or [r["model"] for r in protocol["worker_results"]] != list(MODELS)
    ):
        raise StemScopeError("A complete two-model study is required.")
    if protocol["source_provenance"].get("split") != "train":
        raise StemScopeError("The explored Phase 4 test set cannot train this selector.")
    if (
        protocol["repeats"] != plan["benchmark"]["repeats"]
        or protocol["threads"] != plan["benchmark"]["threads"]
    ):
        raise StemScopeError("Benchmark settings differ from the frozen plan.")
    for worker in protocol["worker_results"]:
        if worker["manifest"]["devices"] != {worker["model"]: plan["benchmark"]["device"]}:
            raise StemScopeError("Benchmark device differs from the frozen plan.")
    tracks = load_prepared_tracks(dataset)
    names = [t.name for t in tracks]
    if set(names) != set(plan["partitions"]) or names != protocol["expected_tracks"]:
        raise StemScopeError("Study tracks differ from the frozen selector plan.")
    track_map = {t.name: t for t in tracks}
    for worker in protocol["worker_results"]:
        records = worker["manifest"]["tracks"]
        if [r["name"] for r in records] != names:
            raise StemScopeError("Worker track coverage differs.")
        for record in records:
            track = track_map[record["name"]]
            for stem, info in dict(record["references"], mixture=record["mixture"]).items():
                path = track.mixture if stem == "mixture" else track.references[stem]
                if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
                    raise StemScopeError("Worker input hashes differ from selector data.")
    with (study / "results.csv").open() as stream:
        raw = list(csv.DictReader(stream))
    result = aggregate(raw, list(MODELS), names, protocol["repeats"])
    lookup = {(r["track"], r["model"]): [] for r in result["per_track"]}
    for row in result["per_track"]:
        lookup[row["track"], row["model"]].append(row["si_sdr_median_db"])
    kept, excluded, features, quality, timing = [], [], [], [], []
    for track in tracks:
        scores = [lookup[track.name, model] for model in MODELS]
        if any(len(s) != 4 or any(v is None for v in s) for s in scores):
            excluded.append(
                {"track": track.name, "reason": "missing reference metric; excluded before fitting"}
            )
            continue
        times = [
            float(
                np.median(
                    [
                        float(r["wall_seconds"])
                        for r in raw
                        if r["track"] == track.name
                        and r["model"] == model
                        and r["stem"] == "vocals"
                    ]
                )
            )
            for model in MODELS
        ]
        features.append(extract(track.mixture))
        quality.append([float(np.mean(s)) for s in scores])
        timing.append(times)
        kept.append(track.name)
        print(f"Selector features {len(kept)}/{len(tracks)}", flush=True)
    x, q, times = np.asarray(features), np.asarray(quality), np.asarray(timing)
    partitions = np.array([plan["partitions"][name]["split"] for name in kept])
    masks = {split: partitions == split for split in ("train", "validation", "test")}
    if masks["train"].sum() < 20 or masks["validation"].sum() < 5 or masks["test"].sum() < 5:
        raise StemScopeError("Insufficient complete labels in the frozen partitions.")
    groups = {
        split: {
            plan["partitions"][n]["group"] for n, part in zip(kept, partitions) if part == split
        }
        for split in masks
    }
    if any(
        groups[a] & groups[b]
        for a, b in [("train", "test"), ("train", "validation"), ("test", "validation")]
    ):
        raise StemScopeError("Artist groups overlap across partitions.")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_x = imputer.fit_transform(x[masks["train"]])
    forest = RandomForestRegressor(**plan["forest"])
    forest.fit(train_x, (q[:, 0] - q[:, 1])[masks["train"]])
    costs = np.median(times[masks["train"]], axis=0)
    artifact = {
        "version": 1,
        "models": list(MODELS),
        "feature_names": FEATURE_NAMES,
        "feature_protocol": PROTOCOL,
        "imputation": imputer.statistics_.tolist(),
        "training_runtime_seconds": costs.tolist(),
        "policies": {},
        "trees": [
            {
                "left": tree.tree_.children_left.tolist(),
                "right": tree.tree_.children_right.tolist(),
                "feature": tree.tree_.feature.tolist(),
                "threshold": tree.tree_.threshold.tolist(),
                "value": tree.tree_.value[:, 0, 0].tolist(),
            }
            for tree in forest.estimators_
        ],
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "study_sha256": hashlib.sha256((study / "study.json").read_bytes()).hexdigest(),
        "results_sha256": hashlib.sha256((study / "results.csv").read_bytes()).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("scikit-learn", "numpy", "scipy", "librosa")
        },
    }
    validation_prediction = forest.predict(imputer.transform(x[masks["validation"]]))
    for preference, penalty in plan["preferences"].items():
        train_values = utility(q[masks["train"]], times[masks["train"]], penalty)
        baseline = int(np.argmax(train_values.mean(axis=0)))
        choices = (validation_prediction < penalty * (costs[0] - costs[1])).astype(int)
        values = utility(q[masks["validation"]], times[masks["validation"]], penalty)
        learned = evaluate(values, choices)
        constant = evaluate(values, np.full(len(values), baseline))
        gain = learned["mean_utility"] - constant["mean_utility"]
        artifact["policies"][preference] = {
            "penalty": penalty,
            "baseline_index": baseline,
            "validation_gain": gain,
            "use_learned": gain >= plan["validation_min_gain"],
        }
    # Save frozen policy BEFORE evaluating test labels. No refitting or promotion afterward.
    artifact_path = output / "frozen-policy.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, allow_nan=False))
    prediction = np.array([predict_forest(artifact, values) for values in x])
    if not np.allclose(prediction, forest.predict(imputer.transform(x)), atol=1e-9, rtol=0):
        raise StemScopeError("Exported selector differs from the fitted forest.")
    results, detailed = [], []
    for split in ("validation", "test"):
        mask = masks[split]
        for preference, policy in artifact["policies"].items():
            values = utility(q[mask], times[mask], policy["penalty"])
            learned = (prediction[mask] < policy["penalty"] * (costs[0] - costs[1])).astype(int)
            constant = np.full(len(values), policy["baseline_index"])
            options = {
                "learned_candidate": learned,
                "frozen_policy": learned if policy["use_learned"] else constant,
                "training_constant": constant,
                "always_demucs": np.zeros(len(values), dtype=int),
                "always_openunmix": np.ones(len(values), dtype=int),
                "oracle_upper_bound": np.argmax(values, axis=1),
            }
            for label, choices in options.items():
                results.append(
                    {
                        "split": split,
                        "preference": preference,
                        "policy": label,
                        **evaluate(values, choices),
                    }
                )
                for name, row, choice in zip(np.array(kept)[mask], values, choices):
                    detailed.append(
                        {
                            "track": name,
                            "split": split,
                            "preference": preference,
                            "policy": label,
                            "selected_model": MODELS[choice],
                            "utility": float(row[choice]),
                            "regret": float(row.max() - row[choice]),
                        }
                    )
    write_csv(output / "evaluation.csv", results)
    write_csv(output / "predictions.csv", detailed)
    feature_rows = [
        {
            "track": name,
            **plan["partitions"][name],
            **{
                key: None if np.isnan(value) else value for key, value in zip(FEATURE_NAMES, vector)
            },
        }
        for name, vector in zip(kept, features)
    ]
    write_csv(output / "features.csv", feature_rows)
    (output / "exclusions.json").write_text(json.dumps(excluded, indent=2))
    counts = {split: int(mask.sum()) for split, mask in masks.items()}
    lines = [
        "# Phase 6 — experimental model selector",
        "",
        f"Selector partitions: {counts}. Excluded tracks: {len(excluded)}. Name-derived artist groups are disjoint. The explored 50-track Phase 4 corpus is excluded.",
        "",
        "## Method",
        "",
        "A fixed 100-tree random-forest regressor predicts Demucs minus Open-Unmix mean four-stem SI-SDR from 11 mixture-only features. Features use the first seven seconds; median imputation and the forest are fitted on training tracks only. No reference energies, genre, artist, names or benchmark scores enter the input features. The JSON forest is verified against scikit-learn prediction; no executable model serialization is loaded.",
        "",
        "Quality, Balanced and Speed maximize mean four-stem SI-SDR minus 0, 0.5 and 2 dB per measured second respectively. These are engineering preferences, not calibrated perceptual utilities. Predicted choices use training-only median CPU runtime costs. Evaluation uses each held-out call's actual runtime, retaining timing outliers. Feature-extraction overhead is not included in benchmark utility; the app includes it in elapsed time.",
        "",
        "Each preference uses the learned model only if validation mean utility beats the training-selected constant baseline by at least 0.1. Otherwise it falls back to that constant. Hyperparameters, grouping seed and thresholds were frozen before benchmarking. No refit or test-driven promotion was performed.",
        "",
        "## Frozen policy",
        "",
        "| Preference | Learned enabled | Validation gain | Fallback model |",
        "|---|---|---:|---|",
    ]
    for preference, policy in artifact["policies"].items():
        lines.append(
            f"| {preference} | {policy['use_learned']} | {policy['validation_gain']:.3f} | {MODELS[policy['baseline_index']]} |"
        )
    lines += [
        "",
        "## Selector holdout evaluation",
        "",
        "Higher utility is better; lower regret is better. Oracle uses the true outcomes and is an unattainable upper bound, not an implemented selector. Agreement is agreement with this utility oracle, not audio accuracy.",
        "",
        "| Preference | Policy | Tracks | Mean utility | Mean regret | Oracle agreement |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in results:
        if row["split"] == "test":
            lines.append(
                f"| {row['preference']} | {row['policy']} | {row['tracks']} | {row['mean_utility']:.3f} | {row['mean_regret']:.3f} | {row['oracle_agreement']:.3f} |"
            )
    lines += [
        "",
        "## Limits",
        "",
        "The public ZIP contains 94 training mixtures, not the full 100-track corpus. Two archive/metadata spelling aliases are recorded during preparation. Artist identity is inferred from names, not independently audited. These official training previews may have been seen during separator pretraining: the held-out split evaluates the selector, not unseen-song separator generalization. Seven-second results and runtime costs do not validate full songs, other machines or unseen genres. This small exploratory experiment is not evidence that ML necessarily beats a fixed separator. Test results must not be reused for further tuning; reserve new evaluation data for future iterations.",
        "",
        "Method reference: [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).",
        "",
        "Artifacts: plan.json (frozen split/config), selector.json (fitted forest/policy), features.csv, evaluation.csv (all baselines), predictions.csv (per-track evidence), exclusions.json. Input/protocol hashes and package versions are recorded. No Phase 7 quality estimator is included.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    final_path = output / "selector.json"
    final_path.write_bytes(artifact_path.read_bytes())
    return final_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/phase6-selector"))
    args = parser.parse_args()
    try:
        print(f"Trained selector: {train(args.study, args.output)}")
    except (StemScopeError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"Selector training failed: {exc}\n")


if __name__ == "__main__":
    main()
