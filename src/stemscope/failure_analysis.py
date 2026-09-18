"""Exploratory, reference-assisted failure analysis of an existing benchmark."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from stemscope.audio.features import analyze_samples
from stemscope.audio.loader import load_audio
from stemscope.benchmarking.datasets import load_prepared_tracks
from stemscope.benchmarking.metrics import align_estimate, si_sdr
from stemscope.benchmarking.study import MODELS
from stemscope.benchmarking.summary import aggregate, distribution
from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS

FEATURES = (
    "rms",
    "centroid_hz",
    "bandwidth_hz",
    "zcr",
    "vocal_energy_share",
    "drum_energy_share",
    "active_source_mean",
)


def read_csv(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference_features(references: dict[str, np.ndarray], rate: int) -> dict:
    """Energy shares exclude cross terms; activity counts four source classes, not instruments."""
    energies = {
        stem: float(np.mean(samples.astype(float) ** 2)) for stem, samples in references.items()
    }
    total = sum(energies.values())
    block = max(1, round(rate * 0.1))
    activity = []
    for samples in references.values():
        power = np.mean(samples.astype(float) ** 2, axis=1)
        envelope = np.array([power[i : i + block].mean() for i in range(0, len(power), block)])
        peak = float(envelope.max())
        activity.append(
            (envelope > peak * 0.001) if peak > 1e-20 else np.zeros(len(envelope), dtype=bool)
        )
    return {
        "vocal_energy_share": energies["vocals"] / total if total > 1e-20 else None,
        "drum_energy_share": energies["drums"] / total if total > 1e-20 else None,
        "active_source_mean": float(np.sum(activity, axis=0).mean()),
    }


def conditions(features: list[dict]) -> tuple[dict, dict]:
    """Define outcome-independent quartile groups; retain ties and missing values explicitly."""
    rules = [
        ("quiet vocal share", "vocal_energy_share", 0.25, "low"),
        ("strong drum share", "drum_energy_share", 0.75, "high"),
        ("low source activity", "active_source_mean", 0.25, "low"),
        ("high source activity", "active_source_mean", 0.75, "high"),
    ]
    thresholds, membership = (
        {},
        {r["track"]: ["all excerpts", f"genre: {r['genre']}"] for r in features},
    )
    for label, feature, quantile, side in rules:
        values = [r[feature] for r in features if r[feature] is not None]
        threshold = float(np.quantile(values, quantile)) if values else None
        thresholds[label] = {
            "feature": feature,
            "quantile": quantile,
            "threshold": threshold,
            "comparison": "<=" if side == "low" else ">=",
        }
        for row in features:
            value = row[feature]
            if (
                value is not None
                and threshold is not None
                and (value <= threshold if side == "low" else value >= threshold)
            ):
                membership[row["track"]].append(label)
    return thresholds, membership


def analyze(study: Path, output: Path) -> Path:
    """Join verified inputs with track-level outcomes; never run or train a separator."""
    protocol = json.loads((study / "study.json").read_text())
    if protocol["failures"]:
        raise StemScopeError("Resolve failed benchmark workers before analysis.")
    tracks = load_prepared_tracks(Path(protocol["dataset"]))
    names = [t.name for t in tracks]
    if names != protocol["expected_tracks"]:
        raise StemScopeError("Dataset selection differs from the benchmark.")
    if protocol["models_in_order"] != list(MODELS) or [
        w["model"] for w in protocol["worker_results"]
    ] != list(MODELS):
        raise StemScopeError("Expected complete Demucs and Open-Unmix benchmark workers.")
    track_map = {t.name: t for t in tracks}
    for worker in protocol["worker_results"]:
        records = worker["manifest"]["tracks"]
        if [r["name"] for r in records] != names:
            raise StemScopeError("Worker track coverage differs from the study.")
        for record in records:
            track = track_map[record["name"]]
            for stem, info in dict(record["references"], mixture=record["mixture"]).items():
                actual = track.mixture if stem == "mixture" else track.references[stem]
                if hashlib.sha256(actual.read_bytes()).hexdigest() != info["sha256"]:
                    raise StemScopeError("Benchmark input hash changed.")
    raw = read_csv(study / "results.csv")
    result = aggregate(raw, protocol["models_in_order"], names, protocol["repeats"])
    metadata = {r["Track Name"]: r for r in protocol["source_provenance"]["tracks"]}
    features = []
    for index, track in enumerate(tracks, 1):
        samples, rate = load_audio(track.mixture)
        description = analyze_samples(samples, rate)
        references = {stem: load_audio(path)[0] for stem, path in track.references.items()}
        features.append(
            {
                "track": track.name,
                "genre": metadata[track.name].get("Genre", "Unknown"),
                "rms": float(np.mean(description.rms)),
                "centroid_hz": float(np.mean(description.centroid)),
                "bandwidth_hz": float(np.mean(description.bandwidth)),
                "zcr": float(np.mean(description.zcr)),
                **reference_features(references, rate),
            }
        )
        print(f"Features {index}/{len(tracks)}: {track.name}", flush=True)
    thresholds, membership = conditions(features)
    groups = []
    for label in sorted({label for labels in membership.values() for label in labels}):
        for model in protocol["models_in_order"]:
            for stem in STEMS:
                rows = [
                    r
                    for r in result["per_track"]
                    if r["model"] == model and r["stem"] == stem and label in membership[r["track"]]
                ]
                scores = [r["si_sdr_median_db"] for r in rows if r["si_sdr_median_db"] is not None]
                improvements = [
                    r["improvement_median_db"]
                    for r in rows
                    if r["improvement_median_db"] is not None
                ]
                stats = distribution(scores)
                groups.append(
                    {
                        "condition": label,
                        "model": model,
                        "stem": stem,
                        "tracks": len(rows),
                        "scored": stats["n"],
                        "median_si_sdr_db": stats["median"],
                        "q25": stats["q25"],
                        "q75": stats["q75"],
                        "negative_improvement_tracks": sum(v < 0 for v in improvements),
                        "improvement_scored": len(improvements),
                        "small_group": len(scores) < 5,
                    }
                )
    feature_map = {r["track"]: r for r in features}
    correlations = []
    for model in protocol["models_in_order"]:
        for stem in STEMS:
            outcomes = [r for r in result["per_track"] if r["model"] == model and r["stem"] == stem]
            for feature in FEATURES:
                pairs = [
                    (feature_map[r["track"]][feature], r["si_sdr_median_db"])
                    for r in outcomes
                    if feature_map[r["track"]][feature] is not None
                    and r["si_sdr_median_db"] is not None
                ]
                x, y = zip(*pairs) if pairs else ([], [])
                rho = (
                    float(spearmanr(x, y).statistic)
                    if len(pairs) >= 5 and len(set(x)) > 1 and len(set(y)) > 1
                    else None
                )
                correlations.append(
                    {
                        "model": model,
                        "stem": stem,
                        "feature": feature,
                        "n": len(pairs),
                        "spearman_rho": rho,
                        "status": "exploratory" if rho is not None else "insufficient_or_constant",
                    }
                )
    cases = []
    track_map = {t.name: t for t in tracks}
    for model in protocol["models_in_order"]:
        for stem in STEMS:
            ranked = sorted(
                [
                    r
                    for r in result["per_track"]
                    if r["model"] == model
                    and r["stem"] == stem
                    and r["si_sdr_median_db"] is not None
                ],
                key=lambda r: (r["si_sdr_median_db"], r["track"]),
            )
            for rank, row in enumerate(ranked, 1):
                if rank > 5 and (
                    row["improvement_median_db"] is None or row["improvement_median_db"] >= 0
                ):
                    continue
                track = track_map[row["track"]]
                paths = {"mixture": str(track.mixture), "reference": str(track.references[stem])}
                for other in protocol["models_in_order"]:
                    matches = [
                        r
                        for r in raw
                        if r["track"] == track.name
                        and r["stem"] == stem
                        and r["model"] == other
                        and r["status"] == "ok"
                        and r["output"]
                    ]
                    if matches:
                        selected = min(matches, key=lambda r: int(r["repeat"]))
                        verify_estimate(
                            Path(selected["output"]), track.references[stem], selected["si_sdr_db"]
                        )
                        paths[other] = selected["output"]
                cases.append(
                    dict(
                        row,
                        id=str(len(cases)),
                        rank=rank,
                        reason="bottom five SI-SDR" if rank <= 5 else "worse than unseparated mix",
                        paths=paths,
                        hashes={
                            key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                            for key, path in paths.items()
                        },
                    )
                )
    output.mkdir(parents=True, exist_ok=True)
    save_csv(output / "features.csv", features)
    save_csv(output / "conditions.csv", groups)
    save_csv(output / "correlations.csv", correlations)
    save_csv(
        output / "cases.csv",
        [{k: v for k, v in r.items() if k not in {"paths", "hashes"}} for r in cases],
    )
    manifest = {
        "study": str(study.resolve()),
        "study_sha256": hashlib.sha256((study / "study.json").read_bytes()).hexdigest(),
        "results_sha256": hashlib.sha256((study / "results.csv").read_bytes()).hexdigest(),
        "thresholds": thresholds,
        "membership": membership,
        "models": protocol["models_in_order"],
        "cases": cases,
        "feature_protocol": "v1; mixture features from Phase 2; reference energy shares=sum stem mean squares; source activity=100ms blocks above -30dB of each source peak; digital silence floor 1e-20; no semantic instrument labels; playback earliest successful repeat",
    }
    (output / "analysis.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    lines = [
        "# Phase 5 — exploratory failure analysis",
        "",
        f"Analyzed {len(tracks)} existing test excerpts; repetitions collapsed within track. No new inference or training.",
        "",
        "## What these groups mean",
        "",
        "Quiet vocals and strong drums refer to original reference energy shares. High/low activity counts active source classes among vocals, drums, bass and other; it does not count instruments or establish musical density. Quartiles use features only, with inclusive ties, so groups can overlap. These reference-assisted features are unavailable for a new upload before separation.",
        "",
        "Genre labels come from the dataset. Groups with fewer than five scored tracks are marked small in the CSV. Reverb, backing vocals, distortion and acoustic instrumentation are not automatically labeled; listening is required. Correlations are exploratory associations without significance or causal claims, and 56 comparisons are exported without selecting only large effects.",
        "",
        "## Reference-defined conditions",
        "",
        "| Condition | Model | Stem | Scored | Median SI-SDR dB | Worse than mix / scored improvements |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in groups:
        if row["condition"].startswith("genre:"):
            continue
        score = (
            "Unavailable" if row["median_si_sdr_db"] is None else f"{row['median_si_sdr_db']:.2f}"
        )
        lines.append(
            f"| {row['condition']} | {row['model']} | {row['stem']} | {row['scored']} | {score} | {row['negative_improvement_tracks']}/{row['improvement_scored']} |"
        )
    lines += [
        "",
        "## Listening review",
        "",
        f"{len(cases)} review cases: bottom five scores per model/stem plus all additional negative-improvement cases. These are diagnostic flags, not confirmed audible artifacts. The app plays the mix, original reference and both model estimates from the earliest successful repetition. Compare leakage, missing target sound and watery/muffled artifacts; no listening conclusions have been fabricated.",
        "",
        "## Limits and artifacts",
        "",
        "All results inherit the Phase 4 seven-second excerpt, source annotation and checkpoint-overlap limits. Negative SI-SDR alone is not a failure definition: negative improvement means worse than the unseparated mix under this metric. Group medians can be driven by reference level, genre and other confounders. This is exploratory analysis of the test set; future selector development needs separate training/validation data and an untouched test set.",
        "",
        "- features.csv: one row per excerpt.",
        "- conditions.csv: all groups, coverage, quartiles and negative-improvement counts.",
        "- correlations.csv: every feature/model/stem rank correlation.",
        "- cases.csv: deterministic review priorities.",
        "- analysis.json: thresholds, membership, source hashes and playback file hashes.",
        "",
        "Reproduce from the project root:",
        "```sh",
        f"python -m stemscope.failure_analysis {study.resolve()} --output {output.resolve()}",
        "```",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    return output


def verify_estimate(path: Path, reference_path: Path, expected: str) -> None:
    """Recheck playback estimates against the saved score before recording evidence hashes."""
    estimate, rate = load_audio(path)
    reference, reference_rate = load_audio(reference_path)
    estimate, reference = align_estimate(estimate, rate, reference, reference_rate)
    score = si_sdr(reference, estimate)
    if (
        score.db is None
        or not expected
        or not np.isclose(score.db, float(expected), atol=1e-6, rtol=0)
    ):
        raise StemScopeError("Review estimate does not reproduce its benchmark score.")


def load_review(output: Path):
    """Return saved report/table and stable case choices for the interface."""
    try:
        manifest = json.loads((output / "analysis.json").read_text())
        rows = read_csv(output / "conditions.csv")
        table = [
            [
                r[k]
                for k in (
                    "condition",
                    "model",
                    "stem",
                    "scored",
                    "median_si_sdr_db",
                    "negative_improvement_tracks",
                )
            ]
            for r in rows
        ]
        choices = [
            (f"{r['model']} · {r['stem']} · {r['track']} ({r['si_sdr_median_db']:.2f} dB)", r["id"])
            for r in manifest["cases"]
        ]
        return table, choices, str(output / "report.md"), str(output / "conditions.csv")
    except (OSError, ValueError, KeyError) as exc:
        raise StemScopeError("Run the Phase 5 analysis command in the README first.") from exc


def review_audio(output: Path, case_id: str):
    """Verify evidence hashes and copy only the selected files into controlled UI storage."""
    manifest = json.loads((output / "analysis.json").read_text())
    case = next((r for r in manifest["cases"] if r["id"] == case_id), None)
    if case is None:
        raise StemScopeError("Choose a listed review case.")
    folder = output / "listening" / case_id
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, key in enumerate(["mixture", "reference", *manifest["models"]]):
        source = Path(case["paths"][key])
        if hashlib.sha256(source.read_bytes()).hexdigest() != case["hashes"][key]:
            raise StemScopeError(
                "A review audio file changed; regenerate analysis from verified results."
            )
        destination = folder / f"{index}.wav"
        shutil.copyfile(source, destination)
        paths.append(str(destination))
    return tuple(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/phase5-analysis"))
    args = parser.parse_args()
    try:
        print(f"Saved analysis: {analyze(args.study, args.output)}")
    except (StemScopeError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Analysis failed: {exc}\n")


if __name__ == "__main__":
    main()
