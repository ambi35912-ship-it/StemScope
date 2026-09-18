"""Summarize a benchmark without counting repeated runs as independent songs."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path, help="Folder with results.csv and manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (args.benchmark / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    manifest = json.loads((args.benchmark / "manifest.json").read_text())
    groups = defaultdict(list)
    for row in rows:
        groups[row["track"], row["stem"], row["model"]].append(row)
    models = manifest["models"]
    tracks = sorted({row["track"] for row in rows})
    expected_tracks = {
        "The Easton Ellises (Baumi) - SDRNR",
        "The Easton Ellises - Falcon 69",
    }
    if set(tracks) != expected_tracks:
        parser.error("This report template is for the prepared two-track public pilot only")
    lines = [
        "# StemScope — real-music pilot evaluation",
        "",
        (
            f"Evaluated {len(tracks)} official MUSDB18 test excerpts with original reference stems, "
            f"{len(models)} models, and {manifest['repeats']} repetitions per model/clip."
        ),
        "",
        (
            "**Scope:** two short excerpts by the same artist. This is a narrow pilot, "
            "not a representative music benchmark or an accuracy percentage. "
            "The repetitions assess repeatability; they do not increase the number of independent tracks."
        ),
        "",
        "## Reference source and selection",
        "",
        (
            "[SiSEC18-MUS 7s Excerpts, version 1.0.0](https://zenodo.org/records/1256064), "
            "`MUSDB18-7-WAV.zip`. Official track/source/license metadata: "
            "[SigSep track list](https://github.com/sigsep/website/blob/master/content/datasets/assets/tracklist.csv)."
        ),
        "",
        (
            "Selection was made before inference: alphabetically ordered test tracks whose official "
            "metadata lists CC BY-NC-SA, up to five tracks. Only two qualified. Both are credited to "
            "The Easton Ellises and listed under CC BY-NC-SA 3.0. No track was excluded based on its scores."
        ),
        "",
        (
            "These previews were selected for source activity, not randomly sampled. They are not full "
            "songs or the full MUSDB18-HQ dataset. Training overlap beyond the official split was not "
            "independently audited. Reference audio remains in the ignored data folder; this report does not redistribute it."
        ),
        "",
        "## Scores against original reference stems",
        "",
        (
            "Full-excerpt, zero-mean SI-SDR in dB; higher is better. Each table cell is the median across "
            "repetitions of that same track/model/stem. Silence/undefined scores are omitted explicitly. "
            "A negative score is possible and is not a percentage."
        ),
        "",
        "| Track | Stem | " + " | ".join(models) + " |",
        "|---|---|" + "---:|" * len(models),
    ]
    wins = {model: 0 for model in models}
    for track in tracks:
        for stem in ("vocals", "drums", "bass", "other"):
            values = {}
            for model in models:
                scores = [
                    float(r["si_sdr_db"])
                    for r in groups[track, stem, model]
                    if r["status"] == "ok" and r["metric_status"] == "ok" and r["si_sdr_db"]
                ]
                values[model] = median(scores) if scores else None
            if all(value is not None for value in values.values()):
                best = max(values.values())
                winners = [model for model, score in values.items() if score == best]
                if len(winners) == 1:
                    wins[winners[0]] += 1
            cells = [
                f"{values[model]:.2f}" if values[model] is not None else "Unavailable"
                for model in models
            ]
            lines.append("| " + track + " | " + stem + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Track/stem comparisons with the highest score (ties not counted): "
        + "; ".join(f"{model}: {count}" for model, count in wins.items())
        + ".",
        "",
        "## Improvement over the unseparated mix",
        "",
        (
            "Positive values mean the separator beats simply returning the mixture for that stem. "
            "Below are medians across the unique track results; repeated runs are collapsed first."
        ),
        "",
        "| Model | Stem | Median improvement (dB) | Unique tracks scored |",
        "|---|---|---:|---:|",
    ]
    for model in models:
        for stem in ("vocals", "drums", "bass", "other"):
            values = []
            for track in tracks:
                numbers = [
                    float(r["si_sdr_improvement_db"])
                    for r in groups[track, stem, model]
                    if r["status"] == "ok" and r["si_sdr_improvement_db"]
                ]
                if numbers:
                    values.append(median(numbers))
            text = f"{median(values):.2f}" if values else "Unavailable"
            lines.append(f"| {model} | {stem} | {text} | {len(values)} |")
    lines += [
        "",
        "## Runtime observations",
        "",
        (
            "Weights were already downloaded. Repetitions after the first are summarized as warm observations. "
            "Timings cover the separation service, not scoring/export. CPU process memory includes "
            "both cached models and libraries; it is not isolated model memory. GPU utilization is unavailable. These are uncontrolled local timing observations, not a hardware speed claim."
        ),
        "",
        "| Model | Median warm seconds per clip | Warm calls |",
        "|---|---:|---:|",
    ]
    for model in models:
        calls = {
            (r["track"], r["repeat"]): float(r["wall_seconds"])
            for r in rows
            if r["model"] == model and int(r["repeat"]) > 1 and r["status"] == "ok"
        }
        value = f"{median(calls.values()):.2f}" if calls else "Unavailable"
        lines.append(f"| {model} | {value} | {len(calls)} |")
    failures = sum(r["status"] != "ok" for r in rows)
    missing = sum(r["metric_status"] != "ok" for r in rows)
    lines += [
        "",
        f"Result rows: {len(rows)}; failed rows: {failures}; non-finite/unavailable score rows: {missing}.",
        "",
        "## Interpretation and limits",
        "",
        (
            "These scores support comparisons only within this pilot and protocol. The low/negative "
            "scores for some stems identify cases worth listening to, not a complete artifact diagnosis. "
            "SI-SDR does not measure every perceptual defect. No listening assessment was performed for this report."
        ),
        "",
        (
            "For stronger portfolio claims, extend to diverse held-out tracks and full songs, "
            "report per-stem distributions and failure cases, and listen to aligned references/estimates. "
            "Do not train a selector on these two clips or present these numbers as published MUSDB BSS Eval SDR."
        ),
        "",
        "## Reproduction",
        "",
        "Run from the StemScope project directory:",
        "",
        "```sh",
        'python -m stemscope.benchmarking.cli data/musdb7-pilot --output outputs/real-music-pilot --repeats 3 --provenance "Official CC-licensed test-excerpt pilot; see data/musdb7-pilot/provenance.json"',
        "```",
        "",
        (
            "Exact input hashes, versions, device, seed and adapter parameters are in the accompanying "
            "`manifest.json`; individual observations are in `results.csv`. Source member hashes and "
            "the copied license metadata are in `data/musdb7-pilot/provenance.json` and `source-tracklist.csv`."
        ),
        "",
        "Metric basis: [Le Roux et al., SDR — half-baked or well done?](https://arxiv.org/abs/1811.02508).",
        (
            "Dataset attribution: Stöter, Liutkus and Ito, SiSEC18-MUS 7s Excerpts (2018), "
            "DOI 10.5281/zenodo.1256064; MUSDB18 creators Rafii, Liutkus, Stöter, Mimilakis and Bittner (2017)."
        ),
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
