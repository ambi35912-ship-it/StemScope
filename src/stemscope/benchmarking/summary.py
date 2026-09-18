"""Aggregate repeated observations at track level and report complete study coverage."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS


def distribution(values: list[float]) -> dict:
    """Summarize finite values; never replace missing scores with zero."""
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(finite):
        return {
            "n": 0,
            "median": None,
            "q25": None,
            "q75": None,
            "mean": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "n": len(finite),
        "median": float(np.median(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "mean": float(finite.mean()),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
    }


def aggregate(rows: list[dict], models: list[str], tracks: list[str], repeats: int) -> dict:
    """Validate coverage, collapse repetitions, then summarize paired track/stem scores."""
    expected = {
        (track, model, str(repeat), stem)
        for track in tracks
        for model in models
        for repeat in range(1, repeats + 1)
        for stem in STEMS
    }
    observed = [(r["track"], r["model"], str(r["repeat"]), r["stem"]) for r in rows]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise StemScopeError(
            "Study CSV has missing, duplicate, or unexpected rows; cannot claim complete coverage."
        )
    for row in rows:
        if row["status"] == "ok" and row["metric_status"] == "ok":
            try:
                valid = np.isfinite(float(row["si_sdr_db"]))
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise StemScopeError("A score marked ok is missing or non-finite.")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["track"], row["model"], row["stem"]].append(row)
    per_track, quality = [], []
    for track in tracks:
        for model in models:
            for stem in STEMS:
                group = grouped[track, model, stem]
                scores = [
                    float(r["si_sdr_db"])
                    for r in group
                    if r["status"] == "ok"
                    and r["metric_status"] == "ok"
                    and r["si_sdr_db"] not in ("", None)
                ]
                improvements = [
                    float(r["si_sdr_improvement_db"])
                    for r in group
                    if r["status"] == "ok" and r["si_sdr_improvement_db"] not in ("", None)
                ]
                per_track.append(
                    {
                        "track": track,
                        "model": model,
                        "stem": stem,
                        "si_sdr_median_db": distribution(scores)["median"],
                        "improvement_median_db": distribution(improvements)["median"],
                        "valid_repetitions": distribution(scores)["n"],
                        "expected_repetitions": repeats,
                    }
                )
    for model in models:
        for stem in STEMS:
            entries = [r for r in per_track if r["model"] == model and r["stem"] == stem]
            values = [r["si_sdr_median_db"] for r in entries if r["si_sdr_median_db"] is not None]
            improvements = [
                r["improvement_median_db"]
                for r in entries
                if r["improvement_median_db"] is not None
            ]
            quality.append(
                dict(
                    model=model,
                    stem=stem,
                    **distribution(values),
                    expected_tracks=len(tracks),
                    missing_tracks=len(tracks) - len(values),
                    median_improvement_db=distribution(improvements)["median"],
                )
            )
    paired = []
    if len(models) == 2:
        lookup = {(r["track"], r["model"], r["stem"]): r["si_sdr_median_db"] for r in per_track}
        for stem in STEMS:
            deltas = [
                lookup[t, models[0], stem] - lookup[t, models[1], stem]
                for t in tracks
                if lookup[t, models[0], stem] is not None and lookup[t, models[1], stem] is not None
            ]
            paired.append(
                {
                    "stem": stem,
                    "first_model": models[0],
                    "second_model": models[1],
                    "first_wins": sum(d > 1e-8 for d in deltas),
                    "second_wins": sum(d < -1e-8 for d in deltas),
                    "ties": sum(abs(d) <= 1e-8 for d in deltas),
                    "median_delta_db": distribution(deltas)["median"],
                    "n_pairs": len(deltas),
                }
            )
    calls = {}
    for row in rows:
        key = row["track"], row["model"], str(row["repeat"])
        previous = calls.get(key)
        if previous and any(
            previous[field] != row[field]
            for field in ("wall_seconds", "cpu_seconds", "sampled_peak_rss_bytes")
        ):
            raise StemScopeError(
                "Repeated per-stem resource observations disagree within one call."
            )
        calls[key] = row
    timing = []
    for model in models:
        entries = [
            r for r in calls.values() if r["model"] == model and r["status"] != "separation_failed"
        ]
        stats = distribution([float(r["wall_seconds"]) for r in entries])
        timing.append(
            dict(
                model=model,
                **stats,
                median_real_time_factor=distribution(
                    [float(r["real_time_factor"]) for r in entries]
                )["median"],
                max_sampled_rss_mib=max(
                    (float(r["sampled_peak_rss_bytes"]) / 1024**2 for r in entries), default=None
                ),
            )
        )
    failures = [r for r in rows if r["status"] != "ok" or r["metric_status"] != "ok"]
    return {
        "quality": quality,
        "per_track": per_track,
        "paired": paired,
        "timing": timing,
        "failures": failures,
        "row_status_counts": dict(Counter(r["status"] for r in rows)),
        "metric_status_counts": dict(Counter(r["metric_status"] for r in rows)),
        "tracks": len(tracks),
        "measured_calls": len(calls),
        "rows": len(rows),
    }


def _csv(path: Path, rows: list[dict], fields=None):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(directory: Path) -> dict:
    """Save CSV distributions, per-track results, exceptions, charts and a cited report."""
    protocol = json.loads((directory / "study.json").read_text())
    if protocol["failures"] or len(protocol["worker_results"]) != len(protocol["models_in_order"]):
        raise StemScopeError(
            "Study has failed or missing workers; inspect worker logs before summarizing."
        )
    if [w["model"] for w in protocol["worker_results"]] != protocol["models_in_order"]:
        raise StemScopeError("Worker model identities do not match the study plan.")
    signatures = []
    for worker in protocol["worker_results"]:
        manifest = worker["manifest"]
        experiment = manifest["experiment"]
        if (
            experiment.get("kind") != "isolated_warm_cpu_study_v1"
            or experiment.get("model") != worker["model"]
            or experiment.get("torch_interop_threads") != 1
            or manifest.get("models") != [worker["model"]]
            or manifest.get("repeats") != protocol["repeats"]
            or experiment.get("warmup_calls") != 1
            or experiment.get("torch_threads") != protocol["threads"]
        ):
            raise StemScopeError("Worker does not match the declared warm-up/thread protocol.")
        signatures.append(
            {
                t["name"]: {
                    name: info["sha256"]
                    for name, info in dict(t["references"], mixture=t["mixture"]).items()
                }
                for t in manifest["tracks"]
            }
        )
    if any(set(signature) != set(protocol["expected_tracks"]) for signature in signatures):
        raise StemScopeError("Worker input tracks do not match the study plan.")
    if any(signature != signatures[0] for signature in signatures):
        raise StemScopeError("Input hashes differ between models.")
    with (directory / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    result = aggregate(
        rows, protocol["models_in_order"], protocol["expected_tracks"], protocol["repeats"]
    )
    _csv(directory / "summary.csv", result["quality"])
    _csv(directory / "per-track.csv", result["per_track"])
    _csv(directory / "paired-comparisons.csv", result["paired"])
    _csv(directory / "runtime-summary.csv", result["timing"])
    _csv(directory / "failures-and-missing.csv", result["failures"], fields=list(rows[0]))
    (directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    render_chart(result, directory)
    write_report(result, protocol, directory)
    return result


def render_chart(result: dict, directory: Path):
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(directory / ".matplotlib"))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 7), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    models = list(dict.fromkeys(r["model"] for r in result["quality"]))
    for ax, stem in zip(axes.flat, STEMS):
        values = [
            [
                r["si_sdr_median_db"]
                for r in result["per_track"]
                if r["model"] == model and r["stem"] == stem and r["si_sdr_median_db"] is not None
            ]
            for model in models
        ]
        for index, data in enumerate(values, 1):
            if data:
                ax.boxplot([data], positions=[index], widths=0.45, showfliers=False)
                jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(data))
                ax.scatter(index + jitter, data, s=12, alpha=0.5)
        ax.set_xticks(range(1, len(models) + 1), models, fontsize=8)
        ax.set(title=stem.title(), ylabel="SI-SDR (dB)")
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.grid(axis="y", alpha=0.2)
    figure.suptitle(
        f"Reference-based quality across {result['tracks']} test excerpts\nEach dot is one track; repeated runs collapsed"
    )
    figure.savefig(directory / "quality-distributions.png", dpi=140)
    figure.clear()


def write_report(result, protocol, directory):
    def number(value):
        return "Unavailable" if value is None else f"{value:.2f}"

    sources = protocol["source_provenance"]["tracks"]
    genres = Counter(record.get("Genre", "Unknown") for record in sources)
    lines = [
        "# StemScope — Phase 4 completed evaluation",
        "",
        (
            f"Evaluated **{result['tracks']} real test excerpts**, both separators, and **{protocol['repeats']} measured repetitions** after one excluded warm-up per model. "
            f"This produced {result['measured_calls']} measured separation calls and {result['rows']} per-stem observations."
        ),
        "",
        "## Scope and data",
        "",
        (
            "Source: [SiSEC18-MUS 7s Excerpts v1.0.0](https://zenodo.org/records/1256064), "
            "`MUSDB18-7-WAV.zip`, all 50 official test previews. Selection was fixed before scoring; "
            "no tracks were dropped based on results. These are seven-second activity-selected "
            "excerpts, not full songs or the MUSDB18-HQ corpus."
        ),
        "",
        "Official genre metadata coverage: "
        + ", ".join(f"{name}: {count}" for name, count in sorted(genres.items()))
        + ".",
        "",
        (
            "The public research previews contain source-restricted material as well as CC tracks. "
            "Original artist names, source restrictions, archive member CRCs and SHA-256 hashes "
            "are retained in the dataset provenance. Audio and stems remain ignored local research files; "
            "this report does not redistribute them. Dataset citation: Stöter, Liutkus and Ito (2018), "
            "DOI 10.5281/zenodo.1256064; MUSDB18 creators Rafii et al. (2017), DOI 10.5281/zenodo.1117372."
        ),
        "",
        "## Per-stem quality distributions",
        "",
        (
            "Higher SI-SDR is better. Values below summarize one median score per unique track/model/stem; "
            "repetitions are not counted as independent songs. Q1–Q3 shows variation across tracks. "
            "Improvement compares with returning the original unseparated mix."
        ),
        "",
        "| Model | Stem | Tracks scored | Median SI-SDR dB | Q1–Q3 dB | Median improvement dB |",
        "|---|---|---:|---:|---|---:|",
    ]
    for r in result["quality"]:
        lines.append(
            f"| {r['model']} | {r['stem']} | {r['n']}/{r['expected_tracks']} | {number(r['median'])} | {number(r['q25'])}–{number(r['q75'])} | {number(r['median_improvement_db'])} |"
        )
    lines += [
        "",
        "![Per-track SI-SDR distributions](quality-distributions.png)",
        "",
        "## Paired model comparison",
        "",
        "Each comparison uses the same track and stem. Positive median difference favors the first model.",
        "",
        "| Stem | First model | Second model | First wins | Second wins | Ties | Pairs | Median difference dB |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in result["paired"]:
        lines.append(
            f"| {r['stem']} | {r['first_model']} | {r['second_model']} | {r['first_wins']} | {r['second_wins']} | {r['ties']} | {r['n_pairs']} | {number(r['median_delta_db'])} |"
        )
    lines += [
        "",
        "## Warm CPU performance",
        "",
        (
            f"Each model ran in a separate fresh process, sequentially, with {protocol['threads']} PyTorch compute threads and one inter-op thread. "
            "The first test mix was used for a single excluded warm-up. No training occurred. "
            "Measured timing includes decode, inference, WAV export and output validation, but excludes scoring/hashing/reporting. "
            "Samples below are calls, not the repeated resource fields on each stem row."
        ),
        "",
        "| Model | Measured calls | Median seconds | Q1–Q3 seconds | Maximum seconds | Median real-time factor | Highest sampled process RSS MiB |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for r in result["timing"]:
        lines.append(
            f"| {r['model']} | {r['n']} | {number(r['median'])} | {number(r['q25'])}–{number(r['q75'])} | {number(r['maximum'])} | {number(r['median_real_time_factor'])} | {number(r['max_sampled_rss_mib'])} |"
        )
    lines += [
        "",
        (
            "RSS is sampled every 50 ms and includes libraries plus one loaded model; it is not "
            "exact tensor allocation or a guaranteed peak. CPU usage is exported in raw results; "
            "it can exceed 100% across cores. GPU utilization is unavailable because this is a CPU study. "
            "The machine was not exclusively reserved; CPU affinity, thermal state and frequency were not locked. "
            "Sequential model order is recorded. Timing outliers are retained; the maximum column exposes slow calls that medians hide. These controls improve comparability without claiming laboratory-grade timing."
        ),
        "",
        "## Coverage, failures and missing scores",
        "",
        "Run statuses: " + json.dumps(result["row_status_counts"]) + ".",
        "Metric statuses: " + json.dumps(result["metric_status_counts"]) + ".",
        "",
        (
            f"{len(result['failures'])} rows are listed in `failures-and-missing.csv`. Empty/missing scores were never replaced with zero. "
            "The summary rejects missing/duplicate rows or different source hashes across models. "
            "Silent references/estimates and mathematical infinities have explicit statuses."
        ),
        "",
        "## Metric and interpretation limits",
        "",
        (
            "Protocol: full-excerpt zero-mean SI-SDR, per-channel DC removal, one scale over stereo channels. "
            "Estimates are resampled to the reference rate; at most one rounding sample is trimmed; "
            "there is no delay search. This is not BSS Eval SDR and must not be compared numerically with published "
            "MUSDB windowed SDR tables. [Metric definition](https://arxiv.org/abs/1811.02508)."
        ),
        "",
        (
            "The official test split is used, but external checkpoint training overlap was not independently audited. "
            "MUSDB source annotations have [documented errors and leakage](https://sigsep.github.io/datasets/musdb.html#errata); "
            "those tracks remain included. Seven-second activity-selected excerpts do not measure full-song behavior, "
            "long-context effects, all music genres, or human-perceived quality. No listening study was conducted. "
            "This completes an excerpt-based Phase 4 study, not proof of universal accuracy."
        ),
        "",
        "## Artifacts and reproduction",
        "",
        "- `results.csv`: all measured observations, output paths, CPU/RSS and scores.",
        "- `summary.csv`: per-stem distributions across unique tracks.",
        "- `per-track.csv`: repetitions collapsed within each track/model/stem.",
        "- `paired-comparisons.csv`: paired wins and median score differences.",
        "- `runtime-summary.csv`: warmed CPU timing distributions.",
        "- `failures-and-missing.csv`: every failed or unavailable-score row.",
        "- `study.json`: source provenance, worker protocols, versions, hardware, hashes and warm-up jobs.",
        "",
        "From the project directory:",
        "",
        "```sh",
        "python scripts/prepare_research_testset.py --output data/musdb7-test",
        "python -m stemscope.benchmarking.study data/musdb7-test --output outputs/phase4-study --repeats 2 --threads 4",
        "python -m stemscope.benchmarking.summary outputs/phase4-study/study-RUN_ID",
        "```",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.study)
        print(f"Summary saved: {args.study / 'report.md'} ({result['tracks']} tracks)")
    except StemScopeError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
