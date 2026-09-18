"""Audit fixed diagnostic flags against explored reference results; do not tune thresholds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from stemscope.audio.loader import load_audio
from stemscope.benchmarking.metrics import align_estimate, si_sdr
from stemscope.errors import StemScopeError
from stemscope.quality.diagnostics import PROTOCOL, inspect_files
from stemscope.quality.service import fingerprint
from stemscope.separators.base import STEMS


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(study: Path, output: Path) -> Path:
    protocol = json.loads((study / "study.json").read_text())
    if protocol["failures"] or len(protocol["worker_results"]) != 2:
        raise StemScopeError("Audit requires a completed two-model study.")
    with (study / "results.csv").open() as stream:
        raw = list(csv.DictReader(stream))
    tracks = protocol["worker_results"][0]["manifest"]["tracks"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2))
    evidence = []
    observations = []
    for track in tracks:
        for info in [track["mixture"], *track["references"].values()]:
            if fingerprint(Path(info["path"])) != info["sha256"]:
                raise StemScopeError("Reference audit input changed.")
        for model in protocol["models_in_order"]:
            selected = [
                r
                for r in raw
                if r["track"] == track["name"] and r["model"] == model and r["repeat"] == "1"
            ]
            if (
                len(selected) != 4
                or {r["stem"] for r in selected} != set(STEMS)
                or any(r["status"] != "ok" or r["metric_status"] != "ok" for r in selected)
            ):
                raise StemScopeError(
                    "Audit requires four scored stems for each first repetition; no silent exclusions."
                )
            paths = {r["stem"]: Path(r["output"]) for r in selected}
            result = inspect_files(Path(track["mixture"]["path"]), paths)
            for row in selected:
                stem = row["stem"]
                estimate, rate = load_audio(paths[stem])
                reference, reference_rate = load_audio(Path(track["references"][stem]["path"]))
                estimate, reference = align_estimate(estimate, rate, reference, reference_rate)
                score = si_sdr(reference, estimate)
                if score.db is None or not np.isclose(
                    score.db, float(row["si_sdr_db"]), atol=1e-6, rtol=0
                ):
                    raise StemScopeError("Audit output no longer matches its saved score.")
                item = next(r for r in result["rows"] if r["stem"] == stem)
                observations.append(
                    {
                        "track": track["name"],
                        "model": model,
                        "stem": stem,
                        "si_sdr_db": score.db,
                        "improvement_db": float(row["si_sdr_improvement_db"]),
                        "flagged": bool(item["flags"]),
                        "flags": "; ".join(item["flags"]),
                        **{
                            key: item[key]
                            for key in (
                                "rms_dbfs",
                                "over_full_scale_fraction",
                                "max_waveform_similarity",
                                "high_band_fraction",
                                "spectral_flatness",
                            )
                        },
                    }
                )
            evidence.append(
                {
                    "track": track["name"],
                    "model": model,
                    "outputs": {
                        name: {"path": str(path), "sha256": fingerprint(path)}
                        for name, path in paths.items()
                    },
                    "diagnostics": result,
                }
            )
        print(f"Audited {track['name']}", flush=True)
    summaries, correlations = [], []
    for model in protocol["models_in_order"]:
        for stem in STEMS:
            rows = [r for r in observations if r["model"] == model and r["stem"] == stem]
            flagged = [r for r in rows if r["flagged"]]
            underperform = [r for r in rows if r["improvement_db"] < 0]
            hits = sum(r["flagged"] for r in underperform)
            summaries.append(
                {
                    "model": model,
                    "stem": stem,
                    "tracks": len(rows),
                    "flagged": len(flagged),
                    "negative_improvement": len(underperform),
                    "flagged_negative_improvement": hits,
                    "missed_negative_improvement": len(underperform) - hits,
                    "precision_for_negative_improvement": hits / len(flagged) if flagged else None,
                    "recall_for_negative_improvement": hits / len(underperform)
                    if underperform
                    else None,
                }
            )
            for feature in (
                "rms_dbfs",
                "max_waveform_similarity",
                "high_band_fraction",
                "spectral_flatness",
            ):
                pairs = [(r[feature], r["si_sdr_db"]) for r in rows if r[feature] is not None]
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
                    }
                )
    save_csv(output / "observations.csv", observations)
    save_csv(output / "summary.csv", summaries)
    save_csv(output / "correlations.csv", correlations)
    (output / "audit.json").write_text(
        json.dumps(
            {
                "study": str(study.resolve()),
                "study_sha256": fingerprint(study / "study.json"),
                "results_sha256": fingerprint(study / "results.csv"),
                "evidence": evidence,
            },
            indent=2,
            allow_nan=False,
        )
    )
    flagged = sum(r["flagged"] for r in observations)
    negatives = sum(r["improvement_db"] < 0 for r in observations)
    hits = sum(r["flagged"] and r["improvement_db"] < 0 for r in observations)
    lines = [
        "# Phase 7 — interpretable signal-quality diagnostics",
        "",
        f"Audited {len(observations)} track/model/stem outputs from {len(tracks)} already-explored excerpts, first repetition only. {flagged} outputs triggered a per-stem review flag. Of {negatives} outputs worse than the unseparated mix by SI-SDR improvement, {hits} were flagged and {negatives - hits} were missed.",
        "",
        "These checks are not a reliable general separation-quality score. The reference audit explicitly exposes missed problems; no configured flags must not be interpreted as high quality. Negative improvement is a metric-based audit target, not a human artifact label. A flag on another output is not necessarily a false audible alarm.",
        "",
        "## What is measured",
        "",
        "- RMS below -60 dBFS: very quiet or absent target, which may be correct.",
        "- At least 0.1% of samples reach/exceed full scale: review playback headroom; float WAV overs are not proof of clipping distortion.",
        "- Absolute zero-lag waveform correlation at least 0.9 with another stem: review shared content; not proof of instrument leakage.",
        "- Relative RMS of mix minus summed stems greater than 0.1: review consistency/alignment. Mixture-consistent leakage can evade this check, and original datasets may not sum exactly.",
        "- High-band power fraction above 4 kHz and median spectral flatness: descriptions only; normal bass can have little high-frequency energy. Watery/muffled artifacts cannot be confirmed from these values.",
        "",
        "The first 30 seconds are inspected at no more than 22,050 Hz. Peak/full-scale checks use original-rate samples. Mono is duplicated for compatible stereo comparisons. Durations must match, and only one resampling-rounding sample may be trimmed. No delay search, gain fitting, reference stems or benchmark scores are used by the inference checks.",
        "",
        "## Reference audit",
        "",
        "| Model | Stem | Tracks | Flagged | Negative improvement | Flagged negatives | Missed negatives |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        lines.append(
            f"| {r['model']} | {r['stem']} | {r['tracks']} | {r['flagged']} | {r['negative_improvement']} | {r['flagged_negative_improvement']} | {r['missed_negative_improvement']} |"
        )
    lines += [
        "",
        "Thresholds were fixed before this audit and were not tuned afterward. The already-explored Phase 4 corpus is not an independent validation set for perceptual quality. This is a diagnostic prototype, not a calibrated HIGH/MEDIUM/LOW quality estimator. Listening annotations and new validation data are needed before such ratings are justified.",
        "",
        "Artifact files: protocol.json (fixed thresholds), observations.csv (every output), summary.csv (coverage and missed negatives), correlations.csv (all 32 exploratory correlations, no significance claims), audit.json (source/output hashes and detailed evidence). Original scores were recomputed to verify the audio still matches the benchmark.",
        "",
        "Reproduce from the project directory:",
        "```sh",
        f"python -m stemscope.quality.audit {study.resolve()} --output {output.resolve()}",
        "```",
        "",
    ]
    report = output / "report.md"
    report.write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/phase7-audit"))
    args = parser.parse_args()
    try:
        print(audit(args.study, args.output))
    except (StemScopeError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Quality audit failed: {exc}\n")


if __name__ == "__main__":
    main()
