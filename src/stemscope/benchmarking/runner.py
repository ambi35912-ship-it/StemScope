"""Repeatable multi-model runs with CSV results and a provenance manifest."""

import csv
import hashlib
import importlib.metadata
import json
import logging
import platform
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from stemscope.audio.loader import load_audio
from stemscope.audio.validation import validate_path
from stemscope.benchmarking.metrics import align_estimate, si_sdr
from stemscope.benchmarking.resources import ResourceMonitor
from stemscope.config import Settings
from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS
from stemscope.service import SeparationService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkTrack:
    name: str
    mixture: Path
    references: dict[str, Path] = field(default_factory=dict)
    provenance: str = "user-supplied; authenticity not verified"


@dataclass(frozen=True)
class BenchmarkResult:
    rows: list[dict]
    csv_path: Path
    manifest_path: Path


def _fingerprint(path: Path) -> dict:
    with path.open("rb") as stream:
        hasher = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
        digest = hasher.hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


def validate_track(track: BenchmarkTrack, settings: Settings) -> None:
    """Check all references before spending time on model inference."""
    mixture = validate_path(track.mixture, settings.max_bytes)
    samples, rate = load_audio(mixture, settings.max_seconds)
    if not track.references:
        return
    if set(track.references) != set(STEMS):
        raise StemScopeError("Provide all four original reference stems, or leave all four empty.")
    if mixture.suffix.lower() != ".wav" or sf.info(mixture).format not in {"WAV", "WAVEX", "RF64"}:
        raise StemScopeError(
            "Use an aligned WAV mix when evaluating reference stems (MP3 can add delay)."
        )
    for name, path in track.references.items():
        validate_path(path, settings.max_bytes)
        if path.suffix.lower() != ".wav" or sf.info(path).format not in {"WAV", "WAVEX", "RF64"}:
            raise StemScopeError("Reference stems must be WAV files.")
        ref, ref_rate = load_audio(path, settings.max_seconds)
        if ref_rate != rate or ref.shape != samples.shape:
            raise StemScopeError(
                f"The {name} reference must match the mix sample rate, channels, and exact frame count."
            )


class BenchmarkRunner:
    """Run sequentially; reuse services and keep failed rows visible in reports."""

    def __init__(
        self,
        services: Mapping[str, SeparationService],
        settings: Settings,
        *,
        experiment: dict | None = None,
    ):
        if not services:
            raise ValueError("At least one model is required")
        self.services = services
        self.settings = settings
        self.experiment = experiment or {}

    def run(self, tracks: Sequence[BenchmarkTrack], repeats: int = 1) -> BenchmarkResult:
        if not tracks or not 1 <= repeats <= 10:
            raise StemScopeError("Choose at least one track and 1–10 repetitions.")
        try:
            for track in tracks:
                validate_track(track, self.settings)
            self.settings.output_dir.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="benchmark-", dir=self.settings.output_dir))
            manifest = {
                "experiment": self.experiment,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "seed": 0,
                "repeats": repeats,
                "models": list(self.services),
                "devices": {
                    name: service.settings.device for name, service in self.services.items()
                },
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "stemscope",
                        "torch",
                        "torchaudio",
                        "demucs",
                        "openunmix",
                        "numpy",
                        "scipy",
                        "psutil",
                    )
                },
                "protocol": "SI-SDR v1: full track; per-channel DC removal; shared stereo scale; estimate resampled to reference rate; at most one rounding sample trimmed; no delay search",
                "resources": "50 ms sampled process RSS (not isolated model RAM); CPU percent may exceed 100; GPU usage unavailable",
                "timing": self.experiment.get(
                    "timing",
                    "SeparationService call only; excludes scoring, hashing and report export; no warmup or model reset; first use may include downloads/loading; models reused across jobs",
                ),
                "adapter_settings": {
                    name: service.separator.configuration for name, service in self.services.items()
                },
                "tracks": [
                    {
                        "name": track.name,
                        "provenance": track.provenance,
                        "mixture": _fingerprint(track.mixture),
                        "references": {
                            name: _fingerprint(path) for name, path in track.references.items()
                        },
                    }
                    for track in tracks
                ],
            }
            manifest_path = directory / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            rows = []
            csv_path = directory / "results.csv"
            for track in tracks:
                for repetition in range(1, repeats + 1):
                    for name, service in self.services.items():
                        rows.extend(self._run_model(track, name, service, repetition))
                        with csv_path.open("w", newline="", encoding="utf-8") as stream:
                            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                            writer.writeheader()
                            writer.writerows(rows)
            return BenchmarkResult(rows, csv_path, manifest_path)
        except StemScopeError:
            raise
        except Exception as exc:
            logger.exception("Benchmark setup or export failed")
            raise StemScopeError(
                "Cannot run or save benchmark. Check input files, output permissions, and logs."
            ) from exc

    def _run_model(self, track, name, service, repetition) -> list[dict]:
        import torch

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        result = None
        error = ""
        with ResourceMonitor() as monitor:
            try:
                result = service.process(track.mixture)
            except Exception as exc:
                logger.exception("Benchmark separation failed for %s / %s", track.name, name)
                error = (
                    str(exc) if isinstance(exc, StemScopeError) else "Separation failed; see logs."
                )
        duration = sf.info(track.mixture).duration
        rows = []
        for stem in STEMS:
            row = {
                "track": track.name,
                "provenance": track.provenance,
                "model": name,
                "repeat": repetition,
                "stem": stem,
                "status": "ok" if result else "separation_failed",
                "error": error,
                "wall_seconds": monitor.seconds,
                "real_time_factor": monitor.seconds / duration,
                "cpu_seconds": monitor.cpu_seconds,
                "cpu_percent": monitor.cpu_percent,
                "baseline_rss_bytes": monitor.baseline_rss,
                "sampled_peak_rss_bytes": monitor.peak_rss,
                "gpu_usage_percent": None,
                "si_sdr_db": None,
                "metric_status": "no_reference",
                "mixture_si_sdr_db": None,
                "mixture_metric_status": "no_reference",
                "si_sdr_improvement_db": None,
                "output": str(result.stems[stem]) if result else "",
            }
            if result is None:
                row["metric_status"] = "not_scored"
            elif stem in track.references:
                try:
                    reference, ref_rate = load_audio(
                        track.references[stem], self.settings.max_seconds
                    )
                    estimate, rate = load_audio(result.stems[stem], self.settings.max_seconds)
                    estimate, aligned_ref = align_estimate(estimate, rate, reference, ref_rate)
                    score = si_sdr(aligned_ref, estimate)
                    mixture, mix_rate = load_audio(track.mixture, self.settings.max_seconds)
                    mixture, mix_ref = align_estimate(mixture, mix_rate, reference, ref_rate)
                    baseline = si_sdr(mix_ref, mixture)
                    row.update(
                        si_sdr_db=score.db,
                        metric_status=score.status,
                        mixture_si_sdr_db=baseline.db,
                        mixture_metric_status=baseline.status,
                    )
                    if score.db is not None and baseline.db is not None:
                        row["si_sdr_improvement_db"] = score.db - baseline.db
                except Exception as exc:
                    logger.exception("Benchmark scoring failed for %s / %s", name, stem)
                    row.update(
                        status="metric_failed",
                        metric_status="not_scored",
                        error=str(exc)
                        if isinstance(exc, StemScopeError)
                        else "Scoring failed; see logs.",
                    )
            rows.append(row)
        return rows
