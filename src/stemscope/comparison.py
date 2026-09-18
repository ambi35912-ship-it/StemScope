"""Compare two separators on one captured upload and retain partial failures."""

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import stft

from stemscope.audio.validation import validate_path
from stemscope.config import Settings
from stemscope.errors import StemScopeError
from stemscope.quality.diagnostics import inspect_files, read_prefix, resample
from stemscope.quality.service import fingerprint
from stemscope.separators.base import STEMS
from stemscope.service import SeparationService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComparisonResult:
    directory: Path
    manifest: Path
    rows: list[dict]
    status: str


class ComparisonService:
    """Use common adapters; retain successful audio when another model fails."""

    def __init__(self, services: Mapping[str, SeparationService], settings: Settings):
        self.services = services
        self.settings = settings

    def run(self, upload: str | Path) -> ComparisonResult:
        if len(self.services) != 2:
            raise StemScopeError("Comparison requires two available separators.")
        source = validate_path(upload, self.settings.max_bytes)
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="comparison-", dir=self.settings.output_dir))
        manifest_path = directory / "comparison.json"
        try:
            snapshot = directory / f"input{source.suffix.lower()}"
            shutil.copyfile(source, snapshot)
            protocol = {
                "version": "comparison-v1",
                "original_filename": source.name,
                "input": {"path": str(snapshot.resolve()), "sha256": fingerprint(snapshot)},
                "models": [],
                "notes": "Sequential interactive calls, shared process, no warm-up or isolation. First use may include model loading/downloads. Times are not controlled benchmark comparisons. Diagnostics and figures use up to the first 30 seconds; no reference-based quality scores are computed.",
            }
            rows = []
            for name, service in self.services.items():
                started = perf_counter()
                entry = {
                    "model": name,
                    "adapter_settings": service.separator.configuration,
                    "device": service.settings.device,
                    "stems": {},
                    "status": "failed",
                    "error": "",
                    "diagnostic_error": "",
                }
                try:
                    result = service.process(snapshot)
                    entry.update(
                        status="ok",
                        separation_seconds=result.seconds,
                        stems={stem: str(path.resolve()) for stem, path in result.stems.items()},
                        output_hashes={
                            stem: fingerprint(path) for stem, path in result.stems.items()
                        },
                    )
                    try:
                        entry["diagnostics"] = inspect_files(snapshot, result.stems)
                    except Exception:
                        logger.exception("Comparison diagnostics failed for %s", name)
                        entry["diagnostic_error"] = (
                            "Diagnostics unavailable; separated audio is retained."
                        )
                except Exception as exc:
                    logger.exception("Comparison separator failed: %s", name)
                    entry["error"] = (
                        str(exc)
                        if isinstance(exc, StemScopeError)
                        else "Separation failed; see application logs."
                    )
                    entry["failed_call_seconds"] = perf_counter() - started
                for stem in STEMS:
                    diagnostic = next(
                        (
                            r
                            for r in entry.get("diagnostics", {}).get("rows", [])
                            if r["stem"] == stem
                        ),
                        {},
                    )
                    rows.append(
                        {
                            "model": name,
                            "stem": stem,
                            "status": entry["status"],
                            "separation_seconds": entry.get("separation_seconds"),
                            "review_status": diagnostic.get("review_status", "Unavailable"),
                            "evidence": "; ".join(diagnostic.get("flags", []))
                            or entry["error"]
                            or entry["diagnostic_error"]
                            or "No configured flags; quality unverified",
                            "rms_dbfs": diagnostic.get("rms_dbfs"),
                            "waveform_similarity": diagnostic.get("max_waveform_similarity"),
                        }
                    )
                protocol["models"].append(entry)
                protocol["rows"] = rows
                manifest_path.write_text(json.dumps(protocol, indent=2, allow_nan=False))
            status = (
                "Complete"
                if all(e["status"] == "ok" for e in protocol["models"])
                else "Finished with model errors; available results retained"
            )
            if any(e["diagnostic_error"] for e in protocol["models"]):
                status += " · some diagnostics unavailable"
            return ComparisonResult(directory, manifest_path, rows, status)
        except Exception as exc:
            logger.exception("Comparison setup/export failed")
            if not manifest_path.exists():
                try:
                    shutil.rmtree(directory)
                except OSError:
                    logger.exception("Cannot clean failed comparison directory")
            if isinstance(exc, StemScopeError):
                raise
            raise StemScopeError(
                "Cannot save the comparison. Check input files and output storage."
            ) from exc

    def view(self, directory: Path, stem: str) -> tuple[list[str | None], Path]:
        """Switch stems without rerunning inference; verify saved evidence hashes."""
        if (
            stem not in STEMS
            or directory.resolve().parent != self.settings.output_dir.resolve()
            or not directory.name.startswith("comparison-")
        ):
            raise StemScopeError("Choose a valid saved comparison and stem.")
        protocol = json.loads((directory / "comparison.json").read_text())
        if len(protocol["models"]) != 2:
            raise StemScopeError("The comparison is not finished yet.")
        paths = [entry["stems"].get(stem) for entry in protocol["models"]]
        for entry, path in zip(protocol["models"], paths):
            if path and fingerprint(Path(path)) != entry["output_hashes"][stem]:
                raise StemScopeError("A compared stem changed. Run a new comparison.")
        snapshot = Path(protocol["input"]["path"])
        if fingerprint(snapshot) != protocol["input"]["sha256"]:
            raise StemScopeError("The captured mix changed. Run a new comparison.")
        figure = directory / f"spectrogram-{stem}.png"
        render_spectrograms(
            snapshot,
            list(zip([entry["model"] for entry in protocol["models"]], paths)),
            stem,
            figure,
        )
        return paths, figure


def spectrum(samples: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Channel-power average with a fixed amplitude-one reference, not per-panel peaks."""
    size = min(1024, len(samples))
    frequencies, times, values = stft(
        samples.T, fs=rate, nperseg=size, noverlap=size // 2, boundary=None, padded=False
    )
    power = np.mean(abs(values) ** 2, axis=0)
    return frequencies, times, 10 * np.log10(np.maximum(power, 1e-10))


def render_spectrograms(
    mixture: Path, models: list[tuple[str, str | None]], stem: str, destination: Path
) -> None:
    """Render original mix and model stems on one frequency and color scale."""
    os.environ.setdefault("MPLCONFIGDIR", str(destination.parent.parent / ".matplotlib"))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    entries = [("Original mix", str(mixture)), *models]
    loaded = {name: read_prefix(Path(path)) for name, path in entries if path}
    rate = min(22050, *[value[1] for value in loaded.values()])
    figure = Figure(figsize=(11, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(entries), 1, sharex=True, sharey=True)
    artist = None
    for axis, (name, path) in zip(axes, entries):
        axis.set_title(name)
        axis.set_ylabel("Frequency (Hz)")
        if path is None:
            axis.text(
                0.5,
                0.5,
                "Model failed — audio unavailable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            continue
        samples, source_rate, _ = loaded[name]
        frequencies, times, db = spectrum(resample(samples, source_rate, rate), rate)
        artist = axis.pcolormesh(
            times, frequencies, db, shading="nearest", cmap="magma", vmin=-100, vmax=0
        )
    axes[-1].set_xlabel("Time from start (seconds)")
    axes[-1].set_xlim(0, min(30, loaded["Original mix"][2]))
    axes[-1].set_ylim(0, rate / 2)
    if artist is not None:
        figure.colorbar(artist, ax=list(axes), label="STFT level (dB relative to amplitude 1)")
    figure.suptitle(
        f"{stem.title()} comparison · first 30 seconds or track end\nShared scale; panels are not normalized independently"
    )
    figure.savefig(destination, dpi=130)
    figure.clear()
