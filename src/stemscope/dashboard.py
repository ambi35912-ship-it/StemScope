"""Build a coherent technical dashboard from one captured upload."""

import json
import logging
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from stemscope.analysis import AnalysisService
from stemscope.application import choose_model
from stemscope.audio.validation import validate_path
from stemscope.benchmarking.results import completed_study
from stemscope.comparison import ComparisonService
from stemscope.config import Settings
from stemscope.errors import StemScopeError
from stemscope.quality.diagnostics import inspect_files
from stemscope.quality.service import fingerprint
from stemscope.service import SeparationService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DashboardResult:
    manifest: Path
    payload: dict


class DashboardService:
    """Compose existing analysis/selection/separation services without duplicate inference."""

    def __init__(
        self,
        services: Mapping[str, SeparationService],
        settings: Settings,
        analysis: AnalysisService | None = None,
    ):
        self.services = services
        self.settings = settings
        self.analysis = analysis or AnalysisService(settings)
        self.comparison = ComparisonService(services, settings)

    def build(self, upload: str | Path, choice: str, compare_both: bool = False) -> DashboardResult:
        source = validate_path(upload, self.settings.max_bytes)
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="dashboard-", dir=self.settings.output_dir))
        started = perf_counter()
        manifest_path = directory / "dashboard.json"
        try:
            captured = directory / f"input{source.suffix.lower()}"
            shutil.copyfile(source, captured)
            analysis = self.analysis.process(captured)
            audio = analysis.audio
            payload = {
                "version": "dashboard-v1",
                "input": {
                    "original_filename": source.name,
                    "path": str(captured.resolve()),
                    "sha256": fingerprint(captured),
                },
                "audio": {
                    "duration_seconds": audio.duration,
                    "sample_rate": audio.sample_rate,
                    "channels": audio.channels,
                    "analysis_rate": audio.analysis_rate,
                    "rms_mean": float(np.mean(audio.rms)),
                    "centroid_mean_hz": float(np.mean(audio.centroid)),
                    "bandwidth_mean_hz": float(np.mean(audio.bandwidth)),
                    "zcr_mean": float(np.mean(audio.zcr)),
                    "tempo_bpm": audio.tempo_bpm,
                    "tempo_note": audio.tempo_note,
                },
                "charts": {name: str(path.resolve()) for name, path in analysis.plots.items()},
                "analysis_seconds": analysis.seconds,
                "requested_choice": choice,
                "selected_model": None,
                "selection_note": "",
                "compare_both": compare_both,
                "models": [],
                "selected_stems": {},
                "selected_diagnostics": None,
                "comparison_chart": None,
                "comparison_manifest": None,
                "errors": [],
                "upload_reference_quality": "Unavailable: no original reference stems supplied. Historical benchmark results below concern different recordings.",
            }
            try:
                selected, reason = choose_model(captured, choice, self.settings)
                if selected not in self.services:
                    raise StemScopeError("Choose one of the available models.")
                payload["selected_model"], payload["selection_note"] = (
                    selected,
                    reason or "Manual model choice",
                )
            except StemScopeError as exc:
                payload["errors"].append(f"Selection: {exc}")
                selected = None

            if selected is not None and compare_both:
                try:
                    comparison = self.comparison.run(captured)
                    evidence = json.loads(comparison.manifest.read_text())
                    payload["models"] = evidence["models"]
                    payload["comparison_manifest"] = str(comparison.manifest.resolve())
                    if evidence["input"]["sha256"] != payload["input"]["sha256"]:
                        raise StemScopeError(
                            "Comparison input differs from the captured dashboard input."
                        )
                    for model in payload["models"]:
                        if model["status"] != "ok":
                            payload["errors"].append(f"{model['model']}: {model['error']}")
                        if model.get("diagnostic_error"):
                            payload["errors"].append(
                                f"{model['model']}: {model['diagnostic_error']}"
                            )
                    try:
                        _, plot = self.comparison.view(comparison.directory, "vocals")
                        payload["comparison_chart"] = str(plot.resolve())
                    except Exception:
                        logger.exception("Dashboard comparison figure unavailable")
                        payload["errors"].append(
                            "Comparison chart unavailable; other evidence is retained."
                        )
                except Exception as exc:
                    logger.exception("Dashboard comparison failed")
                    payload["errors"].append(
                        str(exc)
                        if isinstance(exc, StemScopeError)
                        else "Comparison failed; input analysis is retained."
                    )
            elif selected is not None:
                service = self.services[selected]
                model = {
                    "model": selected,
                    "status": "failed",
                    "stems": {},
                    "device": service.settings.device,
                    "adapter_settings": service.separator.configuration,
                    "error": "",
                    "diagnostic_error": "",
                }
                try:
                    result = service.process(captured)
                    model.update(
                        status="ok",
                        stems={name: str(path.resolve()) for name, path in result.stems.items()},
                        separation_seconds=result.seconds,
                        output_hashes={
                            name: fingerprint(path) for name, path in result.stems.items()
                        },
                    )
                    try:
                        model["diagnostics"] = inspect_files(captured, result.stems)
                    except Exception:
                        logger.exception("Dashboard diagnostics failed")
                        model["diagnostic_error"] = (
                            "Diagnostics unavailable; separated audio is retained."
                        )
                        payload["errors"].append(model["diagnostic_error"])
                except Exception as exc:
                    logger.exception("Dashboard separation failed")
                    model["error"] = (
                        str(exc)
                        if isinstance(exc, StemScopeError)
                        else "Separation failed; input analysis is retained."
                    )
                    payload["errors"].append(model["error"])
                payload["models"].append(model)

            entry = next((model for model in payload["models"] if model["model"] == selected), None)
            if entry and entry["status"] == "ok":
                payload["selected_stems"] = entry["stems"]
                payload["selected_diagnostics"] = entry.get("diagnostics")
            try:
                status, table, chart, report, csv = completed_study(
                    self.settings.output_dir / "phase4-study"
                )
                payload["historical_benchmark"] = {
                    "status": status,
                    "rows": table,
                    "report": report,
                    "results_csv": csv,
                    "chart": chart,
                    "scope": "Historical reference corpus; these are not quality scores for the uploaded song.",
                    "results_sha256": fingerprint(Path(csv)),
                }
            except (StemScopeError, OSError, ValueError, KeyError) as exc:
                payload["historical_benchmark"] = {
                    "status": f"Unavailable: {exc}",
                    "rows": [],
                    "scope": "No historical benchmark was loaded; this does not affect current-file processing.",
                }
            payload["total_seconds"] = perf_counter() - started
            payload["status"] = (
                "Complete"
                if not payload["errors"]
                else "Finished with issues; available results retained"
            )
            payload["timing_note"] = (
                "Total is dashboard wall time up to final manifest export. Model times are individual interactive calls, may include loading and are not controlled benchmark timings. Input analysis, selection, diagnostics and chart/export overhead are separate from model call timings."
            )
            manifest_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            return DashboardResult(manifest_path, payload)
        except Exception as exc:
            logger.exception("Dashboard build failed")
            if not manifest_path.exists():
                try:
                    shutil.rmtree(directory)
                except OSError:
                    logger.exception("Could not clean failed dashboard input")
            if isinstance(exc, StemScopeError):
                raise
            raise StemScopeError(
                "Cannot build or save the dashboard. Check input audio and output storage."
            ) from exc


def dashboard_tables(payload: dict) -> tuple[list, list, list]:
    """Present recorded evidence without substituting corpus scores for upload quality."""
    audio = payload["audio"]
    summary = [
        ["File", payload["input"]["original_filename"]],
        ["Duration (s)", audio["duration_seconds"]],
        ["Sample rate (Hz)", audio["sample_rate"]],
        ["Channels", audio["channels"]],
        ["RMS", audio["rms_mean"]],
        ["Spectral centroid (Hz)", audio["centroid_mean_hz"]],
        ["Spectral bandwidth (Hz)", audio["bandwidth_mean_hz"]],
        ["Zero-crossing rate", audio["zcr_mean"]],
        ["Tempo (BPM)", audio["tempo_bpm"] if audio["tempo_bpm"] is not None else "Unavailable"],
        ["Selected model", payload["selected_model"] or "Unavailable"],
        ["Input analysis (s)", payload["analysis_seconds"]],
        ["Total dashboard time (s)", payload["total_seconds"]],
    ]
    models = [
        [
            row["model"],
            row["status"],
            row.get("separation_seconds"),
            row["device"],
            row.get("error") or row.get("diagnostic_error") or "",
        ]
        for row in payload["models"]
    ]
    diagnostic_rows = (
        payload.get("selected_diagnostics", {}).get("rows", [])
        if payload.get("selected_diagnostics")
        else []
    )
    diagnostics = [
        [
            row["stem"],
            row["review_status"],
            "; ".join(row["flags"]) or "No configured flags; quality unverified",
            row["rms_dbfs"],
            row["max_waveform_similarity"],
        ]
        for row in diagnostic_rows
    ]
    return summary, models, diagnostics
