"""Analysis orchestration; usable without a model or Gradio."""

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from stemscope.audio.features import AudioAnalysis, analyze_samples
from stemscope.audio.loader import load_audio
from stemscope.audio.plots import render_analysis
from stemscope.audio.validation import validate_path
from stemscope.config import Settings
from stemscope.errors import StemScopeError
from stemscope.utils.timing import elapsed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisResult:
    filename: str
    audio: AudioAnalysis
    plots: dict[str, Path]
    seconds: float


class AnalysisService:
    """Validate and analyze an upload, retaining charts only for successful jobs."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def process(self, upload: str | Path) -> AnalysisResult:
        directory = None
        start = perf_counter()
        try:
            source = validate_path(upload, self.settings.max_bytes)
            samples, rate = load_audio(source, self.settings.max_seconds)
            self.settings.output_dir.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="analysis-", dir=self.settings.output_dir))
            audio = analyze_samples(samples, rate)
            plots = render_analysis(audio, directory)
            result = AnalysisResult(source.name, audio, plots, elapsed(start))
            logger.info("Analysis completed job=%s seconds=%.3f", directory.name, result.seconds)
            return result
        except Exception as exc:
            logger.exception("Audio analysis failed")
            if directory is not None:
                try:
                    shutil.rmtree(directory)
                except OSError:
                    logger.exception("Could not remove failed analysis %s", directory)
            if isinstance(exc, StemScopeError):
                raise
            if isinstance(exc, OSError):
                raise StemScopeError(
                    "Cannot write analysis charts. Check storage permissions and free space."
                ) from exc
            raise StemScopeError(
                "Audio analysis failed. See the application logs for details."
            ) from exc
