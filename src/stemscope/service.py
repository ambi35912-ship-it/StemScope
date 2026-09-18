import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf

from stemscope.audio.loader import load_audio
from stemscope.audio.validation import validate_path
from stemscope.config import Settings
from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS, Separator
from stemscope.utils.timing import elapsed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeparationResult:
    filename: str
    stems: dict[str, Path]
    seconds: float
    model: str


def validate_outputs(paths: dict[str, Path], directory: Path, duration: float) -> None:
    """Check containment, decodability, duration and consistent stem formats."""
    if set(paths) != set(STEMS):
        raise StemScopeError("Separation did not produce all four expected stems.")
    formats = set()
    for path in paths.values():
        if path.resolve().parent != directory.resolve() or path.suffix != ".wav":
            raise StemScopeError("Separator returned an invalid output path.")
        try:
            with sf.SoundFile(path) as audio:
                if audio.frames == 0 or abs(audio.frames / audio.samplerate - duration) > 0.1:
                    raise StemScopeError("A generated stem is empty or has an incorrect duration.")
                formats.add((audio.frames, audio.samplerate, audio.channels))
                for block in audio.blocks(blocksize=65536):
                    if not np.isfinite(block).all():
                        raise StemScopeError("A generated stem contains invalid samples.")
        except (OSError, RuntimeError) as exc:
            raise StemScopeError("A generated stem is missing or unreadable.") from exc
    if len(formats) != 1:
        raise StemScopeError("Generated stems have inconsistent audio formats.")


class SeparationService:
    """Coordinate validation, execution and job lifecycle independently of the UI."""

    def __init__(self, separator: Separator, settings: Settings):
        self.separator = separator
        self.settings = settings

    def process(self, upload: str | Path) -> SeparationResult:
        start = perf_counter()
        directory = None
        try:
            source = validate_path(upload, self.settings.max_bytes)
            samples, rate = load_audio(source, self.settings.max_seconds)
            self.settings.output_dir.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="job-", dir=self.settings.output_dir))
            paths = self.separator.separate(samples, rate, directory)
            validate_outputs(paths, directory, len(samples) / rate)
            result = SeparationResult(source.name, paths, elapsed(start), self.separator.name)
            (directory / "result.json").write_text(
                json.dumps(
                    {
                        "original_filename": result.filename,
                        "model": result.model,
                        "processing_seconds": result.seconds,
                        "stems": {name: path.name for name, path in paths.items()},
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            logger.info("Separation completed job=%s seconds=%.3f", directory.name, result.seconds)
            return result
        except Exception as exc:
            logger.exception("Separation failed")
            if directory is not None:
                try:
                    shutil.rmtree(directory)
                except OSError:
                    logger.exception("Could not remove failed job %s", directory)
            if isinstance(exc, StemScopeError):
                raise
            if isinstance(exc, OSError):
                raise StemScopeError(
                    "Cannot create or write output files. Check storage permissions and free space."
                ) from exc
            raise StemScopeError(
                "Separation failed unexpectedly. See the application logs."
            ) from exc
