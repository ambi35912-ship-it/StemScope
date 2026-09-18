"""Render synchronized, bounded practice mixes from existing stems."""

import importlib.metadata
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from time import perf_counter

import librosa
import numpy as np
import soundfile as sf

from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS

logger = logging.getLogger(__name__)
MAX_SECTION_SECONDS = 120


@dataclass(frozen=True)
class PracticeOptions:
    gains: dict[str, float] = field(default_factory=lambda: dict.fromkeys(STEMS, 1.0))
    muted: tuple[str, ...] = ()
    solo: tuple[str, ...] = ()
    start_seconds: float = 0
    end_seconds: float = 0
    speed: float = 1
    pitch_semitones: float = 0

    def validate(self) -> None:
        values = [
            *self.gains.values(),
            self.start_seconds,
            self.end_seconds,
            self.speed,
            self.pitch_semitones,
        ]
        if set(self.gains) != set(STEMS) or not all(
            isinstance(v, Real) and np.isfinite(v) for v in values
        ):
            raise StemScopeError("Provide finite volume, section, speed and pitch values.")
        if set(self.muted) - set(STEMS) or set(self.solo) - set(STEMS):
            raise StemScopeError("Unknown mute or solo stem.")
        if any(not 0 <= gain <= 1.5 for gain in self.gains.values()):
            raise StemScopeError("Stem volumes must be between 0% and 150%.")
        if not 0.5 <= self.speed <= 1.5 or not -12 <= self.pitch_semitones <= 12:
            raise StemScopeError("Use speed 0.5–1.5× and pitch within ±12 semitones.")
        if self.start_seconds < 0 or self.end_seconds < 0:
            raise StemScopeError("Section times cannot be negative.")

    def active_gains(self) -> dict[str, float]:
        """Mute wins over solo; selected solo stems exclude all other stems."""
        self.validate()
        return {
            name: gain if name not in self.muted and (not self.solo or name in self.solo) else 0
            for name, gain in self.gains.items()
        }


@dataclass(frozen=True)
class PracticeResult:
    audio: Path
    manifest: Path
    source_start: float
    source_end: float
    output_seconds: float
    normalization_gain: float
    render_seconds: float


def transform_mix(samples: np.ndarray, rate: int, speed: float, pitch: float) -> np.ndarray:
    """Change pitch and tempo independently, processing the synchronized mix once."""
    audio = samples.T
    if pitch != 0:
        audio = librosa.effects.pitch_shift(audio, sr=rate, n_steps=pitch)
    if speed != 1:
        audio = librosa.effects.time_stretch(audio, rate=speed)
    return np.asarray(audio.T, dtype=np.float32)


def render_practice(
    paths: dict[str, Path], options: PracticeOptions, output: Path
) -> PracticeResult:
    """Seek/read at most 120 source seconds, mix once, transform and export a new WAV."""
    start_time = perf_counter()
    directory = None
    try:
        gains = options.active_gains()
        if set(paths) != set(STEMS):
            raise StemScopeError("Separate a song first; practice mode needs all four stems.")
        if not any(gains.values()):
            raise StemScopeError(
                "No stems are audible. Unmute a stem, clear Solo, or raise its volume."
            )
        infos = {name: sf.info(path) for name, path in paths.items()}
        reference = infos[STEMS[0]]
        if reference.channels not in (1, 2) or reference.frames < 1:
            raise StemScopeError("Practice requires nonempty mono or stereo stems.")
        if any(
            (info.frames, info.samplerate, info.channels)
            != (reference.frames, reference.samplerate, reference.channels)
            for info in infos.values()
        ):
            raise StemScopeError(
                "Practice stems must have the same rate, channels and exact length."
            )
        rate, duration = reference.samplerate, reference.duration
        if options.start_seconds >= duration:
            raise StemScopeError(f"Start must be before the track ends ({duration:.2f} seconds).")
        end = min(
            duration,
            options.end_seconds
            if options.end_seconds
            else options.start_seconds + MAX_SECTION_SECONDS,
        )
        length = end - options.start_seconds
        if length < 0.1 - 1e-8 or length > MAX_SECTION_SECONDS + 1e-8:
            raise StemScopeError("Choose a section between 0.1 and 120 seconds long.")
        first, last = round(options.start_seconds * rate), min(reference.frames, round(end * rate))
        mix = np.zeros((last - first, reference.channels), dtype=np.float32)
        fingerprints = {}
        import hashlib

        for name in STEMS:
            with sf.SoundFile(paths[name]) as stream:
                stream.seek(first)
                samples = stream.read(last - first, dtype="float32", always_2d=True)
            if samples.shape != mix.shape or not np.isfinite(samples).all():
                raise StemScopeError(
                    "A selected stem section is missing or contains invalid samples."
                )
            # Hash exactly the decoded segment used, so long files are not reread in full.
            fingerprints[name] = {
                "path": str(paths[name].resolve()),
                "decoded_segment_sha256": hashlib.sha256(
                    np.asarray(samples, dtype="<f4").tobytes()
                ).hexdigest(),
            }
            mix += samples * gains[name]
        processed = transform_mix(mix, rate, options.speed, options.pitch_semitones)
        if not processed.size or not np.isfinite(processed).all():
            raise StemScopeError("Practice transformation produced invalid audio.")
        # Short edge fades soften loop clicks without deleting or shifting samples.
        fade = min(round(rate * 0.005), len(processed) // 2)
        if fade:
            ramp = np.linspace(0, 1, fade, dtype=np.float32)[:, None]
            processed[:fade] *= ramp
            processed[-fade:] *= ramp[::-1]
        peak = float(np.max(abs(processed)))
        normalization = min(1.0, 0.98 / peak) if peak else 1.0
        processed *= normalization
        output.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="practice-", dir=output))
        audio_path, manifest_path = directory / "practice.wav", directory / "practice.json"
        sf.write(audio_path, processed, rate, subtype="PCM_16")
        elapsed = perf_counter() - start_time
        manifest = {
            "protocol": "practice-render-v1",
            "versions": {
                name: importlib.metadata.version(name) for name in ("librosa", "numpy", "soundfile")
            },
            "inputs": fingerprints,
            "source_frames": [first, last],
            "source_start_seconds": first / rate,
            "source_end_seconds": last / rate,
            "sample_rate": rate,
            "channels": reference.channels,
            "effective_gains": gains,
            "requested_gains": options.gains,
            "muted": list(options.muted),
            "solo": list(options.solo),
            "speed": options.speed,
            "pitch_semitones": options.pitch_semitones,
            "output_frames": len(processed),
            "output_seconds": len(processed) / rate,
            "peak_before_headroom_gain": peak,
            "normalization_gain": normalization,
            "edge_fade_seconds": fade / rate,
            "render_seconds": elapsed,
            "notes": "Speed preserves pitch; pitch shift preserves requested tempo. Transform artifacts may be audible. Loop repeats this rendered section in the player; browser looping is not guaranteed sample-gapless. New control values take effect after rendering again.",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        return PracticeResult(
            audio_path,
            manifest_path,
            first / rate,
            last / rate,
            len(processed) / rate,
            normalization,
            elapsed,
        )
    except Exception as exc:
        logger.exception("Practice rendering failed")
        if directory is not None:
            try:
                shutil.rmtree(directory)
            except OSError:
                logger.exception("Cannot clean up failed practice render %s", directory)
        if isinstance(exc, StemScopeError):
            raise
        raise StemScopeError(
            "Cannot render the practice mix. Check that the stems are readable and storage is available."
        ) from exc
