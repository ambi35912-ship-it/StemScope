from pathlib import Path

import numpy as np
import soundfile as sf

from stemscope.errors import StemScopeError


def load_audio(path: Path, max_seconds: int = 1200) -> tuple[np.ndarray, int]:
    """Decode bounded mono/stereo WAV or MP3 into frames × channels float32."""
    try:
        with sf.SoundFile(path) as audio:
            if audio.format not in {"WAV", "WAVEX", "RF64", "MP3"}:
                raise StemScopeError("The file contents are not WAV or MP3 audio.")
            if audio.channels not in (1, 2):
                raise StemScopeError("Only mono and stereo audio are supported.")
            if not audio.frames or audio.frames > max_seconds * audio.samplerate:
                raise StemScopeError(
                    f"Audio must be nonempty and at most {max_seconds // 60} minutes."
                )
            samples = audio.read(dtype="float32", always_2d=True)
            rate = audio.samplerate
    except (OSError, RuntimeError, ValueError) as exc:
        raise StemScopeError("Audio cannot be decoded. Try exporting a fresh WAV or MP3.") from exc
    if samples.size == 0 or not np.isfinite(samples).all():
        raise StemScopeError("Audio contains empty or invalid samples.")
    return samples, rate
