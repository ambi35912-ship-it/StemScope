"""Fixed mixture-only features shared by selector training and prediction."""

from pathlib import Path

import numpy as np
import soundfile as sf

from stemscope.audio.features import analyze_samples
from stemscope.errors import StemScopeError

FEATURE_NAMES = [
    "rms_mean",
    "rms_std",
    "rms_q10",
    "rms_q90",
    "centroid_mean",
    "centroid_std",
    "bandwidth_mean",
    "bandwidth_std",
    "zcr_mean",
    "zcr_std",
    "tempo_bpm",
]
PROTOCOL = "mixture-v1: first 7 seconds; Phase2 channel-aware features; tempo missing=NaN; training-only median imputation"


def extract(path: Path) -> list[float]:
    """Read at most seven seconds; do not inspect names, references or separator outputs."""
    try:
        with sf.SoundFile(path) as audio:
            rate = audio.samplerate
            samples = audio.read(frames=7 * rate, dtype="float32", always_2d=True)
        if len(samples) < rate:
            raise StemScopeError("The experimental selector needs at least one second of audio.")
        analysis = analyze_samples(samples, rate)
        return [
            float(np.mean(analysis.rms)),
            float(np.std(analysis.rms)),
            float(np.quantile(analysis.rms, 0.1)),
            float(np.quantile(analysis.rms, 0.9)),
            float(np.mean(analysis.centroid)),
            float(np.std(analysis.centroid)),
            float(np.mean(analysis.bandwidth)),
            float(np.std(analysis.bandwidth)),
            float(np.mean(analysis.zcr)),
            float(np.std(analysis.zcr)),
            analysis.tempo_bpm if analysis.tempo_bpm is not None else float("nan"),
        ]
    except (OSError, RuntimeError) as exc:
        raise StemScopeError("Cannot decode audio for model selection.") from exc
