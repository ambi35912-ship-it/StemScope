"""Generate known synthetic components solely to smoke-test evaluation plumbing."""

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from stemscope.benchmarking.runner import BenchmarkTrack
from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS


def create_demo(root: Path) -> BenchmarkTrack:
    """Create a unique four-second example; never overwrite a user's recordings."""
    try:
        root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="synthetic-check-", dir=root))
        rate = 44100
        times = np.arange(rate * 4) / rate
        rng = np.random.default_rng(0)
        signals = [
            0.08 * np.sin(2 * np.pi * 440 * times),
            0.03 * rng.normal(size=len(times)) * (times % 0.5 < 0.05),
            0.08 * np.sin(2 * np.pi * 110 * times),
            0.06 * np.sin(2 * np.pi * 660 * times),
        ]
        references = {}
        for name, signal in zip(STEMS, signals):
            path = directory / f"{name}.wav"
            sf.write(path, np.column_stack([signal, signal]), rate, subtype="FLOAT")
            references[name] = path
        mixture = directory / "mixture.wav"
        signal = np.sum(signals, axis=0)
        sf.write(mixture, np.column_stack([signal, signal]), rate, subtype="FLOAT")
        return BenchmarkTrack(
            "SYNTHETIC CHECK — not music",
            mixture,
            references,
            "Generated tones/noise; stem names are placeholders, not real instruments; not musical quality evidence",
        )
    except OSError as exc:
        raise StemScopeError(
            "Cannot create synthetic audio. Check output storage permissions."
        ) from exc
