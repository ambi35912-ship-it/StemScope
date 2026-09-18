"""Interpretable, bounded signal checks for separated audio."""

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, stft

from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS

PROTOCOL = {
    "version": "signal-diagnostics-v1",
    "max_seconds": 30,
    "analysis_rate": 22050,
    "quiet_rms_dbfs": -60,
    "similarity_threshold": 0.9,
    "over_full_scale_fraction": 0.001,
    "reconstruction_relative_rms_threshold": 0.1,
    "interpretation": "Engineering review thresholds fixed before reference audit; no calibrated quality probability or confirmed leakage/muffling detection.",
}


def read_prefix(path: Path) -> tuple[np.ndarray, int, float]:
    """Decode at most 30 seconds, retaining original full duration for alignment checks."""
    try:
        with sf.SoundFile(path) as stream:
            if stream.channels not in (1, 2):
                raise StemScopeError("Quality checks support mono or stereo audio.")
            rate, duration = stream.samplerate, stream.frames / stream.samplerate
            samples = stream.read(
                frames=round(PROTOCOL["max_seconds"] * rate), dtype="float32", always_2d=True
            )
    except (OSError, RuntimeError) as exc:
        raise StemScopeError("A quality-check audio file cannot be read.") from exc
    if not samples.size or not np.isfinite(samples).all():
        raise StemScopeError("Quality checks require nonempty, finite audio.")
    return samples, rate, duration


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    divisor = gcd(source_rate, target_rate)
    return resample_poly(samples, target_rate // divisor, source_rate // divisor, axis=0)


def similarity(first: np.ndarray, second: np.ndarray) -> float | None:
    """Absolute zero-lag waveform correlation after per-channel DC removal; no delay search."""
    a, b = first.astype(float), second.astype(float)
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)
    aa, bb = float(np.sum(a * a)), float(np.sum(b * b))
    if aa / a.size < 1e-20 or bb / b.size < 1e-20:
        return None
    return float(np.clip(abs(np.sum(a * b)) / np.sqrt(aa * bb), 0, 1))


def spectral_description(samples: np.ndarray, rate: int) -> dict:
    """Descriptive brightness/noise measures only; low bandwidth is normal for bass."""
    if np.mean(samples.astype(float) ** 2) < 1e-20 or len(samples) < 32:
        return {"high_band_fraction": None, "spectral_flatness": None}
    window = min(1024, len(samples))
    frequencies, _, spectrum = stft(
        samples.T, fs=rate, nperseg=window, noverlap=window // 2, boundary=None, padded=False
    )
    power = np.mean(abs(spectrum) ** 2, axis=0).astype(float)
    mean_power = power.mean(axis=0)
    active = mean_power > max(float(mean_power.max()) * 1e-6, 1e-20)
    if not active.any():
        return {"high_band_fraction": None, "spectral_flatness": None}
    power = power[:, active]
    flatness = np.exp(np.mean(np.log(np.maximum(power, 1e-20)), axis=0)) / power.mean(axis=0)
    return {
        "high_band_fraction": float(power[frequencies >= 4000].sum() / power.sum())
        if rate > 8000
        else None,
        "spectral_flatness": float(np.median(flatness)),
    }


def inspect_arrays(
    mixture: np.ndarray, stems: dict[str, np.ndarray], rate: int, original_stats: dict | None = None
) -> dict:
    """Compute evidence and review flags on aligned samples without original references."""
    if (
        set(stems) != set(STEMS)
        or rate <= 0
        or mixture.ndim != 2
        or not mixture.size
        or mixture.shape[1] not in (1, 2)
    ):
        raise StemScopeError("Provide a mix and all four aligned mono/stereo stems.")
    if not np.isfinite(mixture).all() or any(
        s.shape != mixture.shape or not np.isfinite(s).all() for s in stems.values()
    ):
        raise StemScopeError("Quality-check samples must be finite and exactly aligned.")
    shared = {
        name: {other: similarity(stems[name], stems[other]) for other in STEMS if other != name}
        for name in STEMS
    }
    mixture_rms = float(np.sqrt(np.mean(mixture.astype(float) ** 2)))
    residual = mixture.astype(float) - sum(s.astype(float) for s in stems.values())
    residual_rms = float(np.sqrt(np.mean(residual**2)))
    relative = residual_rms / mixture_rms if mixture_rms > 1e-10 else None
    rows = []
    for name, samples in stems.items():
        rms = float(np.sqrt(np.mean(samples.astype(float) ** 2)))
        db = float(20 * np.log10(rms)) if rms > 1e-10 else None
        peak = float(np.max(abs(samples)))
        over = float(np.mean(abs(samples) >= 1))
        if original_stats is not None:
            peak, over = (
                original_stats[name]["sample_peak"],
                original_stats[name]["over_full_scale_fraction"],
            )
        available = {k: v for k, v in shared[name].items() if v is not None}
        peer = max(available, key=available.get) if available else None
        correlation = available[peer] if peer else None
        flags = []
        if db is None or db <= PROTOCOL["quiet_rms_dbfs"]:
            flags.append("very quiet/absent target; may be legitimate")
        if over >= PROTOCOL["over_full_scale_fraction"]:
            flags.append("samples reach/exceed full scale; check playback headroom")
        if correlation is not None and correlation >= PROTOCOL["similarity_threshold"]:
            flags.append(f"strong waveform similarity with {peer}; possible shared content")
        rows.append(
            {
                "stem": name,
                "review_status": "Review recommended" if flags else "No configured flags",
                "flags": flags,
                "rms_dbfs": db,
                "sample_peak": peak,
                "over_full_scale_fraction": over,
                "most_similar_stem": peer,
                "max_waveform_similarity": correlation,
                **spectral_description(samples, rate),
            }
        )
    return {
        "protocol": PROTOCOL,
        "analyzed_seconds": len(mixture) / rate,
        "analysis_rate": rate,
        "rows": rows,
        "pairwise_similarity": shared,
        "reconstruction_relative_rms": relative,
        "reconstruction_status": "Unavailable: mix is silent"
        if relative is None
        else "Review mix/stem consistency"
        if relative > PROTOCOL["reconstruction_relative_rms_threshold"]
        else "Within configured reconstruction threshold",
        "limitations": "No configured flags does not imply clean separation. Shared rhythm or source annotations can produce similarity; silence can be correct. Reconstruction cannot detect mixture-consistent leakage. Brightness/flatness do not establish muffling or watery artifacts. No human-quality rating or accuracy percentage is estimated.",
    }


def inspect_files(mixture_path: Path, stem_paths: dict[str, Path]) -> dict:
    """Align rates/channels with explicit duration checks and bounded prefix decoding."""
    if set(stem_paths) != set(STEMS):
        raise StemScopeError("Supply all four separated stem files.")
    mix, mix_rate, duration = read_prefix(mixture_path)
    loaded = {name: read_prefix(path) for name, path in stem_paths.items()}
    if any(abs(full_duration - duration) > 0.1 for _, _, full_duration in loaded.values()):
        raise StemScopeError("The mix and stems have different durations. Use the matching mix.")
    channels = max([mix.shape[1], *[data.shape[1] for data, _, _ in loaded.values()]])
    target_rate = min(PROTOCOL["analysis_rate"], mix_rate, *[r for _, r, _ in loaded.values()])

    def convert(data, source_rate):
        if data.shape[1] == 1 and channels == 2:
            data = np.repeat(data, 2, axis=1)
        return resample(data, source_rate, target_rate)

    mix = convert(mix, mix_rate)
    stems = {name: convert(data, source_rate) for name, (data, source_rate, _) in loaded.items()}
    lengths = [len(mix), *[len(s) for s in stems.values()]]
    if max(lengths) - min(lengths) > 1:
        raise StemScopeError("Quality-check prefixes do not align; no delay correction is applied.")
    length = min(lengths)
    original = {
        name: {
            "sample_peak": float(np.max(abs(data))),
            "over_full_scale_fraction": float(np.mean(abs(data) >= 1)),
        }
        for name, (data, _, _) in loaded.items()
    }
    result = inspect_arrays(
        mix[:length], {name: data[:length] for name, data in stems.items()}, target_rate, original
    )
    result["full_duration_seconds"] = duration
    return result
