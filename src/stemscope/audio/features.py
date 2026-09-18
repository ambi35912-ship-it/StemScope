"""Descriptive full-track features, independent of separators and Gradio."""

from dataclasses import dataclass
from math import ceil, gcd

import librosa
import numpy as np
from scipy.signal import resample_poly

from stemscope.errors import StemScopeError

FRAME_LENGTH = 2048
HOP_LENGTH = 512
ANALYSIS_RATE = 22050
TEMPO_SECONDS = 120


@dataclass(frozen=True)
class AudioAnalysis:
    """Original metadata and descriptive measurements, never quality scores."""

    duration: float
    sample_rate: int
    channels: int
    analysis_rate: int
    times: np.ndarray
    rms: np.ndarray
    centroid: np.ndarray
    bandwidth: np.ndarray
    zcr: np.ndarray
    waveform_times: np.ndarray
    waveform_min: np.ndarray
    waveform_max: np.ndarray
    spectrogram_db: np.ndarray
    tempo_bpm: float | None
    tempo_note: str


def _tempo(samples: np.ndarray, rate: int) -> tuple[float | None, str]:
    """Estimate rhythm from up to 120 seconds without cancelling stereo channels."""
    if len(samples) < 4 * rate:
        return None, "Unavailable: at least 4 seconds of audio is needed."
    excerpt = samples[: TEMPO_SECONDS * rate]
    # Average channel onset envelopes rather than mixing anti-phase waveforms.
    envelopes = [
        librosa.onset.onset_strength(y=channel, sr=rate, hop_length=HOP_LENGTH)
        for channel in excerpt.T
    ]
    onset = np.mean(envelopes, axis=0)
    if np.max(onset) < 1e-5:
        return None, "Unavailable: no clear rhythmic onsets were detected."
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=rate, hop_length=HOP_LENGTH)
    bpm = float(np.asarray(tempo).reshape(-1)[0])
    if len(beats) < 3 or not np.isfinite(bpm) or bpm <= 0:
        return None, "Unavailable: too few consistent beats were detected."
    return bpm, (
        f"Estimate from the first {len(excerpt) / rate:.1f} seconds; "
        "may be half or double the musical tempo, or inaccurate for changing rhythms."
    )


def analyze_samples(samples: np.ndarray, sample_rate: int) -> AudioAnalysis:
    """Analyze frames × channels audio using bounded STFT blocks.

    Preserve channel polarity in the waveform. Aggregate channel spectral power
    before spectral features; RMS combines channel energy and ZCR averages channel
    rates. Feature audio is low-pass resampled to at most 22,050 Hz. Overview plots
    use peak-preserving bins; summary curves cover the full recording.
    """
    if (
        samples.ndim != 2
        or samples.shape[1] not in (1, 2)
        or len(samples) == 0
        or sample_rate <= 0
        or not np.isfinite(samples).all()
    ):
        raise StemScopeError("Analysis requires nonempty, finite mono or stereo audio.")
    duration = len(samples) / sample_rate
    bins = min(2000, len(samples))
    edges = np.linspace(0, len(samples), bins + 1, dtype=int)
    wave_min = np.minimum.reduceat(samples, edges[:-1], axis=0)
    wave_max = np.maximum.reduceat(samples, edges[:-1], axis=0)
    wave_times = (edges[:-1] + edges[1:]) / (2 * sample_rate)

    rate = min(sample_rate, ANALYSIS_RATE)
    if rate != sample_rate:
        factor = gcd(rate, sample_rate)
        audio = resample_poly(samples, rate // factor, sample_rate // factor, axis=0)
    else:
        audio = samples
    audio = np.asarray(audio, dtype=np.float32)
    frames = max(1, ceil(len(audio) / HOP_LENGTH))
    times = np.arange(frames) * HOP_LENGTH / rate
    rms, centroid, bandwidth, zcr = (np.zeros(frames) for _ in range(4))
    stride = max(1, ceil(frames / 1200))
    overview = np.zeros((FRAME_LENGTH // 2 + 1, ceil(frames / stride)), dtype=np.float32)
    padded = np.pad(audio.T, ((0, 0), (FRAME_LENGTH // 2, FRAME_LENGTH // 2)))

    for start in range(0, frames, 256):
        stop = min(frames, start + 256)
        segment = padded[:, start * HOP_LENGTH : (stop - 1) * HOP_LENGTH + FRAME_LENGTH]
        spectrum = np.abs(
            librosa.stft(segment, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH, center=False)
        )
        magnitude = np.sqrt(np.mean(spectrum**2, axis=0))
        channel_rms = librosa.feature.rms(
            y=segment, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=False
        )[:, 0, :]
        rms[start:stop] = np.sqrt(np.mean(channel_rms**2, axis=0))
        centroid[start:stop] = librosa.feature.spectral_centroid(S=magnitude, sr=rate)[0]
        bandwidth[start:stop] = librosa.feature.spectral_bandwidth(S=magnitude, sr=rate)[0]
        zcr[start:stop] = librosa.feature.zero_crossing_rate(
            y=segment, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, center=False
        )[:, 0, :].mean(axis=0)
        np.maximum.at(overview.T, np.arange(start, stop) // stride, magnitude.T)

    if overview.max() > 0:
        db = librosa.amplitude_to_db(overview, ref=np.max, top_db=80)
    else:
        db = np.full_like(overview, -80)
    bpm, note = _tempo(audio, rate)
    return AudioAnalysis(
        duration,
        sample_rate,
        samples.shape[1],
        rate,
        times,
        rms,
        centroid,
        bandwidth,
        zcr,
        wave_times,
        wave_min,
        wave_max,
        db,
        bpm,
        note,
    )
