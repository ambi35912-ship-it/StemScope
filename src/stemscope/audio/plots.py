"""Render standalone charts without a GUI backend or pyplot global figures."""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from stemscope.audio.features import AudioAnalysis


def _save(figure: "Figure", path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(figure)
    figure.savefig(path, dpi=130, facecolor="white")
    figure.clear()


def render_analysis(analysis: AudioAnalysis, directory: Path) -> dict[str, Path]:
    """Write waveform, spectrogram and feature charts into a job directory."""
    os.environ.setdefault("MPLCONFIGDIR", str(directory.parent / ".matplotlib"))
    from matplotlib.figure import Figure

    paths = {name: directory / f"{name}.png" for name in ("waveform", "spectrogram", "features")}
    duration = max(analysis.duration, 1 / analysis.sample_rate)
    fig = Figure(figsize=(11, 2 + 1.5 * analysis.channels), layout="constrained")
    axes = fig.subplots(analysis.channels, 1, squeeze=False)
    for channel, ax in enumerate(axes[:, 0]):
        ax.vlines(
            analysis.waveform_times,
            analysis.waveform_min[:, channel],
            analysis.waveform_max[:, channel],
            color="#2563eb",
            linewidth=0.7,
        )
        # Envelope outlines also make silence and single-sample bins visible.
        ax.plot(
            analysis.waveform_times,
            analysis.waveform_min[:, channel],
            color="#2563eb",
            linewidth=0.5,
        )
        ax.plot(
            analysis.waveform_times,
            analysis.waveform_max[:, channel],
            color="#2563eb",
            linewidth=0.5,
        )
        ax.set(
            xlim=(0, duration),
            ylabel="Amplitude",
            title="Mono waveform" if analysis.channels == 1 else f"Channel {channel + 1}",
        )
        ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Time (seconds)")
    _save(fig, paths["waveform"])

    fig = Figure(figsize=(11, 4), layout="constrained")
    ax = fig.subplots()
    image = ax.imshow(
        analysis.spectrogram_db,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=(0, duration, 0, analysis.analysis_rate / 2),
        vmin=-80,
        vmax=0,
    )
    ax.set(
        xlabel="Time (seconds)",
        ylabel="Frequency (Hz)",
        title="Spectrogram · relative level (peak-preserving time bins)",
    )
    fig.colorbar(image, ax=ax, label="dB relative to track peak")
    _save(fig, paths["spectrogram"])

    fig = Figure(figsize=(11, 8), layout="constrained")
    axes = fig.subplots(4, 1, sharex=True)
    curves = (
        (analysis.rms, "RMS amplitude", "#2563eb"),
        (analysis.centroid, "Centroid (Hz)", "#7c3aed"),
        (analysis.bandwidth, "Bandwidth (Hz)", "#059669"),
        (analysis.zcr, "ZCR (crossings/sample)", "#d97706"),
    )
    for ax, (values, label, color) in zip(axes, curves):
        ax.plot(analysis.times, values, color=color, linewidth=0.9)
        ax.set(ylabel=label, xlim=(0, duration))
        ax.set_ylim(bottom=0, top=max(float(np.max(values)) * 1.1, 1e-6))
        ax.grid(alpha=0.2)
    axes[0].set_title("Audio features over time")
    axes[-1].set_xlabel("Time (seconds)")
    _save(fig, paths["features"])
    return paths
