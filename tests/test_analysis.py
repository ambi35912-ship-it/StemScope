import numpy as np
import pytest
import soundfile as sf

from stemscope.analysis import AnalysisService
from stemscope.audio.features import HOP_LENGTH, analyze_samples
from stemscope.config import Settings
from stemscope.errors import StemScopeError


def tone(rate=22050, seconds=2, frequency=440):
    times = np.arange(int(rate * seconds)) / rate
    return (0.5 * np.sin(2 * np.pi * frequency * times)).astype(np.float32)[:, None]


def test_tone_features_have_physical_meaning():
    result = analyze_samples(tone(), 22050)
    interior = slice(3, -3)
    assert result.duration == 2
    assert result.sample_rate == result.analysis_rate == 22050
    assert result.channels == 1
    assert np.mean(result.rms[interior]) == pytest.approx(0.5 / np.sqrt(2), rel=0.02)
    assert np.mean(result.centroid[interior]) == pytest.approx(440, abs=10)
    assert np.mean(result.bandwidth[interior]) < 30
    assert np.mean(result.zcr[interior]) == pytest.approx(2 * 440 / 22050, abs=0.002)
    assert result.tempo_bpm is None
    assert "4 seconds" in result.tempo_note


def test_antiphase_stereo_keeps_energy():
    mono = tone()
    stereo = np.concatenate([mono, -mono], axis=1)
    original = analyze_samples(mono, 22050)
    result = analyze_samples(stereo, 22050)
    np.testing.assert_allclose(result.rms, original.rms, rtol=1e-6)
    np.testing.assert_allclose(result.centroid, original.centroid, rtol=1e-6)
    np.testing.assert_allclose(result.waveform_min[:, 0], -result.waveform_max[:, 1])
    assert result.channels == 2


def test_silence_has_no_fake_tempo_or_bright_spectrogram():
    result = analyze_samples(np.zeros((22050 * 5, 2), dtype=np.float32), 22050)
    assert result.tempo_bpm is None
    for values in (result.rms, result.centroid, result.bandwidth, result.zcr):
        assert np.all(values == 0)
    assert np.all(result.spectrogram_db == -80)


@pytest.mark.parametrize("frames", [1, 17, 511, 512, 513, 2048])
def test_tiny_clips_are_finite_and_aligned(frames):
    result = analyze_samples(np.zeros((frames, 1), dtype=np.float32), 8000)
    for values in (result.rms, result.centroid, result.bandwidth, result.zcr):
        assert values.shape == result.times.shape
        assert np.isfinite(values).all()
    assert result.times[-1] < result.duration


@pytest.mark.parametrize(
    "samples,rate",
    [
        (np.empty((0, 1)), 22050),
        (np.zeros((10, 3)), 22050),
        (np.array([[np.nan]]), 22050),
        (np.zeros((10, 1)), 0),
        (np.zeros(10), 22050),
    ],
)
def test_invalid_arrays(samples, rate):
    with pytest.raises(StemScopeError):
        analyze_samples(samples, rate)


def test_resampling_preserves_original_metadata():
    result = analyze_samples(tone(rate=48000), 48000)
    assert result.sample_rate == 48000
    assert result.analysis_rate == 22050
    assert result.duration == 2
    assert np.mean(result.centroid[3:-3]) == pytest.approx(440, abs=10)


def test_block_boundary_matches_continuous_features():
    import librosa

    samples = tone(seconds=7)
    result = analyze_samples(samples, 22050)
    expected = librosa.feature.rms(y=samples[:, 0], frame_length=2048, hop_length=HOP_LENGTH)[0]
    np.testing.assert_allclose(result.rms, expected[: len(result.rms)], rtol=1e-5)
    assert len(result.times) > 256


def test_waveform_bins_preserve_isolated_peak():
    samples = np.zeros((22050, 1), dtype=np.float32)
    samples[12345, 0] = 0.9
    result = analyze_samples(samples, 22050)
    assert len(result.waveform_times) <= 2000
    assert result.waveform_max.max() == pytest.approx(0.9)


def test_regular_clicks_have_plausible_tempo():
    import librosa

    rate = 22050
    samples = librosa.clicks(times=np.arange(0.5, 12, 0.5), sr=rate, length=12 * rate)
    result = analyze_samples(samples[:, None], rate)
    assert result.tempo_bpm == pytest.approx(120, abs=5)


def test_service_generates_pngs_and_unique_jobs(tmp_path):
    from PIL import Image

    path = tmp_path / "tone.wav"
    sf.write(path, tone(), 22050)
    service = AnalysisService(Settings(tmp_path / "out"))
    first = service.process(path)
    second = service.process(path)
    assert first.filename == "tone.wav"
    assert first.seconds > 0
    assert first.plots != second.plots
    assert set(first.plots) == {"waveform", "spectrogram", "features"}
    for plot in first.plots.values():
        with Image.open(plot) as image:
            image.verify()


def test_plot_failure_cleans_up(tmp_path, monkeypatch):
    import stemscope.analysis

    path = tmp_path / "tone.wav"
    sf.write(path, tone(), 22050)

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(stemscope.analysis, "render_analysis", fail)
    output = tmp_path / "out"
    with pytest.raises(StemScopeError, match="storage"):
        AnalysisService(Settings(output)).process(path)
    assert list(output.iterdir()) == []


def test_corrupt_analysis_input(tmp_path):
    path = tmp_path / "bad.mp3"
    path.write_bytes(b"not audio")
    with pytest.raises(StemScopeError, match="decoded"):
        AnalysisService(Settings(tmp_path / "out")).process(path)
    assert not (tmp_path / "out").exists()
