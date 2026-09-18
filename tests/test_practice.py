import json

import numpy as np
import pytest
import soundfile as sf

from stemscope.errors import StemScopeError
from stemscope.practice import PracticeOptions, render_practice, transform_mix
from stemscope.separators.base import STEMS


def stem_files(tmp_path, amplitude=0.1):
    rate = 22050
    time = np.arange(rate) / rate
    arrays = {
        name: np.tile((amplitude * np.sin(2 * np.pi * hz * time))[:, None], (1, 2)).astype(
            np.float32
        )
        for name, hz in zip(STEMS, [440, 700, 110, 1000])
    }
    paths = {}
    for name, samples in arrays.items():
        path = tmp_path / f"{name}.wav"
        sf.write(path, samples, rate, subtype="FLOAT")
        paths[name] = path
    return paths, arrays, rate


def dominant_frequency(samples, rate):
    audio = samples[len(samples) // 4 : 3 * len(samples) // 4, 0]
    spectrum = abs(np.fft.rfft(audio * np.hanning(len(audio))))
    return np.fft.rfftfreq(len(audio), 1 / rate)[np.argmax(spectrum)]


def test_mute_solo_volume_and_section(tmp_path):
    paths, arrays, rate = stem_files(tmp_path)
    options = PracticeOptions(
        gains={"vocals": 0.5, "drums": 1, "bass": 1, "other": 1},
        solo=("vocals", "drums"),
        muted=("drums",),
        start_seconds=0.2,
        end_seconds=0.8,
    )
    result = render_practice(paths, options, tmp_path / "out")
    audio, output_rate = sf.read(result.audio)
    assert output_rate == rate
    assert len(audio) == round(0.6 * rate)
    expected = arrays["vocals"][round(0.2 * rate) : round(0.8 * rate)] * 0.5
    np.testing.assert_allclose(audio[200:-200], expected[200:-200], atol=4e-5)
    data = json.loads(result.manifest.read_text())
    assert data["effective_gains"] == {"vocals": 0.5, "drums": 0, "bass": 0, "other": 0}
    assert len(data["inputs"]) == 4


@pytest.mark.parametrize("speed,pitch,frequency", [(0.5, 0, 440), (1, 12, 880), (0.75, 12, 880)])
def test_speed_and_pitch_are_independent(tmp_path, speed, pitch, frequency):
    _, arrays, rate = stem_files(tmp_path)
    changed = transform_mix(arrays["vocals"], rate, speed, pitch)
    assert len(changed) == round(rate / speed)
    assert changed.shape[1] == 2
    assert dominant_frequency(changed, rate) == pytest.approx(frequency, abs=10)


def test_shared_gain_prevents_clipping_and_retains_balance(tmp_path):
    paths, _, _ = stem_files(tmp_path, amplitude=0.8)
    result = render_practice(paths, PracticeOptions(), tmp_path / "out")
    samples, _ = sf.read(result.audio)
    assert result.normalization_gain < 1
    assert np.max(abs(samples)) <= 0.9801
    np.testing.assert_array_equal(samples[:, 0], samples[:, 1])
    assert samples[0, 0] == 0 and samples[-1, 0] == 0


@pytest.mark.parametrize(
    "options",
    [
        PracticeOptions(muted=STEMS),
        PracticeOptions(start_seconds=2),
        PracticeOptions(start_seconds=0.8, end_seconds=0.2),
        PracticeOptions(speed=float("nan")),
        PracticeOptions(speed=2),
        PracticeOptions(solo=("unknown",)),
    ],
)
def test_invalid_practice_options_are_actionable(tmp_path, options):
    paths, _, _ = stem_files(tmp_path)
    with pytest.raises(StemScopeError):
        render_practice(paths, options, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_end_clamped_and_exports_do_not_overwrite_sources(tmp_path):
    paths, _, _ = stem_files(tmp_path)
    before = {name: path.read_bytes() for name, path in paths.items()}
    first = render_practice(paths, PracticeOptions(end_seconds=30), tmp_path / "out")
    second = render_practice(paths, PracticeOptions(), tmp_path / "out")
    assert first.source_end == 1
    assert first.audio != second.audio
    assert before == {name: path.read_bytes() for name, path in paths.items()}


def test_stem_mismatch_rejected(tmp_path):
    paths, _, _ = stem_files(tmp_path)
    sf.write(paths["bass"], np.zeros((1000, 2)), 22050)
    with pytest.raises(StemScopeError, match="exact length"):
        render_practice(paths, PracticeOptions(), tmp_path / "out")


def test_decimal_minimum_section_is_not_rejected_by_float_rounding(tmp_path):
    paths, _, rate = stem_files(tmp_path)
    result = render_practice(
        paths, PracticeOptions(start_seconds=0.2, end_seconds=0.3), tmp_path / "out"
    )
    assert sf.info(result.audio).frames == round(0.1 * rate)


def test_blank_section_value_has_friendly_error(tmp_path):
    paths, _, _ = stem_files(tmp_path)
    with pytest.raises(StemScopeError, match="finite volume, section"):
        render_practice(paths, PracticeOptions(start_seconds=None), tmp_path / "out")
