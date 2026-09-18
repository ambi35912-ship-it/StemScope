import numpy as np
import pytest
import soundfile as sf

from stemscope.errors import StemScopeError
from stemscope.quality.diagnostics import inspect_arrays, inspect_files, similarity
from stemscope.separators.base import STEMS


def sources():
    time = np.arange(8000) / 8000
    return {
        name: np.tile((0.1 * np.sin(2 * np.pi * frequency * time))[:, None], (1, 2))
        for name, frequency in zip(STEMS, [200, 370, 90, 610])
    }


def test_clean_sources_reconstruct_without_flags():
    stems = sources()
    result = inspect_arrays(sum(stems.values()), stems, 8000)
    assert result["reconstruction_relative_rms"] < 1e-10
    assert all(not r["flags"] for r in result["rows"])


def test_mixture_consistent_leakage_can_evade_reconstruction():
    stems = sources()
    mix = sum(stems.values())
    stems["vocals"] = stems["drums"].copy()
    stems["other"] = mix - stems["vocals"] - stems["drums"] - stems["bass"]
    result = inspect_arrays(mix, stems, 8000)
    assert result["reconstruction_relative_rms"] < 1e-10
    assert result["rows"][0]["max_waveform_similarity"] == pytest.approx(1)
    assert any("similarity" in f for f in result["rows"][0]["flags"])


def test_silence_is_not_given_high_quality_or_invalid_numbers():
    stems = {name: np.zeros((100, 2)) for name in STEMS}
    result = inspect_arrays(stems["vocals"], stems, 8000)
    assert result["reconstruction_relative_rms"] is None
    assert all(r["rms_dbfs"] is None and r["flags"] for r in result["rows"])
    assert similarity(stems["vocals"], stems["bass"]) is None


def test_full_scale_samples_flag_headroom_not_confirmed_clipping():
    stems = sources()
    stems["vocals"] *= 15
    result = inspect_arrays(sum(stems.values()), stems, 8000)
    assert any("headroom" in f for f in result["rows"][0]["flags"])


def test_alignment_and_nonfinite_rejected():
    stems = sources()
    mix = sum(stems.values())
    stems["vocals"] = stems["vocals"][:-3]
    with pytest.raises(StemScopeError, match="aligned"):
        inspect_arrays(mix, stems, 8000)
    stems = sources()
    stems["vocals"][0] = np.nan
    with pytest.raises(StemScopeError, match="finite"):
        inspect_arrays(mix, stems, 8000)


def test_file_resampling_and_duration_checks(tmp_path):
    stems = sources()
    paths = {}
    for name, data in stems.items():
        path = tmp_path / f"{name}.wav"
        sf.write(path, data, 8000, subtype="FLOAT")
        paths[name] = path
    mix = tmp_path / "mix.wav"
    sf.write(mix, sum(stems.values()), 8000, subtype="FLOAT")
    result = inspect_files(mix, paths)
    assert result["analyzed_seconds"] == 1
    sf.write(paths["bass"], np.zeros((2000, 2)), 8000, subtype="FLOAT")
    with pytest.raises(StemScopeError, match="durations"):
        inspect_files(mix, paths)


def test_antiphase_stereo_is_not_mistaken_for_silence():
    stems = sources()
    for value in stems.values():
        value[:, 1] *= -1
    result = inspect_arrays(sum(stems.values()), stems, 8000)
    assert all(r["rms_dbfs"] is not None and not r["flags"] for r in result["rows"])


def test_file_rate_conversion_preserves_original_peak(tmp_path):
    from scipy.signal import resample_poly

    stems = sources()
    stems["vocals"] *= 15
    paths = {}
    for name, data in stems.items():
        path = tmp_path / f"{name}.wav"
        sf.write(path, data, 8000, subtype="FLOAT")
        paths[name] = path
    mixture = tmp_path / "mix.wav"
    sf.write(mixture, resample_poly(sum(stems.values()), 2, 1, axis=0), 16000, subtype="FLOAT")
    result = inspect_files(mixture, paths)
    assert result["analysis_rate"] == 8000
    assert result["rows"][0]["sample_peak"] == pytest.approx(1.5)
    assert result["rows"][0]["over_full_scale_fraction"] > 0.001


def test_long_audio_scope_is_bounded_and_disclosed(tmp_path):
    samples = np.ones((31 * 1000, 1), dtype=np.float32) * 0.01
    path = tmp_path / "long.wav"
    sf.write(path, samples, 1000, subtype="FLOAT")
    result = inspect_files(path, dict.fromkeys(STEMS, path))
    assert result["analyzed_seconds"] == 30
    assert result["full_duration_seconds"] == 31
