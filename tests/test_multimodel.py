import json

import numpy as np
import pytest
import soundfile as sf

from stemscope.application import create_services
from stemscope.config import Settings
from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS, Separator
from stemscope.separators.openunmix import OpenUnmixSeparator
from stemscope.service import SeparationService


@pytest.fixture
def mocked_umx(monkeypatch):
    import openunmix.utils

    calls = []

    class Model:
        sample_rate = 44100

        def eval(self):
            return self

        def __call__(self, audio):
            calls.append(("inference", audio.shape[-1]))
            return audio

        def to_dict(self, audio):
            return {name: audio / 4 for name in STEMS}

    def load(**kwargs):
        calls.append(("load", kwargs))
        return Model()

    monkeypatch.setattr(openunmix.utils, "load_separator", load)
    return calls


def test_registry_is_lazy_and_models_share_interface(tmp_path):
    services = create_services(Settings(tmp_path))
    assert set(services) == {"Demucs (htdemucs)", "Open-Unmix (umxhq)"}
    for name, service in services.items():
        assert isinstance(service.separator, Separator)
        assert service.separator.name == name
        assert service.separator._model is None


@pytest.mark.parametrize(
    "rate,channels,frames", [(8000, 1, 1), (22050, 1, 5000), (44100, 2, 44100 * 3 + 23)]
)
def test_umx_resample_padding_and_chunk_continuity(tmp_path, mocked_umx, rate, channels, frames):
    import torch
    from torchaudio.functional import resample

    times = np.arange(frames) / rate
    samples = np.tile((0.1 * np.sin(2 * np.pi * 220 * times))[:, None], (1, channels)).astype(
        np.float32
    )
    adapter = OpenUnmixSeparator(chunk_seconds=1)
    paths = adapter.separate(samples, rate, tmp_path)
    expected = torch.from_numpy(samples.T.copy())
    if channels == 1:
        expected = expected.repeat(2, 1)
    if rate != 44100:
        expected = resample(expected, rate, 44100)
    for path in paths.values():
        actual, actual_rate = sf.read(path)
        assert actual_rate == 44100
        np.testing.assert_allclose(actual, expected.T.numpy() / 4, atol=1e-7)
    assert max(value for kind, value in mocked_umx if kind == "inference") <= 3 * 44100


def test_umx_reuses_loaded_model(tmp_path, mocked_umx):
    adapter = OpenUnmixSeparator()
    for _ in range(2):
        adapter.separate(np.zeros((4410, 2), dtype=np.float32), 44100, tmp_path)
    loads = [value for kind, value in mocked_umx if kind == "load"]
    assert len(loads) == 1
    assert loads[0]["model_str_or_path"] == "umxhq"
    assert loads[0]["targets"] == list(STEMS)


def test_umx_missing_dependency(tmp_path, monkeypatch):
    import builtins

    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "openunmix.utils":
            raise ImportError("missing")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(StemScopeError, match="dependencies"):
        OpenUnmixSeparator().separate(np.zeros((4410, 2), dtype=np.float32), 44100, tmp_path)


def test_umx_load_failure(tmp_path, monkeypatch):
    import openunmix.utils

    def fail(**kwargs):
        raise RuntimeError("download failure")

    monkeypatch.setattr(openunmix.utils, "load_separator", fail)
    with pytest.raises(StemScopeError, match="Open-Unmix failed"):
        OpenUnmixSeparator().separate(np.zeros((4410, 2), dtype=np.float32), 44100, tmp_path)


@pytest.mark.parametrize("failure", ["missing", "nan", "short", "shape"])
def test_umx_bad_output_cleans_job(tmp_path, monkeypatch, failure):
    import openunmix.utils
    import torch

    class Broken:
        sample_rate = 44100

        def eval(self):
            return self

        def __call__(self, audio):
            return audio

        def to_dict(self, audio):
            if failure == "missing":
                return {}
            if failure == "nan":
                audio = torch.full_like(audio, float("nan"))
            if failure == "short":
                audio = audio[..., :10]
            if failure == "shape":
                audio = audio[0]
            return {name: audio for name in STEMS}

    monkeypatch.setattr(openunmix.utils, "load_separator", lambda **kwargs: Broken())
    path = tmp_path / "input.wav"
    sf.write(path, np.zeros((4410, 2)), 44100)
    output = tmp_path / "out"
    with pytest.raises(StemScopeError):
        SeparationService(OpenUnmixSeparator(), Settings(output)).process(path)
    assert list(output.iterdir()) == []


def test_result_records_actual_model(tmp_path, mocked_umx):
    path = tmp_path / "input.wav"
    sf.write(path, np.zeros((4410, 2)), 44100)
    result = SeparationService(OpenUnmixSeparator(), Settings(tmp_path / "out")).process(path)
    manifest = json.loads((result.stems["vocals"].parent / "result.json").read_text())
    assert result.model == manifest["model"] == "Open-Unmix (umxhq)"
    assert manifest["original_filename"] == "input.wav"
    assert manifest["stems"] == {name: f"{name}.wav" for name in STEMS}


def test_ui_requires_a_model():
    from stemscope.ui import build_app

    with pytest.raises(ValueError, match="At least one"):
        build_app({})


def test_ui_routes_choice_and_keeps_result_model_label(tmp_path):
    from stemscope.ui import build_app

    class Fake(Separator):
        def __init__(self, name):
            self.name = name
            self.calls = 0

        def separate(self, samples, sample_rate, output_dir):
            self.calls += 1
            paths = {name: output_dir / f"{name}.wav" for name in STEMS}
            for path in paths.values():
                sf.write(path, samples, sample_rate)
            return paths

    first, second = Fake("First"), Fake("Second")
    settings = Settings(tmp_path / "out")
    app = build_app(
        {
            "First": SeparationService(first, settings),
            "Second": SeparationService(second, settings),
        }
    )
    callback = next(fn.fn for fn in app.fns.values() if fn.fn.__name__ == "separate")
    path = tmp_path / "input.wav"
    sf.write(path, np.zeros((4410, 2)), 44100)
    result = list(callback(str(path), "Second"))[-1]
    assert result[1] == "Complete"
    assert result[3] == "Second"
    assert first.calls == 0 and second.calls == 1
    error = list(callback(str(path), "Unknown"))[-1]
    assert "Choose one" in error[1]
    assert error[3] == "" and all(value is None for value in error[4:])
    empty = list(callback(None, "Second"))[-1]
    assert "Choose a WAV" in empty[1]
