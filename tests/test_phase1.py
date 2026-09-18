import numpy as np
import pytest
import soundfile as sf

from stemscope.audio.loader import load_audio
from stemscope.audio.validation import validate_path
from stemscope.config import Settings
from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS, Separator
from stemscope.separators.demucs import DemucsSeparator
from stemscope.service import SeparationService, validate_outputs


@pytest.fixture
def song(tmp_path):
    path = tmp_path / "song.wav"
    sf.write(path, np.zeros((800, 2)), 8000)
    return path


@pytest.mark.parametrize("extension", [".wav", ".WAV", ".mp3", ".MP3"])
def test_extensions(tmp_path, extension):
    path = tmp_path / ("song" + extension)
    path.write_bytes(b"audio")
    assert validate_path(path) == path


@pytest.mark.parametrize("name", ["song.txt", "missing.wav", "folder.mp3"])
def test_invalid_paths(tmp_path, name):
    if name == "folder.mp3":
        (tmp_path / name).mkdir()
    with pytest.raises(StemScopeError):
        validate_path(tmp_path / name)


def test_size_limits(song, tmp_path):
    with pytest.raises(StemScopeError, match="limit"):
        validate_path(song, 1)
    empty = tmp_path / "empty.wav"
    empty.touch()
    with pytest.raises(StemScopeError, match="empty"):
        validate_path(empty)


def test_corrupt(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not audio")
    with pytest.raises(StemScopeError, match="decoded"):
        load_audio(path)


def test_decode(song):
    samples, rate = load_audio(song)
    assert samples.shape == (800, 2)
    assert rate == 8000


def test_interface():
    with pytest.raises(TypeError):
        Separator()
    assert isinstance(DemucsSeparator(), Separator)
    assert DemucsSeparator()._model is None


class FakeSeparator(Separator):
    def separate(self, samples, sample_rate, output_dir):
        paths = {name: output_dir / f"{name}.wav" for name in STEMS}
        for path in paths.values():
            sf.write(path, samples, sample_rate)
        return paths


def test_service_and_collisions(song, tmp_path):
    service = SeparationService(FakeSeparator(), Settings(tmp_path / "out"))
    first, second = service.process(song), service.process(song)
    assert first.filename == "song.wav"
    assert first.seconds >= 0
    assert set(first.stems) == set(STEMS)
    assert first.stems != second.stems
    assert all(p.is_file() for p in first.stems.values())


@pytest.mark.parametrize("failure", ["exception", "missing"])
def test_failure_cleanup(song, tmp_path, failure):
    class Broken(Separator):
        def separate(self, samples, sample_rate, output_dir):
            (output_dir / "partial.wav").touch()
            if failure == "exception":
                raise RuntimeError("private traceback detail")
            return {}

    output = tmp_path / "out"
    with pytest.raises(StemScopeError):
        SeparationService(Broken(), Settings(output)).process(song)
    assert list(output.iterdir()) == []


def test_unwritable_output(song, tmp_path):
    path = tmp_path / "file"
    path.touch()
    with pytest.raises(StemScopeError, match="storage"):
        SeparationService(FakeSeparator(), Settings(path)).process(song)


def test_output_checks(song, tmp_path):
    with pytest.raises(StemScopeError, match="path"):
        validate_outputs(dict.fromkeys(STEMS, song), tmp_path / "elsewhere", 0.1)
    with pytest.raises(StemScopeError, match="unreadable"):
        validate_outputs({name: tmp_path / f"{name}.wav" for name in STEMS}, tmp_path, 0.1)


def test_duration_and_nan(tmp_path):
    paths = {name: tmp_path / f"{name}.wav" for name in STEMS}
    for path in paths.values():
        sf.write(path, np.zeros((800, 2)), 8000)
    with pytest.raises(StemScopeError, match="duration"):
        validate_outputs(paths, tmp_path, 10)
    sf.write(paths["vocals"], np.full((800, 2), np.nan), 8000, subtype="FLOAT")
    with pytest.raises(StemScopeError, match="invalid samples"):
        validate_outputs(paths, tmp_path, 0.1)


def test_ui_build(tmp_path):
    from stemscope.ui import build_app

    app = build_app({"Fake": SeparationService(FakeSeparator(), Settings(tmp_path))})
    assert app is not None


def test_demucs_mocked_inference(tmp_path, monkeypatch):
    import demucs.apply
    import demucs.pretrained
    import torch

    class Model:
        sources = STEMS
        samplerate = 8000
        audio_channels = 2

        def eval(self):
            return self

    calls = []
    monkeypatch.setattr(demucs.pretrained, "get_model", lambda name: calls.append(name) or Model())
    monkeypatch.setattr(
        demucs.apply,
        "apply_model",
        lambda model, audio, **kwargs: audio[:, None].repeat(1, 4, 1, 1),
    )
    adapter = DemucsSeparator()
    for _ in range(2):
        paths = adapter.separate(np.zeros((800, 2), dtype=np.float32), 8000, tmp_path)
        validate_outputs(paths, tmp_path, 0.1)
    assert calls == ["htdemucs"]
    assert torch.isfinite(torch.tensor(sf.read(paths["vocals"])[0])).all()


def test_missing_dependency(tmp_path, monkeypatch):
    import builtins

    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "demucs.apply":
            raise ImportError("missing")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(StemScopeError, match="dependencies"):
        DemucsSeparator().separate(np.zeros((800, 2), dtype=np.float32), 8000, tmp_path)


def test_model_failure(tmp_path, monkeypatch):
    import demucs.pretrained

    def fail(name):
        raise RuntimeError("download failed")

    monkeypatch.setattr(demucs.pretrained, "get_model", fail)
    with pytest.raises(StemScopeError, match="Demucs failed"):
        DemucsSeparator().separate(np.zeros((800, 2), dtype=np.float32), 8000, tmp_path)


def test_real_mp3_decode(tmp_path):
    import lameenc

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(44100)
    encoder.set_channels(1)
    samples = (1000 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)).astype(np.int16)
    path = tmp_path / "song.mp3"
    path.write_bytes(encoder.encode(samples.tobytes()) + encoder.flush())
    decoded, rate = load_audio(validate_path(path))
    assert rate == 44100
    assert decoded.shape[1] == 1
    assert len(decoded) >= 44100
