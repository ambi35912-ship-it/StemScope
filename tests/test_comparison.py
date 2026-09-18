import json

import numpy as np
import pytest
import soundfile as sf

from stemscope.comparison import ComparisonService, spectrum
from stemscope.config import Settings
from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS, Separator
from stemscope.service import SeparationService


class FakeSeparator(Separator):
    def __init__(self, name, fail=False):
        self.name = name
        self.fail = fail
        self.calls = 0
        self.inputs = []

    def separate(self, samples, sample_rate, output_dir):
        self.calls += 1
        self.inputs.append(samples.copy())
        if self.fail:
            raise StemScopeError("Fixture failure")
        paths = {}
        for stem in STEMS:
            path = output_dir / f"{stem}.wav"
            sf.write(path, samples / 4, sample_rate, subtype="FLOAT")
            paths[stem] = path
        return paths


def setup_comparison(tmp_path, fail=False):
    source = tmp_path / "upload.wav"
    time = np.arange(8000) / 8000
    sf.write(source, 0.1 * np.sin(2 * np.pi * 440 * time), 8000, subtype="FLOAT")
    settings = Settings(tmp_path / "outputs")
    adapters = [FakeSeparator("First"), FakeSeparator("Second", fail)]
    services = {a.name: SeparationService(a, settings) for a in adapters}
    return source, ComparisonService(services, settings), adapters


def test_identical_captured_input_and_switch_without_inference(tmp_path):
    source, service, adapters = setup_comparison(tmp_path)
    result = service.run(source)
    protocol = json.loads(result.manifest.read_text())
    assert result.status == "Complete"
    assert len(result.rows) == 8
    np.testing.assert_array_equal(adapters[0].inputs[0], adapters[1].inputs[0])
    assert protocol["input"]["path"] != str(source)
    assert all(len(m["output_hashes"]) == 4 for m in protocol["models"])
    # Later changes to the upload cannot change the captured evidence.
    source.write_bytes(b"changed upload")
    for stem in ["vocals", "bass"]:
        paths, figure = service.view(result.directory, stem)
        assert all(path.endswith(f"{stem}.wav") for path in paths)
        assert figure.is_file()
    assert [a.calls for a in adapters] == [1, 1]


def test_partial_failure_retains_successful_audio(tmp_path):
    source, service, _ = setup_comparison(tmp_path, fail=True)
    result = service.run(source)
    assert "errors" in result.status
    assert sum(r["status"] == "failed" for r in result.rows) == 4
    paths, plot = service.view(result.directory, "drums")
    assert paths[0] and paths[1] is None
    assert plot.exists()


def test_diagnostic_failure_is_not_model_failure(tmp_path, monkeypatch):
    import stemscope.comparison as module

    def fail(*args):
        raise RuntimeError("fixture")

    monkeypatch.setattr(module, "inspect_files", fail)
    source, service, _ = setup_comparison(tmp_path)
    result = service.run(source)
    assert all(r["status"] == "ok" for r in result.rows)
    assert "diagnostics unavailable" in result.status
    assert all(r["review_status"] == "Unavailable" for r in result.rows)


def test_changed_output_rejected(tmp_path):
    source, service, _ = setup_comparison(tmp_path)
    result = service.run(source)
    protocol = json.loads(result.manifest.read_text())
    from pathlib import Path

    Path(protocol["models"][0]["stems"]["vocals"]).write_bytes(b"changed")
    with pytest.raises(StemScopeError, match="changed"):
        service.view(result.directory, "vocals")


def test_shared_spectrogram_reference_preserves_level_difference():
    samples = np.sin(2 * np.pi * 440 * np.arange(8000) / 8000)[:, None]
    _, _, loud = spectrum(samples, 8000)
    _, _, quiet = spectrum(samples * 0.1, 8000)
    assert loud.max() - quiet.max() == pytest.approx(20, abs=1e-5)


def test_comparison_view_rejects_other_directories(tmp_path):
    _, service, _ = setup_comparison(tmp_path)
    with pytest.raises(StemScopeError, match="valid saved"):
        service.view(tmp_path, "vocals")
