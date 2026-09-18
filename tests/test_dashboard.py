import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from stemscope.config import Settings
from stemscope.dashboard import DashboardService, dashboard_tables
from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS, Separator
from stemscope.service import SeparationService


class CountingSeparator(Separator):
    def __init__(self, name, fail=False):
        self.name, self.fail, self.calls = name, fail, 0

    def separate(self, samples, rate, output):
        self.calls += 1
        if self.fail:
            raise StemScopeError("Fixture model failure")
        paths = {}
        for name in STEMS:
            path = output / f"{name}.wav"
            sf.write(path, samples / 4, rate, subtype="FLOAT")
            paths[name] = path
        return paths


def setup(tmp_path, fail=False):
    source = tmp_path / "song.wav"
    sf.write(source, 0.1 * np.sin(2 * np.pi * 440 * np.arange(8000) / 8000), 8000, subtype="FLOAT")
    settings = Settings(tmp_path / "out")
    adapters = [CountingSeparator("First", fail), CountingSeparator("Second")]
    services = {a.name: SeparationService(a, settings) for a in adapters}
    return source, DashboardService(services, settings), adapters


def test_selected_model_only_and_missing_history_is_not_fatal(tmp_path):
    source, dashboard, adapters = setup(tmp_path)
    result = dashboard.build(source, "Second")
    payload = result.payload
    assert [a.calls for a in adapters] == [0, 1]
    assert payload["selected_model"] == "Second"
    assert len(payload["selected_stems"]) == 4
    assert payload["status"] == "Complete"
    assert payload["historical_benchmark"]["rows"] == []
    assert "Unavailable" in payload["upload_reference_quality"]
    assert payload["input"]["path"] != str(source)
    assert json.loads(result.manifest.read_text())["selected_model"] == "Second"
    summary, models, diagnostics = dashboard_tables(payload)
    assert len(summary) == 12 and len(models) == 1 and len(diagnostics) == 4


def test_both_models_are_run_once_and_selected_result_is_reused(tmp_path):
    source, dashboard, adapters = setup(tmp_path)
    result = dashboard.build(source, "First", compare_both=True)
    assert [a.calls for a in adapters] == [1, 1]
    assert len(result.payload["models"]) == 2
    assert result.payload["comparison_chart"]
    assert result.payload["selected_stems"] == result.payload["models"][0]["stems"]
    evidence = json.loads(Path(result.payload["comparison_manifest"]).read_text())
    assert evidence["input"]["sha256"] == result.payload["input"]["sha256"]


def test_model_failure_retains_input_features_and_other_model(tmp_path):
    source, dashboard, _ = setup(tmp_path, fail=True)
    result = dashboard.build(source, "First", compare_both=True)
    assert result.payload["selected_stems"] == {}
    assert result.payload["audio"]["duration_seconds"] == 1
    assert result.payload["models"][1]["status"] == "ok"
    assert "issues" in result.payload["status"]
    assert result.payload["errors"]


def test_missing_auto_artifact_preserves_analysis_without_fake_fallback(tmp_path):
    source, dashboard, adapters = setup(tmp_path)
    result = dashboard.build(source, "Auto: Balanced")
    assert result.payload["selected_model"] is None
    assert result.payload["errors"]
    assert result.payload["charts"]["waveform"]
    assert [a.calls for a in adapters] == [0, 0]


def test_historical_scores_stay_separate_from_upload(tmp_path, monkeypatch):
    import stemscope.dashboard as module

    source, dashboard, _ = setup(tmp_path)
    history = tmp_path / "history.csv"
    history.write_text("fixture")
    monkeypatch.setattr(
        module,
        "completed_study",
        lambda _: (
            "Historical fixture",
            [["Other recording", "vocals", "50", "99", "98", "100", "90"]],
            str(history),
            str(history),
            str(history),
        ),
    )
    result = dashboard.build(source, "First")
    assert result.payload["historical_benchmark"]["rows"][0][3] == "99"
    assert "Unavailable" in result.payload["upload_reference_quality"]
    assert "si_sdr" not in result.payload["audio"]
    assert result.payload["selected_diagnostics"]["rows"][0]["review_status"] != "99"


def test_invalid_upload_is_actionable(tmp_path):
    _, dashboard, _ = setup(tmp_path)
    with pytest.raises(StemScopeError):
        dashboard.build(tmp_path / "missing.wav", "First")
