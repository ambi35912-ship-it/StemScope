import json
import logging

import pytest

from stemscope.config import Settings
from stemscope.logging_config import JsonFormatter


def test_config_file_and_environment_precedence(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"output_dir": str(tmp_path / "jobs"), "max_seconds": 30}))
    monkeypatch.setenv("STEMSCOPE_CONFIG", str(path))
    monkeypatch.setenv("STEMSCOPE_MAX_SECONDS", "45")
    settings = Settings.from_env()
    assert settings.max_seconds == 45
    assert settings.output_dir == tmp_path / "jobs"


@pytest.mark.parametrize("value", [{"unknown": 1}, [], {"max_seconds": 0}])
def test_bad_configuration_rejected(tmp_path, monkeypatch, value):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(value))
    monkeypatch.setenv("STEMSCOPE_CONFIG", str(path))
    with pytest.raises(ValueError):
        Settings.from_env()


def test_json_logging():
    record = logging.LogRecord("stemscope", logging.INFO, "", 0, "Completed %s", ("audio",), None)
    record.job_id = "job-example"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "Completed audio"
    assert payload["job_id"] == "job-example"
    assert payload["level"] == "INFO"
