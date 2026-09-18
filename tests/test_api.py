import io
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from stemscope.api import create_app
from stemscope.config import Settings
from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS, Separator
from stemscope.service import SeparationService


class FixtureSeparator(Separator):
    name = "Fixture"

    def separate(self, samples, rate, output):
        paths = {stem: output / f"{stem}.wav" for stem in STEMS}
        for path in paths.values():
            sf.write(path, samples / 4, rate)
        return paths


def setup(tmp_path, adapter=None, max_bytes=100000):
    settings = Settings(tmp_path / "out", max_bytes=max_bytes)
    service = SeparationService(adapter or FixtureSeparator(), settings)
    return TestClient(create_app(settings, {"Fixture": service})), settings


def wav():
    stream = io.BytesIO()
    sf.write(stream, np.sin(np.arange(8000) * 0.1) * 0.1, 8000, format="WAV")
    return stream.getvalue()


def submit(client, name="song.wav", content=None, model="Fixture"):
    return client.post(
        "/separations",
        data={"model": model},
        files={"audio": (name, wav() if content is None else content, "audio/wav")},
    )


def test_upload_download_and_restart(tmp_path):
    client, settings = setup(tmp_path)
    assert client.get("/health").status_code == 200
    assert client.get("/models").json()["models"] == ["Fixture"]
    response = submit(client, name="../../song.wav")
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["filename"] == "song.wav"
    assert set(payload["stems"]) == set(STEMS)
    restarted = TestClient(create_app(settings, {}))
    for url in payload["stems"].values():
        audio = restarted.get(url)
        assert audio.status_code == 200
        samples, rate = sf.read(io.BytesIO(audio.content))
        assert len(samples) == 8000 and rate == 8000
    assert not list(settings.output_dir.glob("upload-*"))


def test_invalid_requests_cleanup_and_recovery(tmp_path):
    client, settings = setup(tmp_path)
    assert submit(client, name="bad.exe").status_code == 400
    assert submit(client, content=b"corrupt").status_code == 400
    assert submit(client, content=b"").status_code == 400
    assert submit(client, model="unknown").status_code == 400
    assert submit(client).status_code == 201
    assert not list(settings.output_dir.glob("upload-*"))


def test_size_limit(tmp_path):
    client, settings = setup(tmp_path, max_bytes=10)
    assert submit(client).status_code == 413
    assert not list(settings.output_dir.glob("upload-*"))


def test_download_rejects_unknown_and_symlink(tmp_path):
    client, settings = setup(tmp_path)
    payload = submit(client).json()
    url = payload["stems"]["vocals"]
    path = settings.output_dir / payload["job_id"] / "vocals.wav"
    path.unlink()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"private")
    path.symlink_to(outside)
    assert client.get(url).status_code == 404
    assert client.get("/separations/job-missing/stems/vocals").status_code == 404
    assert client.get(f"/separations/{payload['job_id']}/stems/result.json").status_code == 404


def test_busy_request_and_model_failure_release_capacity(tmp_path):
    entered, release = Event(), Event()

    class Blocking(FixtureSeparator):
        def separate(self, samples, rate, output):
            entered.set()
            assert release.wait(10)
            raise StemScopeError("Fixture unavailable")

    client, settings = setup(tmp_path, Blocking())
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(submit, client)
        try:
            assert entered.wait(5)
            assert submit(client).status_code == 409
            assert client.get("/health").status_code == 200
        finally:
            release.set()
        assert pending.result().status_code == 400
    assert submit(client).status_code == 400
    assert not list(settings.output_dir.glob("upload-*"))
    assert not list(settings.output_dir.glob("job-*"))
