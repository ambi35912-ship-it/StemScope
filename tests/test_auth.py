import json
import time

import gradio as gr
import httpx
import pytest
from fastapi.testclient import TestClient
from test_api import FixtureSeparator, wav

from stemscope.api import create_app
from stemscope.auth import AuthSettings, SupabaseAuth
from stemscope.auth_web import create_authenticated_ui
from stemscope.config import Settings
from stemscope.service import SeparationService

A = "11111111-1111-4111-8111-111111111111"
B = "22222222-2222-4222-8222-222222222222"
ORIGIN = "http://testserver"


def auth_fixture():
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path.endswith("/token"):
            payload = json.loads(request.content)
            if payload["password"] != "good-password":
                return httpx.Response(400, json={"message": "private upstream error"})
            return httpx.Response(200, json={"access_token": "alice", "expires_in": 3600})
        if request.url.path.endswith("/signup"):
            return httpx.Response(200, json={"id": A})
        token = request.headers.get("authorization")
        if token == "Bearer unavailable":
            return httpx.Response(503)
        ids = {"Bearer alice": A, "Bearer bob": B}
        if token in ids:
            return httpx.Response(200, json={"id": ids[token]})
        return httpx.Response(401)

    return SupabaseAuth(
        AuthSettings("https://example.supabase.co", "sb_publishable_example", ORIGIN),
        httpx.MockTransport(handler),
    ), calls


def setup(tmp_path):
    auth, calls = auth_fixture()
    settings = Settings(tmp_path / "out")
    services = {"Fixture": SeparationService(FixtureSeparator(), settings)}
    return auth, calls, settings, services


def login(client):
    return client.post(
        "/auth/login",
        data={"email": "alice@example.com", "password": "good-password"},
        headers={"origin": ORIGIN},
        follow_redirects=False,
    )


def test_api_identity_and_file_isolation(tmp_path):
    auth, calls, settings, services = setup(tmp_path)
    with TestClient(create_app(settings, services, auth)) as client:
        assert client.get("/models").status_code == 401
        assert client.get("/health").status_code == 200
        assert client.get("/models", headers={"authorization": "Bearer invalid"}).status_code == 401
        assert (
            client.get("/models", headers={"authorization": "Bearer unavailable"}).status_code
            == 503
        )
        response = client.post(
            "/separations",
            data={"model": "Fixture"},
            files={"audio": ("song.wav", wav())},
            headers={"authorization": "Bearer alice"},
        )
        assert response.status_code == 201, response.text
        url = response.json()["stems"]["vocals"]
        assert client.get(url, headers={"authorization": "Bearer alice"}).status_code == 200
        assert client.get(url, headers={"authorization": "Bearer bob"}).status_code == 404
        assert client.get(url).status_code == 401
        assert list((settings.output_dir / "users" / A).glob("job-*"))
        assert any(r.url.path.endswith("/user") for r in calls)


def test_login_logout_csrf_expiry_and_signup(tmp_path):
    auth, _calls, settings, services = setup(tmp_path)
    with TestClient(create_app(settings, services, auth)) as client:
        assert "Sign in" in client.get("/").text
        data = {"email": "alice@example.com", "password": "good-password"}
        assert client.post("/auth/login", data=data).status_code == 403
        assert (
            client.post(
                "/auth/login", data=data, headers={"origin": "https://evil.example"}
            ).status_code
            == 403
        )
        response = login(client)
        assert response.status_code == 303
        assert "httponly" in response.headers["set-cookie"].lower()
        assert "samesite=strict" in response.headers["set-cookie"].lower()
        assert "alice" not in client.cookies["stemscope_session"]
        assert client.get("/auth/me").json()["id"] == A
        assert (
            client.post(
                "/separations", data={"model": "Fixture"}, files={"audio": ("song.wav", wav())}
            ).status_code
            == 403
        )
        session_id = client.cookies["stemscope_session"]
        token, _ = auth.sessions[session_id]
        auth.sessions[session_id] = (token, time.time() - 1)
        assert client.get("/auth/me").status_code == 401
        login(client)
        old_cookie = client.cookies["stemscope_session"]
        assert (
            client.post(
                "/auth/logout", headers={"origin": ORIGIN}, follow_redirects=False
            ).status_code
            == 303
        )
        client.cookies.set("stemscope_session", old_cookie)
        assert client.get("/auth/me").status_code == 401
        assert client.post("/auth/signup", data=data, headers={"origin": ORIGIN}).status_code == 200


def test_login_errors_hide_upstream_details(tmp_path):
    auth, _, settings, services = setup(tmp_path)
    with TestClient(create_app(settings, services, auth)) as client:
        response = client.post(
            "/auth/login",
            data={"email": "a@example.com", "password": "bad-password"},
            headers={"origin": ORIGIN},
        )
        assert response.status_code == 401
        assert "private upstream" not in response.text
        assert "stemscope_session" not in client.cookies


def test_gradio_workspace_auth_and_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "cache"))
    auth, _, settings, services = setup(tmp_path)

    def builder(services):
        with gr.Blocks() as blocks:
            file = gr.File()
            gr.Button().click(lambda value: value, file, file, api_name="echo", queue=False)
        return blocks.queue()

    app = create_authenticated_ui(settings, services, auth, builder)
    with TestClient(app) as client:
        assert client.get("/app/", follow_redirects=False).status_code == 303
        for token, user in [("alice", A), ("bob", B)]:
            response = client.get("/app/config", headers={"authorization": f"Bearer {token}"})
            assert response.status_code == 200, response.text
            root = settings.output_dir / "users" / user
            (root / "private.txt").write_text(user)
        alice_file = settings.output_dir / "users" / A / "private.txt"
        url = "/app/gradio_api/file=" + str(alice_file)
        assert client.get(url, headers={"authorization": "Bearer alice"}).text == A
        assert client.get(url, headers={"authorization": "Bearer bob"}).status_code == 404
        assert (
            client.get(
                url.replace("/file=", "/file/"), headers={"authorization": "Bearer bob"}
            ).status_code
            == 404
        )
        assert client.get(url).status_code == 401
        headers = {"authorization": "Bearer alice"}
        uploaded = client.post(
            "/app/gradio_api/upload", files={"files": ("song.wav", wav())}, headers=headers
        )
        assert uploaded.status_code == 200, uploaded.text
        payload = {"data": [{"path": uploaded.json()[0], "meta": {"_type": "gradio.FileData"}}]}
        echoed = client.post("/app/gradio_api/api/echo", json=payload, headers=headers)
        assert echoed.status_code == 200, echoed.text
        denied = client.post(
            "/app/gradio_api/api/echo", json=payload, headers={"authorization": "Bearer bob"}
        )
        assert denied.status_code == 400, denied.text
        for content_type in (None, "application/vnd.api+json"):
            headers = {"authorization": "Bearer bob"}
            if content_type:
                headers["content-type"] = content_type
            denied = client.post(
                "/app/gradio_api/api/echo", content=json.dumps(payload), headers=headers
            )
            assert denied.status_code == 400, denied.text


def test_incomplete_environment_fails_closed(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.delenv("SUPABASE_PUBLISHABLE_KEY", raising=False)
    with pytest.raises(ValueError):
        AuthSettings.from_env()


def test_full_authenticated_ui_separation(tmp_path, monkeypatch):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "cache"))
    auth, _, settings, services = setup(tmp_path)
    app = create_authenticated_ui(settings, services, auth)
    with TestClient(app) as client:
        login(client)
        headers = {"origin": ORIGIN}
        uploaded = client.post(
            "/app/gradio_api/upload", files={"files": ("song.wav", wav())}, headers=headers
        )
        assert uploaded.status_code == 200, uploaded.text
        response = client.post(
            "/app/gradio_api/call/separate",
            json={
                "data": [
                    {"path": uploaded.json()[0], "meta": {"_type": "gradio.FileData"}},
                    "Fixture",
                ]
            },
            headers=headers,
        )
        assert response.status_code == 200, response.text
        completed = client.get("/app/gradio_api/call/separate/" + response.json()["event_id"])
        assert "event: complete" in completed.text, completed.text
        assert len(list((settings.output_dir / "users" / A).glob("job-*/vocals.wav"))) == 1


def test_auth_required_without_credentials(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_PUBLISHABLE_KEY", raising=False)
    monkeypatch.setenv("STEMSCOPE_AUTH_REQUIRED", "true")
    with pytest.raises(ValueError):
        AuthSettings.from_env()


@pytest.mark.parametrize(
    "key", ["sb_secret_example", "e30.eyJyb2xlIjoic2VydmljZV9yb2xlIn0.signature"]
)
def test_secret_keys_rejected(key):
    with pytest.raises(ValueError):
        AuthSettings("https://example.supabase.co", key)
