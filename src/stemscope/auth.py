"""Supabase Auth transport and short-lived, server-side browser sessions."""

import base64
import json
import secrets
import time
from dataclasses import dataclass
from threading import Lock
from urllib.parse import urlparse
from uuid import UUID

import httpx
from fastapi import HTTPException, Request


@dataclass(frozen=True)
class AuthSettings:
    url: str
    key: str
    origin: str = "http://127.0.0.1:7860"

    def __post_init__(self):
        if urlparse(self.url).scheme != "https" or not urlparse(self.url).netloc:
            raise ValueError("SUPABASE_URL must be an HTTPS project URL.")
        if not self.key or self.key.startswith("sb_secret_"):
            raise ValueError("Use a Supabase publishable/anon key, never a secret key.")
        if self.key.count(".") == 2:
            try:
                encoded = self.key.split(".")[1]
                role = json.loads(
                    base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
                ).get("role")
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError("Invalid Supabase anon key format.") from exc
            if role != "anon":
                raise ValueError("Use an anon key, never a service-role key.")
        origin = urlparse(self.origin)
        if (
            origin.scheme not in {"http", "https"}
            or not origin.netloc
            or origin.path
            or origin.query
            or origin.fragment
        ):
            raise ValueError("STEMSCOPE_AUTH_ORIGIN must include http:// or https://.")

    @classmethod
    def from_env(cls):
        import os

        url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_PUBLISHABLE_KEY")
        if not url and not key:
            if os.getenv("STEMSCOPE_AUTH_REQUIRED", "false").lower() == "true":
                raise ValueError(
                    "Supabase authentication is required. Configure its URL and publishable key."
                )
            return None
        if not url or not key:
            raise ValueError("Set both SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY.")
        return cls(
            url.rstrip("/"),
            key,
            os.getenv("STEMSCOPE_AUTH_ORIGIN", "http://127.0.0.1:7860").rstrip("/"),
        )


class SupabaseAuth:
    def __init__(self, settings: AuthSettings, transport=None):
        self.settings = settings
        self.transport = transport
        self.sessions = {}
        self.session_lock = Lock()

    def call(self, method, path, *, token=None, data=None):
        headers = {"apikey": self.settings.key}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with httpx.Client(timeout=15, transport=self.transport) as client:
                response = client.request(
                    method, self.settings.url + "/auth/v1/" + path, headers=headers, json=data
                )
            if response.status_code == 429:
                raise HTTPException(429, "Too many attempts. Please try again later.")
            if response.status_code >= 500:
                raise HTTPException(503, "Sign-in service is unavailable. Please try again.")
            if response.status_code >= 400:
                raise HTTPException(
                    401, "Sign-in failed. Check your details and confirm your email."
                )
            return response.json() if response.content else {}
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(503, "Sign-in service is unavailable. Please try again.") from exc

    def user(self, token):
        result = self.call("GET", "user", token=token)
        try:
            return str(UUID(result["id"]))
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(401, "Invalid user identity.") from exc

    def identity(self, request: Request):
        bearer = request.headers.get("authorization", "")
        if bearer:
            if not bearer.startswith("Bearer "):
                raise HTTPException(401, "A Bearer access token is required.")
            return self.user(bearer[7:])
        with self.session_lock:
            session = self.sessions.get(request.cookies.get("stemscope_session"))
        if not session or session[1] <= time.time():
            raise HTTPException(401, "Please sign in.")
        return self.user(session[0])

    def new_session(self, payload):
        token = payload.get("access_token")
        if not token:
            raise HTTPException(401, "Confirm your email before signing in.")
        self.user(token)
        now = time.time()
        sid = secrets.token_urlsafe(32)
        lifetime = min(max(int(payload.get("expires_in", 3600)), 1), 3600)
        with self.session_lock:
            self.sessions = {k: v for k, v in self.sessions.items() if v[1] > now}
            if len(self.sessions) >= 1000:
                raise HTTPException(503, "Session capacity reached. Please try later.")
            self.sessions[sid] = (token, now + lifetime)
        return sid, lifetime

    def forget(self, session_id):
        with self.session_lock:
            self.sessions.pop(session_id, None)
