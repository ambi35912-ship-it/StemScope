"""Browser login pages and isolated authenticated Gradio workspaces."""

import asyncio
import html
import json
import os
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from stemscope.service import SeparationService


def page(message="", signed_in=False, workspace_url="/app/"):
    body = (
        f'<p><a href="{workspace_url}">Open StemScope</a></p><form method="post" action="/auth/logout"><button>Sign out</button></form>'
        if signed_in
        else '<form method="post" action="/auth/login">'
        '<label>Email<input name="email" type="email" autocomplete="email" required maxlength="254"></label>'
        '<label>Password<input name="password" type="password" autocomplete="current-password" required minlength="1" maxlength="128"></label>'
        '<button>Sign in</button><button formaction="/auth/signup">Create account</button></form>'
        "<p>After creating an account, confirm your email if requested, then sign in.</p>"
    )
    return HTMLResponse(
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>StemScope — Account</title>
<style>body{font:17px system-ui;background:#f4f6fa;color:#172033;margin:0}main{max-width:430px;margin:8vh auto;padding:32px;background:white;border-radius:16px}label{display:block;margin:18px 0}input{display:block;width:95%;padding:12px;border:1px solid #aaa;border-radius:6px}button{padding:12px;margin:8px 8px 0 0;cursor:pointer}a{color:#294abb}</style>
<main><h1>StemScope</h1><h2>Your music workspace</h2>"""
        + f'<p role="status">{html.escape(message)}</p>'
        + body
        + "</main></html>",
        headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
    )


def install_auth_pages(app, auth):
    @app.middleware("http")
    async def protect_origin(request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            auth_form = request.url.path.startswith("/auth/")
            cookie_request = "stemscope_session" in request.cookies
            if (
                auth_form or (cookie_request and not request.headers.get("authorization"))
            ) and request.headers.get("origin") != auth.settings.origin:
                return JSONResponse({"detail": "Request origin is not allowed."}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/", response_class=HTMLResponse)
    def account(request: Request):
        try:
            auth.identity(request)
            return page(signed_in=True, workspace_url=getattr(app.state, "workspace_url", "/docs"))
        except HTTPException:
            return page()

    @app.post("/auth/login")
    def login(request: Request, email: str = Form(...), password: str = Form(...)):
        if len(email) > 254 or not 1 <= len(password) <= 128:
            return page("Check your email and password.")
        try:
            payload = auth.call(
                "POST", "token?grant_type=password", data={"email": email, "password": password}
            )
            sid, lifetime = auth.new_session(payload)
        except HTTPException as exc:
            response = page(exc.detail)
            response.status_code = exc.status_code
            return response
        auth.forget(request.cookies.get("stemscope_session"))
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(
            "stemscope_session",
            sid,
            max_age=lifetime,
            httponly=True,
            secure=auth.settings.origin.startswith("https:"),
            samesite="strict",
            path="/",
        )
        return response

    @app.post("/auth/signup")
    def signup(email: str = Form(...), password: str = Form(...)):
        if len(email) > 254 or not 8 <= len(password) <= 128:
            return page("Use an email address and a password of 8–128 characters.")
        try:
            auth.call("POST", "signup", data={"email": email, "password": password})
        except HTTPException as exc:
            response = page("Could not create an account. Please try again later.")
            response.status_code = exc.status_code
            return response
        return page("If your account needs confirmation, check your email. Then sign in.")

    @app.post("/auth/logout")
    def logout(request: Request):
        auth.forget(request.cookies.get("stemscope_session"))
        response = RedirectResponse("/", status_code=303)
        response.delete_cookie("stemscope_session", path="/")
        return response

    @app.get("/auth/me")
    def me(request: Request):
        return {"id": auth.identity(request)}


class UserWorkspaces:
    """One Gradio app/cache per user; model adapters remain shared within the process."""

    def __init__(self, auth, settings, services, stack, builder=None):
        self.auth, self.settings, self.services, self.stack = auth, settings, services, stack
        self.builder = builder
        self.apps = {}
        self.lock = asyncio.Lock()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await send({"type": "websocket.close", "code": 1008})
            return
        request = Request(scope, receive)
        try:
            user = await run_in_threadpool(self.auth.identity, request)
        except HTTPException as exc:
            response = (
                RedirectResponse("/", status_code=303)
                if request.url.path.rstrip("/") == "/app"
                else JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
            )
            await response(scope, receive, send)
            return
        root = (self.settings.output_dir / "users" / user).resolve()
        # Gradio's global temporary directory is not a cross-user file allowlist.
        path = unquote(scope.get("path", ""))
        file_marker = next((marker for marker in ("/file=", "/file/") if marker in path), None)
        if file_marker:
            candidate = path.split(file_marker, 1)[1]
            try:
                Path(candidate).resolve().relative_to(root)
            except ValueError:
                await JSONResponse({"detail": "File not found."}, status_code=404)(
                    scope, receive, send
                )
                return
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if request.method == "POST" and (
            not content_type or content_type == "application/json" or content_type.endswith("+json")
        ):
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 2 * 1024 * 1024:
                    await JSONResponse({"detail": "Request too large."}, status_code=413)(
                        scope, receive, send
                    )
                    return

            def validate_files(value):
                if isinstance(value, dict):
                    if "path" in value and isinstance(value["path"], str):
                        Path(value["path"]).resolve().relative_to(root)
                    for item in value.values():
                        validate_files(item)
                elif isinstance(value, list):
                    for item in value:
                        validate_files(item)

            try:
                validate_files(json.loads(body))
            except (ValueError, TypeError, RecursionError):
                await JSONResponse(
                    {"detail": "Invalid workspace file or request."}, status_code=400
                )(scope, receive, send)
                return
            original_receive = receive
            replayed = False

            async def replay_receive():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await original_receive()

            receive = replay_receive
        async with self.lock:
            if user not in self.apps:
                if len(self.apps) >= 20:
                    await JSONResponse({"detail": "Workspace capacity reached."}, status_code=503)(
                        scope, receive, send
                    )
                    return
                scoped = replace(self.settings, output_dir=root)
                services = {
                    name: SeparationService(service.separator, scoped)
                    for name, service in self.services.items()
                }
                root.mkdir(parents=True, exist_ok=True)
                import gradio as gr

                from stemscope.ui import build_app

                blocks = await run_in_threadpool(self.builder or build_app, services)
                with blocks:
                    gr.Markdown("[Account / sign out](/)")
                cache = root / ".gradio"
                cache.mkdir(exist_ok=True)
                for block in [blocks, *blocks.blocks.values()]:
                    block.GRADIO_CACHE = str(cache)
                mounted = gr.mount_gradio_app(
                    FastAPI(),
                    blocks,
                    path="/",
                    allowed_paths=[str(root)],
                    auth_dependency=lambda request: request.state.stemscope_user,
                    max_file_size=scoped.max_bytes,
                    show_error=False,
                )
                for route in mounted.routes:
                    if hasattr(route, "app") and hasattr(route.app, "uploaded_file_dir"):
                        route.app.uploaded_file_dir = str(cache)
                await self.stack.enter_async_context(mounted.router.lifespan_context(mounted))
                self.apps[user] = mounted
        scope.setdefault("state", {})["stemscope_user"] = user
        await self.apps[user](scope, receive, send)


def create_authenticated_ui(settings, services, auth, builder=None):
    from stemscope.api import create_app

    # Gradio 5 checks all uploaded paths against this process-wide parent.
    # The dispatcher additionally restricts each request to its own user root.
    os.environ["GRADIO_TEMP_DIR"] = str((settings.output_dir / "users").resolve())
    stack = AsyncExitStack()
    app = create_app(settings, services, auth=auth)

    @asynccontextmanager
    async def lifespan(app):
        async with stack:
            yield

    app.router.lifespan_context = lifespan
    app.state.workspace_url = "/app/"
    app.mount("/app", UserWorkspaces(auth, settings, services, stack, builder))
    return app
