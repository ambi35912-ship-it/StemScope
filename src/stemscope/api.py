"""Local HTTP interface to the existing separation service."""

import logging
import re
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from stemscope.application import choose_model, create_services
from stemscope.auth import AuthSettings, SupabaseAuth
from stemscope.config import Settings
from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS
from stemscope.service import SeparationService

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, services=None, auth=None) -> FastAPI:
    """Build a single-process local API; injectable adapters keep tests lightweight."""
    settings = settings or Settings.from_env()
    services = create_services(settings) if services is None else services
    app = FastAPI(title="StemScope", version="0.4.0")
    busy = Lock()
    auth_config = AuthSettings.from_env()
    auth = auth or (SupabaseAuth(auth_config) if auth_config else None)
    app.state.auth = auth

    def current_user(request: Request):
        return auth.identity(request) if auth else None

    def user_settings(user):
        return (
            replace(settings, output_dir=settings.output_dir / "users" / user) if user else settings
        )

    @app.get("/health")
    def health():
        return {"status": "ok", "device": settings.device, "models_loaded": "lazy"}

    @app.get("/models")
    def models(user: Annotated[str | None, Depends(current_user)]):
        return {"models": list(services)}

    @app.post("/separations", status_code=201)
    def separate(
        audio: Annotated[UploadFile, File()],
        model: Annotated[str, Form()],
        user: Annotated[str | None, Depends(current_user)],
    ):
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "Another separation is running. Retry when it finishes.")
        directory = None
        scoped = user_settings(user)
        try:
            suffix = Path(audio.filename or "").suffix.lower()
            if suffix not in {".wav", ".mp3"}:
                raise HTTPException(400, "Upload a WAV or MP3 file.")
            scoped.output_dir.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="upload-", dir=scoped.output_dir))
            source = directory / ("input" + suffix)
            size = 0
            with source.open("wb") as destination:
                while chunk := audio.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_bytes:
                        raise HTTPException(413, "Audio exceeds the configured upload limit.")
                    destination.write(chunk)
            selected, reason = choose_model(source, model, scoped)
            if selected not in services:
                raise HTTPException(400, "Unknown model. See /models for available names.")
            service = (
                SeparationService(services[selected].separator, scoped)
                if user
                else services[selected]
            )
            result = service.process(source)
            job_id = result.stems["vocals"].parent.name
            logger.info("API separation completed", extra={"job_id": job_id})
            return {
                "job_id": job_id,
                "filename": Path(audio.filename or "audio").name,
                "model": result.model,
                "selection_reason": reason,
                "seconds": result.seconds,
                "stems": {stem: f"/separations/{job_id}/stems/{stem}" for stem in STEMS},
            }
        except StemScopeError as exc:
            raise HTTPException(400, str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("API separation failed")
            raise HTTPException(500, "Processing failed. See the server logs.") from exc
        finally:
            if directory is not None:
                try:
                    shutil.rmtree(directory)
                except OSError:
                    logger.exception("Could not remove captured API upload %s", directory)
            audio.file.close()
            busy.release()

    @app.get("/separations/{job_id}/stems/{stem}")
    def download(job_id: str, stem: str, user: Annotated[str | None, Depends(current_user)]):
        if not re.fullmatch(r"job-[a-zA-Z0-9_-]+", job_id) or stem not in STEMS:
            raise HTTPException(404, "Stem not found.")
        root = user_settings(user).output_dir.resolve()
        directory = root / job_id
        path = directory / f"{stem}.wav"
        if (
            directory.is_symlink()
            or path.is_symlink()
            or path.resolve().parent.parent != root
            or not (directory / "result.json").is_file()
            or not path.is_file()
        ):
            raise HTTPException(404, "Stem not found.")
        return FileResponse(path, media_type="audio/wav", filename=f"{job_id}-{stem}.wav")

    if auth:
        from stemscope.auth_web import install_auth_pages

        install_auth_pages(app, auth)
    return app


def main() -> None:
    import uvicorn

    from stemscope.logging_config import configure_logging

    configure_logging()
    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
