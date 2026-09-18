from pathlib import Path

from stemscope.errors import StemScopeError


def validate_path(path: str | Path, max_bytes: int = 200 * 1024 * 1024) -> Path:
    """Reject unsupported, missing, empty, or oversized inputs before decoding."""
    source = Path(path)
    if source.suffix.lower() not in {".wav", ".mp3"}:
        raise StemScopeError("Upload a WAV or MP3 file.")
    try:
        if not source.is_file():
            raise StemScopeError("The uploaded file no longer exists. Please upload it again.")
        size = source.stat().st_size
    except OSError as exc:
        raise StemScopeError("The uploaded file cannot be accessed.") from exc
    if size == 0:
        raise StemScopeError("The uploaded file is empty.")
    if size > max_bytes:
        raise StemScopeError(f"The upload exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    return source
