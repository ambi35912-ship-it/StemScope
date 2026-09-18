"""Save diagnostic evidence separately from separator execution and timing."""

import hashlib
import json
import tempfile
from pathlib import Path

from stemscope.quality.diagnostics import inspect_files


def inspect_and_save(mixture: Path, stems: dict[str, Path], output: Path) -> tuple[dict, Path]:
    result = inspect_files(mixture, stems)
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="quality-", dir=output))
    files = dict(stems, mixture=mixture)
    result["inputs"] = {
        name: {"path": str(path.resolve()), "sha256": fingerprint(path)}
        for name, path in files.items()
    }
    path = directory / "diagnostics.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result, path


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
