"""Load prepared reference datasets with provenance and integrity checks."""

import hashlib
import json
from pathlib import Path

from stemscope.benchmarking.runner import BenchmarkTrack
from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS


def load_prepared_tracks(root: Path, *, pilot: bool = False) -> list[BenchmarkTrack]:
    """Use downloaded references; never substitute synthetic or separated audio."""
    try:
        metadata = json.loads((root / "provenance.json").read_text())
        if "planned_tracks" in metadata:
            recorded = [record["Track Name"] for record in metadata["tracks"]]
            if sorted(recorded) != sorted(metadata["planned_tracks"]):
                raise StemScopeError("The research dataset download is incomplete.")
        split = metadata.get("split", "test")
        if split not in {"train", "test"} or (pilot and split != "test"):
            raise StemScopeError("Invalid dataset split.")
        tracks = []
        for record in metadata["tracks"]:
            name = record["Track Name"]
            if name in {".", ".."} or "/" in name or "\\" in name:
                raise StemScopeError("Invalid track name in pilot metadata.")
            paths = {}
            for stem in ("mixture", *STEMS):
                path = root / name / f"{stem}.wav"
                if record["files"][stem]["member"] != f"{split}/{name}/{stem}.wav":
                    raise StemScopeError(
                        "Pilot metadata must identify original files from the declared split."
                    )
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != record["files"][stem]["sha256"]:
                    raise StemScopeError(
                        "A pilot audio file changed. Restore the original download before scoring."
                    )
                paths[stem] = path
            tracks.append(
                BenchmarkTrack(
                    name,
                    paths.pop("mixture"),
                    paths,
                    f"{metadata['dataset']}; official {split} excerpt; {record['License']}; "
                    f"{metadata['record']}; "
                    + (
                        "small same-artist pilot, not a full benchmark"
                        if pilot
                        else metadata.get("selection", "prepared reference dataset")
                    ),
                )
            )
        if not tracks:
            raise StemScopeError("The prepared pilot has no tracks.")
        return tracks
    except StemScopeError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StemScopeError(
            "Prepared real-music references are missing or invalid. See the README public-pilot setup."
        ) from exc


def load_public_pilot(root: Path) -> list[BenchmarkTrack]:
    """Compatibility entry point for the small CC-only UI pilot."""
    return load_prepared_tracks(root, pilot=True)
