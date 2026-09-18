"""Prepare public MUSDB18 research previews: 50 test or 94 available train mixtures.

Source licensing is preserved verbatim. Use separate output directories per split.
"""

import argparse
import csv
import hashlib
import http.client
import json
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from prepare_public_pilot import METADATA_URL, URL, RemoteZip


class ResearchArchive(RemoteZip):
    """Retry transient public-download failures while respecting server backoff."""

    def read(self, n=-1):
        position = self.tell()
        for attempt in range(5):
            try:
                return super().read(n)
            except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead) as exc:
                self.seek(position)
                if isinstance(exc, urllib.error.HTTPError) and exc.code not in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }:
                    raise
                if attempt == 4:
                    raise
                raw = getattr(exc, "headers", {}).get("Retry-After", "10")
                delay = int(raw) if raw.isdigit() else 10
                print(f"Transient download error; retry in {delay}s", flush=True)
                while delay > 0:
                    pause = min(delay, 30)
                    time.sleep(pause)
                    delay -= pause


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/musdb7-test"))
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    split = args.split
    expected_count = 50 if split == "test" else 94
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "source-tracklist.csv"
    if not metadata_path.exists():
        with urllib.request.urlopen(METADATA_URL, timeout=60) as response:
            metadata_path.write_bytes(response.read())
    metadata_bytes = metadata_path.read_bytes()
    metadata = {r["Track Name"]: r for r in csv.DictReader(metadata_bytes.decode().splitlines())}
    # The public ZIP has 94 train mixtures and two names spelled differently in the source CSV.
    aliases = {
        "Jokers, Jacks & Kings - Sea Of Leaves": "Jokers Jacks & Kings - Sea Of Leaves",
        "Patrick Talbot - Set Me Free": "Patrick Talbot - Set Free Me",
    }
    for archive_name, metadata_name in aliases.items():
        if metadata_name in metadata:
            metadata[archive_name] = dict(
                metadata[metadata_name],
                **{"Track Name": archive_name, "metadata_track_name": metadata_name},
            )
    state_path = args.output / "provenance.json"
    records = json.loads(state_path.read_text())["tracks"] if state_path.exists() else []
    if records and any(
        not info["member"].startswith(f"{split}/")
        for record in records
        for info in record["files"].values()
    ):
        raise ValueError(
            "Existing output contains another split; choose a separate output directory"
        )
    done = {record["Track Name"]: record for record in records}
    remote = ResearchArchive()
    with zipfile.ZipFile(remote) as archive:
        names = archive.namelist()
        selected = sorted(
            {
                PurePosixPath(n).parts[1]
                for n in names
                if len(PurePosixPath(n).parts) == 3
                and PurePosixPath(n).parts[0] == split
                and PurePosixPath(n).parts[-1] == "mixture.wav"
            }
        )
        if len(selected) != expected_count or any(name not in metadata for name in selected):
            raise ValueError(
                f"Expected exactly {expected_count} {split} tracks with official metadata"
            )
        plan = {
            "dataset": f"SiSEC18-MUS 7s WAV excerpts v1.0.0; full {expected_count}-track {split} split",
            "split": split,
            "record": "https://zenodo.org/records/1256064",
            "url": URL,
            "metadata_url": METADATA_URL,
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "selection": f"All {expected_count} {split} previews, alphabetically; no selection by score",
            "usage": "Local research evaluation only; source restrictions apply; do not redistribute audio",
            "planned_tracks": selected,
            "tracks": records,
        }
        (args.output / "selection-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        local = threading.local()

        def fetch(name):
            if name in {".", ".."} or "/" in name or "\\" in name:
                raise ValueError("Unsafe track name")
            folder = args.output / name
            folder.mkdir(exist_ok=True)
            if name in done:
                for stem, info in done[name]["files"].items():
                    if (
                        hashlib.sha256((folder / f"{stem}.wav").read_bytes()).hexdigest()
                        != info["sha256"]
                    ):
                        raise ValueError(f"Previously downloaded input changed: {name}/{stem}")
                return done[name]
            if not hasattr(local, "archive"):
                local.remote = ResearchArchive()
                local.archive = zipfile.ZipFile(local.remote)
            source = local.archive
            members = {
                stem: source.getinfo(f"{split}/{name}/{stem}.wav")
                for stem in ("mixture", "vocals", "drums", "bass", "other")
            }
            if any(info.file_size > 10 * 1024**2 for info in members.values()):
                raise ValueError("Unexpected excerpt size")
            start = min(info.header_offset for info in members.values())
            end = max(info.header_offset + info.compress_size + 4096 for info in members.values())
            local.remote.seek(start)
            local.remote.read(end - start)
            record = dict(metadata[name], files={})
            for stem, info in members.items():
                data = source.read(info)
                target = folder / f"{stem}.wav"
                temporary = target.with_suffix(".partial")
                temporary.write_bytes(data)
                temporary.replace(target)
                record["files"][stem] = {
                    "member": info.filename,
                    "zip_crc32": info.CRC,
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            return record

        # Each worker has its own ZIP cursor/cache; only the caller updates the manifest.
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, record in enumerate(executor.map(fetch, selected), 1):
                name = record["Track Name"]
                if name not in done:
                    records.append(record)
                    done[name] = record
                plan["tracks"] = records
                temporary = state_path.with_suffix(".partial")
                temporary.write_text(json.dumps(plan, indent=2) + "\n")
                temporary.replace(state_path)
                print(f"{index}/{expected_count} verified {name}", flush=True)
        print(f"COMPLETE: {len(records)} {split} tracks prepared", flush=True)


if __name__ == "__main__":
    main()
