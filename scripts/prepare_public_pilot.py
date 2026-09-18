"""Fetch a small, explicitly licensed official MUSDB18 test-excerpt pilot.

Only selected ZIP members are downloaded with HTTP ranges. No model outputs or
training tracks are substituted for original test references. Run from the project root.
"""

import argparse
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

URL = "https://zenodo.org/records/1256064/files/MUSDB18-7-WAV.zip?download=1"


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.pos = 0
        self.start = 0
        self.cache = b""
        request = urllib.request.Request(URL, headers={"Range": "bytes=-65536"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise RuntimeError("Server does not support ranges")
            self.size = int(response.headers["Content-Range"].split("/")[-1])
            self.cache = response.read()
            self.start = self.size - len(self.cache)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (
            offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        )
        return self.pos

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if not (self.start <= self.pos and self.pos + n <= self.start + len(self.cache)):
            end = min(self.size, self.pos + max(n, 65536))
            request = urllib.request.Request(URL, headers={"Range": f"bytes={self.pos}-{end - 1}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206 or not response.headers["Content-Range"].startswith(
                    f"bytes {self.pos}-"
                ):
                    raise RuntimeError("Unexpected response to range request")
                self.cache = response.read()
                self.start = self.pos
        result = self.cache[self.pos - self.start : self.pos - self.start + n]
        self.pos += len(result)
        return result


METADATA_URL = (
    "https://raw.githubusercontent.com/sigsep/website/master/content/datasets/assets/tracklist.csv"
)


def select_tracks(names, metadata, limit=5):
    """Select alphabetically before scoring, requiring an explicitly listed CC license."""
    tracks = sorted(
        {
            PurePosixPath(name).parts[1]
            for name in names
            if len(PurePosixPath(name).parts) == 3
            and PurePosixPath(name).parts[0] == "test"
            and PurePosixPath(name).parts[2] == "mixture.wav"
        }
    )
    return [
        track
        for track in tracks
        if track in metadata and metadata[track]["License"].startswith("CC BY-NC-SA")
    ][:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/musdb7-pilot"))
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.limit <= 50:
        parser.error("limit must be between 1 and 50")
    if args.output.exists():
        parser.error("output already exists; choose a new folder to avoid overwriting data")
    with urllib.request.urlopen(METADATA_URL, timeout=60) as response:
        metadata_bytes = response.read()
    metadata = {
        row["Track Name"]: row for row in csv.DictReader(metadata_bytes.decode().splitlines())
    }
    with zipfile.ZipFile(RemoteZip()) as archive:
        selected = select_tracks(archive.namelist(), metadata, args.limit)
        if not selected:
            parser.error("no eligible test tracks found")
        args.output.mkdir(parents=True)
        (args.output / "source-tracklist.csv").write_bytes(metadata_bytes)
        records = []
        for track in selected:
            if track in {".", ".."} or "/" in track or "\\" in track:
                raise ValueError("Unsafe track name")
            directory = args.output / track
            directory.mkdir()
            record = dict(metadata[track], files={})
            for stem in ("mixture", "vocals", "drums", "bass", "other"):
                member = f"test/{track}/{stem}.wav"
                info = archive.getinfo(member)
                if info.file_size > 10 * 1024 * 1024:
                    raise ValueError("Unexpectedly large excerpt")
                data = archive.read(member)  # ZipFile verifies the member's CRC.
                path = directory / f"{stem}.wav"
                path.write_bytes(data)
                record["files"][stem] = {
                    "member": member,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "zip_crc32": info.CRC,
                }
                print(f"Downloaded {member}", flush=True)
            records.append(record)
        provenance = {
            "dataset": "SiSEC18-MUS 7s Excerpts, version 1.0.0, MUSDB18-7-WAV.zip",
            "record": "https://zenodo.org/records/1256064",
            "url": URL,
            "metadata_url": METADATA_URL,
            "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "selection": "Alphabetical test tracks with CC BY-NC-SA listed in official metadata; selected before inference",
            "requested_limit": args.limit,
            "actual_tracks": len(records),
            "caveats": "Short activity-selected excerpts; WAV preview archive is not full MUSDB18-HQ; no independent training-overlap audit; retain attribution and noncommercial/share-alike terms",
            "tracks": records,
        }
        (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"Prepared {len(records)} real test excerpts in {args.output}", flush=True)


if __name__ == "__main__":
    main()
