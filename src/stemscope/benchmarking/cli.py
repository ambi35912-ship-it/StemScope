"""Command-line evaluation of a folder of aligned tracks."""

import argparse
import logging
from pathlib import Path

from stemscope.application import create_services
from stemscope.benchmarking.runner import BenchmarkRunner, BenchmarkTrack
from stemscope.config import Settings
from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark both separators; scores require original reference stems."
    )
    parser.add_argument(
        "dataset", type=Path, help="Directory with one subfolder per track and mixture.wav"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--provenance", required=True, help="Dataset/source, split, and any relevant caveats"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        if not args.dataset.is_dir():
            raise StemScopeError("The dataset directory does not exist.")
        tracks = []
        for directory in sorted(args.dataset.iterdir()):
            if directory.is_dir() and (directory / "mixture.wav").is_file():
                refs = {
                    stem: directory / f"{stem}.wav"
                    for stem in STEMS
                    if (directory / f"{stem}.wav").exists()
                }
                tracks.append(
                    BenchmarkTrack(directory.name, directory / "mixture.wav", refs, args.provenance)
                )
        settings = Settings(args.output.resolve(), device=args.device)
        result = BenchmarkRunner(create_services(settings), settings).run(tracks, args.repeats)
        print(f"Results: {result.csv_path}\nProtocol and input hashes: {result.manifest_path}")
        if any(row["status"] != "ok" for row in result.rows):
            parser.exit(1, "Some runs failed; inspect the saved CSV.\n")
    except StemScopeError as exc:
        parser.exit(2, f"Benchmark error: {exc}\n")


if __name__ == "__main__":
    main()
