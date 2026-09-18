"""Run an isolated, warmed CPU study; workers execute sequentially, one model each."""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from stemscope.benchmarking.datasets import load_prepared_tracks
from stemscope.benchmarking.runner import BenchmarkRunner
from stemscope.config import Settings
from stemscope.errors import StemScopeError

logger = logging.getLogger(__name__)

MODELS = ("Demucs (htdemucs)", "Open-Unmix (umxhq)")


def worker(dataset: Path, output: Path, model: str, repeats: int, threads: int) -> None:
    import psutil
    import torch

    from stemscope.application import create_services

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    settings = Settings(output.resolve(), device="cpu")
    tracks = load_prepared_tracks(dataset)
    services = create_services(settings)
    service = services[model]
    logger.info("Warm-up (excluded): %s", model)
    warmup = service.process(tracks[0].mixture)
    experiment = {
        "kind": "isolated_warm_cpu_study_v1",
        "pid": os.getpid(),
        "model": model,
        "warmup_calls": 1,
        "warmup_source": str(tracks[0].mixture),
        "warmup_job": str(warmup.stems["vocals"].parent),
        "warmup_seconds_excluded": warmup.seconds,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "system_memory_bytes": psutil.virtual_memory().total,
        "timing": "Measured SeparationService calls after one excluded warm-up; includes input decode, inference, export and output checks; excludes reference scoring and report export. Single model loaded per fresh worker process; workers sequential; CPU threads fixed; no CPU affinity or exclusive-machine control.",
    }
    result = BenchmarkRunner({model: service}, settings, experiment=experiment).run(tracks, repeats)
    (output / "worker-result.json").write_text(
        json.dumps(
            {
                "csv": str(result.csv_path),
                "manifest": str(result.manifest_path),
            },
            indent=2,
        )
        + "\n"
    )


def run_study(dataset: Path, output: Path, repeats: int = 2, threads: int = 4) -> Path:
    """Run both models in separate sequential processes and retain partial/failure evidence."""
    if repeats < 1 or repeats > 10 or threads < 1:
        raise StemScopeError("Use 1–10 measured repetitions and at least one CPU thread.")
    tracks = load_prepared_tracks(dataset)
    if not tracks:
        raise StemScopeError("No prepared reference tracks found.")
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="study-", dir=output)).resolve()
    protocol = {
        "kind": "isolated_warm_cpu_study_v1",
        "dataset": str(dataset.resolve()),
        "models_in_order": list(MODELS),
        "repeats": repeats,
        "threads": threads,
        "expected_tracks": [track.name for track in tracks],
        "expected_rows": len(tracks) * len(MODELS) * repeats * 4,
        "source_provenance": json.loads((dataset / "provenance.json").read_text()),
        "worker_results": [],
        "failures": [],
    }
    manifest_path = directory / "study.json"
    manifest_path.write_text(json.dumps(protocol, indent=2) + "\n")
    combined = []
    for index, model in enumerate(MODELS):
        target = directory / f"model-{index + 1}"
        target.mkdir()
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
        command = [
            sys.executable,
            "-m",
            "stemscope.benchmarking.study",
            str(dataset.resolve()),
            "--output",
            str(target),
            "--repeats",
            str(repeats),
            "--threads",
            str(threads),
            "--worker",
            model,
        ]
        print(
            f"Running {model}: excluded warm-up, then {len(tracks) * repeats} measured calls",
            flush=True,
        )
        with (target / "worker.log").open("w") as log:
            completed = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        if completed.returncode != 0:
            protocol["failures"].append(
                {
                    "model": model,
                    "exit_code": completed.returncode,
                    "log": str(target / "worker.log"),
                }
            )
        else:
            locations = json.loads((target / "worker-result.json").read_text())
            model_manifest = json.loads(Path(locations["manifest"]).read_text())
            protocol["worker_results"].append(
                {"model": model, "locations": locations, "manifest": model_manifest}
            )
            with Path(locations["csv"]).open() as stream:
                combined.extend(csv.DictReader(stream))
        manifest_path.write_text(json.dumps(protocol, indent=2) + "\n")
        if combined:
            with (directory / "results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
                writer.writeheader()
                writer.writerows(combined)
        print(f"Finished {model}, exit={completed.returncode}", flush=True)
    if protocol["failures"]:
        raise StemScopeError(
            f"Study has failed workers; all available evidence is saved in {directory}"
        )
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/phase4-study"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--worker", choices=MODELS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        if args.worker:
            worker(args.dataset, args.output, args.worker, args.repeats, args.threads)
        else:
            directory = run_study(args.dataset, args.output, args.repeats, args.threads)
            print(f"Study completed: {directory}", flush=True)
    except StemScopeError as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
