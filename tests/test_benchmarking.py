import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from stemscope.benchmarking.demo import create_demo
from stemscope.benchmarking.metrics import align_estimate, si_sdr
from stemscope.benchmarking.runner import BenchmarkRunner, validate_track
from stemscope.config import Settings
from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS, Separator
from stemscope.service import SeparationService


def signals():
    t = np.arange(8000) / 8000
    return np.sin(2 * np.pi * 100 * t)[:, None], np.sin(2 * np.pi * 300 * t)[:, None]


def test_si_sdr_known_20db_and_scale_invariance():
    reference, noise = signals()
    estimate = reference + 0.1 * noise
    score = si_sdr(reference, estimate)
    assert score.status == "ok"
    assert score.db == pytest.approx(20, abs=1e-8)
    assert si_sdr(reference, -3 * estimate).db == pytest.approx(score.db)
    assert si_sdr(reference + 2, estimate - 1).db == pytest.approx(score.db)


@pytest.mark.parametrize(
    "kind,status",
    [
        ("perfect", "perfect_positive_infinity"),
        ("reference", "silent_reference"),
        ("estimate", "silent_estimate"),
        ("orthogonal", "zero_projection_negative_infinity"),
    ],
)
def test_metric_edge_cases_are_explicit(kind, status):
    ref, est = signals()
    if kind == "perfect":
        est = ref * 2
    elif kind == "reference":
        ref = np.zeros_like(ref)
    elif kind == "estimate":
        est = np.zeros_like(est)
    score = si_sdr(ref, est)
    assert score.db is None and score.status == status


def test_stereo_preserves_channel_balance_in_score():
    ref, _ = signals()
    ref = np.column_stack([ref, -ref])
    est = ref * np.array([1, 0.5])
    assert si_sdr(ref, est).db is not None
    assert si_sdr(ref, ref).status == "perfect_positive_infinity"


@pytest.mark.parametrize("shape", [(10,), (10, 3)])
def test_metric_rejects_mismatched_arrays(shape):
    with pytest.raises(StemScopeError):
        si_sdr(np.zeros((10, 2)), np.zeros(shape))


def test_metric_rejects_nan():
    with pytest.raises(StemScopeError):
        si_sdr(np.ones((10, 1)), np.full((10, 1), np.nan))


def test_alignment_resamples_and_only_trims_rounding():
    ref, _ = signals()
    estimate, aligned = align_estimate(np.repeat(ref, 2, axis=1), 8000, ref, 8000)
    assert estimate.shape == aligned.shape == (8000, 2)
    estimate, aligned = align_estimate(np.zeros((16000, 2)), 16000, ref, 8000)
    assert len(estimate) == len(aligned) == 8000
    with pytest.raises(StemScopeError, match="durations"):
        align_estimate(np.zeros((7000, 1)), 8000, ref, 8000)
    with pytest.raises(StemScopeError, match="channel"):
        align_estimate(ref, 8000, np.repeat(ref, 2, axis=1), 8000)


class CopyMix(Separator):
    name = "Copy mix baseline"

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def separate(self, samples, sample_rate, output_dir):
        self.calls.append(samples.copy())
        if self.fail:
            raise StemScopeError("Deliberate inference failure")
        paths = {stem: output_dir / f"{stem}.wav" for stem in STEMS}
        for path in paths.values():
            sf.write(path, samples, sample_rate, subtype="FLOAT")
        return paths


@pytest.fixture
def demo(tmp_path):
    return create_demo(tmp_path / "inputs")


def test_runner_scores_baseline_and_records_protocol(tmp_path, demo):
    settings = Settings(tmp_path / "out")
    first, second = CopyMix(), CopyMix()
    runner = BenchmarkRunner(
        {"one": SeparationService(first, settings), "two": SeparationService(second, settings)},
        settings,
    )
    result = runner.run([demo], repeats=2)
    assert len(result.rows) == 16
    assert len(first.calls) == len(second.calls) == 2
    np.testing.assert_array_equal(first.calls[0], second.calls[0])
    for row in result.rows:
        assert row["status"] == "ok"
        assert row["si_sdr_improvement_db"] == pytest.approx(0, abs=1e-7)
        assert row["sampled_peak_rss_bytes"] >= row["baseline_rss_bytes"] > 0
        assert row["wall_seconds"] > 0
    manifest = json.loads(result.manifest_path.read_text())
    assert len(manifest["tracks"][0]["mixture"]["sha256"]) == 64
    assert manifest["tracks"][0]["provenance"].startswith("Generated")
    with result.csv_path.open() as stream:
        assert len(list(csv.DictReader(stream))) == 16


def test_no_references_means_no_quality_score(tmp_path, demo):
    settings = Settings(tmp_path / "out")
    runner = BenchmarkRunner({"copy": SeparationService(CopyMix(), settings)}, settings)
    result = runner.run([replace(demo, references={})])
    assert all(
        row["si_sdr_db"] is None and row["metric_status"] == "no_reference" for row in result.rows
    )


def test_failed_model_does_not_disappear_or_stop_other_models(tmp_path, demo):
    settings = Settings(tmp_path / "out")
    runner = BenchmarkRunner(
        {
            "bad": SeparationService(CopyMix(True), settings),
            "good": SeparationService(CopyMix(), settings),
        },
        settings,
    )
    result = runner.run([demo])
    assert [row["status"] for row in result.rows[:4]] == ["separation_failed"] * 4
    assert all(row["status"] == "ok" for row in result.rows[4:])


def test_partial_or_mismatched_references_fail_before_inference(tmp_path, demo):
    settings = Settings(tmp_path / "out")
    adapter = CopyMix()
    runner = BenchmarkRunner({"copy": SeparationService(adapter, settings)}, settings)
    with pytest.raises(StemScopeError, match="all four"):
        runner.run([replace(demo, references={"vocals": demo.references["vocals"]})])
    sf.write(demo.references["bass"], np.zeros((100, 2)), 44100)
    with pytest.raises(StemScopeError, match="exact frame count"):
        runner.run([demo])
    assert not adapter.calls


def test_mp3_references_not_accepted(tmp_path, demo):
    path = tmp_path / "reference.mp3"
    path.write_bytes(demo.references["vocals"].read_bytes())
    refs = dict(demo.references, vocals=path)
    with pytest.raises(StemScopeError, match="must be WAV"):
        validate_track(replace(demo, references=refs), Settings(tmp_path))


def test_invalid_repetitions(tmp_path, demo):
    settings = Settings(tmp_path)
    runner = BenchmarkRunner({"copy": SeparationService(CopyMix(), settings)}, settings)
    with pytest.raises(StemScopeError):
        runner.run([demo], repeats=0)


def test_synthetic_example_is_explicit_and_sums_to_mix(demo):
    mixture, _ = sf.read(demo.mixture)
    total = sum(sf.read(path)[0] for path in demo.references.values())
    np.testing.assert_allclose(total, mixture, atol=2e-8)
    assert "not music" in demo.name and "placeholders" in demo.provenance


def test_renamed_mp3_cannot_bypass_wav_reference_requirement(tmp_path, demo):
    import lameenc

    encoder = lameenc.Encoder()
    encoder.set_in_sample_rate(44100)
    encoder.set_channels(2)
    path = tmp_path / "renamed.wav"
    path.write_bytes(
        encoder.encode(np.zeros((44100, 2), dtype=np.int16).tobytes()) + encoder.flush()
    )
    with pytest.raises(StemScopeError, match="aligned WAV"):
        validate_track(replace(demo, mixture=path), Settings(tmp_path))
