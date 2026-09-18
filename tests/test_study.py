import pytest

from stemscope.benchmarking.results import completed_study
from stemscope.benchmarking.summary import aggregate, distribution
from stemscope.errors import StemScopeError
from stemscope.separators.base import STEMS


def observations():
    return [
        {
            "track": track,
            "model": model,
            "repeat": repeat,
            "stem": stem,
            "status": "ok",
            "metric_status": "ok",
            "si_sdr_db": str(value + repeat),
            "si_sdr_improvement_db": "2",
            "wall_seconds": "1",
            "cpu_seconds": "2",
            "sampled_peak_rss_bytes": "1048576",
            "real_time_factor": "0.14",
        }
        for track, value in [("a", 0), ("b", 10)]
        for model in ["first", "second"]
        for repeat in [1, 2]
        for stem in STEMS
    ]


def test_repetitions_are_not_independent_tracks_or_stem_timings():
    result = aggregate(observations(), ["first", "second"], ["a", "b"], 2)
    assert result["rows"] == 32
    assert result["measured_calls"] == 8
    assert result["quality"][0]["n"] == 2
    assert result["quality"][0]["median"] == 6.5
    assert result["timing"][0]["n"] == 4
    assert result["paired"][0]["ties"] == 2


@pytest.mark.parametrize("change", ["missing", "duplicate", "resources"])
def test_corrupt_coverage_rejected(change):
    rows = observations()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["wall_seconds"] = "9"
    with pytest.raises(StemScopeError):
        aggregate(rows, ["first", "second"], ["a", "b"], 2)


def test_missing_scores_retained_not_zero():
    rows = observations()
    for row in rows:
        if row["track"] == "a" and row["model"] == "first":
            row.update(metric_status="silent_reference", si_sdr_db="", si_sdr_improvement_db="")
    result = aggregate(rows, ["first", "second"], ["a", "b"], 2)
    assert len(result["failures"]) == 8
    assert result["quality"][0]["missing_tracks"] == 1
    assert result["quality"][0]["median"] == 11.5
    assert result["paired"][0]["n_pairs"] == 1
    assert distribution([])["median"] is None


def test_missing_report_is_actionable(tmp_path):
    with pytest.raises(StemScopeError, match="No completed"):
        completed_study(tmp_path)


@pytest.mark.parametrize("score", ["", "nan", "inf", None])
def test_invalid_success_score_rejected(score):
    rows = observations()
    rows[0]["si_sdr_db"] = score
    with pytest.raises(StemScopeError, match="non-finite"):
        aggregate(rows, ["first", "second"], ["a", "b"], 2)


@pytest.mark.parametrize("fault", ["hash", "threads", "model", "tracks", "workers"])
def test_inconsistent_worker_protocol_rejected(tmp_path, fault):
    import json

    from stemscope.benchmarking.summary import summarize

    workers = [
        {
            "model": model,
            "manifest": {
                "models": [model],
                "repeats": 2,
                "experiment": {
                    "kind": "isolated_warm_cpu_study_v1",
                    "model": model,
                    "warmup_calls": 1,
                    "torch_threads": 4,
                    "torch_interop_threads": 1,
                },
                "tracks": [
                    {
                        "name": "a",
                        "mixture": {"sha256": "same"},
                        "references": {stem: {"sha256": "same"} for stem in STEMS},
                    }
                ],
            },
        }
        for model in ["first", "second"]
    ]
    protocol = {
        "failures": [],
        "worker_results": workers,
        "models_in_order": ["first", "second"],
        "threads": 4,
        "repeats": 2,
        "expected_tracks": ["a"],
    }
    if fault == "hash":
        workers[1]["manifest"]["tracks"][0]["mixture"]["sha256"] = "different"
    elif fault == "threads":
        workers[1]["manifest"]["experiment"]["torch_threads"] = 8
    elif fault == "model":
        workers[1]["model"] = "first"
    elif fault == "tracks":
        protocol["expected_tracks"] = ["b"]
    else:
        workers.pop()
    (tmp_path / "study.json").write_text(json.dumps(protocol))
    with pytest.raises(StemScopeError):
        summarize(tmp_path)
