import json

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from stemscope.errors import StemScopeError
from stemscope.selection.features import FEATURE_NAMES, PROTOCOL
from stemscope.selection.inference import predict_forest, recommend
from stemscope.selection.plan import artist_group, partition, prepare
from stemscope.selection.train import evaluate, utility


def test_artist_groups_never_cross_splits():
    names = [f"Artist{i} - Song{j}" for i in range(20) for j in range(2)]
    split = partition(names)
    assert partition(list(reversed(names))) == split
    for i in range(20):
        assert split[f"Artist{i} - Song0"]["split"] == split[f"Artist{i} - Song1"]["split"]
    assert artist_group("A Band (Remix) - Song") == artist_group("A Band - Original")


def test_explored_test_set_cannot_prepare_selector(tmp_path):
    (tmp_path / "provenance.json").write_text(json.dumps({"split": "test"}))
    with pytest.raises(StemScopeError, match="fresh"):
        prepare(tmp_path, tmp_path / "selector")


def test_insufficient_artist_groups_rejected():
    with pytest.raises(StemScopeError, match="15"):
        partition(["Same Artist - One", "Same Artist - Two"])


def test_utility_regret_and_ties():
    values = utility(np.array([[8.0, 6.0], [5.0, 7.0]]), np.array([[5.0, 1.0], [5.0, 1.0]]), 0.5)
    result = evaluate(values, np.array([0, 1]))
    assert result["mean_regret"] == 0
    assert result["oracle_agreement"] == 1
    assert result["mean_utility"] == 6


def test_json_forest_matches_sklearn():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(50, len(FEATURE_NAMES))).astype(np.float32)
    forest = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=1).fit(x, x[:, 0] * 3)
    artifact = {
        "feature_names": FEATURE_NAMES,
        "feature_protocol": PROTOCOL,
        "imputation": [0.0] * len(FEATURE_NAMES),
        "trees": [
            {
                "left": t.tree_.children_left.tolist(),
                "right": t.tree_.children_right.tolist(),
                "feature": t.tree_.feature.tolist(),
                "threshold": t.tree_.threshold.tolist(),
                "value": t.tree_.value[:, 0, 0].tolist(),
            }
            for t in forest.estimators_
        ],
    }
    assert np.allclose([predict_forest(artifact, row) for row in x], forest.predict(x))
    missing = x[0].copy()
    missing[0] = np.nan
    filled = np.nan_to_num(missing)
    assert predict_forest(artifact, missing) == pytest.approx(forest.predict([filled])[0])
    artifact["feature_protocol"] = "wrong"
    with pytest.raises(StemScopeError, match="version"):
        predict_forest(artifact, x[0])


def test_fallback_does_not_claim_learning_or_extract_features(tmp_path):
    path = tmp_path / "selector.json"
    path.write_text(
        json.dumps(
            {
                "models": ["first", "second"],
                "policies": {"Quality": {"use_learned": False, "baseline_index": 0}},
            }
        )
    )
    model, reason = recommend(tmp_path / "unused.wav", path, "Quality")
    assert model == "first"
    assert "fallback" in reason


def test_missing_selector_actionable(tmp_path):
    with pytest.raises(StemScopeError, match="unavailable"):
        recommend(tmp_path / "song.wav", tmp_path / "missing.json", "Quality")


def test_test_labels_cannot_change_fitted_forest_or_validation_policy(tmp_path, monkeypatch):
    import csv
    import hashlib
    import importlib

    from stemscope.benchmarking.runner import BenchmarkTrack
    from stemscope.benchmarking.study import MODELS
    from stemscope.separators.base import STEMS

    module = importlib.import_module("stemscope.selection.train")
    dataset, study, output = (tmp_path / folder for folder in ("data", "study", "selector"))
    dataset.mkdir()
    study.mkdir()
    names = [f"Artist{i:02} - Song" for i in range(40)]
    metadata = {
        "split": "train",
        "planned_tracks": names,
        "tracks": [{"Track Name": name} for name in names],
    }
    (dataset / "provenance.json").write_text(json.dumps(metadata))
    plan_path = prepare(dataset, output)
    plan = json.loads(plan_path.read_text())
    tracks, records = [], []
    for name in names:
        audio = dataset / f"{name}.wav"
        audio.write_bytes(b"fixture")
        track = BenchmarkTrack(name, audio, dict.fromkeys(STEMS, audio))
        tracks.append(track)
        fingerprint = {"path": str(audio), "sha256": hashlib.sha256(audio.read_bytes()).hexdigest()}
        records.append(
            {"name": name, "mixture": fingerprint, "references": dict.fromkeys(STEMS, fingerprint)}
        )
    protocol = {
        "failures": [],
        "models_in_order": list(MODELS),
        "source_provenance": metadata,
        "expected_tracks": names,
        "repeats": 1,
        "threads": 4,
        "worker_results": [
            {"model": model, "manifest": {"tracks": records, "devices": {model: "cpu"}}}
            for model in MODELS
        ],
    }
    (study / "study.json").write_text(json.dumps(protocol))
    rows = [
        {"track": name, "model": model, "stem": stem, "wall_seconds": "1"}
        for name in names
        for model in MODELS
        for stem in STEMS
    ]
    with (study / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(module, "load_prepared_tracks", lambda _: tracks)
    monkeypatch.setattr(
        module, "extract", lambda path: [float(path.name[6:8])] * len(FEATURE_NAMES)
    )
    shift = [0]

    def fake_aggregate(*args):
        return {
            "per_track": [
                {
                    "track": name,
                    "model": model,
                    "si_sdr_median_db": float(i % 7)
                    + (2 if model == MODELS[0] else 0)
                    + (
                        shift[0]
                        if plan["partitions"][name]["split"] == "test" and model == MODELS[0]
                        else 0
                    ),
                }
                for i, name in enumerate(names)
                for model in MODELS
                for _ in STEMS
            ]
        }

    monkeypatch.setattr(module, "aggregate", fake_aggregate)
    first = json.loads(module.train(study, output).read_text())
    shift[0] = 1000
    second = json.loads(module.train(study, output).read_text())
    assert first["trees"] == second["trees"]
    assert first["policies"] == second["policies"]
    assert first["imputation"] == second["imputation"]


def test_cpu_selector_not_silently_applied_to_gpu(tmp_path):
    from stemscope.application import choose_model
    from stemscope.config import Settings

    with pytest.raises(StemScopeError, match="CPU"):
        choose_model(tmp_path / "song.wav", "Auto: Speed", Settings(tmp_path, device="cuda"))
