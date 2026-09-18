import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from stemscope.benchmarking.metrics import si_sdr
from stemscope.errors import StemScopeError
from stemscope.failure_analysis import conditions, reference_features, review_audio, verify_estimate
from stemscope.separators.base import STEMS


def test_reference_energy_and_activity_preserve_stereo_polarity():
    tone = np.ones((100, 2))
    tone[:, 1] = -1
    references = {stem: np.zeros_like(tone) for stem in STEMS}
    references["vocals"] = tone
    reference = reference_features(references, 100)
    assert reference["vocal_energy_share"] == 1
    assert reference["drum_energy_share"] == 0
    assert reference["active_source_mean"] == 1


def test_silent_reference_shares_are_missing():
    result = reference_features({stem: np.zeros((100, 2)) for stem in STEMS}, 100)
    assert result["vocal_energy_share"] is None
    assert result["active_source_mean"] == 0


def test_quartile_groups_keep_ties_and_exclude_missing():
    rows = [
        {
            "track": str(i),
            "genre": "fixture",
            "vocal_energy_share": None if i == 4 else i,
            "drum_energy_share": 1,
            "active_source_mean": 4,
        }
        for i in range(5)
    ]
    thresholds, membership = conditions(rows)
    assert thresholds["quiet vocal share"]["threshold"] == 0.75
    assert "quiet vocal share" in membership["0"]
    assert "quiet vocal share" not in membership["4"]
    assert all(
        "high source activity" in labels and "low source activity" in labels
        for labels in membership.values()
    )


def test_changed_estimate_rejected(tmp_path):
    rng = np.random.default_rng(4)
    ref = rng.normal(size=(1000, 2))
    estimate = ref + rng.normal(size=ref.shape) * 0.2
    a, b = tmp_path / "reference.wav", tmp_path / "estimate.wav"
    sf.write(a, ref, 8000, subtype="FLOAT")
    sf.write(b, estimate, 8000, subtype="FLOAT")
    saved_ref, _ = sf.read(a)
    saved_estimate, _ = sf.read(b)
    score = str(si_sdr(saved_ref, saved_estimate).db)
    verify_estimate(b, a, score)
    sf.write(b, rng.normal(size=ref.shape), 8000, subtype="FLOAT")
    with pytest.raises(StemScopeError, match="reproduce"):
        verify_estimate(b, a, score)


def test_playback_rejects_unknown_case_and_changed_evidence(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"original")
    keys = ["mixture", "reference", "one", "two"]
    case = {
        "id": "0",
        "paths": dict.fromkeys(keys, str(path)),
        "hashes": dict.fromkeys(keys, hashlib.sha256(path.read_bytes()).hexdigest()),
    }
    (tmp_path / "analysis.json").write_text(json.dumps({"models": ["one", "two"], "cases": [case]}))
    with pytest.raises(StemScopeError, match="Choose"):
        review_audio(tmp_path, "../escape")
    paths = review_audio(tmp_path, "0")
    assert len(paths) == 4
    path.write_bytes(b"changed")
    with pytest.raises(StemScopeError, match="changed"):
        review_audio(tmp_path, "0")
