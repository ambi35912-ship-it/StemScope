import hashlib
import json
import runpy
from pathlib import Path

import pytest

from stemscope.benchmarking.datasets import load_public_pilot
from stemscope.benchmarking.demo import create_demo
from stemscope.errors import StemScopeError


def prepared(tmp_path):
    track = create_demo(tmp_path)
    root = track.mixture.parent.parent
    name = track.mixture.parent.name
    files = dict(track.references, mixture=track.mixture)
    record = {
        "Track Name": name,
        "License": "CC BY-NC-SA 3.0",
        "files": {
            stem: {
                "member": f"test/{name}/{stem}.wav",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for stem, path in files.items()
        },
    }
    (root / "provenance.json").write_text(
        json.dumps({"dataset": "test fixture", "record": "fixture", "tracks": [record]})
    )
    return root, files


def test_prepared_data_preserves_reference_identity(tmp_path):
    root, files = prepared(tmp_path)
    tracks = load_public_pilot(root)
    assert len(tracks) == 1
    assert tracks[0].mixture == files["mixture"]
    assert tracks[0].references == {stem: path for stem, path in files.items() if stem != "mixture"}


def test_changed_reference_is_rejected(tmp_path):
    root, files = prepared(tmp_path)
    files["vocals"].write_bytes(b"replaced")
    with pytest.raises(StemScopeError, match="changed"):
        load_public_pilot(root)


def test_missing_dataset_explains_setup(tmp_path):
    with pytest.raises(StemScopeError, match="missing or invalid"):
        load_public_pilot(tmp_path)


def test_selection_excludes_training_and_restricted_tracks():
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_public_pilot.py"
    select = runpy.run_path(str(script))["select_tracks"]
    metadata = {
        name: {"License": license}
        for name, license in [
            ("A", "Restricted"),
            ("B", "CC BY-NC-SA 3.0"),
            ("C", "CC BY-NC-SA"),
            ("D", "CC BY-NC-SA"),
        ]
    }
    names = [
        "test/C/mixture.wav",
        "train/D/mixture.wav",
        "test/A/mixture.wav",
        "test/B/mixture.wav",
    ]
    assert select(names, metadata, 5) == ["B", "C"]
    assert select(names, metadata, 1) == ["B"]


def test_incomplete_research_selection_rejected(tmp_path):
    from stemscope.benchmarking.datasets import load_prepared_tracks

    root, _ = prepared(tmp_path)
    path = root / "provenance.json"
    metadata = json.loads(path.read_text())
    metadata["planned_tracks"] = [metadata["tracks"][0]["Track Name"], "missing track"]
    path.write_text(json.dumps(metadata))
    with pytest.raises(StemScopeError, match="incomplete"):
        load_prepared_tracks(root)


def test_train_references_require_explicit_split_and_cannot_be_pilot(tmp_path):
    from stemscope.benchmarking.datasets import load_prepared_tracks

    root, _ = prepared(tmp_path)
    path = root / "provenance.json"
    metadata = json.loads(path.read_text())
    metadata["split"] = "train"
    for info in metadata["tracks"][0]["files"].values():
        info["member"] = info["member"].replace("test/", "train/", 1)
    path.write_text(json.dumps(metadata))
    assert len(load_prepared_tracks(root)) == 1
    with pytest.raises(StemScopeError, match="split"):
        load_public_pilot(root)
