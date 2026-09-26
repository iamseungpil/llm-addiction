import json

import numpy as np

from multilayer_causal.src.checkpoint import ArmCheckpoint, VectorStore


def test_record_resume_roundtrip(tmp_path):
    ck = ArmCheckpoint("e1", "full", tmp_path, sync_every=2, hf_enabled=False)
    assert ck.done_seeds() == set()
    ck.record({"seed": 42, "x": 1})
    ck.record({"seed": 1039, "x": 2})
    ck2 = ArmCheckpoint("e1", "full", tmp_path, hf_enabled=False)
    assert ck2.done_seeds() == {42, 1039}
    lines = [json.loads(l) for l in open(tmp_path / "full.jsonl")]
    assert [l["x"] for l in lines] == [1, 2]


def test_sync_called_every_n(tmp_path, monkeypatch):
    calls = []
    ck = ArmCheckpoint("e1", "full", tmp_path, sync_every=2, hf_enabled=True)
    monkeypatch.setattr(ck, "_upload", lambda: calls.append(1))
    for s in range(5):
        ck.record({"seed": s})
    assert len(calls) == 2          # after 2 and 4


def test_vector_store_roundtrip(tmp_path):
    vs = VectorStore(tmp_path / "vec.npz", n_layers=3, d_model=4)
    vs.append(seed=1, minus=np.ones((3, 4)), plus=2 * np.ones((3, 4)))
    vs.save()
    vs2 = VectorStore(tmp_path / "vec.npz", n_layers=3, d_model=4)
    assert vs2.seeds == [1]
    vs2.append(seed=2, minus=np.zeros((3, 4)), plus=np.zeros((3, 4)))
    vs2.save()
    vs3 = VectorStore(tmp_path / "vec.npz", n_layers=3, d_model=4)
    assert vs3.seeds == [1, 2] and vs3.minus.shape == (2, 3, 4)
