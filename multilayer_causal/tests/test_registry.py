from multilayer_causal.src.registry import load_arms


def test_e1_registry_complete():
    arms = load_arms()
    e1 = [a for a in arms.values() if a["phase"] == "e1"]
    assert len(e1) == 19                                   # 17 interventions + 2 anchors
    full = arms["e1_full"]
    assert full["mode"] == "patch" and full["layers"] == list(range(42))
    assert full.get("log_vectors") is True
    assert arms["e1_anchor_minus"]["mode"] == "anchor_minus"
    sliding = [a for a in e1 if a["id"].startswith("e1_win")]
    assert len(sliding) == 7
    covered = sorted(l for a in sliding for l in a["layers"])
    assert covered == list(range(42))                      # tiling, no gaps/overlap


def test_ids_unique_and_layers_valid():
    arms = load_arms()
    for a in arms.values():
        assert a["n"] == 50
        for l in a.get("layers", []):
            assert 0 <= l <= 41
