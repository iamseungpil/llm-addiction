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


def test_e2_e3a_registry():
    arms = load_arms()
    e2 = [a for a in arms.values() if a["phase"] == "e2"]
    e3a = [a for a in arms.values() if a["phase"] == "e3a"]
    assert len(e2) == 35                                   # 5 pca-r + 30 random ctrl
    assert len(e3a) == 19                                  # 7 alphas + 10 random dirs
    for a in e2 + e3a:
        assert a["layers"] == list(range(18, 24))          # S* fixed from E1
    assert sorted(a["alpha"] for a in e3a if a.get("direction") != "random") == \
        [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    rnd = [a for a in e2 if a.get("basis") == "random"]
    assert len(rnd) == 30 and len({a["basis_seed"] for a in rnd}) == 30


def test_ids_unique_and_layers_valid():
    arms = load_arms()
    for a in arms.values():
        assert a["n"] == (200 if a["phase"] == "e1c" else 50)
        for l in a.get("layers", []):
            assert 0 <= l <= 41


def test_e1c_confirmatory_held_out():
    arms = load_arms()
    e1c = [a for a in arms.values() if a["phase"] == "e1c"]
    assert len(e1c) == 3
    for a in e1c:
        # held-out: distinct seeds AND state slice disjoint from discovery (0..99)
        assert a["seed_base"] == 1000042 and a["state_offset"] == 100
