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
        if a["phase"] in ("e1c", "w2"):       # confirmatory: 200 (50 controls)
            assert a["n"] in (50, 200)
        else:
            assert a["n"] == 50
        for l in a.get("layers", []):
            assert 0 <= l <= 41


def test_e1c_confirmatory_held_out():
    arms = load_arms()
    e1c = [a for a in arms.values() if a["phase"] == "e1c"]
    assert len(e1c) == 3
    for a in e1c:
        # held-out: distinct seeds AND state slice disjoint from discovery (0..99)
        assert a["seed_base"] == 1000042 and a["state_offset"] == 100


def test_w1_registry():
    arms = load_arms()
    by_phase = {}
    for a in arms.values():
        by_phase.setdefault(a["phase"], []).append(a)
    assert len(by_phase["w1ro"]) == 4
    assert len(by_phase["w1lc"]) == 7
    assert len(by_phase["w1s"]) == 4
    assert len(by_phase["w1ic"]) == 6
    assert len(by_phase["w1b"]) == 2
    assert len(by_phase["w1e"]) == 7
    # layers_list expansion: non-contiguous set stored verbatim into 'layers'
    assert arms["w1b_patch5"]["layers"] == [8, 12, 22, 25, 30]
    assert "layers_list" not in arms["w1b_patch5"]
    assert arms["w1b_steer5"]["layers"] == [8, 12, 22, 25, 30]
    for a in by_phase["w1lc"]:
        assert a.get("probe") is True and a["task"] == "sm"
    for a in by_phase["w1ic"]:
        assert a["task"] == "ic" and not a.get("probe")
        assert a["mode"] in {"anchor_minus", "steer"}
    for a in by_phase["w1ro"] + by_phase["w1s"]:
        assert a["mode"] == "steer" and a["layers"] == list(range(18, 24))
    assert sorted(a["alpha"] for a in by_phase["w1ro"]) == [-2.0, 0.0, 2.0, 4.0]
    rnd = [a for a in arms.values()
           if a["phase"].startswith("w1") and a.get("direction") == "random"]
    assert {a["id"] for a in rnd} == {"w1lc_rnd", "w1ic_rnd_s0", "w1ic_rnd_s1"}
    assert len({a["dir_seed"] for a in rnd}) == 3


def test_w2_confirmatory_held_out():
    arms = load_arms()
    w2 = [a for a in arms.values() if a["phase"] == "w2"]
    assert len(w2) == 8
    for a in w2:
        # third seed set AND state slice disjoint from discovery (0..99)
        # and e1c confirmatory (100..299)
        assert a["seed_base"] == 2000042 and a["state_offset"] == 300
        # any axis used at offset 300 must be the re-estimated W2 build
        if "directions_npz" in a:
            assert "/assets/w2/" in a["directions_npz"], a["id"]
    assert arms["w2e_1617"]["layers"] == [16, 17]
    assert arms["w2e_1821"]["layers"] == [18, 19, 20, 21]
    assert arms["w2ro5_p20"]["layers"] == [8, 12, 22, 25, 30]
    # confirmatory mains are n=200; controls (random / ro5 closing arm) n=50
    assert {a["id"]: a["n"] for a in w2 if a["n"] == 50} == \
        {"w2rnd_p40": 50, "w2ro5_p20": 50}
    probes = {a["id"] for a in w2 if a.get("probe")}
    assert probes == {"w2_anchor_minus", "w2_anchor_plus",
                      "w2lc_iba_p40", "w2lc_ilc_p20", "w2rnd_p40"}
