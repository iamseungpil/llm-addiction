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
        elif a["phase"] == "w3L":             # power-sized gate anchors (878)
            assert a["n"] == (878 if "anchor" in a["id"] else 50)
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


def test_w3_registry():
    arms = load_arms()
    w3 = {ph: [a for a in arms.values() if a["phase"] == ph]
          for ph in ("w3a", "w3bk", "w3pos", "w3rnd", "w3m", "w3L")}
    assert {ph: len(v) for ph, v in w3.items()} == \
        {"w3a": 6, "w3bk": 8, "w3pos": 1, "w3rnd": 8, "w3m": 3, "w3L": 10}
    # w3a pairs the LITERAL stored Table-1 direction with its eval-pool-
    # excluded refit twin at the same alphas (round-1 leakage fix: PR-1
    # positive requires BOTH directions to move behavior).
    saerd = {a["id"]: a for a in w3["w3a"]}
    for suffix in ("m20", "p20", "p40"):
        lit, twin = saerd[f"w3a_saerd_{suffix}"], saerd[f"w3a_saerdx_{suffix}"]
        assert lit["directions_npz"].endswith("directions_saerd.npz")
        assert twin["directions_npz"].endswith("directions_saerd_excl.npz")
        assert lit["alpha"] == twin["alpha"] and lit["layers"] == twin["layers"]
    # Gemma arms ride the W2 anchor protocol (third seeds, offset-300 states);
    # IC arms keep the seeds but their own corpus at offset 0.
    for a in w3["w3a"] + w3["w3bk"] + w3["w3pos"] + w3["w3rnd"] + w3["w3m"]:
        assert a["model"] == "gemma" and a["n"] == 50, a["id"]
        assert a["seed_base"] == 2000042, a["id"]
        assert a["state_offset"] == (0 if a.get("task") == "ic" else 300), a["id"]
    assert {a["id"] for a in w3["w3bk"] if a.get("task") == "ic"} == \
        {"w3bk_ic_p20", "w3bk_ic_p40"}
    # New direction assets live under assets/w3/; w3pos + the random-direction
    # nulls reuse the already-built W2 I_BA axis (scales only, for w3rnd).
    for a in w3["w3a"] + w3["w3bk"]:
        assert "/assets/w3/" in a["directions_npz"], a["id"]
    for a in w3["w3pos"] + w3["w3rnd"]:
        assert "/assets/w2/" in a["directions_npz"], a["id"]
    assert arms["w3bk_smres_p40"]["directions_npz"].endswith(
        "directions_bk_sm_balres.npz")
    # round-3 discriminative balance controls (balres is a partial control):
    # balorth = bk_sm axis orthogonal to the target balance-slope direction,
    # balslope = the pure balance-slope direction (confound-positive), both
    # at the bk_sm alpha=+4 protocol with manipulation-check logging.
    assert arms["w3bk_smorth_p40"]["directions_npz"].endswith(
        "directions_bk_sm_balorth.npz")
    assert arms["w3bk_smbal_p40"]["directions_npz"].endswith(
        "directions_bk_sm_balslope.npz")
    for aid in ("w3bk_smorth_p40", "w3bk_smbal_p40"):
        assert arms[aid]["alpha"] == 4.0 and arms[aid]["layers"] == \
            arms["w3bk_sm_p40"]["layers"], aid
    assert {a["id"] for a in w3["w3bk"] if a.get("log_vectors")} == \
        {"w3bk_sm_p40", "w3bk_smres_p40", "w3bk_smorth_p40", "w3bk_smbal_p40"}
    w3all = [a for v in w3.values() for a in v]
    assert {a["id"] for a in w3all if a.get("probe")} == \
        {"w3pos_iba_p40"} | {f"w3rnd_s{i}" for i in range(8)}
    assert sorted(a["dir_seed"] for a in w3["w3rnd"]) == \
        [2026067000 + i for i in range(8)]
    for a in w3["w3rnd"]:
        assert a["direction"] == "random" and a["alpha"] == 4.0
        assert a["layers"] == list(range(16, 22))
    # twin_component appears on exactly the w3m arms (patch mode, value M)
    twins = [a for a in arms.values() if "twin_component" in a]
    assert {a["id"] for a in twins} == {"w3m_0813", "w3m_1621", "w3m_2429"}
    for a in twins:
        assert a["twin_component"] == "M" and a["mode"] == "patch"
    assert arms["w3m_0813"]["layers"] == list(range(8, 14))
    assert arms["w3m_1621"]["layers"] == list(range(16, 22))
    assert arms["w3m_2429"]["layers"] == list(range(24, 30))
    # LLaMA wave: own anchors, fresh corpus (offset 0, seed_base 42),
    # layers within the 32-layer budget, 8 width-4 windows tiling 0..31.
    for a in w3["w3L"]:
        assert a["model"] == "llama"
        # Gate anchors are power-sized from configs/llama_gap_estimate.json
        # (d=0.134 → 878/arm for 0.8 power); sweep windows stay n=50 discovery.
        assert a["n"] == (878 if "anchor" in a["id"] else 50)
        assert a["seed_base"] == 42 and a["state_offset"] == 0
        assert all(0 <= l <= 31 for l in a.get("layers", [])), a["id"]
    tiled = sorted(l for a in w3["w3L"] if a["mode"] == "patch"
                   for l in a["layers"])
    assert tiled == list(range(32))
    assert {a["mode"] for a in w3["w3L"] if "anchor" in a["id"]} == \
        {"anchor_minus", "anchor_plus"}


def test_w3_validation_rejects_bad_fields(tmp_path):
    import pytest
    import yaml

    def _load(arm):
        p = tmp_path / "arms.yaml"
        p.write_text(yaml.safe_dump(
            {"defaults": {"model": "gemma", "task": "sm", "n": 50},
             "arms": [arm]}))
        return load_arms(p)

    _load({"id": "ok", "phase": "p", "mode": "patch", "model": "llama",
           "layers": [28, 31]})                       # llama top window is legal
    with pytest.raises(AssertionError):               # llama layers must be <= 31
        _load({"id": "x", "phase": "p", "mode": "patch", "model": "llama",
               "layers": [28, 35]})
    with pytest.raises(AssertionError):               # unknown model
        _load({"id": "x", "phase": "p", "mode": "patch", "model": "qwen",
               "layers": [0, 3]})
    with pytest.raises(AssertionError):               # twin_component value
        _load({"id": "x", "phase": "p", "mode": "patch", "layers": [8, 13],
               "twin_component": "H"})
    with pytest.raises(AssertionError):               # twin_component patch-only
        _load({"id": "x", "phase": "p", "mode": "steer", "layers": [8, 13],
               "alpha": 2.0, "twin_component": "M"})
    with pytest.raises(AssertionError):               # log_vectors must be bool
        _load({"id": "x", "phase": "p", "mode": "patch", "layers": [8, 13],
               "log_vectors": "yes"})


def test_w3L_gate_power_asset_matches_preregistration():
    """configs/llama_gap_estimate.json is the pre-registered (tracked copy of the out/ scratch original) power basis for w3L
    (RUN_PLAN_W3 PR-2): anchor n must equal the asset's 80%-power n, and the
    round-2 claimability rule requires simulation-verified MDEs at the
    discovery (n=50) and W4 confirmatory (n=200) stages, both materially
    larger than the +-G gap, with adequate power for the Gemma-magnitude
    target effect at discovery n."""
    import json
    from pathlib import Path

    asset = Path(__file__).resolve().parents[1] / "configs" / "llama_gap_estimate.json"
    est = json.loads(asset.read_text())
    arms = load_arms()
    anchors = [a for a in arms.values()
               if a["phase"] == "w3L" and "anchor" in a["id"]]
    assert len(anchors) == 2
    for a in anchors:
        assert a["n"] == est["n_per_arm_for_80pct"]
    assert est["welch_power"][str(est["n_per_arm_for_80pct"])] >= 0.8
    mde = est["mde"]
    for stage in ("n50_discovery", "n200_confirmatory"):
        cell = mde[stage]
        assert cell["sim_power"] >= 0.8
        assert cell["shift"] > est["gap"]              # claimable >> observed gap
        assert cell["gap_multiple"] > 1.0
    # discovery n=50 is sized for the Gemma-magnitude target effect, not the gap
    assert est["gemma_magnitude_power"]["n50"] >= 0.8
