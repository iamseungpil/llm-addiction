"""Submit-time guard for direction assets (RUN_PLAN_W3.md 구현 원칙 + audit r1).

The W3 audit caught registered arms pointing at npz files that did not exist
yet (assets/w3/ empty) — a FileNotFoundError that would only surface on the
GPU box. npz assets are gitignored and ship via the push_code_to_hf.py
tarball built from THIS checkout, so pytest here is the submit-time check:
every registered asset path must exist, W3 assets must carry the standard
schema (unit-row directions + 3%-norm scales) with gate results embedded,
and the runner's gate auto-exclusion (plan line: 재현-게이트 실패 시 자동
제외) must actually fire for gate-failed assets.
"""
from pathlib import Path

import numpy as np
import pytest

from multilayer_causal.src.registry import load_arms
from multilayer_causal.src.runner import _load_steer_assets

REPO_ROOT = Path(__file__).resolve().parents[2]
W3_ASSETS = REPO_ROOT / "multilayer_causal" / "assets" / "w3"
N_LAYERS, D_MODEL = 42, 3584                       # gemma-2-9b decoder stack


def _w3_npz_paths():
    return sorted({REPO_ROOT / a["directions_npz"]
                   for a in load_arms().values()
                   if "/assets/w3/" in a.get("directions_npz", "")})


def test_all_registered_npz_assets_exist():
    missing = []
    for a in load_arms().values():
        for key in ("directions_npz", "basis_npz"):
            if key in a and not (REPO_ROOT / a[key]).is_file():
                missing.append(f"{a['id']}: {a[key]}")
    assert not missing, f"registered arms reference absent npz: {missing}"


BK_CONTROL_NPZ = {"directions_bk_sm_balorth.npz",
                  "directions_bk_sm_balslope.npz"}


def test_w3_assets_cover_all_seven_axes():
    # saerd_excl = eval-pool-excluded refit twin of the literal Table-1
    # direction (round-1 leakage fix, RUN_PLAN_W3.md PR-1 both-directions rule);
    # bk_sm_balorth/balslope = round-3 discriminative balance controls (balres
    # alone is a partial control, cos(plain, balres) ~ 0.95).
    names = {p.name for p in _w3_npz_paths()}
    assert names == {"directions_saerd.npz", "directions_saerd_excl.npz",
                     "directions_bk_sm.npz", "directions_bk_sm_balres.npz",
                     "directions_bk_ic.npz"} | BK_CONTROL_NPZ
    on_disk = {p.name for p in W3_ASSETS.glob("*.npz")}
    assert names <= on_disk


def test_w3_assets_schema_and_gate_fields():
    for p in _w3_npz_paths():
        z = np.load(p, allow_pickle=True)
        d, s = z["directions"], z["scales"]
        assert d.shape == (N_LAYERS, D_MODEL) and d.dtype == np.float32, p.name
        assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-4), p.name
        assert s.shape == (N_LAYERS,) and np.all(s > 0) and np.all(np.isfinite(s)), p.name
        assert int(z["schema_version"]) == 1, p.name
        assert "gate_passed" in z.files, p.name
        if p.name.startswith("directions_saerd"):
            assert np.isfinite(float(z["r2_achieved"])), p.name
            assert z["fold_r2"].shape == (5,), p.name
            # round-1 leakage fix: both saerd npz carry the exclusion-refit
            # gate alongside the disclosed in-corpus (leaky) reproduction R^2
            # and the eval-window overlap counts.
            assert np.isfinite(float(z["r2_in_corpus"])), p.name
            assert z["fold_r2_in_corpus"].shape == (5,), p.name
            assert int(z["eval_exclude_n"]) == 500, p.name
            assert int(z["eval_overlap_rows"]) >= int(z["eval_overlap_games"]), p.name
            assert float(z["decoder_check_cos"]) > 0.999, p.name
        elif p.name in BK_CONTROL_NPZ:
            # round-3 discriminative balance controls: construction-validity
            # gates + disclosure (not gate) AUC/balance numbers.
            assert abs(float(z["cos_to_plain_axis"])) <= 1.0, p.name
            assert np.isfinite(float(z["plain_cos_balance_slope_target"])), p.name
            assert 0.0 < float(z["auc_game_p_disclosure"]) <= 1.0, p.name
            for k in ("cos_balance_slope_target",
                      "balance_align_pearson_target",
                      "balance_align_spearman_target",
                      "auc_game_disclosure"):
                assert np.isfinite(float(z[k])), (p.name, k)
            if p.name.endswith("balorth.npz"):
                # orthogonal to the target balance-slope BY CONSTRUCTION and
                # balance-decorrelated on target rows (the linear residual of
                # the regression is uncorrelated with balance along any
                # direction orthogonal to the slope).
                assert abs(float(z["cos_balance_slope_target"])) < 1e-8, p.name
                assert abs(float(z["balance_align_pearson_target"])) < 1e-6, p.name
                assert float(z["cos_to_plain_axis"]) > float(z["orth_residual_min"]), p.name
            else:
                # sign-aligned with the plain axis's balance component
                assert float(z["cos_to_plain_axis"]) > float(z["bal_component_min"]), p.name
                assert np.isclose(abs(float(z["cos_balance_slope_target"])),
                                  1.0, atol=1e-9), p.name
        else:
            # round-1 gate fix: bk assets carry the pre-registered construct
            # gates (relevance game-perm p, existence pair-cos bootstrap,
            # stability) + stop-ward orientation calibration — the Table-2
            # "reproduction" AUC target is gone (different estimand).
            assert np.isfinite(float(z["auc_game_raw"])), p.name
            assert 0.0 < float(z["auc_game_p"]) <= 1.0, p.name
            assert float(z["orientation_sign"]) in (-1.0, 1.0), p.name
            assert float(z["auc_game_oriented"]) >= 0.5, p.name
            assert z["bootstrap_cosines"].shape == (200,), p.name
            assert z["bootstrap_pair_cos"].shape == (200,), p.name
            for g in ("relevance_gate_passed", "existence_gate_passed",
                      "stability_gate_passed"):
                assert g in z.files, p.name
            assert bool(z["gate_passed"]) == (
                bool(z["relevance_gate_passed"])
                and bool(z["existence_gate_passed"])
                and bool(z["stability_gate_passed"])), p.name
            # round-2 balance disclosure: bk assets must QUANTIFY the partial
            # balance control (cos of plain vs balres axes + projection-balance
            # alignment on target/sources + balance-slope cosines) — the old
            # auc_on_balres_states was a BK row-AUC, not a balance readout.
            assert abs(float(z["cos_plain_vs_balres"])) <= 1.0, p.name
            for k in ("balance_align_pearson_target",
                      "balance_align_spearman_target",
                      "cos_balance_slope_target",
                      "counterpart_balance_align_pearson_target",
                      "counterpart_balance_align_spearman_target",
                      "counterpart_cos_balance_slope_target"):
                assert np.isfinite(float(z[k])), (p.name, k)
            n_src = len(z["source_tasks"])
            for k in ("balance_align_pearson_sources",
                      "balance_align_spearman_sources",
                      "cos_balance_slope_sources"):
                assert z[k].shape == (n_src,) and np.all(np.isfinite(z[k])), \
                    (p.name, k)


def test_bk_sm_balres_pair_discloses_partial_control():
    """The plain and balres bk_sm assets must agree on cos_plain_vs_balres,
    and the recorded numbers must reflect the round-2 finding: the two axes
    are nearly identical (cos >> 0), so the balres arm is a PARTIAL control
    whose agreement with the plain arm cannot exclude a balance account."""
    plain = np.load(W3_ASSETS / "directions_bk_sm.npz", allow_pickle=True)
    balres = np.load(W3_ASSETS / "directions_bk_sm_balres.npz", allow_pickle=True)
    c_p = float(plain["cos_plain_vs_balres"])
    c_b = float(balres["cos_plain_vs_balres"])
    assert np.isclose(c_p, c_b, atol=1e-6)
    # each file's own-axis target alignment is the counterpart number of the
    # other file (before/after recorded in BOTH, per the audit fix hint)
    assert np.isclose(float(plain["balance_align_pearson_target"]),
                      float(balres["counterpart_balance_align_pearson_target"]),
                      atol=1e-9)
    assert np.isclose(float(balres["balance_align_pearson_target"]),
                      float(plain["counterpart_balance_align_pearson_target"]),
                      atol=1e-9)


def test_bk_sm_balance_controls_are_discriminative():
    """Round-3 fix: the balance-confound account of a positive w3bk SM
    result must be answerable. balorth = plain bk_sm axis made EXACTLY
    orthogonal to the target balance-slope direction (target-side, unlike
    source-side residualisation whose axis kept cos ~ -0.2 with it);
    balslope = the pure balance-slope direction, sign-aligned with the
    plain axis's balance component (confound-positive control). Both ride
    the plain asset's scales (same Delta-norm)."""
    plain = np.load(W3_ASSETS / "directions_bk_sm.npz", allow_pickle=True)
    orth = np.load(W3_ASSETS / "directions_bk_sm_balorth.npz", allow_pickle=True)
    slope = np.load(W3_ASSETS / "directions_bk_sm_balslope.npz", allow_pickle=True)
    d_p, d_o, d_s = (z["directions"][22].astype(np.float64)
                     for z in (plain, orth, slope))
    # exact orthogonalisation: cos(balorth, balslope) = 0, unlike balres
    assert abs(float(d_o @ d_s)) < 1e-4
    # recorded decomposition is consistent across the three saved assets
    assert np.isclose(float(d_o @ d_p), float(orth["cos_to_plain_axis"]), atol=1e-4)
    assert np.isclose(float(d_s @ d_p), float(slope["cos_to_plain_axis"]), atol=1e-4)
    assert float(slope["cos_to_plain_axis"]) > 0          # sign-aligned
    assert np.isclose(float(slope["cos_to_plain_axis"]),
                      abs(float(plain["cos_balance_slope_target"])), atol=1e-3)
    assert np.isclose(float(orth["cos_to_plain_axis"]),
                      np.sqrt(1.0 - float(plain["cos_balance_slope_target"]) ** 2),
                      atol=1e-3)
    # same Delta-norm as the plain bk_sm arm (identical per-layer scales)
    np.testing.assert_allclose(orth["scales"], plain["scales"], rtol=1e-5)
    np.testing.assert_allclose(slope["scales"], plain["scales"], rtol=1e-5)


def test_runner_gate_auto_exclusion_matches_npz():
    """gate_passed=False npz → arm fails fast at asset load (auto-exclusion);
    gate_passed=True npz → loads unit directions + per-layer scales."""
    for a in load_arms().values():
        if "/assets/w3/" not in a.get("directions_npz", ""):
            continue
        arm = dict(a, directions_npz=str(REPO_ROOT / a["directions_npz"]))
        passed = bool(np.load(arm["directions_npz"])["gate_passed"])
        if passed:
            dirs, scales = _load_steer_assets(arm)
            assert set(dirs) == set(scales) == set(arm["layers"])
            for li in arm["layers"]:
                assert abs(float(dirs[li].norm()) - 1.0) < 1e-4
                assert scales[li] > 0
        else:
            with pytest.raises(AssertionError, match="reproduction gate"):
                _load_steer_assets(arm)
