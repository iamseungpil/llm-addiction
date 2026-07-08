"""analyze_w14: §4.3 G-specificity on the G∩M-free matched pool.

Synthetic ladders only — the test pins the ESTIMATOR's contract, not the run's
numbers: (a) +G vs +M uses every shared dose, (b) comparisons against the
3-point −G control are computed on the common grid so curvature is not
confounded with condition, (c) the impute_stop arm neutralises the parse-
selection collider, (d) the verdict/sign-stability flags follow the CIs.
"""
import json

import pytest

from multilayer_causal.src.sec4_stats import (W14_COMMON_DOSES, _restrict_doses,
                                              _w10b_cells, _w14_cells,
                                              analyze_w14, analyze_w14_model)

DOSES_7 = [(-3, "am3"), (-2, "am2"), (-1, "am1"), (0, "a0"),
           (1, "ap1"), (2, "ap2"), (3, "ap3")]
DOSES_3 = [(-3, "am3"), (0, "a0"), (3, "ap3")]


def _write(d, model, cond, doses, slope, n=60, parse=1.0, base=0.1):
    """A linear ladder: bet_ratio = base + slope*alpha. The first
    round((1-parse)*n) rows of each dose are parse failures."""
    for alpha, sfx in doses:
        n_bad = round((1 - parse) * n)
        rows = []
        for i in range(n):
            ok = i >= n_bad
            rows.append({"alpha": float(alpha), "parse_ok": ok,
                         "bet_ratio": (base + slope * alpha) if ok else None,
                         "action": "bet", "parse_fail": not ok})
        (d / f"sec4_w14_{model}_{cond}_{sfx}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.fixture
def gspecific(tmp_path):
    """gemma: +G steeper than +M (G_SPECIFIC). llama: the mirror (M_SPECIFIC),
    so the test also proves the verdict is not hard-coded to one sign."""
    _write(tmp_path, "gemma", "minusG", DOSES_3, 0.030)
    _write(tmp_path, "gemma", "plusG", DOSES_7, 0.050)
    _write(tmp_path, "gemma", "plusM", DOSES_7, 0.020)
    _write(tmp_path, "llama", "minusG", DOSES_3, 0.030)
    _write(tmp_path, "llama", "plusG", DOSES_3, 0.010)
    _write(tmp_path, "llama", "plusM", DOSES_3, 0.040)
    return tmp_path


def test_plusM_ladder_is_read_at_all_seven_doses(gspecific):
    """Regression guard on the cell builder's glob: a *m3/*p3 suffix must not
    silently drop doses (a half-read ladder would fake a slope difference)."""
    cells = _w14_cells(gspecific, "gemma", "plusM")
    assert sorted(cells) == [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    assert all(c["n"] == 60 for c in cells.values())


def test_primary_contrast_uses_shared_doses_not_the_common_grid(gspecific):
    res = analyze_w14_model(gspecific, "gemma")
    assert res["shared_doses"] == [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    # slope(+G) - slope(+M) recovers the planted 0.050 - 0.020
    assert res["primary_plusG_minus_plusM"]["diff"] == pytest.approx(0.030,
                                                                    abs=2e-3)
    assert res["verdict"] == "G_SPECIFIC"
    assert res["sign_survives_collider_correction"]


def test_vs_minusG_is_restricted_to_the_common_grid(gspecific):
    res = analyze_w14_model(gspecific, "gemma")
    assert res["common_doses"] == list(W14_COMMON_DOSES)
    # -G only ever has 3 doses; +G is truncated to the same 3 before comparison
    assert res["vs_minusG_common_grid"]["plusG"]["diff"] == pytest.approx(
        0.050 - 0.030, abs=2e-3)
    assert res["slopes_common_grid"]["minusG"] == pytest.approx(0.030, abs=2e-3)


def test_restrict_doses_drops_only_the_unlisted_doses(gspecific):
    cells = _w14_cells(gspecific, "gemma", "plusG")
    assert sorted(_restrict_doses(cells, W14_COMMON_DOSES)) == [-3.0, 0.0, 3.0]
    assert sorted(_restrict_doses(cells, ())) == []


def test_verdict_follows_the_sign_so_M_specific_is_reachable(gspecific):
    res = analyze_w14_model(gspecific, "llama")
    assert res["verdict"] == "M_SPECIFIC"
    assert res["primary_plusG_minus_plusM"]["diff"] < 0
    both = analyze_w14(gspecific)
    assert both["verdicts"] == {"gemma": "G_SPECIFIC", "llama": "M_SPECIFIC"}
    assert both["models_agree"] is False


def test_no_specificity_when_the_two_twins_share_a_slope(tmp_path):
    _write(tmp_path, "gemma", "minusG", DOSES_3, 0.030)
    _write(tmp_path, "gemma", "plusG", DOSES_7, 0.030)
    _write(tmp_path, "gemma", "plusM", DOSES_7, 0.030)
    res = analyze_w14_model(tmp_path, "gemma")
    assert res["verdict"] == "NO_SPECIFICITY"
    assert not res["primary_plusG_minus_plusM"]["excludes_zero"]


def test_impute_stop_scores_parse_failures_as_stops(tmp_path):
    # 50% parse failures at every dose: parse_ok keeps 30 rows, impute keeps 60
    _write(tmp_path, "gemma", "plusG", DOSES_7, 0.050, n=60, parse=0.5)
    kept = _w14_cells(tmp_path, "gemma", "plusG")
    imputed = _w14_cells(tmp_path, "gemma", "plusG", impute_stop=True)
    assert len(kept[3.0]["values"]["m"]) == 30
    assert len(imputed[3.0]["values"]["m"]) == 60
    assert imputed[3.0]["values"]["m"].count(0.0) == 30
    # the true parse_rate is preserved either way, so the gate still sees it
    assert kept[3.0]["parse_rate"] == pytest.approx(0.5)
    assert imputed[3.0]["parse_rate"] == pytest.approx(0.5)


def test_impute_stop_is_off_for_every_pre_w14_caller(tmp_path):
    """_w10b_cells is shared with the W10 llama analysis — its default must stay
    parse_ok-only, or W10's published numbers would move."""
    _write(tmp_path, "gemma", "plusG", DOSES_7, 0.050, n=60, parse=0.5)
    default = _w10b_cells(tmp_path, "sec4_w14_gemma_plusG_")
    assert len(default[3.0]["values"]["m"]) == 30


def test_parse_degraded_cells_are_disclosed_not_silently_dropped(tmp_path):
    _write(tmp_path, "gemma", "minusG", DOSES_3, 0.030)
    _write(tmp_path, "gemma", "plusG", DOSES_3, 0.050)
    _write(tmp_path, "gemma", "plusM", DOSES_3, 0.020, parse=0.72)
    res = analyze_w14_model(tmp_path, "gemma")
    assert res["parse_degraded_cells"] == ["plusM@+0", "plusM@+3", "plusM@-3"]
    # still above W14_PARSE_GATE (0.5), so the cells are USED, only flagged
    assert res["primary_plusG_minus_plusM"]["diff"] == pytest.approx(0.030,
                                                                     abs=2e-3)


def test_missing_ladder_raises_rather_than_returning_a_partial_verdict(tmp_path):
    _write(tmp_path, "gemma", "minusG", DOSES_3, 0.030)
    with pytest.raises(FileNotFoundError, match="w14"):
        analyze_w14_model(tmp_path, "gemma")
