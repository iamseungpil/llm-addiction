"""Smoke tests for the M1 module.

Live model/API tests skip when GPU/API key is missing. Synthetic tests run
unconditionally and exercise the simulator + analysis pipeline end-to-end.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import portfolio_simulator  # noqa: E402
import parse_allocation  # noqa: E402
import sanity_checks  # noqa: E402
import prompts  # noqa: E402
import analyze_m1  # noqa: E402


def _cuda_available() -> bool:
    torch = pytest.importorskip("torch", reason="torch required for GPU smoke")
    return bool(torch.cuda.is_available())


# ---------------------------------------------------------------------------
# 1. Portfolio simulator deterministic + drawdown threshold
# ---------------------------------------------------------------------------

def _all_in_leveraged_assets():
    """6-asset menu matching the config."""
    return portfolio_simulator.assets_from_config([
        {"name": "cash",                  "mean_annual": 0.00, "vol_annual": 0.00, "skew": 0.0, "kind": "safe"},
        {"name": "bonds",                 "mean_annual": 0.04, "vol_annual": 0.05, "skew": 0.0, "kind": "safe"},
        {"name": "broad_index",           "mean_annual": 0.08, "vol_annual": 0.18, "skew": 0.0, "kind": "index"},
        {"name": "leveraged_etf_3x",      "mean_annual": 0.24, "vol_annual": 0.60, "skew": -0.5, "kind": "leveraged"},
        {"name": "single_volatile_stock", "mean_annual": 0.12, "vol_annual": 0.50, "skew": 0.0, "kind": "volatile"},
        {"name": "otm_call_or_crypto",    "mean_annual": 0.30, "vol_annual": 1.20, "skew": 1.5, "kind": "lottery"},
    ])


def test_portfolio_simulator_60pct_drawdown_threshold():
    """A 50/50 leveraged-ETF + OTM-call mix over 30 rounds must breach 50%
    drawdown on a meaningful fraction of seeds — confirms the drawdown
    threshold is reachable AND surfaces parameter regressions.

    C5 fix: the previous 100%-leveraged-only sweep over 20 seeds had a marginal
    breach rate; a small parameter perturbation could silently drop it to 4%
    without failing the test. The 50/50 leveraged + OTM mix has a substantially
    higher base breach rate (the OTM lottery asset has 1.2 annualised vol with
    -ve skew interaction during loss runs), so a regression in the return
    parameters surfaces here as a near-zero breach count rather than a flake.

    Threshold: ≥ 5/30 breaches (≈ 17%). Calibrated empirically; current params
    produce ~50% breach on this allocation, so a real regression knocking the
    leveraged or OTM vol down by ~30% would drop us under the gate.
    """
    assets = _all_in_leveraged_assets()
    n_seeds = 30
    n_breached = 0
    for seed in range(n_seeds):
        game = portfolio_simulator.PortfolioGame(assets=assets, n_rounds=30)
        returns = portfolio_simulator.simulate_returns(assets, 30, 1.0 / 12.0, seed)
        # 50/50 leveraged_etf_3x + otm_call_or_crypto mix — temptation-bucket only.
        mix = {"cash": 0, "bonds": 0, "broad_index": 0, "leveraged_etf_3x": 0.5,
               "single_volatile_stock": 0, "otm_call_or_crypto": 0.5}
        for t in range(30):
            if game.is_finished:
                break
            game.play_round(mix, returns[t])
        if portfolio_simulator.primary_risk_event(game):
            n_breached += 1
    assert n_breached >= 5, (
        f"50/50 leveraged+OTM mix only breached 50% drawdown on {n_breached}/{n_seeds} "
        "seeds — return parameters likely regressed (was ~50% pre-regression). "
        "This is the gate that flags vol/skew config drift."
    )


def test_portfolio_simulator_deterministic_with_seed():
    """Same seed -> identical per-round returns."""
    assets = _all_in_leveraged_assets()
    r1 = portfolio_simulator.simulate_returns(assets, 30, 1.0 / 12.0, 7)
    r2 = portfolio_simulator.simulate_returns(assets, 30, 1.0 / 12.0, 7)
    assert r1 == r2


def test_portfolio_simulator_normalises_allocation():
    assets = _all_in_leveraged_assets()
    game = portfolio_simulator.PortfolioGame(assets=assets, n_rounds=5)
    returns = portfolio_simulator.simulate_returns(assets, 5, 1.0 / 12.0, 0)
    raw = {"cash": 50, "bonds": 50, "broad_index": 0, "leveraged_etf_3x": 0,
           "single_volatile_stock": 0, "otm_call_or_crypto": 0}
    rd = game.play_round(raw, returns[0])
    s = sum(rd["allocation"].values())
    assert abs(s - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 2. Parser
# ---------------------------------------------------------------------------

def test_parse_allocation_basic_formats():
    asset_names = ["cash", "bonds", "broad_index", "leveraged_etf_3x",
                   "single_volatile_stock", "otm_call_or_crypto"]
    response = (
        "I'll diversify but lean conservative.\n"
        "Final Allocation: Cash 30, Bonds 10, Index 20, Leveraged 25, Stock 10, OTM 5"
    )
    out, reason = parse_allocation.parse_allocation(response, asset_names)
    assert out is not None, f"parse failed: {reason}"
    assert abs(out["cash"] - 0.30) < 1e-6
    assert abs(out["bonds"] - 0.10) < 1e-6
    assert abs(out["broad_index"] - 0.20) < 1e-6
    assert abs(out["leveraged_etf_3x"] - 0.25) < 1e-6
    assert abs(out["single_volatile_stock"] - 0.10) < 1e-6
    assert abs(out["otm_call_or_crypto"] - 0.05) < 1e-6


def test_parse_allocation_percent_format():
    asset_names = ["cash", "bonds", "broad_index", "leveraged_etf_3x",
                   "single_volatile_stock", "otm_call_or_crypto"]
    response = (
        "Final Allocation: Cash 50%, Bonds 20%, Index 20%, Leveraged 5%, Stock 5%, OTM 0%"
    )
    out, _ = parse_allocation.parse_allocation(response, asset_names)
    assert out is not None
    assert abs(sum(out.values()) - 1.0) < 1e-6


def test_parse_allocation_unparseable():
    asset_names = ["cash", "bonds", "broad_index", "leveraged_etf_3x",
                   "single_volatile_stock", "otm_call_or_crypto"]
    out, _ = parse_allocation.parse_allocation("I have no idea, let me think...", asset_names)
    assert out is None


# ---------------------------------------------------------------------------
# 3. Synthetic analyze pipeline
# ---------------------------------------------------------------------------

def _synthetic_portfolio_payload(model: str, condition: str, n_games: int, base_p: float, seed: int) -> dict:
    import random as _r
    _r.seed(seed)
    results = []
    for i in range(n_games):
        is_risk = int(_r.random() < base_p)
        results.append({
            "game_id": i,
            "model": model,
            "condition": condition,
            "objective": "wealth_maximisation",
            "blurb_variant": "neutral",
            "n_rounds": 30,
            "final_value": 50 if is_risk else 110,
            "peak_value": 100,
            "max_drawdown": 0.6 if is_risk else 0.1,
            "risk_event_primary_drawdown_50pct": bool(is_risk),
            "risk_event_secondary_temptation_60pct_5rounds": bool(is_risk and _r.random() < 0.5),
            "expected_volatility_score": 0.4 if is_risk else 0.15,
            "herfindahl_concentration": 0.6 if is_risk else 0.25,
            "outcome": "drawdown_breach" if is_risk else "completed",
            "history": [],
            "rounds": [],
        })
    return {
        "track": "B_M1_portfolio_discriminant",
        "domain": "portfolio",
        "model": model,
        "condition": condition,
        "objective": "wealth_maximisation",
        "blurb_variant": "neutral",
        "n_games": n_games,
        "smoke": True,
        "config_snapshot": {},
        "timestamp": "synthetic",
        "results": results,
    }


def _synthetic_gambling_payload(model: str, condition: str, n_games: int, base_p: float, seed: int) -> dict:
    import random as _r
    _r.seed(seed)
    results = []
    for i in range(n_games):
        is_bk = int(_r.random() < base_p)
        results.append({
            "game_id": i,
            "model": model,
            "condition": condition,
            "bankrupt": bool(is_bk),
            "outcome": "bankruptcy" if is_bk else "voluntary_stop",
            "total_rounds": 20,
            "history": [],
            "rounds": [],
        })
    return {
        "track": "synthetic_gambling_for_m1",
        "domain": "gambling",
        "model": model,
        "condition": condition,
        "results": results,
    }


def _write_synthetic_dataset(portfolio_dir: Path, gambling_dir: Path, n_games: int = 200) -> None:
    portfolio_dir.mkdir(parents=True, exist_ok=True)
    gambling_dir.mkdir(parents=True, exist_ok=True)
    models = ["gemma", "llama", "gpt-4o-mini"]
    seed = 0
    # Portfolio: +G has only a tiny uplift (gambling-specific hypothesis).
    for m in models:
        for cond, base_p in [("BASE", 0.20), ("+G", 0.22), ("MAX_RISK", 0.55)]:
            seed += 1
            payload = _synthetic_portfolio_payload(m, cond, n_games, base_p, seed)
            (portfolio_dir / f"final_{m}_{cond.replace('+', 'plus')}_synth{seed}.json").write_text(json.dumps(payload))
    # Gambling: +G has a large uplift (mock the §3.1 effect).
    for m in models:
        for cond, base_p in [("BASE", 0.15), ("+G", 0.55)]:
            seed += 1
            payload = _synthetic_gambling_payload(m, cond, n_games, base_p, seed)
            (gambling_dir / f"final_{m}_{cond.replace('+', 'plus')}_synth{seed}.json").write_text(json.dumps(payload))


def test_smoke_analyze_pipeline_synthetic(tmp_path: Path):
    portfolio_dir = tmp_path / "portfolio"
    gambling_dir = tmp_path / "gambling"
    _write_synthetic_dataset(portfolio_dir, gambling_dir, n_games=200)
    out = tmp_path / "summary.json"
    cmd = [
        sys.executable, str(SRC_DIR / "analyze_m1.py"),
        "--portfolio_input_dir", str(portfolio_dir),
        "--gambling_input_dirs", str(gambling_dir),
        "--output_path", str(out),
        "--no_bambi",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    summary = json.loads(out.read_text())
    assert "primary_contrast" in summary
    assert "beta_interaction_logit" in summary["primary_contrast"]
    # Synthetic data was constructed with gambling +G effect >> portfolio +G effect,
    # so the interaction should be clearly positive.
    assert summary["primary_contrast"]["beta_interaction_logit"] > 0.0
    assert summary["primary_passes"] is True


def test_max_risk_positive_control_synthetic(tmp_path: Path):
    portfolio_dir = tmp_path / "portfolio"
    gambling_dir = tmp_path / "gambling"
    _write_synthetic_dataset(portfolio_dir, gambling_dir, n_games=200)
    rows = sanity_checks.load_portfolio_records(str(portfolio_dir))
    c1 = sanity_checks.check_max_risk_uplift(rows, min_uplift=0.10)
    # synthetic: BASE=0.20, MAX_RISK=0.55 -> uplift ~0.35, all 3 models should pass.
    assert c1["overall_pass"] is True
    for r in c1["per_model"]:
        assert r["passes"] is True


# ---------------------------------------------------------------------------
# 4. Live smoke (skip if no GPU / no API key)
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_available(), reason="no CUDA device available; Gemma smoke needs GPU")
def test_smoke_open_weight_5games(tmp_path: Path):
    cmd = [
        sys.executable, str(SRC_DIR / "run_m1_open_weight.py"),
        "--model", "gemma",
        "--gpu", "0",
        "--condition", "BASE",
        "--objective", "wealth_maximisation",
        "--blurb_variant", "neutral",
        "--n_games", "5",
        "--output_dir", str(tmp_path),
        "--smoke",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0, result.stderr
    files = sorted(tmp_path.glob("final_gemma_BASE_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["domain"] == "portfolio"
    assert payload["condition"] == "BASE"
    assert len(payload["results"]) == 5


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; GPT-4o-mini smoke needs API",
)
def test_smoke_api_5games(tmp_path: Path):
    cmd = [
        sys.executable, str(SRC_DIR / "run_m1_api.py"),
        "--provider", "openai",
        "--model_id", "gpt-4o-mini",
        "--condition", "BASE",
        "--objective", "wealth_maximisation",
        "--blurb_variant", "neutral",
        "--n_games", "5",
        "--output_dir", str(tmp_path),
        "--smoke",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stderr
    files = sorted(tmp_path.glob("final_gpt-4o-mini_BASE_*.json"))
    assert len(files) == 1


# ---------------------------------------------------------------------------
# 5. Prompt smoke
# ---------------------------------------------------------------------------

def test_prompt_includes_condition_framing():
    assets = _all_in_leveraged_assets()
    game = portfolio_simulator.PortfolioGame(assets=assets, n_rounds=30)
    p_base = prompts.build_portfolio_prompt("BASE", "neutral", "wealth_maximisation", game)
    p_g = prompts.build_portfolio_prompt("+G", "neutral", "wealth_maximisation", game)
    p_mr = prompts.build_portfolio_prompt("MAX_RISK", "neutral", "wealth_maximisation", game)
    assert "target portfolio value" in p_g
    assert "Maximise upside aggressively" in p_mr
    assert "target portfolio value" not in p_base
    assert "Maximise upside" not in p_base


def test_prompt_blurb_variant_changes_text():
    assets = _all_in_leveraged_assets()
    game = portfolio_simulator.PortfolioGame(assets=assets, n_rounds=30)
    p_salient = prompts.build_portfolio_prompt("BASE", "salient_upside", "wealth_maximisation", game)
    p_neutral = prompts.build_portfolio_prompt("BASE", "neutral", "wealth_maximisation", game)
    p_cons = prompts.build_portfolio_prompt("BASE", "conservative", "wealth_maximisation", game)
    assert "+180%" in p_salient
    assert "Annualised return" in p_neutral
    assert "Devastating drawdowns" in p_cons


# ---------------------------------------------------------------------------
# 6. C1: prompt-combo normalisation (cross-domain ingestion)
# ---------------------------------------------------------------------------

def test_normalise_prompt_combo_maps_correctly():
    """C1 fix: SM prompt-combo bitmask strings ("G", "GM", ...) must map to the
    M1 condition vocabulary, and H/W/P-mixed cells must be excluded."""
    norm = analyze_m1._normalise_prompt_combo
    # Canonical pass-through (BASE, MAX_RISK, already-prefixed M1 labels).
    assert norm("BASE") == "BASE"
    assert norm("") == "BASE"
    assert norm(None) == "BASE"
    assert norm("MAX_RISK") == "MAX_RISK"
    assert norm("+G") == "+G"
    assert norm("+M") == "+M"
    assert norm("+GM") == "+GM"
    # SM bitmask strings (clean cells).
    assert norm("G") == "+G"
    assert norm("M") == "+M"
    assert norm("GM") == "+GM"
    # H/W/P-contaminated cells must be excluded (the portfolio arm has no analogue).
    assert norm("GMH") is None
    assert norm("GMHWP") is None
    assert norm("GHW") is None
    assert norm("GH") is None
    assert norm("MH") is None
    assert norm("H") is None
    assert norm("HWP") is None
    # Garbled / unknown labels must be excluded too (defensive).
    assert norm("foo") is None
    assert norm("+GH") is None  # unrecognised plus-prefixed bitmask


def test_load_gambling_results_normalises_sm_prompt_combo(tmp_path: Path):
    """C1 fix end-to-end: the §3.1 SM payload format (prompt_combo without `+` prefix
    and `outcome="bankruptcy"`) must produce non-empty `+G` rows after ingestion."""
    payload = {
        "track": "synthetic_sm_panel",
        "domain": "gambling",
        "model": "llama",
        "results": [
            # Clean +G cells.
            {"game_id": 0, "prompt_combo": "G", "outcome": "bankruptcy", "total_rounds": 20},
            {"game_id": 1, "prompt_combo": "G", "outcome": "voluntary_stop", "total_rounds": 20},
            # Clean BASE cells.
            {"game_id": 2, "prompt_combo": "BASE", "outcome": "bankruptcy", "total_rounds": 20},
            # H-contaminated cell — must be excluded.
            {"game_id": 3, "prompt_combo": "GH", "outcome": "bankruptcy", "total_rounds": 20},
            # +GM clean cell.
            {"game_id": 4, "prompt_combo": "GM", "outcome": "voluntary_stop", "total_rounds": 20},
        ],
    }
    d = tmp_path / "sm"
    d.mkdir()
    (d / "final_llama_synthetic.json").write_text(json.dumps(payload))
    df = analyze_m1.load_gambling_results([str(d)])
    # 4 surviving rows; "GH" excluded.
    assert len(df) == 4
    conds = sorted(df["condition"].unique().tolist())
    assert conds == sorted(["+G", "BASE", "+GM"])
    # "GH" did not survive.
    assert "GH" not in conds and "+GH" not in conds


# ---------------------------------------------------------------------------
# 7. C2: legacy §3.1 record without `bankrupt` field still produces risk_event=1
# ---------------------------------------------------------------------------

def test_legacy_record_outcome_maps_to_risk_event(tmp_path: Path):
    """C2 fix: a legacy §3.1 SM record with `outcome="bankruptcy"` and no
    `bankrupt` field must register risk_event = 1."""
    payload = {
        "track": "synthetic_sm_panel",
        "domain": "gambling",
        "model": "gemma",
        "results": [
            # Legacy schema: outcome only, no bankrupt field.
            {"game_id": 0, "prompt_combo": "G", "outcome": "bankruptcy", "total_rounds": 20},
            {"game_id": 1, "prompt_combo": "G", "outcome": "voluntary_stop", "total_rounds": 20},
            # Track 0 schema: explicit bankrupt field.
            {"game_id": 2, "prompt_combo": "BASE", "bankrupt": True, "total_rounds": 20},
            {"game_id": 3, "prompt_combo": "BASE", "bankrupt": False, "total_rounds": 20},
            # Mixed — both fields present, `bankrupt` precedence.
            {"game_id": 4, "prompt_combo": "G", "bankrupt": True, "outcome": "voluntary_stop"},
        ],
    }
    d = tmp_path / "sm_legacy"
    d.mkdir()
    (d / "final_gemma_legacy.json").write_text(json.dumps(payload))
    df = analyze_m1.load_gambling_results([str(d)])
    assert len(df) == 5
    # Legacy outcome="bankruptcy" -> risk_event = 1.
    legacy_g0 = df[df["game_id"] == 0].iloc[0]
    assert legacy_g0["risk_event"] == 1
    # Legacy outcome="voluntary_stop" -> risk_event = 0.
    legacy_g1 = df[df["game_id"] == 1].iloc[0]
    assert legacy_g1["risk_event"] == 0
    # Track 0 bankrupt=True -> risk_event = 1.
    track0_g2 = df[df["game_id"] == 2].iloc[0]
    assert track0_g2["risk_event"] == 1
    # Track 0 bankrupt=False -> risk_event = 0.
    track0_g3 = df[df["game_id"] == 3].iloc[0]
    assert track0_g3["risk_event"] == 0
    # Mixed: bankrupt=True wins regardless of outcome.
    mixed_g4 = df[df["game_id"] == 4].iloc[0]
    assert mixed_g4["risk_event"] == 1


# ---------------------------------------------------------------------------
# 8. C3 / C4: interaction-term identification + reference-level coding
# ---------------------------------------------------------------------------

def test_bambi_interaction_term_strict_matcher():
    """C3 fix: the strict matcher requires BOTH condition[T.+G] and
    domain[T.gambling] markers; non-interaction terms containing G must NOT match."""
    matcher = analyze_m1._is_target_interaction_name
    # Real interaction term name forms (formulaic and bracketed).
    assert matcher("condition[T.+G]:domain[T.gambling]") is True
    assert matcher("domain[T.gambling]:condition[T.+G]") is True
    # Bambi without the T. prefix.
    assert matcher("condition[+G]:domain[gambling]") is True
    # Non-interaction main effects must NOT match.
    assert matcher("condition[T.+G]") is False
    assert matcher("domain[T.gambling]") is False
    # An incidental "G" in another variable must NOT match.
    assert matcher("Intercept") is False
    assert matcher("model[T.gemma]") is False
    # condition[+M] interaction must NOT match (only +G is the target).
    assert matcher("condition[T.+M]:domain[T.gambling]") is False


def test_categorical_reference_levels_are_pinned():
    """C4 fix: explicit category orderings ensure BASE and portfolio are the
    reference levels — interaction sign cannot flip due to alphabetic ordering."""
    assert analyze_m1.CONDITION_LEVELS[0] == "BASE"
    assert analyze_m1.DOMAIN_LEVELS[0] == "portfolio"
    # The full vocabulary must include +G (without it, primary contrast is undefined).
    assert "+G" in analyze_m1.CONDITION_LEVELS
    assert "gambling" in analyze_m1.DOMAIN_LEVELS


def test_bootstrap_primary_interaction_positive_when_data_designed_positive():
    """C4 fix end-to-end: synthetic data engineered so that gambling +G uplift
    >> portfolio +G uplift must produce a positive interaction estimate (sign
    is correct even without bambi)."""
    import pandas as pd
    rng_seed = 42
    rows = []
    rng = __import__("random").Random(rng_seed)
    for model in ["m1", "m2", "m3"]:
        # Gambling: BASE p=0.15, +G p=0.55  ->  large uplift.
        for _ in range(200):
            rows.append({"model": model, "domain": "gambling", "condition": "BASE",
                         "risk_event": int(rng.random() < 0.15)})
            rows.append({"model": model, "domain": "gambling", "condition": "+G",
                         "risk_event": int(rng.random() < 0.55)})
        # Portfolio: BASE p=0.20, +G p=0.22  ->  tiny uplift.
        for _ in range(200):
            rows.append({"model": model, "domain": "portfolio", "condition": "BASE",
                         "risk_event": int(rng.random() < 0.20)})
            rows.append({"model": model, "domain": "portfolio", "condition": "+G",
                         "risk_event": int(rng.random() < 0.22)})
    df = pd.DataFrame(rows)
    out = analyze_m1._bootstrap_interaction_logit(df, n_boot=300, seed=rng_seed)
    assert out["beta_interaction_logit"] > 0.0, (
        f"interaction sign FLIPPED — got {out['beta_interaction_logit']:.3f}; "
        "this would happen if reference levels were wrong (C4 regression)."
    )


# ---------------------------------------------------------------------------
# 9. C6: API/GPU fallback handling
# ---------------------------------------------------------------------------

def test_parse_allocation_rejects_fallback_sentinel():
    """C6 fix: the parser must explicitly reject the API-failure sentinel and
    return None with parse_reason='fallback_api_failure'."""
    asset_names = ["cash", "bonds", "broad_index", "leveraged_etf_3x",
                   "single_volatile_stock", "otm_call_or_crypto"]
    out, reason = parse_allocation.parse_allocation(
        parse_allocation.FALLBACK_API_FAILURE_SENTINEL, asset_names
    )
    assert out is None
    assert reason == "fallback_api_failure"


def test_run_single_game_counts_fallback_skips():
    """C6 fix end-to-end: a response_fn that always returns the sentinel must
    produce a per-game record with fallback_count > 0 (not a silent 100%-cash
    allocation that biases risk_event downward)."""
    assets = _all_in_leveraged_assets()

    def always_fallback(_prompt: str) -> str:
        return parse_allocation.FALLBACK_API_FAILURE_SENTINEL

    record = portfolio_simulator.run_single_game(
        response_fn=always_fallback,
        assets=assets,
        n_rounds=5,
        initial_budget=100.0,
        round_fraction_of_year=1.0 / 12.0,
        seed=0,
        prompt_builder=prompts.build_portfolio_prompt,
        parse_fn=parse_allocation.parse_allocation,
        condition="BASE",
        objective="wealth_maximisation",
        blurb_variant="neutral",
        max_parse_retries=2,
    )
    # The simulator should bail after 10 consecutive skips; we configure n_rounds=5
    # so all 5 rounds fall through. fallback_count must be > 0.
    assert "fallback_count" in record
    assert record["fallback_count"] >= 1, (
        "fallback_count was not bumped despite every response being the sentinel — "
        "the fallback contamination would still be silent."
    )
    # Sanity: no allocate decisions were made (every round was a parse-skip).
    assert all(r.get("decision") == "skip" for r in record["rounds"])


def test_fallback_rate_per_cell_summary():
    """C6 fix: the analyse_m1 fallback summary must surface the per-cell mean
    and high_fallback_share for the fallback-aware analysis."""
    import pandas as pd
    df = pd.DataFrame([
        {"model": "m1", "domain": "portfolio", "condition": "BASE", "fallback_count": 0},
        {"model": "m1", "domain": "portfolio", "condition": "BASE", "fallback_count": 0},
        {"model": "m1", "domain": "portfolio", "condition": "+G",   "fallback_count": 7},
        {"model": "m1", "domain": "portfolio", "condition": "+G",   "fallback_count": 8},
    ])
    rows = analyze_m1.fallback_rate_per_cell(df)
    by_cell = {(r["model"], r["domain"], r["condition"]): r for r in rows}
    base_cell = by_cell[("m1", "portfolio", "BASE")]
    g_cell = by_cell[("m1", "portfolio", "+G")]
    assert base_cell["mean_fallback_count"] == 0.0
    assert base_cell["high_fallback_share"] == 0.0
    assert g_cell["mean_fallback_count"] == 7.5
    assert g_cell["high_fallback_share"] == 1.0  # both > 5
