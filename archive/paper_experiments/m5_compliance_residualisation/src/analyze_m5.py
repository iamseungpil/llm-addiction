"""Compute Δ_G survival ratios and apply pre-registered M5 thresholds.

For each (model, mode) pair:
  Δ_G_dp  = R²_+G  − R²_-G   on the **decision-point** sample (n=3,200)
                              with NO residualisation — this is the M5
                              baseline (see Plan v4 §13 deviation 2026-05-07)
  Δ_G_dp' = R²_+G' − R²_-G'  residualised re-fit on the SAME sample
  ratio_drop = |Δ_G_dp − Δ_G_dp'| / |Δ_G_dp|       (absolute, per §3.2)

Why the decision-point baseline (C1 fix):
  The canonical Table 3 Δ_G = 0.0903 was fit on the round-level
  ``sae_features_L22.npz`` cache (~21,421 rows). M5's residualisation runs
  on the decision-point cache ``hidden_states_dp.npz`` (3,200 rows). A
  residualised Δ_G computed on the decision-point sample is not a
  residualised version of the round-level canonical Δ_G — they live on
  different sample spaces. The fix is to re-baseline on the same 3,200-row
  sample (compute_baseline_dp.py), then compare residualised against that.

Decision rule (Plan v4 §3.2):
  individual modes:  ratio_drop < 0.30  → pass
  joint mode:        ratio_drop < 0.50  → pass
  Stability rule:    if |Δ_G_dp| < 0.005 use absolute thresholds
                     0.01 / 0.015 applied to |Δ_G_dp − Δ_G_dp'|.

Outcome branches (claim_surgery_M5_outcome_branches.md):
  M5-passes:   all 3 individual + joint pass
  M5-partial:  joint passes but ≥1 individual fails (or vice versa)
  M5-fails:    joint fails

Random-direction controls: should always pass. If they don't, residualisation
has a bug or feature space is too low-dimensional and the M5 result is
uninterpretable.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
DEFAULT_CONFIG = ROOT / "configs" / "m5_config.yaml"

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("m5.analyze")


# ----- threshold logic -----------------------------------------------------


def evaluate_threshold(
    delta_orig: float,
    delta_resid: float,
    is_joint: bool,
    thresholds: Dict,
) -> Dict:
    """Apply pre-registered relative threshold; switch to absolute under
    stability rule. Returns dict with ratio_drop, abs_drop, threshold used,
    pass/fail, and which rule fired.
    """
    abs_drop = abs(delta_orig - delta_resid)
    floor = float(thresholds.get("stability_floor_abs_delta", 0.005))
    if abs(delta_orig) < floor:
        rule = "absolute"
        thr = float(
            thresholds.get("joint_absolute" if is_joint else "individual_absolute", 0.015 if is_joint else 0.01)
        )
        passed = abs_drop < thr
        ratio_drop = None
    else:
        rule = "relative"
        thr = float(
            thresholds.get("joint_relative" if is_joint else "individual_relative", 0.50 if is_joint else 0.30)
        )
        # Use ABSOLUTE ratio per Plan v4 §3.2: |Δ_G − Δ_G'| / |Δ_G| < thr.
        # Signed ratio (delta_orig - delta_resid) / delta_orig would silently
        # PASS the case where Δ_G' overshoots Δ_G in the wrong direction
        # (residualisation increases R² unexpectedly), which the pre-reg
        # explicitly treats as a FAIL.
        ratio_drop = abs(delta_orig - delta_resid) / abs(delta_orig)
        # "Drop" sense: pass if compliance projection did NOT change Δ_G by ≥ thr.
        passed = ratio_drop < thr
    return {
        "delta_orig": float(delta_orig),
        "delta_resid": float(delta_resid),
        "abs_drop": float(abs_drop),
        "ratio_drop": None if ratio_drop is None else float(ratio_drop),
        "rule_applied": rule,
        "threshold": float(thr),
        "is_joint": bool(is_joint),
        "pass": bool(passed),
    }


# ----- IO ------------------------------------------------------------------


def load_original_table3(cfg: dict) -> Dict[Tuple[str, str], float]:
    """Returns mapping (model, indicator) → canonical Δ_G from the round-level JSON.

    Kept for reporting/context only — the M5 pass/fail decision uses
    ``load_dp_baselines`` (Plan v4 §13 deviation 2026-05-07). The canonical
    round-level Δ_G is included in the analysis output for transparency
    so reviewers can see both numbers.
    """
    path = Path(cfg["original_table3_path"])
    layer = int(cfg["layer"])
    task = cfg["task"]
    indicator = cfg["indicator"]
    with open(path) as f:
        d = json.load(f)
    out: Dict[Tuple[str, str], float] = {}
    plus_key = cfg.get("plus_g_subset", "plus_G")
    minus_key = cfg.get("minus_g_subset", "minus_G")
    for model in cfg["models"].keys():
        key = cfg["original_table3_key_template"].format(
            model=model, task=task, indicator=indicator, layer=layer,
        )
        cell = d.get(key)
        if cell is None:
            log.warning("Table 3 missing key: %s", key)
            continue
        sub = cell.get("subsets", {})
        plus = sub.get(plus_key, {}).get("r2_mean")
        minus = sub.get(minus_key, {}).get("r2_mean")
        if plus is None or minus is None:
            log.warning("Table 3 missing +G or -G for %s", key)
            continue
        out[(model, indicator)] = float(plus) - float(minus)
    return out


def load_dp_baselines(baseline_path: Path) -> Dict[str, float]:
    """Returns mapping model → Δ_G_dp from the decision-point baseline JSON
    produced by ``compute_baseline_dp.py``.

    This is the headline baseline for M5 pass/fail decisions, replacing the
    hardcoded canonical Δ_G = 0.0903 (round-level, n≈21k) which lived on a
    different sample space than the residualised re-fits (decision-point,
    n=3,200). See Plan v4 §13 deviation 2026-05-07.
    """
    if not Path(baseline_path).exists():
        log.error(
            "missing dp baseline %s — run compute_baseline_dp.py first",
            baseline_path,
        )
        return {}
    with open(baseline_path) as f:
        d = json.load(f)
    out: Dict[str, float] = {}
    for model, cell in d.get("models", {}).items():
        delta = cell.get("delta_g_dp")
        if delta is None:
            log.warning("dp baseline missing Δ_G_dp for model=%s", model)
            continue
        out[model] = float(delta)
    return out


def load_refit_results(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


# ----- analysis ------------------------------------------------------------


def compute_delta_resid(per_mode: Dict, plus_key: str = "plus_G", minus_key: str = "minus_G") -> Optional[float]:
    plus = per_mode.get(plus_key, {}).get("r2_mean")
    minus = per_mode.get(minus_key, {}).get("r2_mean")
    if plus is None or minus is None:
        return None
    return float(plus) - float(minus)


def classify_outcome(per_mode_passes: Dict[str, bool]) -> str:
    """Map mode_tag→pass into one of the three outcome-branch labels."""
    individual_modes = [k for k in per_mode_passes if k.startswith("individual_")]
    joint_modes = [k for k in per_mode_passes if k.startswith("joint_")]
    individual_pass = all(per_mode_passes[m] for m in individual_modes) if individual_modes else False
    joint_pass = all(per_mode_passes[m] for m in joint_modes) if joint_modes else False
    if individual_pass and joint_pass:
        return "M5-passes"
    if joint_pass and not individual_pass:
        return "M5-partial"
    if individual_pass and not joint_pass:
        return "M5-partial"
    return "M5-fails"


def analyze_for_model(
    model: str,
    indicator: str,
    delta_orig: float,
    per_model_modes: Dict[str, Dict],
    thresholds: Dict,
) -> Dict:
    out: Dict = {
        "delta_orig": delta_orig,
        "per_mode": {},
    }
    pass_map: Dict[str, bool] = {}
    for mode_tag, subset_results in per_model_modes.items():
        delta_resid = compute_delta_resid(subset_results)
        if delta_resid is None:
            out["per_mode"][mode_tag] = {"reason": "missing +G or -G"}
            continue
        is_joint = mode_tag.startswith("joint_")
        verdict = evaluate_threshold(delta_orig, delta_resid, is_joint, thresholds)
        # Tag random controls as such (they should pass; if they fail, the
        # whole pipeline result is suspect).
        if mode_tag.startswith("control_random"):
            verdict["is_control"] = True
        out["per_mode"][mode_tag] = verdict
        # Only count direction-bearing modes for outcome classification.
        if not mode_tag.startswith("control_"):
            pass_map[mode_tag] = verdict["pass"]
    out["outcome_branch"] = classify_outcome(pass_map)
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="M5 — analyze residualisation outcomes vs Table 3 Δ_G")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--paradigm-dir", default="slot_machine")
    p.add_argument("--refit-results", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    layer = int(cfg["layer"])
    indicator = cfg["indicator"]
    out_root = Path(args.output or cfg["output"]["root"])
    results_dir = out_root / cfg["output"].get("results_subdir", "results")

    refit_path = args.refit_results or (
        results_dir / f"refit_results_{args.paradigm_dir}_{indicator}_L{layer}.json"
    )
    if not Path(refit_path).exists():
        log.error("missing refit results: %s — run refit_table3_residualised.py first", refit_path)
        return 1

    # Round-level canonical Δ_G — kept only for reporting context.
    canonical_deltas = load_original_table3(cfg)
    # Decision-point baseline Δ_G_dp — the operational baseline (C1 fix).
    baseline_path = results_dir / "delta_g_dp_baseline.json"
    dp_deltas = load_dp_baselines(baseline_path)

    refit = load_refit_results(Path(refit_path))
    thresholds = cfg.get("thresholds", {})

    final: Dict[str, Dict] = {
        "config_paradigm_dir": args.paradigm_dir,
        "indicator": indicator,
        "layer": layer,
        "thresholds": thresholds,
        "baseline_source": "decision_point_n3200 (Plan v4 §13 deviation 2026-05-07)",
        "models": {},
    }
    for model, per_modes in refit.items():
        delta_orig = dp_deltas.get(model)
        if delta_orig is None:
            log.error(
                "no decision-point Δ_G_dp for %s — run compute_baseline_dp.py first; skipping",
                model,
            )
            continue
        per_model = analyze_for_model(model, indicator, delta_orig, per_modes, thresholds)
        # Stash the canonical round-level Δ_G alongside for reporting context.
        canonical = canonical_deltas.get((model, indicator))
        per_model["delta_g_canonical_round_level"] = (
            float(canonical) if canonical is not None else None
        )
        final["models"][model] = per_model
        log.info(
            "[%s] Δ_G_dp=%.4f (canonical round-level Δ_G=%s)  outcome=%s",
            model, delta_orig,
            f"{canonical:.4f}" if canonical is not None else "n/a",
            per_model["outcome_branch"],
        )

    out_path = results_dir / f"m5_analysis_{args.paradigm_dir}_{indicator}_L{layer}.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
