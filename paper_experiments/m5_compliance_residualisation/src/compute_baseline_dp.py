"""Compute the decision-point baseline Δ_G_dp on the 3,200-row sample.

Critical fix (C1) — row-coordinate parity:
=========================================

The canonical Table 3 Δ_G = 0.09029 (Gemma SM I_BA L22) was fit on the
**round-level** ``sae_features_L22.npz`` cache (n ≈ 21,421 rows), but M5
operates on the **decision-point** ``hidden_states_dp.npz`` cache
(n = 3,200 rows). Comparing a residualised Δ_G' fit on the decision-point
sample against the canonical round-level Δ_G is a category mismatch —
they live on different sample spaces, so Δ_G' is *not* a residualised
version of the canonical Δ_G.

Resolution (Plan v4 §13 deviation, 2026-05-07): re-baseline Δ_G on the
*same* decision-point sample by re-encoding H through the SAE without any
direction projection, then fitting the canonical GroupKFold readout. Call
this Δ_G_dp. Then the residualised Δ_G_dp' computed by
``residualise_sae_features.py`` + ``refit_table3_residualised.py`` IS a
true residualised version of the new baseline, and the survival ratio
``|Δ_G_dp − Δ_G_dp'| / |Δ_G_dp|`` is well-defined.

Implementation parity:
* Hidden states: ``data.hidden_states_root/{paradigm}/{model}/hidden_states_dp.npz``,
  layer index from ``cfg["layer"]`` resolved through the cached ``layers`` array.
* SAE encode: identical to ``residualise_sae_features.sae_encode``
  (gemma_scope JumpReLU via sae_lens, fnlp ReLU via safetensors).
* GroupKFold readout: identical to ``refit_table3_residualised.fit_one_residualised_subset``
  (5-fold, RF deconfound, top-200 Spearman, Ridge α=100, group_by=game_id).

Output: ``<results_subdir>/delta_g_dp_baseline.json`` keyed by model →
{plus_G, minus_G, delta_g_dp, n}.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
DEFAULT_CONFIG = ROOT / "configs" / "m5_config.yaml"

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("m5.baseline_dp")

# Reuse canonical pipeline (same as refit_table3_residualised).
SAE_V3_SRC = Path(__file__).resolve().parents[3] / "sae_v3_analysis" / "src"
if str(SAE_V3_SRC) not in sys.path:
    sys.path.insert(0, str(SAE_V3_SRC))


def _make_meta_for_groupkfold(hs: Dict) -> Dict:
    """Build the meta dict expected by ``fit_one_residualised_subset``.

    Mirrors the keys produced by ``residualise_sae_features.save_residualised_sparse``
    so the canonical GroupKFold helpers work identically.
    """
    return {
        "game_ids": hs["game_ids"],
        "round_nums": hs["round_nums"],
        "balances": hs["balances"],
        "bet_types": hs["bet_types"],
        "prompt_conditions": hs["prompt_conditions"],
        "game_outcomes": hs["game_outcomes"],
    }


def compute_baseline_for_model(
    cfg: dict,
    model: str,
    paradigm_dir: str,
    plus_key: str,
    minus_key: str,
) -> Dict:
    """Encode hidden states through SAE without any residualisation, then
    apply the canonical GroupKFold readout to plus_G / minus_G subsets.
    """
    # Lazy imports — pytest collect-only must work without GPU stack.
    from residualise_sae_features import (
        load_hidden_states,
        sae_encode,
        _load_sae_for_model,
    )
    from refit_table3_residualised import fit_one_residualised_subset
    from scipy import sparse

    layer = int(cfg["layer"])
    indicator = cfg["indicator"]
    task_short = cfg["task"]

    hs = load_hidden_states(cfg, model, paradigm_dir)
    H = hs["H"]
    log.info("[%s/%s] decision-point H shape=%s (n=%d)", model, paradigm_dir, H.shape, H.shape[0])

    # SAE-encode without residualisation. This is the baseline F.
    sae_obj, sae_kind = _load_sae_for_model(cfg, model, layer)
    F = sae_encode(
        H,
        sae_obj,
        sae_kind,
        cfg.get("generation", {}).get("device", "cuda"),
    )
    sp = sparse.csr_matrix(F)

    meta = _make_meta_for_groupkfold(hs)
    out: Dict[str, Dict] = {}
    for subset in (plus_key, minus_key):
        log.info("  subset %s ...", subset)
        out[subset] = fit_one_residualised_subset(
            sp, meta, model, task_short, indicator, subset,
        )

    plus_r2 = out.get(plus_key, {}).get("r2_mean")
    minus_r2 = out.get(minus_key, {}).get("r2_mean")
    delta_g_dp: Optional[float] = None
    if plus_r2 is not None and minus_r2 is not None:
        delta_g_dp = float(plus_r2) - float(minus_r2)

    return {
        "model": model,
        "paradigm_dir": paradigm_dir,
        "layer": layer,
        "indicator": indicator,
        "task": task_short,
        "n_decision_points": int(H.shape[0]),
        "plus_subset": plus_key,
        "minus_subset": minus_key,
        "subsets": out,
        "delta_g_dp": delta_g_dp,
    }


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="M5 — compute decision-point baseline Δ_G_dp (no residualisation)",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--model", choices=["gemma", "llama", "all"], default="all")
    p.add_argument("--paradigm-dir", default="slot_machine")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_root = Path(args.output or cfg["output"]["root"])
    results_dir = out_root / cfg["output"].get("results_subdir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    plus_key = cfg.get("plus_g_subset", "plus_G")
    minus_key = cfg.get("minus_g_subset", "minus_G")

    models = ["gemma", "llama"] if args.model == "all" else [args.model]
    baselines: Dict[str, Dict] = {}
    for mk in models:
        baselines[mk] = compute_baseline_for_model(
            cfg, mk, args.paradigm_dir, plus_key, minus_key,
        )
        d = baselines[mk]["delta_g_dp"]
        log.info("[%s] Δ_G_dp = %s", mk, f"{d:.4f}" if d is not None else "None")

    out_path = results_dir / "delta_g_dp_baseline.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "paradigm_dir": args.paradigm_dir,
                "layer": int(cfg["layer"]),
                "indicator": cfg["indicator"],
                "task": cfg["task"],
                "models": baselines,
            },
            f,
            indent=2,
        )
    log.info("wrote %s", out_path)
    return 0


def load_baseline_dp_delta_g(
    baseline_path: Path,
    model: str,
    indicator: str,
) -> Optional[float]:
    """Read Δ_G_dp baseline for one (model, indicator) pair from the JSON.

    Used by analyze_m5.py to replace the canonical Δ_G with the
    decision-point baseline (Plan v4 §13 deviation 2026-05-07).
    Returns None if the baseline file is missing or the cell is incomplete,
    so callers can fall back / warn explicitly.
    """
    if not Path(baseline_path).exists():
        return None
    with open(baseline_path) as f:
        d = json.load(f)
    if d.get("indicator") != indicator:
        log.warning(
            "baseline file indicator=%s != requested indicator=%s",
            d.get("indicator"), indicator,
        )
    cell = d.get("models", {}).get(model)
    if cell is None:
        return None
    return cell.get("delta_g_dp")


if __name__ == "__main__":
    sys.exit(main())
