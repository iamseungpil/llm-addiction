"""Track 0 analysis: hierarchical mixed-logit + per-model bootstrap CIs.

Pre-registered (Plan v4 §1bis.3, §8.1):
    bankrupt ~ condition * cap + (condition * cap | model)

Primary estimand:
    beta_primary = E[bankrupt | variable, cap=70] - E[bankrupt | fixed, cap=70]
    decision rule: lower 95% credible interval > 0 (logit scale).

Cluster-robust SE clustering on (model, game_id) is declared here for the
frequentist cross-check; the bambi/PyMC fit is the primary path.

The bambi/pymc dependency is **soft-imported**. The module loads cleanly even when
those packages are absent so the module-level smoke tests in CI don't need them.
The `fit_mixed_logit` function raises a clear ImportError at call time if missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent


def _load_cfg() -> dict:
    with open(HERE.parent / "configs" / "track0_config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_results(input_dir: str) -> pd.DataFrame:
    """Walk `input_dir`, parse every `final_*.json`, return long-format dataframe.

    Columns: model, game_id, condition (=mode), cap, bankrupt, total_bets, n_rounds,
             total_won, final_balance, file.
    """
    rows = []
    for path in sorted(Path(input_dir).glob("final_*.json")):
        with open(path, "r") as f:
            payload = json.load(f)
        model = payload.get("model")
        cap = payload.get("cap")
        mode = payload.get("mode")
        for record in payload.get("results", []):
            rows.append({
                "model": model,
                "game_id": record.get("game_id"),
                "condition": mode,
                "cap": cap,
                "bankrupt": int(bool(record.get("bankrupt"))),
                "total_bets": record.get("total_bet"),
                "n_rounds": record.get("total_rounds"),
                "total_won": record.get("total_won"),
                "final_balance": record.get("final_balance"),
                "file": path.name,
            })
    df = pd.DataFrame(rows)
    return df


def fit_mixed_logit(df: pd.DataFrame, cfg: Optional[dict] = None) -> "object":
    """Fit `bankrupt ~ condition * cap + (condition * cap | model)` via bambi.

    Returns the fitted bambi.Model wrapping the InferenceData; caller uses
    `.fit_result` (or `model.idata`) for posterior queries.

    Raises ImportError with installation hint if bambi/pymc unavailable.
    """
    try:
        import bambi as bmb  # type: ignore
    except ImportError as e:
        raise ImportError(
            "bambi (and pymc) required for fit_mixed_logit. "
            "Install with: pip install bambi pymc"
        ) from e

    cfg = cfg or _load_cfg()
    bcfg = cfg["bambi_fit"]

    # Cap as categorical to allow 4-level contrast per pre-reg.
    df = df.copy()
    df["cap"] = df["cap"].astype("category")
    df["condition"] = df["condition"].astype("category")
    df["model"] = df["model"].astype("category")

    model = bmb.Model(
        bcfg["formula"],
        data=df,
        family=bcfg["family"],
    )
    idata = model.fit(
        draws=bcfg["draws"],
        tune=bcfg["tune"],
        chains=bcfg["chains"],
        cores=bcfg["cores"],
        random_seed=bcfg["random_seed"],
    )
    model.idata = idata
    return model


def primary_contrast_at_cap(
    df: pd.DataFrame,
    fitted_model: Optional["object"] = None,
    cap_value: int = 70,
) -> Dict[str, float]:
    """Compute beta_primary = P(bankrupt | variable, cap) - P(bankrupt | fixed, cap).

    If a fitted bambi model is provided, uses posterior predictive marginals to
    derive logit-scale + probability-scale contrasts with 95% credible intervals.
    Otherwise falls back to a non-parametric per-model bootstrap of the empirical
    rates pooled by inverse-variance weighting (frequentist sanity).
    """
    sub = df[df["cap"] == cap_value]
    if sub.empty:
        return {"beta_primary_prob": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "method": "no_data"}

    if fitted_model is not None:
        try:
            return _posterior_contrast(fitted_model, df, cap_value)
        except Exception:
            # fall through to bootstrap
            pass

    return _bootstrap_pooled_contrast(sub)


def _posterior_contrast(fitted_model, df: pd.DataFrame, cap_value: int) -> Dict[str, float]:
    import arviz as az  # type: ignore  # noqa: F401  (used implicitly via fitted_model.idata)

    new_data_var = pd.DataFrame({
        "model": df["model"].astype("category").cat.categories,
        "condition": "variable",
        "cap": cap_value,
    })
    new_data_fix = new_data_var.copy()
    new_data_fix["condition"] = "fixed"

    # kind="response" returns posterior on the response (probability) scale; we want the
    # pooled marginal across models, so include_random=False suppresses the per-model
    # random effects and gives the grand-mean P(bankrupt | condition, cap).
    pp_var = fitted_model.predict(
        idata=fitted_model.idata, data=new_data_var, inplace=False,
        kind="response", include_random=False,
    )
    pp_fix = fitted_model.predict(
        idata=fitted_model.idata, data=new_data_fix, inplace=False,
        kind="response", include_random=False,
    )

    var_mean_name = next(iter(pp_var.posterior.data_vars))
    fix_mean_name = next(iter(pp_fix.posterior.data_vars))
    var_arr = pp_var.posterior[var_mean_name]
    fix_arr = pp_fix.posterior[fix_mean_name]
    # Sanity: bambi posterior should be (chain, draw, n_obs).
    assert var_arr.ndim == 3, f"expected (chain, draw, n_obs), got dims={var_arr.dims}"
    assert "chain" in var_arr.dims and "draw" in var_arr.dims, var_arr.dims
    assert var_arr.shape == fix_arr.shape, f"shape mismatch {var_arr.shape} vs {fix_arr.shape}"
    p_var = var_arr.mean(dim=[d for d in var_arr.dims if d not in ("chain", "draw")])
    p_fix = fix_arr.mean(dim=[d for d in fix_arr.dims if d not in ("chain", "draw")])
    diff = (p_var - p_fix).values.flatten()
    eps = 1e-6
    p_var_v = np.clip(p_var.values.flatten(), eps, 1 - eps)
    p_fix_v = np.clip(p_fix.values.flatten(), eps, 1 - eps)
    logit_diff = np.log(p_var_v / (1 - p_var_v)) - np.log(p_fix_v / (1 - p_fix_v))
    return {
        "beta_primary_prob": float(np.mean(diff)),
        "ci_low": float(np.quantile(diff, 0.025)),
        "ci_high": float(np.quantile(diff, 0.975)),
        "beta_primary_logit": float(np.mean(logit_diff)),
        "logit_ci_low": float(np.quantile(logit_diff, 0.025)),
        "logit_ci_high": float(np.quantile(logit_diff, 0.975)),
        "method": "bambi_posterior",
    }


def _bootstrap_pooled_contrast(sub: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    # Pool across models with equal weight per model (Stage 1 design treats models as exchangeable).
    # Pre-reg primary scale is logit (configs/track0_config.yaml primary_contrast.scale: logit),
    # so we must compute logit-scale CI as well as the prob-scale diagnostic.
    eps = 1e-6
    prob_diffs = []
    logit_diffs = []
    models = sub["model"].dropna().unique()
    for _ in range(n_boot):
        per_model_prob = []
        per_model_logit = []
        for m in models:
            mset = sub[sub["model"] == m]
            var = mset[mset["condition"] == "variable"]["bankrupt"].values
            fix = mset[mset["condition"] == "fixed"]["bankrupt"].values
            if len(var) == 0 or len(fix) == 0:
                continue
            v = rng.choice(var, size=len(var), replace=True).mean()
            f = rng.choice(fix, size=len(fix), replace=True).mean()
            per_model_prob.append(v - f)
            v_c = float(np.clip(v, eps, 1 - eps))
            f_c = float(np.clip(f, eps, 1 - eps))
            per_model_logit.append(np.log(v_c / (1 - v_c)) - np.log(f_c / (1 - f_c)))
        if per_model_prob:
            prob_diffs.append(np.mean(per_model_prob))
            logit_diffs.append(np.mean(per_model_logit))
    if not prob_diffs:
        return {
            "beta_primary_prob": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "beta_primary_logit": float("nan"),
            "logit_ci_low": float("nan"),
            "logit_ci_high": float("nan"),
            "method": "bootstrap_failed",
        }
    return {
        "beta_primary_prob": float(np.mean(prob_diffs)),
        "ci_low": float(np.quantile(prob_diffs, 0.025)),
        "ci_high": float(np.quantile(prob_diffs, 0.975)),
        "beta_primary_logit": float(np.mean(logit_diffs)),
        "logit_ci_low": float(np.quantile(logit_diffs, 0.025)),
        "logit_ci_high": float(np.quantile(logit_diffs, 0.975)),
        "method": "bootstrap_pooled",
    }


def per_model_deltas(df: pd.DataFrame, cap_value: int = 70, n_boot: int = 2000, seed: int = 42) -> List[Dict]:
    """Per-model bootstrap CI on Δ at the highest cap. Used for the qualitative ≥4/6 rule."""
    rng = np.random.default_rng(seed)
    out = []
    sub = df[df["cap"] == cap_value]
    for m in sub["model"].dropna().unique():
        mset = sub[sub["model"] == m]
        var = mset[mset["condition"] == "variable"]["bankrupt"].values
        fix = mset[mset["condition"] == "fixed"]["bankrupt"].values
        if len(var) == 0 or len(fix) == 0:
            out.append({"model": m, "delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")})
            continue
        diffs = []
        for _ in range(n_boot):
            v = rng.choice(var, size=len(var), replace=True).mean()
            f = rng.choice(fix, size=len(fix), replace=True).mean()
            diffs.append(v - f)
        out.append({
            "model": m,
            "delta": float(var.mean() - fix.mean()),
            "ci_low": float(np.quantile(diffs, 0.025)),
            "ci_high": float(np.quantile(diffs, 0.975)),
            "n_var": int(len(var)),
            "n_fix": int(len(fix)),
        })
    return out


def cluster_robust_summary(df: pd.DataFrame) -> Dict:
    """Cluster-robust SE cross-check via statsmodels logistic GEE-equivalent.

    Pre-registered alternate: clustering by (model, game_id). statsmodels GEE
    accepts a single grouping variable, so we collapse to a string clustering id.
    Soft-imported.
    """
    try:
        import statsmodels.api as sm  # type: ignore
        from statsmodels.formula.api import logit  # type: ignore
    except ImportError:
        return {"available": False, "note": "statsmodels not installed"}

    work = df.dropna(subset=["bankrupt", "condition", "cap", "model", "game_id"]).copy()
    work["cluster"] = work["model"].astype(str) + "_" + work["game_id"].astype(str)
    try:
        fit = logit("bankrupt ~ C(condition) * C(cap)", data=work).fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": work["cluster"]},
        )
        return {
            "available": True,
            "params": fit.params.to_dict(),
            "bse": fit.bse.to_dict(),
            "n_obs": int(fit.nobs),
            "note": "logit with cluster-robust SE on (model, game_id)",
        }
    except Exception as e:
        return {"available": False, "note": f"fit failed: {e}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--cap", type=int, default=70)
    parser.add_argument("--no_bambi", action="store_true",
                        help="Skip bambi fit (use bootstrap-only for the primary contrast)")
    args = parser.parse_args()

    cfg = _load_cfg()
    df = load_results(args.input_dir)
    if df.empty:
        raise SystemExit(f"no `final_*.json` files in {args.input_dir}")

    fitted = None
    fit_status = "skipped"
    if not args.no_bambi:
        try:
            fitted = fit_mixed_logit(df, cfg)
            fit_status = "ok"
        except Exception as e:
            fit_status = f"failed: {e}"

    primary = primary_contrast_at_cap(df, fitted_model=fitted, cap_value=args.cap)
    per_model = per_model_deltas(df, cap_value=args.cap)

    # Plan §1bis.2 secondary rule: ">=4/6 models with positive Δ at highest cap AND
    # POOLED 95% CI excludes 0". Per-model CIs at n=200 are structurally too wide for
    # the AND clause; the pooled CI from the primary contrast is the gating quantity.
    pos_directional_count = sum(1 for r in per_model if r.get("delta") is not None and r["delta"] > 0)
    logit_low = primary.get("logit_ci_low")
    prob_low = primary.get("ci_low")
    if logit_low is not None and not (isinstance(logit_low, float) and np.isnan(logit_low)):
        pooled_ci_excludes_zero = bool(logit_low > 0)
        pooled_ci_scale = "logit"
    elif prob_low is not None and not (isinstance(prob_low, float) and np.isnan(prob_low)):
        pooled_ci_excludes_zero = bool(prob_low > 0)
        pooled_ci_scale = "prob"
    else:
        pooled_ci_excludes_zero = False
        pooled_ci_scale = "unavailable"

    # Primary contrast pass: prefer logit-scale (pre-reg primary) when present.
    if logit_low is not None and not (isinstance(logit_low, float) and np.isnan(logit_low)):
        primary_passes = bool(logit_low > 0)
        primary_pass_scale = "logit"
    elif prob_low is not None and not (isinstance(prob_low, float) and np.isnan(prob_low)):
        primary_passes = bool(prob_low > 0)
        primary_pass_scale = "prob"
    else:
        primary_passes = False
        primary_pass_scale = "unavailable"
    print(f"[analyze_track0] primary gate scale={primary_pass_scale} passes={primary_passes}")

    cluster = cluster_robust_summary(df)

    summary = {
        "cap": args.cap,
        "n_games_total": int(len(df)),
        "models": sorted(df["model"].dropna().unique().tolist()),
        "primary_contrast": primary,
        "primary_passes": primary_passes,
        "primary_pass_scale": primary_pass_scale,
        "per_model_deltas": per_model,
        "qualitative_secondary": {
            "rule": cfg["qualitative_secondary"]["rule"],
            "pos_directional_count": int(pos_directional_count),
            "pooled_ci_excludes_zero": bool(pooled_ci_excludes_zero),
            "pooled_ci_scale": pooled_ci_scale,
            "passes": bool(pos_directional_count >= 4 and pooled_ci_excludes_zero),
        },
        "bambi_fit_status": fit_status,
        "cluster_robust_se_check": cluster,
    }

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[analyze_track0] wrote {out}")
    print(json.dumps(summary["primary_contrast"], indent=2))


if __name__ == "__main__":
    main()
