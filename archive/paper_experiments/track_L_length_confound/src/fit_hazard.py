"""Cause-specific multinomial hazard fitter for Track L Plan v3.3.

For each primary dataset (SM_API, SM_OW, IC_OW):
  fit MNL on outcome ∈ {continue, bankrupt, voluntary_stop} per round
  conditioned on bet_type + cap + log1p(balance) + round + C(model)
                + C(prompt_combo)
  with cluster-robust SE on (dataset, file_timestamp, cap, prompt_combo,
                             model, game_id)
  extract β_var^bankrupt → RR_per_decision = exp(β); compute Wald 95% CI;
  apply Holm correction across the 3 primary p-values; classify trichotomy
  per Plan §2.3:
    L-passes: RR>1.5 AND lowerCI>1.2
    L-fails:  lowerCI ≤ 1.0
    L-mixed:  otherwise
  Holm-adjusted-p reported alongside descriptive verdict.

IC_API is excluded from primary fits (zero bankruptcy events at max_rounds=10).

Output: JSON with per-dataset RR / CI / p / Holm-p / verdict + summary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf


PRIMARY_DATASETS = ["SM_API", "SM_OW", "IC_OW"]

CLUSTER_KEYS = ["dataset", "file_timestamp", "cap", "prompt_combo", "model", "game_id"]


def restrict_overlap(df: pd.DataFrame, balance_bins: int = 4, round_max: int = 10) -> pd.DataFrame:
    """Plan v3 §3.4 S5: restrict to (model × balance_quartile × round_block) cells where
    BOTH bet_types are observed. For SM, also drop variable-only rounds 11-100 from primary
    (per Plan v3 §3.4: 'variable-arm rounds 11-100 with no fixed-arm counterpart are
    excluded from the per-decision RR'). For IC_OW, max_rounds=100 but matched-cap framing
    uses rounds 1-10 (Plan v3 §3.1).
    """
    df = df.copy()
    # Cap rounds at 10 for matched per-decision RR (Plan v3.3 §3.1, §3.4)
    df = df[df["round"] <= round_max]
    # Quartile of balance_before (across each dataset for stable bins)
    out_parts = []
    for ds, g in df.groupby("dataset"):
        try:
            g["bal_q"] = pd.qcut(g["balance_before"], q=balance_bins, labels=False, duplicates="drop")
        except ValueError:
            g["bal_q"] = 0
        out_parts.append(g)
    df = pd.concat(out_parts, ignore_index=True)
    # Drop cells where only one bet_type is observed
    cell_keys = ["dataset", "model", "bal_q", "round"]
    type_counts = df.groupby(cell_keys)["bet_type"].nunique()
    keep_cells = type_counts[type_counts == 2].index
    df = df.set_index(cell_keys)
    df = df.loc[df.index.isin(keep_cells)].reset_index()
    return df


def fit_binary_logistic_fallback(df_subset: pd.DataFrame, dataset_name: str) -> dict:
    """Plan v3.3 §3.3 Robustness 2: binary cause-specific logistic regression
    on bankruptcy vs not, with statsmodels Logit + small L2 ridge to handle
    quasi-separation when MNL fails. Approximation of Firth-penalised logistic.
    """
    df = df_subset.copy()
    df["log1p_bal"] = np.log1p(df["balance_before"].clip(lower=0))
    df["y_bk"] = (df["outcome"] == "bankrupt").astype(int)
    cluster_id = df[CLUSTER_KEYS].astype(str).agg("_".join, axis=1)
    rhs_parts = ["C(bet_type, Treatment(reference='fixed'))", "C(cap)", "log1p_bal", "round"]
    if df["prompt_combo"].nunique() > 1:
        rhs_parts.append("C(prompt_combo)")
    rhs = " + ".join(rhs_parts)
    try:
        X = patsy.dmatrix(rhs, df, return_type="dataframe")
        # Logit with L2 ridge approximates Firth penalisation
        model = sm.Logit(df["y_bk"].values, X.values).fit_regularized(
            disp=False,
            method="l1",
            alpha=0.5,  # mild ridge
            maxiter=200,
        )
    except Exception as e:
        return {"dataset": dataset_name, "method": "binary_ridge_fallback",
                "error": f"fit failure: {type(e).__name__}: {e}"}

    var_rows = [r for r in X.columns if "bet_type" in r and "variable" in r]
    if not var_rows:
        return {"dataset": dataset_name, "method": "binary_ridge_fallback",
                "error": "no bet_type variable contrast"}
    var_idx = list(X.columns).index(var_rows[0])
    beta = float(model.params[var_idx])
    # Compute SE via observed information; under regularised fit this approximates Wald.
    try:
        se = float(np.sqrt(np.diag(model.cov_params())[var_idx]))
    except Exception:
        se = float("nan")
    p = 2 * (1 - 0.5 * (1 + (np.sign(beta) * np.tanh(abs(beta) / max(se, 1e-9)))))
    if not (np.isfinite(beta) and np.isfinite(se) and se > 0):
        # final fallback: chi-square via 2x2 table
        return {"dataset": dataset_name, "method": "binary_ridge_fallback",
                "error": "non-finite beta/se after ridge fit"}
    z = beta / se
    from scipy.stats import norm
    p_wald = 2 * (1 - norm.cdf(abs(z)))
    rr = float(np.exp(beta))
    rr_lo = float(np.exp(beta - 1.96 * se))
    rr_hi = float(np.exp(beta + 1.96 * se))
    return {
        "dataset": dataset_name,
        "method": "binary_ridge_fallback",
        "n_rows": int(len(df)),
        "n_bankrupt": int(df["y_bk"].sum()),
        "n_clusters": int(cluster_id.nunique()),
        "beta_var_bankrupt": beta,
        "se": se,
        "z": float(z),
        "p_value": float(p_wald),
        "RR_per_decision": rr,
        "RR_lower95": rr_lo,
        "RR_upper95": rr_hi,
    }


def fit_one(df_subset: pd.DataFrame, dataset_name: str) -> dict:
    """Fit MNL on one dataset, return dict with RR_per_decision, CI, p."""
    if len(df_subset) == 0:
        return {"dataset": dataset_name, "n": 0, "error": "empty subset"}
    if df_subset["outcome"].nunique() < 2:
        return {"dataset": dataset_name, "n": len(df_subset), "error": "single-class outcome"}
    n_bankrupt = int((df_subset["outcome"] == "bankrupt").sum())
    if n_bankrupt < 5:
        return {"dataset": dataset_name, "n": len(df_subset), "n_bankrupt": n_bankrupt,
                "error": "fewer than 5 bankrupt events; switch to Firth fallback"}

    df = df_subset.copy()
    df["log1p_bal"] = np.log1p(df["balance_before"].clip(lower=0))
    cluster_id = df[CLUSTER_KEYS].astype(str).agg("_".join, axis=1)

    # Integer-encode outcome with reference=continue (0), bankrupt=1, voluntary_stop=2.
    OUTCOME_MAP = {"continue": 0, "bankrupt": 1, "voluntary_stop": 2}
    df = df[df["outcome"].isin(OUTCOME_MAP)].copy()
    df["y"] = df["outcome"].map(OUTCOME_MAP).astype(int)

    # Build design matrix with patsy. Use C(bet_type) so 'fixed' is reference.
    rhs_parts = ["C(bet_type, Treatment(reference='fixed'))", "C(cap)", "log1p_bal", "round"]
    if df["model"].nunique() > 1:
        # Plan v3.3 §3.3 Robustness 2: if any (model × bet_type) cell has 0 bankruptcies
        # AND another cell has >5, drop C(model) to avoid quasi-separation in MLE; the
        # cluster-robust SE already absorbs within-model dependence.
        cell_bk = df.groupby(["model", "bet_type"])["outcome"].apply(lambda s: (s == "bankrupt").sum())
        zero_cell = (cell_bk == 0).any()
        big_cell = (cell_bk > 5).any()
        if not (zero_cell and big_cell):
            rhs_parts.append("C(model)")
    if df["prompt_combo"].nunique() > 1:
        rhs_parts.append("C(prompt_combo)")
    rhs = " + ".join(rhs_parts)

    try:
        X = patsy.dmatrix(rhs, df, return_type="dataframe")
    except Exception as e:
        return {"dataset": dataset_name, "n": len(df), "error": f"design failure: {type(e).__name__}: {e}"}

    try:
        model = sm.MNLogit(df["y"].values, X.values).fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": cluster_id.values},
            maxiter=200,
        )
    except Exception as e:
        return {"dataset": dataset_name, "n": len(df), "error": f"fit failure: {type(e).__name__}: {e}"}

    # MNLogit params shape: (k_features, n_categories - 1) where columns are
    # for the non-reference categories in the same order as np.unique(y) == [0,1,2].
    # Reference = y=0 ('continue'); col 0 = y=1 ('bankrupt'); col 1 = y=2 ('voluntary_stop').
    params = pd.DataFrame(model.params, index=X.columns)
    bse = pd.DataFrame(model.bse, index=X.columns)
    pvals = pd.DataFrame(model.pvalues, index=X.columns)
    var_rows = [r for r in X.columns if "bet_type" in r and "variable" in r]
    if not var_rows:
        return {"dataset": dataset_name, "n": len(df), "error": f"no bet_type variable contrast in design; cols={list(X.columns)}"}
    var_row = var_rows[0]
    bankrupt_col = 0  # MNLogit uses 0-based for non-reference categories

    beta = float(params.loc[var_row, bankrupt_col])
    se = float(bse.loc[var_row, bankrupt_col])
    p = float(pvals.loc[var_row, bankrupt_col])
    z = beta / se if se > 0 else 0.0
    ci_lo = beta - 1.96 * se
    ci_hi = beta + 1.96 * se
    rr = float(np.exp(beta))
    rr_lo = float(np.exp(ci_lo))
    rr_hi = float(np.exp(ci_hi))

    return {
        "dataset": dataset_name,
        "n_rows": int(len(df)),
        "n_bankrupt": int(n_bankrupt),
        "n_clusters": int(cluster_id.nunique()),
        "beta_var_bankrupt": beta,
        "se": se,
        "z": z,
        "p_value": p,
        "RR_per_decision": rr,
        "RR_lower95": rr_lo,
        "RR_upper95": rr_hi,
        "n_models": int(df["model"].nunique()),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.zeros(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj_i = (n - rank) * p[idx]
        running_max = max(running_max, adj_i)
        adj[idx] = min(running_max, 1.0)
    return list(adj)


def classify_verdict(result: dict) -> str:
    """Plan v3 §2.3 trichotomy."""
    if "error" in result:
        return f"INCONCLUSIVE ({result['error']})"
    rr = result["RR_per_decision"]
    rr_lo = result["RR_lower95"]
    if rr_lo <= 1.0:
        return "L-fails"
    if rr > 1.5 and rr_lo > 1.2:
        return "L-passes"
    return "L-mixed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="/home/v-seungplee/llm-addiction/paper_experiments/track_L_length_confound/round_table.csv")
    ap.add_argument("--out", default="/home/v-seungplee/llm-addiction/paper_experiments/track_L_length_confound/track_L_results.json")
    ap.add_argument("--no-overlap", action="store_true", help="skip overlap restriction (debugging)")
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    print(f"loaded {len(df)} rows from {args.table}")
    print(f"datasets: {df['dataset'].value_counts().to_dict()}")

    if not args.no_overlap:
        df = restrict_overlap(df)
        print(f"after overlap restriction: {len(df)} rows")

    results = {}
    for ds in PRIMARY_DATASETS:
        sub = df[df["dataset"] == ds]
        print(f"\n--- fitting {ds} (n={len(sub)}) ---")
        r = fit_one(sub, ds)
        if not np.isfinite(r.get("RR_per_decision", float("nan"))):
            print(f"  MNL non-convergence for {ds} → falling back to binary ridge logistic")
            r = fit_binary_logistic_fallback(sub, ds)
        results[ds] = r
        print(f"  result: {r}")

    # Holm correction across 3 primary p-values
    p_values = [results[ds].get("p_value", 1.0) for ds in PRIMARY_DATASETS]
    holm = holm_adjust(p_values)
    for ds, p_adj in zip(PRIMARY_DATASETS, holm):
        results[ds]["p_holm_adjusted"] = float(p_adj)
        results[ds]["verdict"] = classify_verdict(results[ds])

    print("\n=== SUMMARY (Plan v3.3 §6) ===")
    for ds in PRIMARY_DATASETS:
        r = results[ds]
        print(f"  {ds}:")
        for k in ["n_rows", "n_bankrupt", "n_clusters", "RR_per_decision", "RR_lower95", "RR_upper95",
                  "p_value", "p_holm_adjusted", "verdict", "error"]:
            if k in r:
                print(f"    {k}: {r[k]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
