"""Track A1 / M2 analysis: condition × framing mixed-logit + per-task primary contrast.

Pre-registered (Plan v4 §2, §8.1 row A1):

    risk_metric ~ condition * framing + (1 | model)

Primary estimand (single test, lower-95%-Wald-CI gating):

    Delta_{+G,first} - Delta_{+G,role}
        where Delta_{cond,frame} = E[risk | cond, frame] - E[risk | BASE, frame]

Lower 95% Wald CI > 0 -> +G-induced shift survives in first-person framing,
not just role-play, supporting the propensity (vs role-uptake) hypothesis.

bambi/pymc are soft-imported. The module loads without them; the bootstrap
fallback in `_bootstrap_primary_contrast` is the frequentist sanity path and
is also what the synthetic CI smoke test exercises (no statistical-stack
dependency for unit tests).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent


def _load_cfg() -> dict:
    with open(HERE.parent / "configs" / "m2_config.yaml", "r") as f:
        return yaml.safe_load(f)


def _gambling_keyword_count(text: str, keywords: List[str]) -> int:
    if not text:
        return 0
    low = text.lower()
    return sum(low.count(k) for k in keywords)


def load_results(input_dir: str, cfg: Optional[dict] = None) -> pd.DataFrame:
    """Walk `input_dir`, parse every `final_*.json`, return long-format dataframe.

    Columns: model, game_id, condition, framing, task, bankrupt, n_rounds,
             total_bet, total_won, final_balance, gambling_kw_count, file.

    `gambling_kw_count` is the per-game total across all stored response strings;
    used by the manipulation-check sanity report.
    """
    cfg = cfg or _load_cfg()
    keywords = cfg.get("manipulation_check", {}).get("gambling_keywords", [])
    rows = []
    for path in sorted(Path(input_dir).glob("final_*.json")):
        with open(path, "r") as f:
            payload = json.load(f)
        model = payload.get("model")
        condition = payload.get("condition")
        framing = payload.get("framing")
        task = payload.get("task", "SM")
        for record in payload.get("results", []):
            kw_count = 0
            for round_rec in record.get("rounds", []):
                kw_count += _gambling_keyword_count(round_rec.get("response", ""), keywords)
            rows.append({
                "model": model,
                "game_id": record.get("game_id"),
                "condition": condition,
                "framing": framing,
                "task": task,
                "bankrupt": int(bool(record.get("bankrupt"))),
                "n_rounds": record.get("total_rounds"),
                "total_bet": record.get("total_bet"),
                "total_won": record.get("total_won"),
                "final_balance": record.get("final_balance"),
                "gambling_kw_count": kw_count,
                "file": path.name,
            })
    return pd.DataFrame(rows)


def fit_mixed_logit(df: pd.DataFrame, cfg: Optional[dict] = None) -> "object":
    """Fit `risk_metric ~ condition * framing + (1 | model)` via bambi.

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

    work = df.copy()
    work["condition"] = work["condition"].astype("category")
    work["framing"] = work["framing"].astype("category")
    work["model"] = work["model"].astype("category")
    # The pre-reg formula uses `risk_metric` as the LHS; we map bankruptcy onto it.
    work["risk_metric"] = work["bankrupt"].astype(int)

    model = bmb.Model(
        bcfg["formula"],
        data=work,
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


def primary_contrast(
    df: pd.DataFrame,
    fitted_model: Optional["object"] = None,
    test_condition: str = "+G",
    reference_condition: str = "BASE",
) -> Dict[str, float]:
    """Compute Delta_{+G,first} - Delta_{+G,role} on probability + logit scales.

    If `fitted_model` is supplied, posterior marginals from bambi are used for
    the credible interval. Otherwise a per-model bootstrap of the empirical
    rates is used. Both routes return the same dict schema.
    """
    sub = df[df["condition"].isin([test_condition, reference_condition])]
    if sub.empty:
        return {
            "delta_gap_prob": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "method": "no_data",
        }

    if fitted_model is not None:
        try:
            return _posterior_primary_contrast(fitted_model, df, test_condition, reference_condition)
        except Exception:
            pass
    return _bootstrap_primary_contrast(sub, test_condition, reference_condition)


def _posterior_primary_contrast(fitted_model, df: pd.DataFrame, test_condition: str, reference_condition: str) -> Dict[str, float]:
    new = []
    for cond in (reference_condition, test_condition):
        for frame in ("first_person", "role_play_gambler"):
            for m in df["model"].astype("category").cat.categories:
                new.append({"model": m, "condition": cond, "framing": frame})
    new_df = pd.DataFrame(new)

    pp = fitted_model.predict(
        idata=fitted_model.idata, data=new_df, inplace=False,
        kind="response", include_random=False,
    )
    var_name = next(iter(pp.posterior.data_vars))
    arr = pp.posterior[var_name]  # (chain, draw, n_obs)

    # Marginalise across model on each (cond, frame) row.
    new_df["_idx"] = np.arange(len(new_df))
    def _grand_mean(cond: str, frame: str):
        idx = new_df[(new_df["condition"] == cond) & (new_df["framing"] == frame)]["_idx"].values
        sub = arr.isel({arr.dims[-1]: idx})
        return sub.mean(dim=arr.dims[-1])  # (chain, draw)

    p_g_first = _grand_mean(test_condition, "first_person")
    p_b_first = _grand_mean(reference_condition, "first_person")
    p_g_role = _grand_mean(test_condition, "role_play_gambler")
    p_b_role = _grand_mean(reference_condition, "role_play_gambler")

    delta_first = (p_g_first - p_b_first).values.flatten()
    delta_role = (p_g_role - p_b_role).values.flatten()
    gap = delta_first - delta_role

    eps = 1e-6
    def _logit(x):
        x = np.clip(x, eps, 1 - eps)
        return np.log(x / (1 - x))
    logit_delta_first = _logit(p_g_first.values.flatten()) - _logit(p_b_first.values.flatten())
    logit_delta_role = _logit(p_g_role.values.flatten()) - _logit(p_b_role.values.flatten())
    logit_gap = logit_delta_first - logit_delta_role

    return {
        "delta_first_prob": float(np.mean(delta_first)),
        "delta_role_prob": float(np.mean(delta_role)),
        "delta_gap_prob": float(np.mean(gap)),
        "ci_low": float(np.quantile(gap, 0.025)),
        "ci_high": float(np.quantile(gap, 0.975)),
        "delta_gap_logit": float(np.mean(logit_gap)),
        "logit_ci_low": float(np.quantile(logit_gap, 0.025)),
        "logit_ci_high": float(np.quantile(logit_gap, 0.975)),
        "method": "bambi_posterior",
    }


def _bootstrap_primary_contrast(
    sub: pd.DataFrame,
    test_condition: str,
    reference_condition: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    eps = 1e-6
    models = sub["model"].dropna().unique()

    gap_prob_draws: List[float] = []
    gap_logit_draws: List[float] = []
    for _ in range(n_boot):
        per_model_gap_prob = []
        per_model_gap_logit = []
        for m in models:
            mset = sub[sub["model"] == m]
            cells = {}
            ok = True
            for cond in (test_condition, reference_condition):
                for frame in ("first_person", "role_play_gambler"):
                    arr = mset[(mset["condition"] == cond) & (mset["framing"] == frame)]["bankrupt"].values
                    if len(arr) == 0:
                        ok = False
                        break
                    cells[(cond, frame)] = rng.choice(arr, size=len(arr), replace=True).mean()
                if not ok:
                    break
            if not ok:
                continue
            d_first = cells[(test_condition, "first_person")] - cells[(reference_condition, "first_person")]
            d_role = cells[(test_condition, "role_play_gambler")] - cells[(reference_condition, "role_play_gambler")]
            per_model_gap_prob.append(d_first - d_role)

            def _l(p): return np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
            ld_first = _l(cells[(test_condition, "first_person")]) - _l(cells[(reference_condition, "first_person")])
            ld_role = _l(cells[(test_condition, "role_play_gambler")]) - _l(cells[(reference_condition, "role_play_gambler")])
            per_model_gap_logit.append(ld_first - ld_role)

        if per_model_gap_prob:
            gap_prob_draws.append(float(np.mean(per_model_gap_prob)))
            gap_logit_draws.append(float(np.mean(per_model_gap_logit)))

    if not gap_prob_draws:
        return {
            "delta_gap_prob": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "delta_gap_logit": float("nan"),
            "logit_ci_low": float("nan"),
            "logit_ci_high": float("nan"),
            "method": "bootstrap_failed",
        }

    return {
        "delta_gap_prob": float(np.mean(gap_prob_draws)),
        "ci_low": float(np.quantile(gap_prob_draws, 0.025)),
        "ci_high": float(np.quantile(gap_prob_draws, 0.975)),
        "delta_gap_logit": float(np.mean(gap_logit_draws)),
        "logit_ci_low": float(np.quantile(gap_logit_draws, 0.025)),
        "logit_ci_high": float(np.quantile(gap_logit_draws, 0.975)),
        "method": "bootstrap_pooled",
    }


def per_model_deltas(
    df: pd.DataFrame,
    test_condition: str = "+G",
    reference_condition: str = "BASE",
    n_boot: int = 2000,
    seed: int = 42,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    out = []
    for m in sorted(df["model"].dropna().unique()):
        mset = df[df["model"] == m]
        record: Dict = {"model": m}
        for frame in ("first_person", "role_play_gambler"):
            test_arr = mset[(mset["condition"] == test_condition) & (mset["framing"] == frame)]["bankrupt"].values
            ref_arr = mset[(mset["condition"] == reference_condition) & (mset["framing"] == frame)]["bankrupt"].values
            if len(test_arr) == 0 or len(ref_arr) == 0:
                record[f"delta_{frame}"] = float("nan")
                record[f"delta_{frame}_ci_low"] = float("nan")
                record[f"delta_{frame}_ci_high"] = float("nan")
                continue
            diffs = []
            for _ in range(n_boot):
                t = rng.choice(test_arr, size=len(test_arr), replace=True).mean()
                r = rng.choice(ref_arr, size=len(ref_arr), replace=True).mean()
                diffs.append(t - r)
            record[f"delta_{frame}"] = float(test_arr.mean() - ref_arr.mean())
            record[f"delta_{frame}_ci_low"] = float(np.quantile(diffs, 0.025))
            record[f"delta_{frame}_ci_high"] = float(np.quantile(diffs, 0.975))
        out.append(record)
    return out


def framing_condition_heatmap(df: pd.DataFrame) -> List[Dict]:
    """Long-format rows for a (condition × framing) bankruptcy-rate heatmap, pooled across models."""
    rows = []
    for cond in df["condition"].dropna().unique():
        for frame in df["framing"].dropna().unique():
            cell = df[(df["condition"] == cond) & (df["framing"] == frame)]
            if cell.empty:
                continue
            rows.append({
                "condition": cond,
                "framing": frame,
                "bankrupt_rate": float(cell["bankrupt"].mean()),
                "n": int(len(cell)),
            })
    return rows


def manipulation_check_kw(df: pd.DataFrame) -> Dict:
    """+G under role_play should boost gambling-keyword frequency more than +G under first-person.

    Returns per-cell mean keyword count and the pooled delta. The check passes
    qualitatively when delta_role > delta_first (positive role-play boost vs
    first-person boost). This is a sanity flag, not a primary inferential test.
    """
    cell_means: Dict = defaultdict(float)
    cell_counts: Dict = defaultdict(int)
    for _, row in df.iterrows():
        key = (row["condition"], row["framing"])
        cell_means[key] += row.get("gambling_kw_count", 0) or 0
        cell_counts[key] += 1
    avg = {k: (cell_means[k] / cell_counts[k]) if cell_counts[k] else float("nan") for k in cell_means}

    delta_first = avg.get(("+G", "first_person"), float("nan")) - avg.get(("BASE", "first_person"), float("nan"))
    delta_role = avg.get(("+G", "role_play_gambler"), float("nan")) - avg.get(("BASE", "role_play_gambler"), float("nan"))
    passes = bool(np.isfinite(delta_role) and np.isfinite(delta_first) and delta_role > delta_first)
    return {
        "cell_mean_kw_count": {f"{c}|{f}": v for (c, f), v in avg.items()},
        "delta_first": float(delta_first) if np.isfinite(delta_first) else float("nan"),
        "delta_role": float(delta_role) if np.isfinite(delta_role) else float("nan"),
        "role_boost_exceeds_first_boost": passes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--task", default=None,
                        help="Filter to a single task (SM/IC/MW). Default: analyze all tasks present.")
    parser.add_argument("--no_bambi", action="store_true",
                        help="Skip bambi fit (use bootstrap-only for the primary contrast)")
    args = parser.parse_args()

    cfg = _load_cfg()
    df = load_results(args.input_dir, cfg)
    if df.empty:
        raise SystemExit(f"no `final_*.json` files in {args.input_dir}")

    tasks = [args.task] if args.task else sorted(df["task"].dropna().unique().tolist())

    per_task_summary = {}
    for t in tasks:
        df_t = df[df["task"] == t] if args.task is None else df
        if df_t.empty:
            continue

        fitted = None
        fit_status = "skipped"
        if not args.no_bambi:
            try:
                fitted = fit_mixed_logit(df_t, cfg)
                fit_status = "ok"
            except Exception as e:
                fit_status = f"failed: {e}"

        primary = primary_contrast(df_t, fitted_model=fitted)
        models = per_model_deltas(df_t)
        heat = framing_condition_heatmap(df_t)
        manip = manipulation_check_kw(df_t)

        logit_low = primary.get("logit_ci_low")
        prob_low = primary.get("ci_low")
        if logit_low is not None and not (isinstance(logit_low, float) and np.isnan(logit_low)):
            primary_passes = bool(logit_low > 0)
            primary_pass_scale = "logit"
        elif prob_low is not None and not (isinstance(prob_low, float) and np.isnan(prob_low)):
            primary_passes = bool(prob_low > 0)
            primary_pass_scale = "prob"
        else:
            primary_passes = False
            primary_pass_scale = "unavailable"

        per_task_summary[t] = {
            "n_games_total": int(len(df_t)),
            "models": sorted(df_t["model"].dropna().unique().tolist()),
            "primary_contrast": primary,
            "primary_passes": primary_passes,
            "primary_pass_scale": primary_pass_scale,
            "per_model_deltas": models,
            "framing_condition_heatmap": heat,
            "manipulation_check": manip,
            "bambi_fit_status": fit_status,
        }

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"per_task": per_task_summary}, f, indent=2)
    print(f"[analyze_m2] wrote {out}")
    for t, s in per_task_summary.items():
        print(f"  task={t} primary_passes={s['primary_passes']} scale={s['primary_pass_scale']}")
        print(f"    primary_contrast={json.dumps(s['primary_contrast'], indent=2)}")


if __name__ == "__main__":
    main()
