"""M1 cross-domain analysis: gambling × condition × portfolio × condition interaction.

Pre-registered (Plan v4 §4.2):
    risk_event ~ domain * condition + (1 | model)

Primary estimand (logit scale, single test, NOT a 2× ratio):
    beta_{+G * gambling}  -  beta_{+G * portfolio}  >  0
    decision rule: lower 95% Wald CI on the logit-scale interaction term > 0
                   -> +G effect is gambling-specific.

Secondary descriptive output (probability scale):
    ( P[risk|+G,gambling] - P[risk|BASE,gambling] )
        - ( P[risk|+G,portfolio] - P[risk|BASE,portfolio] )
with delta-method or bootstrap CI; per-model interaction estimates also reported.

Bambi/PyMC are soft-imported; the module loads without them and the bootstrap path
runs the primary contrast on logit scale via the empirical risk-event rates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


HERE = Path(__file__).resolve().parent


# Pre-registered ordering of categorical levels (Plan v4 §13 deviation log: C4 fix).
# BASE is the reference for `condition`; portfolio is the reference for `domain`. Under
# this coding the bambi interaction term named `condition[T.+G]:domain[T.gambling]` is
# exactly the desired estimand `β_{+G × gambling} − β_{+G × portfolio}`.
CONDITION_LEVELS = ["BASE", "+G", "+M", "+GM", "MAX_RISK"]
DOMAIN_LEVELS = ["portfolio", "gambling"]


def _load_cfg() -> dict:
    with open(HERE.parent / "configs" / "m1_config.yaml", "r") as f:
        return yaml.safe_load(f)


def _normalise_prompt_combo(prompt_combo: Optional[str]) -> Optional[str]:
    """Map raw §3.1 SM `prompt_combo` strings to M1 condition labels.

    Per Plan v4 §13 deviation log (C1 fix): the §3.1 SM panel writes prompt-combo
    bitmasks directly (e.g. ``"G"``, ``"GM"``, ``"GMHWP"``), which the M1 ingestion
    must translate to the condition vocabulary it shares with the portfolio task
    (``"BASE"``, ``"+G"``, ``"+M"``, ``"+GM"``). Cells that mix the H / W / P bit
    flags into the G/M cells are excluded from the cross-domain analysis because
    the portfolio arm has no analogue of those manipulations — including them would
    contaminate the comparison.

    Mapping rules:
        ``"BASE"``, ``""``, ``None``           → ``"BASE"``
        ``"G"`` (G alone, no M)                → ``"+G"``
        ``"M"`` (M alone, no G)                → ``"+M"``
        ``"GM"`` (G and M, no H/W/P)           → ``"+GM"``
        ``"+G"`` / ``"+M"`` / ``"+GM"``        → unchanged (already-canonical M1 input)
        ``"MAX_RISK"``                         → ``"MAX_RISK"``
        anything else (e.g. ``"GMH"``, ``"GHW"``, ``"H"``) → ``None`` (excluded)
    """
    if prompt_combo is None:
        return "BASE"
    s = str(prompt_combo).strip()
    if s == "" or s.upper() == "BASE":
        return "BASE"
    if s == "MAX_RISK":
        return "MAX_RISK"
    # Already-canonical M1 labels pass through.
    if s in ("+G", "+M", "+GM"):
        return s
    # If the string starts with `+` but is not one of the canonical M1 labels,
    # it is something we don't want to touch (e.g. `"+GH"`); exclude.
    if s.startswith("+"):
        return None
    # Treat as a §3.1 SM bitmask string. Allowed alphabet: G, M, H, W, P.
    if not all(c in "GMHWP" for c in s):
        return None
    has_g = "G" in s
    has_m = "M" in s
    has_other = any(c in "HWP" for c in s)
    if has_other:
        # H/W/P contamination — exclude from cross-domain analysis.
        return None
    if has_g and has_m:
        return "+GM"
    if has_g:
        return "+G"
    if has_m:
        return "+M"
    # Neither G nor M and no H/W/P leaves only "" which is BASE; that branch was
    # already handled above. If we reach here, exclude defensively.
    return None


def load_portfolio_results(input_dir: str) -> pd.DataFrame:
    """Walk M1 portfolio outputs into a long-format dataframe.

    Columns: model, game_id, condition, objective, blurb_variant, domain='portfolio',
             risk_event (drawdown > 50%), n_rounds, max_drawdown, file.
    """
    rows = []
    for path in sorted(Path(input_dir).glob("final_*.json")):
        with open(path, "r") as f:
            payload = json.load(f)
        if payload.get("domain") != "portfolio":
            continue
        model = payload.get("model")
        condition = payload.get("condition")
        objective = payload.get("objective")
        blurb = payload.get("blurb_variant")
        for record in payload.get("results", []):
            rows.append({
                "model": model,
                "game_id": record.get("game_id"),
                "condition": condition,
                "objective": objective,
                "blurb_variant": blurb,
                "domain": "portfolio",
                "risk_event": int(bool(record.get("risk_event_primary_drawdown_50pct"))),
                "risk_event_secondary": int(bool(record.get("risk_event_secondary_temptation_60pct_5rounds"))),
                "n_rounds": record.get("n_rounds"),
                "max_drawdown": record.get("max_drawdown"),
                "expected_volatility_score": record.get("expected_volatility_score"),
                "herfindahl_concentration": record.get("herfindahl_concentration"),
                # C6 fix: per-game count of API/GPU-failure sentinel rounds (defaults
                # to 0 for older records that pre-date the field).
                "fallback_count": int(record.get("fallback_count", 0) or 0),
                "file": path.name,
            })
    return pd.DataFrame(rows)


def load_gambling_results(input_dirs: List[str]) -> pd.DataFrame:
    """Walk gambling outputs (Track 0 SM matched-cap and/or §3.1 SM panel) into long-format.

    The risk_event for gambling is `bankrupt`. We tag `condition` from the JSON if
    available; Track 0 outputs use `mode` ∈ {fixed, variable} which is *not* the
    same axis as M1 conditions, so we only ingest Track 0 / §3.1 runs that explicitly
    record a `condition` field (e.g., +G/+M/BASE) at the payload or per-record level.
    Other Track 0 cells are ignored — the M1 primary contrast needs +G vs BASE in the
    gambling domain, which Track 0 does not vary; §3.1 SM 6-model panel does.
    """
    rows = []
    for d in input_dirs:
        if not d:
            continue
        for path in sorted(Path(d).glob("final_*.json")):
            try:
                with open(path, "r") as f:
                    payload = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            payload_condition = payload.get("condition") or payload.get("prompt_combo")
            model = payload.get("model")
            for record in payload.get("results", []):
                raw_cond = record.get("condition") or record.get("prompt_combo") or payload_condition
                # C1 fix: normalise §3.1 SM bitmask strings ("G", "GM", ...) to the
                # M1 condition vocabulary ("+G", "+GM", ...). H/W/P-mixed cells are
                # excluded because the portfolio arm has no analogue.
                cond = _normalise_prompt_combo(raw_cond)
                if cond is None:
                    continue
                # C2 fix: legacy §3.1 SM records emit only `outcome="bankruptcy"`;
                # Track 0 records also emit a boolean `bankrupt` field. Accept either.
                bankrupt_flag = bool(record.get("bankrupt")) or (
                    record.get("outcome") == "bankruptcy"
                )
                rows.append({
                    "model": model,
                    "game_id": record.get("game_id"),
                    "condition": cond,
                    "objective": None,
                    "blurb_variant": None,
                    "domain": "gambling",
                    "risk_event": int(bankrupt_flag),
                    "risk_event_secondary": None,
                    "n_rounds": record.get("total_rounds"),
                    "max_drawdown": None,
                    "expected_volatility_score": None,
                    "herfindahl_concentration": None,
                    # C6: gambling-domain SM games predate the fallback sentinel;
                    # treat as 0 so the column exists when concatenating with
                    # portfolio rows.
                    "fallback_count": 0,
                    "file": path.name,
                })
    return pd.DataFrame(rows)


def build_combined_df(portfolio_dir: str, gambling_dirs: List[str]) -> pd.DataFrame:
    p = load_portfolio_results(portfolio_dir)
    g = load_gambling_results(gambling_dirs)
    if p.empty and g.empty:
        return pd.DataFrame()
    combined = pd.concat([p, g], ignore_index=True, sort=False)
    # C1 fix: condition vocabulary is enforced by `_normalise_prompt_combo` on the
    # gambling side and by the portfolio runner on the portfolio side, both producing
    # one of CONDITION_LEVELS. Anything else (e.g. an unexpected H/W/P-bitmask leaked
    # through) is dropped here. Plan v4 §13 deviation log records this restriction.
    return combined[combined["condition"].isin(CONDITION_LEVELS)].copy()


def fit_mixed_logit_interaction(df: pd.DataFrame, cfg: Optional[dict] = None) -> "object":
    """Fit `risk_event ~ domain * condition + (1 | model)` via bambi."""
    try:
        import bambi as bmb  # type: ignore
    except ImportError as e:
        raise ImportError(
            "bambi (and pymc) required for fit_mixed_logit_interaction. "
            "Install with: pip install bambi pymc"
        ) from e

    cfg = cfg or _load_cfg()
    bcfg = cfg["bambi_fit"]

    df = df.copy()
    # C4 fix: pin reference levels explicitly so the interaction term name is
    # deterministic regardless of which categories happen to appear in the data.
    # `BASE` first → reference for condition; `portfolio` first → reference for
    # domain; the interaction `condition[T.+G]:domain[T.gambling]` is therefore
    # the desired estimand. Without this, alphabetic ordering would put `+G`
    # before `BASE` and flip the sign of the interaction.
    df["domain"] = pd.Categorical(df["domain"], categories=DOMAIN_LEVELS)
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_LEVELS)
    df["model"] = df["model"].astype("category")

    model = bmb.Model(bcfg["formula"], data=df, family=bcfg["family"])
    idata = model.fit(
        draws=bcfg["draws"],
        tune=bcfg["tune"],
        chains=bcfg["chains"],
        cores=bcfg["cores"],
        random_seed=bcfg["random_seed"],
    )
    model.idata = idata
    return model


def primary_interaction_logit(
    df: pd.DataFrame,
    fitted_model: Optional["object"] = None,
) -> Dict[str, float]:
    """Compute beta_{+G x gambling} - beta_{+G x portfolio} on the logit scale.

    With bambi: extract the interaction posterior directly from `idata.posterior`.
    Without bambi: bootstrap empirical rates per (domain, condition), convert to
    logit per resample, take the interaction difference.
    """
    if fitted_model is not None:
        try:
            return _bambi_interaction(fitted_model)
        except Exception:
            pass
    return _bootstrap_interaction_logit(df)


def _is_target_interaction_name(name: str) -> bool:
    """Strict matcher for the (+G × gambling) interaction posterior variable name.

    Requires BOTH `condition[T.+G]`-style and `domain[T.gambling]`-style markers
    to be present, eliminating the original substring-match false positives where
    a non-interaction term containing the letter ``G`` would match.
    """
    has_condition_g = "condition[T.+G]" in name or "condition[+G]" in name
    has_domain_gambling = (
        "domain[T.gambling]" in name or "domain[gambling]" in name
    )
    return has_condition_g and has_domain_gambling


def _find_interaction_term_via_bambi_api(fitted_model) -> Optional[str]:
    """Look up the interaction term via bambi's structured `terms` API.

    Bambi exposes term metadata via ``model.response_component.terms`` (and on some
    versions, ``model.terms``). We iterate, pick interaction terms whose factor
    components are exactly ``{"condition", "domain"}``, then expand to the
    posterior variable name that targets the (+G, gambling) cell.

    Returns ``None`` if the structured API does not surface a usable interaction
    name (caller should fall back to the strict substring matcher).
    """
    candidates: List[str] = []
    seen = set()

    def _consider(term_name: str) -> None:
        if term_name and term_name not in seen:
            seen.add(term_name)
            candidates.append(term_name)

    # Try a few attribute paths that have appeared across bambi versions; we don't
    # know which is exposed at runtime, so we tolerate AttributeError silently.
    term_iters: List = []
    for attr in ("terms", "common_terms"):
        obj = getattr(fitted_model, attr, None)
        if obj is None:
            continue
        try:
            term_iters.append(list(obj.values()) if hasattr(obj, "values") else list(obj))
        except Exception:
            continue
    rc = getattr(fitted_model, "response_component", None)
    if rc is not None:
        for attr in ("terms", "common_terms"):
            obj = getattr(rc, attr, None)
            if obj is None:
                continue
            try:
                term_iters.append(list(obj.values()) if hasattr(obj, "values") else list(obj))
            except Exception:
                continue

    for terms in term_iters:
        for term in terms:
            kind = getattr(term, "kind", None)
            if kind != "interaction":
                continue
            # Recover the set of factor names this term involves. Bambi exposes
            # different attributes across versions; try a few.
            factor_names = None
            for attr in ("term_dict_components", "components", "vars", "name_components"):
                val = getattr(term, attr, None)
                if val is None:
                    continue
                try:
                    factor_names = set(val)
                    break
                except TypeError:
                    continue
            if factor_names is None:
                # Fallback: parse the term name. Bambi names interactions as
                # "condition:domain"; split on ":" gives the components.
                name = getattr(term, "name", "") or ""
                factor_names = {part.split("[")[0] for part in name.split(":")}
            if {"condition", "domain"}.issubset(factor_names):
                _consider(getattr(term, "name", "") or "")
    return candidates[0] if candidates else None


def _bambi_interaction(fitted_model) -> Dict[str, float]:
    """Pull the (+G × gambling) interaction term from the posterior.

    C3 fix: replaces the original fragile substring scan (``"G" in name``) with a
    two-tier strategy:
        1. Use bambi's structured term API to identify the
           (``condition`` × ``domain``) interaction term. If that succeeds and the
           target posterior variable is found, use it.
        2. Otherwise, scan ``posterior.data_vars`` and accept ONLY names that
           contain BOTH ``condition[T.+G]`` and ``domain[T.gambling]`` substrings
           (strict matcher — non-interaction terms containing the letter G no
           longer match).

    Asserts that exactly one matching posterior variable exists; raises
    ``RuntimeError`` if zero or multiple match (the caller catches and falls back
    to the bootstrap-logit estimate so downstream code does not crash).
    """
    post = fitted_model.idata.posterior
    posterior_vars = list(post.data_vars)

    # First-tier: try bambi's structured term API.
    api_term_name = _find_interaction_term_via_bambi_api(fitted_model)

    matches: List[str] = []
    if api_term_name:
        # Expand the structured term name into a concrete posterior variable name.
        for var in posterior_vars:
            if api_term_name in var and _is_target_interaction_name(var):
                matches.append(var)

    # Second-tier: strict substring matcher on posterior variable names.
    if not matches:
        matches = [v for v in posterior_vars if _is_target_interaction_name(v)]

    if len(matches) == 0:
        raise RuntimeError(
            "_bambi_interaction: no posterior variable matches the (+G × gambling) "
            f"interaction (looked at {len(posterior_vars)} data_vars; expected one "
            "containing both 'condition[T.+G]' and 'domain[T.gambling]')."
        )
    if len(matches) > 1:
        raise RuntimeError(
            "_bambi_interaction: ambiguous match — multiple posterior variables "
            f"satisfy the (+G × gambling) interaction matcher: {matches}. "
            "This indicates a category-ordering or formula change; refusing to "
            "guess. Fix `CONDITION_LEVELS` / `DOMAIN_LEVELS` or the bambi formula."
        )
    interaction_name = matches[0]
    arr = post[interaction_name].values.flatten()
    return {
        "beta_interaction_logit": float(np.mean(arr)),
        "logit_ci_low": float(np.quantile(arr, 0.025)),
        "logit_ci_high": float(np.quantile(arr, 0.975)),
        "method": "bambi_posterior",
        "interaction_term_name": interaction_name,
    }


def _logit(p: float, eps: float = 1e-6) -> float:
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1 - p))


def _bootstrap_interaction_logit(df: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    """Bootstrap fallback for the logit-scale interaction term.

    Per resample b:
        delta_g_b   = logit P_b(risk | +G, gambling)  - logit P_b(risk | BASE, gambling)
        delta_p_b   = logit P_b(risk | +G, portfolio) - logit P_b(risk | BASE, portfolio)
        interact_b  = delta_g_b - delta_p_b
    Mean across models is used as the pooled estimate (equal weight per model).
    """
    rng = np.random.default_rng(seed)
    sub = df[df["condition"].isin(["BASE", "+G"])]
    interactions = []
    models = sub["model"].dropna().unique()
    for _ in range(n_boot):
        per_model = []
        for m in models:
            mset = sub[sub["model"] == m]
            try:
                gv = mset[(mset["domain"] == "gambling") & (mset["condition"] == "+G")]["risk_event"].values
                gb = mset[(mset["domain"] == "gambling") & (mset["condition"] == "BASE")]["risk_event"].values
                pv = mset[(mset["domain"] == "portfolio") & (mset["condition"] == "+G")]["risk_event"].values
                pb = mset[(mset["domain"] == "portfolio") & (mset["condition"] == "BASE")]["risk_event"].values
            except KeyError:
                continue
            if len(gv) == 0 or len(gb) == 0 or len(pv) == 0 or len(pb) == 0:
                continue
            gv_b = rng.choice(gv, size=len(gv), replace=True).mean()
            gb_b = rng.choice(gb, size=len(gb), replace=True).mean()
            pv_b = rng.choice(pv, size=len(pv), replace=True).mean()
            pb_b = rng.choice(pb, size=len(pb), replace=True).mean()
            d_gambling = _logit(gv_b) - _logit(gb_b)
            d_portfolio = _logit(pv_b) - _logit(pb_b)
            per_model.append(d_gambling - d_portfolio)
        if per_model:
            interactions.append(float(np.mean(per_model)))
    if not interactions:
        return {
            "beta_interaction_logit": float("nan"),
            "logit_ci_low": float("nan"),
            "logit_ci_high": float("nan"),
            "method": "bootstrap_failed_no_overlap",
        }
    return {
        "beta_interaction_logit": float(np.mean(interactions)),
        "logit_ci_low": float(np.quantile(interactions, 0.025)),
        "logit_ci_high": float(np.quantile(interactions, 0.975)),
        "method": "bootstrap_logit_pooled",
        "n_boot": n_boot,
    }


def secondary_marginal_difference(df: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    """Probability-scale secondary contrast (Plan v4 §4.2 line 346)."""
    rng = np.random.default_rng(seed)
    sub = df[df["condition"].isin(["BASE", "+G"])]
    diffs = []
    models = sub["model"].dropna().unique()
    for _ in range(n_boot):
        per_model = []
        for m in models:
            mset = sub[sub["model"] == m]
            gv = mset[(mset["domain"] == "gambling") & (mset["condition"] == "+G")]["risk_event"].values
            gb = mset[(mset["domain"] == "gambling") & (mset["condition"] == "BASE")]["risk_event"].values
            pv = mset[(mset["domain"] == "portfolio") & (mset["condition"] == "+G")]["risk_event"].values
            pb = mset[(mset["domain"] == "portfolio") & (mset["condition"] == "BASE")]["risk_event"].values
            if len(gv) == 0 or len(gb) == 0 or len(pv) == 0 or len(pb) == 0:
                continue
            d_g = rng.choice(gv, size=len(gv), replace=True).mean() - rng.choice(gb, size=len(gb), replace=True).mean()
            d_p = rng.choice(pv, size=len(pv), replace=True).mean() - rng.choice(pb, size=len(pb), replace=True).mean()
            per_model.append(d_g - d_p)
        if per_model:
            diffs.append(float(np.mean(per_model)))
    if not diffs:
        return {"marginal_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "method": "bootstrap_failed"}
    return {
        "marginal_diff": float(np.mean(diffs)),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "method": "bootstrap_prob",
    }


def per_model_interactions(df: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> List[Dict]:
    """Per-model logit-scale interaction with bootstrap CI."""
    rng = np.random.default_rng(seed)
    sub = df[df["condition"].isin(["BASE", "+G"])]
    out = []
    for m in sub["model"].dropna().unique():
        mset = sub[sub["model"] == m]
        gv = mset[(mset["domain"] == "gambling") & (mset["condition"] == "+G")]["risk_event"].values
        gb = mset[(mset["domain"] == "gambling") & (mset["condition"] == "BASE")]["risk_event"].values
        pv = mset[(mset["domain"] == "portfolio") & (mset["condition"] == "+G")]["risk_event"].values
        pb = mset[(mset["domain"] == "portfolio") & (mset["condition"] == "BASE")]["risk_event"].values
        if min(len(gv), len(gb), len(pv), len(pb)) == 0:
            out.append({"model": m, "interaction_logit": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")})
            continue
        boot = []
        for _ in range(n_boot):
            d_g = _logit(rng.choice(gv, size=len(gv), replace=True).mean()) - _logit(rng.choice(gb, size=len(gb), replace=True).mean())
            d_p = _logit(rng.choice(pv, size=len(pv), replace=True).mean()) - _logit(rng.choice(pb, size=len(pb), replace=True).mean())
            boot.append(d_g - d_p)
        out.append({
            "model": m,
            "interaction_logit": float(np.mean(boot)),
            "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
            "n_gambling_var": int(len(gv)),
            "n_gambling_base": int(len(gb)),
            "n_portfolio_var": int(len(pv)),
            "n_portfolio_base": int(len(pb)),
        })
    return out


def fallback_rate_per_cell(df: pd.DataFrame) -> List[Dict]:
    """C6: per (model, domain, condition) summary of API/GPU-failure rate.

    Reported in the analysis JSON so downstream readers can spot per-model
    contamination of the risk_event signal. Computed as:
        - n_games: rows in the cell
        - mean_fallback_count: average per-game count of fallback-sentinel rounds
        - high_fallback_share: share of games with fallback_count > 5 (the same
          threshold used by `--exclude_high_fallback`)
    """
    if "fallback_count" not in df.columns or df.empty:
        return []
    rows = []
    grouped = df.groupby(["model", "domain", "condition"], dropna=False)
    for (model, domain, condition), g in grouped:
        n = len(g)
        fc = g["fallback_count"].fillna(0).astype(int)
        rows.append({
            "model": model,
            "domain": domain,
            "condition": condition,
            "n_games": int(n),
            "mean_fallback_count": float(fc.mean()) if n else 0.0,
            "high_fallback_share": float((fc > 5).sum()) / n if n else 0.0,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio_input_dir", required=True,
                        help="Directory of M1 portfolio final_*.json")
    parser.add_argument("--gambling_input_dirs", nargs="+", default=None,
                        help="Override config gambling_input_dirs")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--no_bambi", action="store_true",
                        help="Skip bambi fit; use bootstrap-only for the primary contrast")
    parser.add_argument(
        "--exclude_high_fallback", action="store_true",
        help="C6: drop games with fallback_count > 5 from the primary contrast. "
             "Reported in summary regardless; this flag controls whether they "
             "feed the bambi/bootstrap fit.",
    )
    args = parser.parse_args()

    cfg = _load_cfg()
    gambling_dirs = args.gambling_input_dirs or cfg["output"].get("gambling_input_dirs", [])
    df_raw = build_combined_df(args.portfolio_input_dir, gambling_dirs)
    if df_raw.empty:
        raise SystemExit(
            f"no usable rows: portfolio_dir={args.portfolio_input_dir}, gambling_dirs={gambling_dirs}"
        )

    fallback_summary = fallback_rate_per_cell(df_raw)

    if args.exclude_high_fallback and "fallback_count" in df_raw.columns:
        df = df_raw[df_raw["fallback_count"].fillna(0).astype(int) <= 5].copy()
    else:
        df = df_raw

    fitted = None
    fit_status = "skipped"
    if not args.no_bambi:
        try:
            fitted = fit_mixed_logit_interaction(df, cfg)
            fit_status = "ok"
        except Exception as e:
            fit_status = f"failed: {e}"

    primary = primary_interaction_logit(df, fitted_model=fitted)
    secondary = secondary_marginal_difference(df)
    per_model = per_model_interactions(df)

    logit_low = primary.get("logit_ci_low")
    if logit_low is not None and not (isinstance(logit_low, float) and math.isnan(logit_low)):
        primary_passes = bool(logit_low > 0)
    else:
        primary_passes = False

    summary = {
        "n_rows_total": int(len(df)),
        "n_rows_pre_fallback_filter": int(len(df_raw)),
        "n_portfolio": int((df["domain"] == "portfolio").sum()),
        "n_gambling": int((df["domain"] == "gambling").sum()),
        "models": sorted(df["model"].dropna().unique().tolist()),
        "primary_contrast": primary,
        "primary_passes": primary_passes,
        "secondary_marginal_diff": secondary,
        "per_model_interactions": per_model,
        "fallback_rate_per_cell": fallback_summary,
        "exclude_high_fallback_applied": bool(args.exclude_high_fallback),
        "bambi_fit_status": fit_status,
    }
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[analyze_m1] wrote {out}")
    print(json.dumps(summary["primary_contrast"], indent=2))


if __name__ == "__main__":
    main()
