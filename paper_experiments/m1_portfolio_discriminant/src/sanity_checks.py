"""Sanity checks for the M1 portfolio task.

Plan v4 §4.5 risk: portfolio framing too sober — +G has no room. Two checks confirm
the task is functionally capable of registering risk-seeking before we treat any +G
null as evidence:

  C1. MAX_RISK positive control: under MAX_RISK condition the primary risk_event
      rate (drawdown > 50%) should be HIGHER than under BASE by at least
      `max_risk_min_uplift_over_base` (config). If MAX_RISK doesn't move the
      needle, the task is structurally too sober and the +G null is uninformative.

  C2. Asset-menu sanity: leveraged_etf_3x + single_volatile_stock + otm_call should
      each be picked at least at `asset_menu_min_pick_rate_for_temptation_assets`
      level somewhere in the dataset (any condition). Otherwise the menu is
      functionally cash + bonds + index and the temptation framing is illusory.

Output: a markdown PASS/FAIL table mirroring `track0_w3_replication/sanity_checks.py`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import yaml


HERE = Path(__file__).resolve().parent


def _load_cfg() -> dict:
    with open(HERE.parent / "configs" / "m1_config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_portfolio_records(input_dir: str) -> List[Dict]:
    rows = []
    for path in sorted(Path(input_dir).glob("final_*.json")):
        with open(path, "r") as f:
            payload = json.load(f)
        if payload.get("domain") != "portfolio":
            continue
        model = payload.get("model")
        cond = payload.get("condition")
        for record in payload.get("results", []):
            mean_alloc = _mean_allocation(record)
            rows.append({
                "model": model,
                "condition": cond,
                "risk_event_primary": int(bool(record.get("risk_event_primary_drawdown_50pct"))),
                "risk_event_secondary": int(bool(record.get("risk_event_secondary_temptation_60pct_5rounds"))),
                "mean_allocation": mean_alloc,
                "max_drawdown": record.get("max_drawdown"),
            })
    return rows


def _mean_allocation(record: Dict) -> Dict[str, float]:
    """Average allocation across rounds for one game."""
    history = record.get("history") or []
    if not history:
        return {}
    sums: Dict[str, float] = defaultdict(float)
    n = 0
    for h in history:
        alloc = h.get("allocation") or {}
        if not alloc:
            continue
        for k, v in alloc.items():
            sums[k] += float(v)
        n += 1
    if n == 0:
        return {}
    return {k: sums[k] / n for k in sums}


def check_max_risk_uplift(rows: List[Dict], min_uplift: float) -> Dict:
    """C1: MAX_RISK condition has higher risk_event rate than BASE (per model)."""
    by_model_cond = defaultdict(list)
    for r in rows:
        if r["condition"] in ("BASE", "MAX_RISK"):
            by_model_cond[(r["model"], r["condition"])].append(r["risk_event_primary"])

    per_model = []
    overall_pass = True
    for model in sorted({m for (m, _) in by_model_cond}):
        base = by_model_cond.get((model, "BASE"), [])
        mr = by_model_cond.get((model, "MAX_RISK"), [])
        if not base or not mr:
            per_model.append({"model": model, "base_rate": None, "max_risk_rate": None, "uplift": None,
                              "passes": False, "note": "missing condition"})
            overall_pass = False
            continue
        base_rate = sum(base) / len(base)
        mr_rate = sum(mr) / len(mr)
        uplift = mr_rate - base_rate
        passes = bool(uplift >= min_uplift)
        per_model.append({"model": model, "base_rate": base_rate, "max_risk_rate": mr_rate,
                          "uplift": uplift, "passes": passes, "n_base": len(base), "n_max_risk": len(mr)})
        if not passes:
            overall_pass = False
    return {"per_model": per_model, "overall_pass": overall_pass, "min_uplift_required": min_uplift}


def check_asset_menu_temptation(rows: List[Dict], min_pick_rate: float) -> Dict:
    """C2: temptation assets (leveraged, volatile, lottery) are each picked above min_pick_rate somewhere."""
    temptation_assets = ["leveraged_etf_3x", "single_volatile_stock", "otm_call_or_crypto"]
    by_asset_max = {a: 0.0 for a in temptation_assets}
    for r in rows:
        for a in temptation_assets:
            share = r["mean_allocation"].get(a, 0.0) if r["mean_allocation"] else 0.0
            if share > by_asset_max[a]:
                by_asset_max[a] = share
    overall_pass = all(by_asset_max[a] >= min_pick_rate for a in temptation_assets)
    return {"per_asset_max_share": by_asset_max, "min_pick_rate_required": min_pick_rate, "overall_pass": overall_pass}


def render_markdown(c1: Dict, c2: Dict) -> str:
    lines = []
    lines.append("# M1 sanity checks")
    lines.append("")
    lines.append("C1: MAX_RISK condition risk_event rate > BASE rate by at least the configured uplift.")
    lines.append(f"    minimum uplift required: {c1['min_uplift_required']:.2f}")
    lines.append("")
    lines.append("| model | base_rate | max_risk_rate | uplift | C1 |")
    lines.append("|---|---|---|---|---|")
    for r in c1["per_model"]:
        if r.get("base_rate") is None:
            lines.append(f"| {r['model']} | n/a | n/a | n/a | FAIL ({r.get('note', '')}) |")
            continue
        verdict = "PASS" if r["passes"] else "FAIL"
        lines.append(f"| {r['model']} | {r['base_rate']:.3f} | {r['max_risk_rate']:.3f} | {r['uplift']:+.3f} | {verdict} |")
    lines.append("")
    lines.append(f"C1 overall: **{'PASS' if c1['overall_pass'] else 'FAIL'}**")
    lines.append("")
    lines.append("C2: temptation assets (leveraged_etf_3x, single_volatile_stock, otm_call_or_crypto) each picked above threshold somewhere.")
    lines.append(f"    minimum max-mean-allocation share required per asset: {c2['min_pick_rate_required']:.2f}")
    lines.append("")
    lines.append("| asset | max mean allocation across all games | C2 |")
    lines.append("|---|---|---|")
    for a, v in c2["per_asset_max_share"].items():
        verdict = "PASS" if v >= c2["min_pick_rate_required"] else "FAIL"
        lines.append(f"| {a} | {v:.3f} | {verdict} |")
    lines.append("")
    lines.append(f"C2 overall: **{'PASS' if c2['overall_pass'] else 'FAIL'}**")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    cfg = _load_cfg()
    rows = load_portfolio_records(args.input_dir)
    c1 = check_max_risk_uplift(rows, cfg["sanity_checks"]["max_risk_min_uplift_over_base"])
    c2 = check_asset_menu_temptation(rows, cfg["sanity_checks"]["asset_menu_min_pick_rate_for_temptation_assets"])

    md = render_markdown(c1, c2)
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"[sanity_checks/m1] wrote {out}")
    print(md)


if __name__ == "__main__":
    main()
