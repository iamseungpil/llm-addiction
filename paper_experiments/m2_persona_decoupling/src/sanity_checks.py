"""Manipulation check for Track A1 / M2 (Plan v4 §2.3).

The framing manipulation must actually move the model — otherwise a primary-contrast
null is uninformative. The pre-registered manipulation check counts gambling/risk
keyword frequency in model rationales per (model, condition, framing) cell:

    +G under role_play_gambler should boost gambling-keyword frequency more than
    +G under first_person.

If this asymmetry does not hold, the role-play framing is too weak to be
distinguishable from first-person and the §2.5 risk fallback applies (pilot 50
games, redesign framing).

Output: a markdown report with per-cell mean keyword counts and a pooled
PASS/FAIL flag per model.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

import yaml


HERE = Path(__file__).resolve().parent

DEFAULT_KEYWORDS = ["bet", "risk", "loss", "win", "gamble", "luck", "chance"]


def _load_keywords() -> List[str]:
    cfg_path = HERE.parent / "configs" / "m2_config.yaml"
    if not cfg_path.exists():
        return DEFAULT_KEYWORDS
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("manipulation_check", {}).get("gambling_keywords", DEFAULT_KEYWORDS)


def _kw_count(text: str, keywords: List[str]) -> int:
    if not text:
        return 0
    low = text.lower()
    return sum(low.count(k) for k in keywords)


def load_long_records(input_dir: str, keywords: List[str]) -> List[Dict]:
    rows = []
    for path in sorted(Path(input_dir).glob("final_*.json")):
        with open(path, "r") as f:
            payload = json.load(f)
        model = payload.get("model")
        condition = payload.get("condition")
        framing = payload.get("framing")
        task = payload.get("task", "SM")
        for record in payload.get("results", []):
            kw = 0
            n_resp = 0
            for round_rec in record.get("rounds", []):
                resp = round_rec.get("response", "") or ""
                if resp:
                    kw += _kw_count(resp, keywords)
                    n_resp += 1
            rows.append({
                "model": model,
                "condition": condition,
                "framing": framing,
                "task": task,
                "kw_total": kw,
                "n_responses": n_resp,
                "kw_per_response": (kw / n_resp) if n_resp else 0.0,
            })
    return rows


def compute_per_model(rows: List[Dict]) -> List[Dict]:
    bucket: Dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["model"],)
        bucket[key][(r["condition"], r["framing"])].append(r["kw_per_response"])

    out = []
    for (m,), cells in sorted(bucket.items()):
        cell_avg: Dict = {k: (mean(v) if v else 0.0) for k, v in cells.items()}
        d_first = cell_avg.get(("+G", "first_person"), 0.0) - cell_avg.get(("BASE", "first_person"), 0.0)
        d_role = cell_avg.get(("+G", "role_play_gambler"), 0.0) - cell_avg.get(("BASE", "role_play_gambler"), 0.0)
        out.append({
            "model": m,
            "kw_per_response_BASE_first": cell_avg.get(("BASE", "first_person"), float("nan")),
            "kw_per_response_+G_first": cell_avg.get(("+G", "first_person"), float("nan")),
            "kw_per_response_BASE_role": cell_avg.get(("BASE", "role_play_gambler"), float("nan")),
            "kw_per_response_+G_role": cell_avg.get(("+G", "role_play_gambler"), float("nan")),
            "delta_first": d_first,
            "delta_role": d_role,
            "manipulation_passes": d_role > d_first,
        })
    return out


def render_markdown(per_model: List[Dict], keywords: List[str]) -> str:
    lines: List[str] = []
    lines.append("# Track A1 / M2 manipulation check")
    lines.append("")
    lines.append(f"Keywords scored: {keywords}")
    lines.append("")
    lines.append("Pass rule: delta_role (+G role - BASE role) > delta_first (+G first - BASE first)")
    lines.append("")
    lines.append("| model | BASE/first | +G/first | BASE/role | +G/role | delta_first | delta_role | passes |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in per_model:
        passes = "PASS" if r["manipulation_passes"] else "FAIL"
        def f(x):
            try:
                return f"{x:.2f}"
            except Exception:
                return "n/a"
        lines.append(
            f"| {r['model']} | {f(r['kw_per_response_BASE_first'])} | "
            f"{f(r['kw_per_response_+G_first'])} | {f(r['kw_per_response_BASE_role'])} | "
            f"{f(r['kw_per_response_+G_role'])} | {f(r['delta_first'])} | "
            f"{f(r['delta_role'])} | {passes} |"
        )
    lines.append("")
    n_pass = sum(1 for r in per_model if r["manipulation_passes"])
    lines.append(f"**Pooled summary: {n_pass} / {len(per_model)} models pass the framing manipulation check.**")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    keywords = _load_keywords()
    rows = load_long_records(args.input_dir, keywords)
    per_model = compute_per_model(rows)
    md = render_markdown(per_model, keywords)
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"[m2/sanity_checks] wrote {out}")
    print(md)


if __name__ == "__main__":
    main()
