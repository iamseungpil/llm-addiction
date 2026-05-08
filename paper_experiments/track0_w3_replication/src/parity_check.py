"""Track 0 W3 protocol parity gate (Plan v5.2 §3.5.1, §8).

Compares the GPT-4o-mini cells emitted by the v6 launcher (`run_track0_api.py`)
against the legacy parity baseline (`run_legacy_baseline.py`). If protocol parity
fails, do NOT launch the cross-model run — fix the v6 code first.

Per-cell pass conditions (8 cells = 4 caps × 2 modes):
  - |bankruptcy_v6 - bankruptcy_legacy| <= 0.05
  - KS distance on `total_rounds` <= 0.10
  - per-cell chi-square on (bankrupt, voluntary_stop, max_rounds) p >= 0.10 / 8
    (Holm correction at family alpha = 0.10), informational fallback only.

Pooled pass condition (the verdict the rebuttal hangs on):
  - Pooled 2x3 chi-square over all 8 cells (rows = arm v6 vs. legacy, columns =
    bankrupt/voluntary_stop/max_rounds) p >= 0.10.

Outputs:
  - {output_path}.json : machine-readable per-cell metrics + pooled chi-square
  - {output_path}.md   : human-readable summary, ends with VERDICT line

Usage:
    python parity_check.py \
        --v6_dir /scratch/x3415a02/data/llm-addiction/track0_w3/ \
        --legacy_dir /scratch/x3415a02/data/llm-addiction/track0_w3/parity_legacy_baseline/ \
        --output_path /scratch/x3415a02/data/llm-addiction/track0_w3/parity_report
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

CAPS = [10, 30, 50, 70]
MODES = ["fixed", "variable"]


def _outcome_label(record: Dict) -> str:
    if record.get("bankrupt"):
        return "bankrupt"
    if record.get("outcome") == "voluntary_stop":
        return "voluntary_stop"
    return "max_rounds"


def _load_cell(path: Path) -> Optional[Dict]:
    """Return parsed payload or None if not parseable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _find_cell_files(base_dir: Path, pattern_prefix: str, cap: int, mode: str) -> List[Path]:
    """Locate JSON files matching `<prefix>_gpt-4o-mini_cap{cap}_{mode}_*.json`."""
    return sorted(base_dir.glob(f"{pattern_prefix}_gpt-4o-mini_cap{cap}_{mode}_*.json"))


def _bankruptcy_rate(records: List[Dict]) -> float:
    if not records:
        return float("nan")
    return float(np.mean([1 if r.get("bankrupt") else 0 for r in records]))


def _outcome_counts(records: List[Dict]) -> Counter:
    return Counter(_outcome_label(r) for r in records)


def _ks_distance_rounds(v6_records: List[Dict], legacy_records: List[Dict]) -> float:
    """Two-sample KS distance on `total_rounds`."""
    a = np.asarray([r.get("total_rounds", 0) for r in v6_records], dtype=float)
    b = np.asarray([r.get("total_rounds", 0) for r in legacy_records], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    res = stats.ks_2samp(a, b)
    return float(res.statistic)


def _per_cell_chi2(v6_counts: Counter, legacy_counts: Counter) -> Tuple[float, float]:
    """Build a 2x3 contingency (rows = v6, legacy; cols = three outcomes)."""
    cols = ["bankrupt", "voluntary_stop", "max_rounds"]
    table = np.array([
        [v6_counts.get(c, 0) for c in cols],
        [legacy_counts.get(c, 0) for c in cols],
    ], dtype=float)
    if table.sum() == 0 or (table.sum(axis=1) == 0).any() or (table.sum(axis=0) == 0).any():
        return float("nan"), float("nan")
    chi2, p, _dof, _exp = stats.chi2_contingency(table)
    return float(chi2), float(p)


def _holm_correction(pvals: List[float], alpha: float) -> List[bool]:
    """Holm step-down adjustment at family alpha. Returns boolean reject flags."""
    valid = [(i, p) for i, p in enumerate(pvals) if not np.isnan(p)]
    valid.sort(key=lambda kv: kv[1])
    m = len(valid)
    reject = [False] * len(pvals)
    for k, (i, p) in enumerate(valid):
        threshold = alpha / (m - k)
        if p < threshold:
            reject[i] = True
        else:
            break
    return reject


def _evaluate_cell(
    cap: int,
    mode: str,
    v6: List[Dict],
    legacy: List[Dict],
) -> Dict:
    v6_rate = _bankruptcy_rate(v6)
    legacy_rate = _bankruptcy_rate(legacy)
    delta = float(abs(v6_rate - legacy_rate)) if not (np.isnan(v6_rate) or np.isnan(legacy_rate)) else float("nan")
    ks = _ks_distance_rounds(v6, legacy)
    v6_counts = _outcome_counts(v6)
    legacy_counts = _outcome_counts(legacy)
    chi2, p = _per_cell_chi2(v6_counts, legacy_counts)

    bankruptcy_pass = (not np.isnan(delta)) and delta <= 0.05
    ks_pass = (not np.isnan(ks)) and ks <= 0.10

    return {
        "cap": cap,
        "mode": mode,
        "n_v6": len(v6),
        "n_legacy": len(legacy),
        "bankruptcy_v6": v6_rate,
        "bankruptcy_legacy": legacy_rate,
        "bankruptcy_delta_abs": delta,
        "bankruptcy_pass": bool(bankruptcy_pass),
        "ks_total_rounds": ks,
        "ks_pass": bool(ks_pass),
        "outcome_counts_v6": dict(v6_counts),
        "outcome_counts_legacy": dict(legacy_counts),
        "per_cell_chi2_stat": chi2,
        "per_cell_chi2_p": p,
    }


def _pool_outcome_counts(cells_v6: List[Counter], cells_legacy: List[Counter]) -> Tuple[float, float]:
    cols = ["bankrupt", "voluntary_stop", "max_rounds"]
    v6_row = np.zeros(3)
    legacy_row = np.zeros(3)
    for c in cells_v6:
        for j, col in enumerate(cols):
            v6_row[j] += c.get(col, 0)
    for c in cells_legacy:
        for j, col in enumerate(cols):
            legacy_row[j] += c.get(col, 0)
    table = np.vstack([v6_row, legacy_row])
    if table.sum() == 0 or (table.sum(axis=1) == 0).any() or (table.sum(axis=0) == 0).any():
        return float("nan"), float("nan")
    chi2, p, _dof, _exp = stats.chi2_contingency(table)
    return float(chi2), float(p)


def _format_cell_md(cell: Dict) -> str:
    lines = []
    head = f"### cap={cell['cap']}, mode={cell['mode']}"
    lines.append(head)
    lines.append(
        f"- n_v6={cell['n_v6']}, n_legacy={cell['n_legacy']}"
    )
    lines.append(
        f"- bankruptcy: v6={cell['bankruptcy_v6']:.3f} legacy={cell['bankruptcy_legacy']:.3f} "
        f"|Δ|={cell['bankruptcy_delta_abs']:.3f} "
        f"({'PASS' if cell['bankruptcy_pass'] else 'FAIL'} <= 0.05)"
    )
    lines.append(
        f"- KS(total_rounds)={cell['ks_total_rounds']:.3f} "
        f"({'PASS' if cell['ks_pass'] else 'FAIL'} <= 0.10)"
    )
    lines.append(
        f"- per-cell chi2={cell['per_cell_chi2_stat']:.2f} p={cell['per_cell_chi2_p']:.3f}"
    )
    lines.append(
        f"- outcomes v6={cell['outcome_counts_v6']} legacy={cell['outcome_counts_legacy']}"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track 0 W3 protocol parity gate"
    )
    parser.add_argument(
        "--v6_dir",
        default="/scratch/x3415a02/data/llm-addiction/track0_w3/",
        help="Directory containing run_track0_api.py outputs (gpt-4o-mini cells)",
    )
    parser.add_argument(
        "--legacy_dir",
        default="/scratch/x3415a02/data/llm-addiction/track0_w3/parity_legacy_baseline/",
        help="Directory containing run_legacy_baseline.py outputs",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output prefix; .json and .md will be written next to it",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Family alpha for per-cell Holm correction and pooled chi-square gate",
    )
    args = parser.parse_args()

    v6_dir = Path(args.v6_dir)
    legacy_dir = Path(args.legacy_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cells: List[Dict] = []
    cells_v6_counts: List[Counter] = []
    cells_legacy_counts: List[Counter] = []
    missing: List[str] = []

    for cap in CAPS:
        for mode in MODES:
            v6_files = _find_cell_files(v6_dir, "final", cap, mode)
            legacy_files = _find_cell_files(legacy_dir, "legacy", cap, mode)
            if not v6_files or not legacy_files:
                missing.append(f"cap={cap} mode={mode} (v6={len(v6_files)} legacy={len(legacy_files)})")
                continue
            v6_payload = _load_cell(v6_files[-1])
            legacy_payload = _load_cell(legacy_files[-1])
            if v6_payload is None or legacy_payload is None:
                missing.append(f"cap={cap} mode={mode} (parse error)")
                continue
            v6_records = v6_payload.get("results", [])
            legacy_records = legacy_payload.get("results", [])
            cell = _evaluate_cell(cap, mode, v6_records, legacy_records)
            cells.append(cell)
            cells_v6_counts.append(_outcome_counts(v6_records))
            cells_legacy_counts.append(_outcome_counts(legacy_records))

    # Per-cell Holm-corrected fallback breakdown.
    per_cell_pvals = [c["per_cell_chi2_p"] for c in cells]
    per_cell_reject = _holm_correction(per_cell_pvals, args.alpha)
    for cell, rej in zip(cells, per_cell_reject):
        cell["per_cell_chi2_holm_reject"] = bool(rej)

    # Pooled chi-square: the gate.
    pooled_chi2, pooled_p = _pool_outcome_counts(cells_v6_counts, cells_legacy_counts)
    pooled_pass = (not np.isnan(pooled_p)) and pooled_p >= args.alpha

    # Per-cell metric pass: bankruptcy AND KS.
    metric_pass = all(c["bankruptcy_pass"] and c["ks_pass"] for c in cells) and not missing

    overall_pass = metric_pass and pooled_pass and not missing
    verdict = "PASS" if overall_pass else "FAIL — fix v6 code, do not launch cross-model run"

    report = {
        "v6_dir": str(v6_dir),
        "legacy_dir": str(legacy_dir),
        "alpha": args.alpha,
        "missing_cells": missing,
        "cells": cells,
        "pooled_chi2_stat": pooled_chi2,
        "pooled_chi2_p": pooled_p,
        "pooled_pass": bool(pooled_pass),
        "metric_pass": bool(metric_pass),
        "overall_pass": bool(overall_pass),
        "verdict": verdict,
    }

    json_path = output_path.with_suffix(".json")
    md_path = output_path.with_suffix(".md")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# Track 0 W3 protocol parity report",
        "",
        f"- v6_dir: `{v6_dir}`",
        f"- legacy_dir: `{legacy_dir}`",
        f"- family alpha: {args.alpha}",
        f"- pooled chi2 stat: {pooled_chi2:.2f}, p = {pooled_p:.3f} "
        f"({'PASS' if pooled_pass else 'FAIL'} >= {args.alpha})",
        f"- per-cell metric pass (bankruptcy and KS): "
        f"{'PASS' if metric_pass else 'FAIL'}",
        "",
    ]
    if missing:
        md_lines.append("## Missing cells")
        for m in missing:
            md_lines.append(f"- {m}")
        md_lines.append("")

    md_lines.append("## Cells")
    md_lines.append("")
    for cell in cells:
        md_lines.append(_format_cell_md(cell))
        md_lines.append("")

    md_lines.append(f"VERDICT: {verdict}")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"[parity_check] wrote {json_path}")
    print(f"[parity_check] wrote {md_path}")
    print(f"[parity_check] VERDICT: {verdict}")


if __name__ == "__main__":
    main()
