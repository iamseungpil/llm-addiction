#!/usr/bin/env python3
"""Analyze layer_sweep_3metrics.py output.

For each (model, task, metric):
  - Print per-layer R^2 row
  - Identify peak layer
  - Compare against paper_neural_audit.json claims
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PAPER_AUDIT = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/paper_neural_audit.json")


def load_audit() -> dict:
    if not PAPER_AUDIT.exists():
        return {}
    return json.loads(PAPER_AUDIT.read_text())


def load_sweep(path: str) -> list[dict]:
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def cell_metric_value(cell: dict, metric: str):
    """Extract R^2 (float) or None."""
    if metric not in cell:
        return None
    m = cell[metric]
    if not isinstance(m, dict):
        return None
    v = m.get("r2")
    return v if isinstance(v, (int, float)) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_jsonl")
    args = ap.parse_args()

    rows = load_sweep(args.sweep_jsonl)
    audit = load_audit()
    audit_ilc = audit.get("rq1_ilc", {})
    audit_direct = audit.get("rq1_direct", {})

    # Group by (model, task)
    cells = {}
    for r in rows:
        key = (r["model"], r["task"])
        cells.setdefault(key, []).append(r)

    print("=" * 78)
    print(f"V17-MATCHED SWEEP RESULTS  ({len(rows)} cells from {args.sweep_jsonl})")
    print("=" * 78)

    for (model, task), task_rows in sorted(cells.items()):
        task_rows.sort(key=lambda r: r["layer"])
        print(f"\n--- {model.upper()}/{task.upper()} ---")
        # Header
        layers = [r["layer"] for r in task_rows]
        print(f"  layers: {layers}")
        for metric in ("i_lc", "i_ba", "i_ec"):
            vals = [cell_metric_value(c, metric) for c in task_rows]
            valid_pairs = [(l, v) for l, v in zip(layers, vals) if v is not None]
            if not valid_pairs:
                print(f"    {metric.upper():>5}: all None")
                continue
            best_layer, best_v = max(valid_pairs, key=lambda x: x[1])
            row_str = "  ".join(f"L{l:>2}={v:+.3f}" for l, v in zip(layers, vals)
                                if v is not None)
            print(f"    {metric.upper():>5}: peak L{best_layer:>2} ({best_v:+.3f})")
            print(f"           {row_str}")

            # Compare to paper audit
            audit_key = f"{model}_{task}"
            if metric == "i_lc" and audit_key in audit_ilc:
                a = audit_ilc[audit_key]
                print(f"      paper_neural_audit V17: L{a['layer']} R^2={a['r2']:.3f}")
            elif metric == "i_ba":
                key = f"{model}_{task}_i_ba"
                if key in audit_direct:
                    a = audit_direct[key]
                    print(f"      paper_neural_audit V17: L{a['layer']} R^2={a['r2']:.3f}")
            elif metric == "i_ec":
                key = f"{model}_{task}_i_ec"
                if key in audit_direct:
                    a = audit_direct[key]
                    print(f"      paper_neural_audit V17: L{a['layer']} R^2={a['r2']:.3f}")


if __name__ == "__main__":
    main()
