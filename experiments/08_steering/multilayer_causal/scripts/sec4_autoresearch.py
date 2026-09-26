#!/usr/bin/env python3
"""§4 causal-wave autoresearch loop: poll HF -> analyse -> log -> ledger -> GO/NO-GO.

One pass: pull the completed sec4 arm jsonls from HF into the local results dir,
run ``sec4_stats.analyze``, log the summary to W&B (no-op unless enabled), append
a rung row to ``experiments/sec4_causal/INDEX.md``, and print a GO / NO-GO verdict
against the pre-registered §3 criteria (H1 behavioural writes + H2 readout inert
=> GO). NO auto-launch of Phase 2 — expansion is a human gate.

Usage:
  python multilayer_causal/scripts/sec4_autoresearch.py [--once] [--interval 300]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

_MLC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_MLC.parents[0]))  # repo root -> import multilayer_causal.*

from multilayer_causal.src import sec4_stats, wandb_logger  # noqa: E402

HF_REPO = "llm-addiction-research/llm-addiction"
HF_PREFIX = "experiments/sec4_causal/checkpoints/sec4_p0"
RESULTS_DIR = _MLC / "results" / "sec4_p0"
ASSETS_DIR = _MLC / "assets" / "sec4"
INDEX_MD = _MLC / "experiments" / "sec4_causal" / "INDEX.md"


def pull_completed_arms(results_dir: Path) -> int:
    """Download every sec4_p0 arm jsonl from HF into results_dir. Returns the
    number of cells pulled (0 when HF is unreachable — the caller reports it)."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as e:
        print(f"[sec4-auto] huggingface_hub unavailable: {e}")
        return 0
    results_dir.mkdir(parents=True, exist_ok=True)
    try:
        files = HfApi().list_repo_files(HF_REPO, repo_type="dataset")
    except Exception as e:
        print(f"[sec4-auto] HF listing failed: {e}")
        return 0
    cells = [f for f in files
             if f.startswith(HF_PREFIX + "/") and f.endswith(".jsonl")]
    for f in cells:
        try:
            p = hf_hub_download(HF_REPO, f, repo_type="dataset")
            (results_dir / Path(f).name).write_bytes(Path(p).read_bytes())
        except Exception as e:
            print(f"[sec4-auto] pull {f} failed: {e}")
    return len(cells)


def go_nogo(res: dict) -> str:
    """§3 gate: GO iff H1 (behavioural writes) AND H2 (readout inert)."""
    if res["h1_behavioural_writes"] and res["h2_readout_inert"]:
        return "GO"
    return "NO-GO"


def append_index_row(res: dict, n_cells: int, decision: str) -> None:
    ax = res["axes"]

    def rho(a):
        v = ax.get(a, {}).get("spearman")
        return f"{v:.2f}" if isinstance(v, float) and v == v else "nan"

    cos = res.get("cos_read_write", {})
    cos_s = "; ".join(f"{k}={v:.2f}" for k, v in cos.items()) or "n/a"
    nums = (f"beh ρ={rho('behavioural')}/write={sec4_stats._writes(ax['behavioural'])}, "
            f"read ρ={rho('readout')}/write={sec4_stats._writes(ax['readout'])}, "
            f"conf ρ={rho('confound')}, cos_rw {cos_s}, "
            f"null_n={res['null_band']['n']}, cells={n_cells}")
    row = (f"| P0 | sec4_p0 | H1 behavioural writes, H2 readout inert | "
           f"arms_sec4_p0.yaml (Gemma SM) | {nums} | {res['verdict']} | "
           f"{decision} ({datetime.now():%Y-%m-%d %H:%M}) |\n")
    text = INDEX_MD.read_text() if INDEX_MD.exists() else ""
    INDEX_MD.parent.mkdir(parents=True, exist_ok=True)
    INDEX_MD.write_text(text + row)


def one_pass() -> str:
    n_cells = pull_completed_arms(RESULTS_DIR)
    res = sec4_stats.analyze(RESULTS_DIR, ASSETS_DIR,
                             out_json=sec4_stats.OUT_JSON,
                             out_png=sec4_stats.OUT_PNG)
    wandb_logger.init_run("llm-addiction-sec4", "sec4_p0_autoresearch")
    wandb_logger.log_arm("sec4_p0_analysis", None, {
        "verdict_write_confirmed": int(res["verdict"] == "WRITE_CONFIRMED"),
        "h1_behavioural_writes": int(res["h1_behavioural_writes"]),
        "h2_readout_inert": int(res["h2_readout_inert"]),
        "n_cells": n_cells,
    })
    decision = go_nogo(res)
    append_index_row(res, n_cells, decision)
    print(f"[sec4-auto] cells={n_cells} verdict={res['verdict']} "
          f"H1={res['h1_behavioural_writes']} H2={res['h2_readout_inert']} "
          f"-> {decision}")
    return decision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single pass then exit")
    ap.add_argument("--interval", type=int, default=300, help="poll seconds")
    args = ap.parse_args()
    if args.once:
        one_pass()
        return
    while True:
        one_pass()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
