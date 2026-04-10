#!/usr/bin/env python3
"""Run a layer/rank sweep for RQ2 aligned hidden transfer."""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["gemma", "llama"])
    parser.add_argument("--basis-method", required=True, choices=["readout_pca", "centroid_pca"])
    parser.add_argument("--layers", required=True, help="Comma-separated layer list")
    parser.add_argument("--ranks", required=True, help="Comma-separated rank list")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--tag-prefix", type=str, default="sweep")
    return parser.parse_args()


def main():
    args = parse_args()
    analysis_root = Path(
        os.environ.get(
            "LLM_ADDICTION_ANALYSIS_ROOT",
            "/home/v-seungplee/llm-addiction/sae_v3_analysis",
        )
    )
    script = analysis_root / "src" / "run_rq2_aligned_hidden_transfer.py"
    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    ranks = [x.strip() for x in args.ranks.split(",") if x.strip()]

    print(
        f"[rq2_sweep] model={args.model} basis={args.basis_method} "
        f"layers={layers} ranks={ranks} n_splits={args.n_splits}",
        flush=True,
    )
    for layer in layers:
        for rank in ranks:
            tag = f"{args.tag_prefix}_L{layer}_r{rank}"
            cmd = [
                sys.executable,
                str(script),
                "--model",
                args.model,
                "--layer",
                str(layer),
                "--rank",
                str(rank),
                "--basis-method",
                args.basis_method,
                "--n-splits",
                str(args.n_splits),
                "--pca-dim",
                str(args.pca_dim),
                "--out-tag",
                tag,
            ]
            if args.tasks:
                cmd.extend(["--tasks", args.tasks])
            print("[rq2_sweep] START", shlex.join(cmd), flush=True)
            subprocess.run(cmd, check=True)
            print(f"[rq2_sweep] DONE layer={layer} rank={rank}", flush=True)
    print("[rq2_sweep] COMPLETE", flush=True)


if __name__ == "__main__":
    main()
