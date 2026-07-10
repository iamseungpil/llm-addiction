#!/usr/bin/env python3
"""CLI: python run_experiment.py --arm e1_full --gpu 0 [--n 50] [--smoke]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from multilayer_causal.src.registry import load_arms          # noqa: E402
from multilayer_causal.src.runner import run_arm              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default=str(Path(__file__).parent / "out"))
    ap.add_argument("--arms", default=None,
                    help="arms yaml override (else MLC_ARMS_YAML / configs/arms.yaml)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    arms = load_arms(Path(args.arms)) if args.arms else load_arms()
    assert args.arm in arms, f"unknown arm {args.arm}; have {sorted(arms)}"
    run_arm(arms[args.arm], args.out, gpu=args.gpu, n=args.n, smoke=args.smoke)


if __name__ == "__main__":
    main()
