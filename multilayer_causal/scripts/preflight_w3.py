#!/usr/bin/env python3
"""Pre-submission gate audit: list every arm's direction-asset gate fields.

RUN_PLAN_W3.md 구현 원칙: 재현-게이트 실패 시 그 축의 arm은 제출 명단에서 자동
제외 — the builder (src/paper_axes.py) records gate_passed inside each npz and
the runner refuses failed assets at load time (_load_steer_assets). This script
surfaces the same verdicts BEFORE `amlt run` so a wave is never staged with
unbuilt assets or silently-failed gates (the W3 audit found assets/w3/ empty
and the bk gates False with nothing reading them).

Usage: python multilayer_causal/scripts/preflight_w3.py [--prefix w3]
Exit 1 if any matched arm references a missing npz or a failed gate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from multilayer_causal.src.registry import load_arms  # noqa: E402

GATE_FIELD_SUFFIXES = ("_achieved", "_target", "_tol")


def _gate_fields(z):
    return {k: z[k].tolist() for k in z.files
            if "gate" in k or k.endswith(GATE_FIELD_SUFFIXES)}


def audit_arm(arm):
    """One verdict row per arm: (arm_id, npz, status, gate_fields).

    status: SUBMIT (gate passed / legacy npz without gate fields / random
    control borrowing only scales / no direction asset), EXCLUDE (recorded
    gate_passed False — runner would refuse with the same verdict), or
    MISSING (asset not built — submission must not proceed).
    """
    npz_rel = arm.get("directions_npz")
    if arm.get("mode") != "steer" or npz_rel is None:
        return arm["id"], npz_rel, "SUBMIT", {"note": "no direction asset"}
    path = REPO_ROOT / npz_rel
    if not path.exists():
        return arm["id"], npz_rel, "MISSING", {}
    z = np.load(path)
    fields = _gate_fields(z)
    if arm.get("direction") == "random":
        fields["note"] = "random control: scales only, gate exempt"
        return arm["id"], npz_rel, "SUBMIT", fields
    if "gate_passed" not in z.files:
        fields["note"] = "pre-gate schema (W1/W2 asset)"
        return arm["id"], npz_rel, "SUBMIT", fields
    status = "SUBMIT" if bool(z["gate_passed"]) else "EXCLUDE"
    return arm["id"], npz_rel, status, fields


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="w3",
                    help="audit arms whose id starts with this (default w3)")
    args = ap.parse_args(argv)
    arms = [a for aid, a in sorted(load_arms().items())
            if aid.startswith(args.prefix)]
    assert arms, f"no arms match prefix {args.prefix!r}"
    rows = [audit_arm(a) for a in arms]
    blockers = [r for r in rows if r[2] in ("EXCLUDE", "MISSING")]
    for arm_id, npz, status, fields in rows:
        print(f"{status:8s} {arm_id:22s} {npz or '-'}")
        for k, v in fields.items():
            print(f"{'':8s} {'':22s}   {k} = {v}")
    print(f"\nsubmit list ({len(rows) - len(blockers)}/{len(rows)}):",
          " ".join(r[0] for r in rows if r[2] == "SUBMIT"))
    if blockers:
        print(f"BLOCKED ({len(blockers)}):",
              " ".join(f"{r[0]}[{r[2]}]" for r in blockers))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
