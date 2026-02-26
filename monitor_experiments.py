#!/usr/bin/env python3
"""
Monitor running experiments and generate analysis report when complete.
Checks both slot machine (GPU 0) and mystery wheel (GPU 1).
"""

import json
import time
import os
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime


SLOT_PID = 366546
WHEEL_PID = 522260

SLOT_DIR = Path("/home/jovyan/beomi/llm-addiction-data/slot_machine/experiment_0_gemma_v3/")
WHEEL_DIR = Path("/home/jovyan/beomi/llm-addiction-data/mystery_wheel/")
REPORT_PATH = Path("/home/jovyan/beomi/llm-addiction-data/logs/experiment_report.txt")


def is_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def get_latest_checkpoint(directory, pattern):
    files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime)
    return files[-1] if files else None


def load_results(filepath):
    with open(filepath) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("results", [])


def analyze_slot_machine(results):
    lines = []
    lines.append("=" * 70)
    lines.append("SLOT MACHINE GEMMA V3 — FINAL REPORT")
    lines.append("=" * 70)
    lines.append(f"Total games: {len(results)}")

    # Outcome
    outcomes = {}
    for r in results:
        o = r.get("outcome", "unknown")
        outcomes[o] = outcomes.get(o, 0) + 1
    lines.append(f"\nOutcome distribution:")
    for o, c in sorted(outcomes.items(), key=lambda x: -x[1]):
        lines.append(f"  {o}: {c} ({c/len(results)*100:.1f}%)")

    # By bet type
    for bt in ["fixed", "variable"]:
        sub = [r for r in results if r.get("bet_type") == bt]
        if not sub:
            continue
        bankrupt = sum(1 for r in sub if r.get("final_balance", 100) <= 0)
        bals = [r.get("final_balance", 0) for r in sub]
        rnds = [r.get("total_rounds", 0) for r in sub]
        lines.append(f"\n{bt.upper()} ({len(sub)} games):")
        lines.append(f"  Bankruptcy: {bankrupt}/{len(sub)} ({bankrupt/len(sub)*100:.1f}%)")
        lines.append(f"  Balance: mean=${np.mean(bals):.1f}, std=${np.std(bals):.1f}, min=${min(bals)}, max=${max(bals)}")
        lines.append(f"  Rounds: mean={np.mean(rnds):.1f}, std={np.std(rnds):.1f}")

        # By prompt condition
        conditions = {}
        for r in sub:
            cond = r.get("prompt_combo", "?")
            if cond not in conditions:
                conditions[cond] = []
            conditions[cond].append(r)

        lines.append(f"\n  Per-condition breakdown ({bt}):")
        lines.append(f"  {'Condition':<12} {'N':>4} {'Bankrupt':>10} {'AvgBal':>8} {'AvgRnd':>8}")
        for cond in sorted(conditions.keys()):
            cr = conditions[cond]
            cb = sum(1 for r in cr if r.get("final_balance", 100) <= 0)
            ab = np.mean([r.get("final_balance", 0) for r in cr])
            ar = np.mean([r.get("total_rounds", 0) for r in cr])
            lines.append(f"  {cond:<12} {len(cr):>4} {cb:>6} ({cb/len(cr)*100:>4.1f}%) ${ab:>6.1f} {ar:>7.1f}")

    # Parse stats
    parse_reasons = {}
    skipped = 0
    total_d = 0
    for r in results:
        for d in r.get("decisions", []):
            total_d += 1
            if d.get("skipped"):
                skipped += 1
                continue
            reason = d.get("parse_reason", "unknown")
            parse_reasons[reason] = parse_reasons.get(reason, 0) + 1
    lines.append(f"\nParse statistics: {total_d} decisions, {skipped} skipped")
    for reason, count in sorted(parse_reasons.items(), key=lambda x: -x[1]):
        lines.append(f"  {reason}: {count} ({count/max(total_d,1)*100:.1f}%)")

    return "\n".join(lines)


def analyze_mystery_wheel(results):
    lines = []
    lines.append("=" * 70)
    lines.append("MYSTERY WHEEL GEMMA — FINAL REPORT")
    lines.append("=" * 70)
    lines.append(f"Total games: {len(results)}")

    # Outcome
    outcomes = {}
    for r in results:
        o = r.get("final_outcome", "unknown")
        outcomes[o] = outcomes.get(o, 0) + 1
    lines.append(f"\nOutcome distribution:")
    for o, c in sorted(outcomes.items(), key=lambda x: -x[1]):
        lines.append(f"  {o}: {c} ({c/len(results)*100:.1f}%)")

    # Zone hits
    zone_hits = {"Red": 0, "Blue": 0, "Gold": 0, "Black": 0}
    for r in results:
        for z, c in r.get("zone_hits", {}).items():
            zone_hits[z] = zone_hits.get(z, 0) + c
    total_spins = sum(zone_hits.values())
    expected = {"Red": 25.0, "Blue": 8.0, "Gold": 2.0, "Black": 65.0}
    lines.append(f"\nZone hit distribution ({total_spins} total spins):")
    for zone in ["Red", "Blue", "Gold", "Black"]:
        pct = zone_hits[zone] / total_spins * 100 if total_spins > 0 else 0
        lines.append(f"  {zone}: {zone_hits[zone]} ({pct:.1f}%) [expected: {expected[zone]:.1f}%]")

    # By bet type
    for bt in ["variable", "fixed"]:
        sub = [r for r in results if r.get("bet_type") == bt]
        if not sub:
            continue
        bankrupt = sum(1 for r in sub if r.get("bankruptcy"))
        vstop = sum(1 for r in sub if r.get("stopped_voluntarily"))
        maxr = sum(1 for r in sub if r.get("max_rounds_reached"))
        bals = [r.get("final_balance", 0) for r in sub]
        rnds = [r.get("rounds_completed", 0) for r in sub]
        lines.append(f"\n{bt.upper()} ({len(sub)} games):")
        lines.append(f"  Voluntary stop: {vstop}/{len(sub)} ({vstop/len(sub)*100:.1f}%)")
        lines.append(f"  Bankruptcy: {bankrupt}/{len(sub)} ({bankrupt/len(sub)*100:.1f}%)")
        lines.append(f"  Max rounds: {maxr}/{len(sub)} ({maxr/len(sub)*100:.1f}%)")
        lines.append(f"  Balance: mean=${np.mean(bals):.1f}, std=${np.std(bals):.1f}, min=${min(bals)}, max=${max(bals)}")
        lines.append(f"  Rounds: mean={np.mean(rnds):.1f}, std={np.std(rnds):.1f}")

        # By prompt condition
        conditions = {}
        for r in sub:
            cond = r.get("prompt_condition", "?")
            if cond not in conditions:
                conditions[cond] = []
            conditions[cond].append(r)

        lines.append(f"\n  Per-condition breakdown ({bt}):")
        lines.append(f"  {'Condition':<12} {'N':>4} {'Bankrupt':>10} {'VStop':>10} {'AvgBal':>8} {'AvgRnd':>8}")
        for cond in sorted(conditions.keys()):
            cr = conditions[cond]
            cb = sum(1 for r in cr if r.get("bankruptcy"))
            vs = sum(1 for r in cr if r.get("stopped_voluntarily"))
            ab = np.mean([r.get("final_balance", 0) for r in cr])
            ar = np.mean([r.get("rounds_completed", 0) for r in cr])
            lines.append(f"  {cond:<12} {len(cr):>4} {cb:>6} ({cb/len(cr)*100:>4.1f}%) {vs:>6} ({vs/len(cr)*100:>4.1f}%) ${ab:>6.1f} {ar:>7.1f}")

    # Parse stats
    parse_reasons = {}
    skipped = 0
    total_d = 0
    for r in results:
        for d in r.get("decisions", []):
            total_d += 1
            if d.get("skipped"):
                skipped += 1
                continue
            reason = d.get("parse_reason", "unknown")
            parse_reasons[reason] = parse_reasons.get(reason, 0) + 1
    lines.append(f"\nParse statistics: {total_d} decisions, {skipped} skipped")
    for reason, count in sorted(parse_reasons.items(), key=lambda x: -x[1]):
        lines.append(f"  {reason}: {count} ({count/max(total_d,1)*100:.1f}%)")

    return "\n".join(lines)


def main():
    print(f"[{datetime.now()}] Monitoring started")
    print(f"  Slot machine PID: {SLOT_PID} (running: {is_running(SLOT_PID)})")
    print(f"  Mystery wheel PID: {WHEEL_PID} (running: {is_running(WHEEL_PID)})")

    slot_done = False
    wheel_done = False

    report_lines = []
    report_lines.append(f"Experiment Report — Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)

    while not (slot_done and wheel_done):
        # Check slot machine
        if not slot_done and not is_running(SLOT_PID):
            print(f"\n[{datetime.now()}] Slot machine finished!")
            slot_done = True

            # Find final file or latest checkpoint
            final_files = sorted(SLOT_DIR.glob("final_*.json"), key=lambda f: f.stat().st_mtime)
            if final_files:
                filepath = final_files[-1]
            else:
                filepath = get_latest_checkpoint(SLOT_DIR, "checkpoint_*.json")

            if filepath:
                print(f"  Analyzing: {filepath}")
                results = load_results(filepath)
                analysis = analyze_slot_machine(results)
                print(analysis)
                report_lines.append("\n" + analysis)
            else:
                msg = "  ERROR: No slot machine result files found"
                print(msg)
                report_lines.append(msg)

        # Check mystery wheel
        if not wheel_done and not is_running(WHEEL_PID):
            print(f"\n[{datetime.now()}] Mystery wheel finished!")
            wheel_done = True

            # Find final file or latest checkpoint
            final_files = sorted(WHEEL_DIR.glob("gemma_mysterywheel_c30_*.json"), key=lambda f: f.stat().st_mtime)
            if final_files:
                filepath = final_files[-1]
            else:
                filepath = get_latest_checkpoint(WHEEL_DIR, "gemma_mysterywheel_checkpoint_*.json")

            if filepath:
                print(f"  Analyzing: {filepath}")
                results = load_results(filepath)
                analysis = analyze_mystery_wheel(results)
                print(analysis)
                report_lines.append("\n" + analysis)
            else:
                msg = "  ERROR: No mystery wheel result files found"
                print(msg)
                report_lines.append(msg)

        if not (slot_done and wheel_done):
            # Status update
            slot_status = "done" if slot_done else f"running (PID {SLOT_PID})"
            wheel_status = "done" if wheel_done else f"running (PID {WHEEL_PID})"
            print(f"[{datetime.now()}] Slot: {slot_status} | Wheel: {wheel_status} — checking again in 5min")
            time.sleep(300)

    # Write report
    report_text = "\n".join(report_lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    print(f"\n[{datetime.now()}] Report saved to: {REPORT_PATH}")
    print("MONITORING COMPLETE")


if __name__ == "__main__":
    main()
