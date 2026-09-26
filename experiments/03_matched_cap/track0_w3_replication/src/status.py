"""One-screen status of the rebuttal work: experiments, data hygiene, and written output.

Written so that every progress report in this session is generated the same way rather than
assembled by hand, which is how stale counts get repeated.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

DATA = Path("/home/v-seungplee/data/llm-addiction")
RB = Path("/home/v-seungplee/llm-addiction/rebuttal_review")
DECISION = re.compile(r"final decision:?\s*(bet\s*\$?\d+|stop)", re.I)


def running(pattern: str) -> int:
    out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if pattern in line and "grep" not in line)


def cell_health(paths: list[str]) -> tuple[int, float]:
    """Return (cells, worst completeness percent)."""
    worst = 100.0
    for p in paths:
        try:
            payload = json.load(open(p))
        except Exception:                                       # noqa: BLE001
            return len(paths), 0.0
        total = complete = 0
        for game in payload.get("results", []):
            for rnd in game.get("rounds", []):
                text = rnd.get("response") or ""
                if not text:
                    continue
                total += 1
                complete += bool(DECISION.search(text))
        if total:
            worst = min(worst, 100.0 * complete / total)
    return len(paths), worst


def main() -> None:
    print(f"=== {time.strftime('%H:%M:%S')} ===")

    mc = sorted(glob.glob(str(DATA / "mc32/final_*.json")))
    e7 = sorted(glob.glob(str(DATA / "e7_factorial/e7_*.json")))
    n_mc, worst_mc = cell_health(mc)
    print(f"mc32 ladder      {n_mc}/64 cells   worst decision completeness {worst_mc:.1f}%")
    print(f"E7 factorial     {len(e7)}/44 cells")
    print(f"runners alive    mc32 {running('run_track0_api.py')}   E7 {running('run_e7.py')}")

    # Which ladder combinations are still missing, so the report says what is left rather than
    # only what is done.
    have = set()
    for p in mc:
        d = json.load(open(p))
        have.add((d["model"], d["cap"], d.get("prompt_combo", "BASE"), d["mode"]))
    missing = []
    for model in ("gpt-4o-mini", "gpt-4.1-mini", "gemini-flash", "claude-haiku-4-5-20251001"):
        for cap in (10, 30, 50, 70):
            for cond in ("BASE", "GMPRW"):
                for mode in ("fixed", "variable"):
                    if (model, cap, cond, mode) not in have:
                        missing.append(f"{model.split('-')[0]}/{cap}/{cond}/{mode}")
    print(f"ladder missing   {len(missing)}" + (f"  e.g. {', '.join(missing[:4])}" if missing else ""))

    for name in ("QUARANTINE", "TRUNCATED_claude_maxtok300", "LEGACY_PARSER_claude"):
        d = DATA / "mc32" / name
        if d.exists():
            print(f"quarantined      {name}: {len(list(d.glob('*.json')))}")

    print()
    for f in sorted((RB / "post").glob("*.md")):
        chars = len(f.read_text())
        age = (time.time() - f.stat().st_mtime) / 60
        flag = "  OVER LIMIT" if chars >= 10000 else ""
        print(f"{f.name:<24} {chars:>6,} chars   touched {age:.0f} min ago{flag}")
    facts = RB / "VERIFIED_FACTS.md"
    if facts.exists():
        print(f"{'VERIFIED_FACTS.md':<24} {len(facts.read_text()):>6,} chars   "
              f"touched {(time.time() - facts.stat().st_mtime)/60:.0f} min ago")


if __name__ == "__main__":
    main()
