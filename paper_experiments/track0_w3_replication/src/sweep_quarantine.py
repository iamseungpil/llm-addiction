"""Apply the cell-quality guard to a whole output directory, not just to cells a driver launched.

The driver in `run_mc_ladder.sh` checks each cell as its lane finishes. That check never runs for a
cell whose lane subshell was killed while the python runner kept going — an orphaned runner writes
its output with nobody left to inspect it. This sweep closes that gap and can be run at any time.

Two rejection criteria, the same ones the driver uses:

  * `manifest.api_fallback_responses` above zero. The runner substitutes a stop response when an
    API call fails after its retries, and a substituted stop is indistinguishable in the data from
    a model that chose to walk away.
  * Fewer than 95% of decisions carrying a complete, parseable `Final Decision:` line. A reply cut
    off before its verdict is read by the parser as a stop, so a truncated cell manufactures
    voluntary stopping. This is what a 300-token limit did to the Claude cells.

Rejected cells move to a QUARANTINE subdirectory rather than being renamed in place: a renamed file
still matches the driver's skip glob, which would leave the cell permanently missing on the next
run.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

DECISION = re.compile(r"final decision:?\s*(bet\s*\$?\d+|stop)", re.I)
MIN_COMPLETE_PCT = 95.0


def inspect(path: Path) -> tuple[bool, str]:
    """Return (keep, reason)."""
    try:
        payload = json.load(open(path))
    except Exception as exc:                                  # noqa: BLE001
        return False, f"unreadable ({type(exc).__name__})"

    fallbacks = payload.get("manifest", {}).get("api_fallback_responses", -1)
    total = complete = 0
    for game in payload.get("results", []):
        for rnd in game.get("rounds", []):
            text = rnd.get("response") or ""
            if not text:
                continue
            total += 1
            complete += bool(DECISION.search(text))

    pct = 100.0 * complete / total if total else 0.0
    if fallbacks != 0:
        return False, f"api_fallback_responses={fallbacks}"
    if total and pct < MIN_COMPLETE_PCT:
        return False, f"only {pct:.1f}% of {total} decisions carry a complete verdict"
    return True, f"{pct:.1f}% complete over {total} decisions"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/v-seungplee/data/llm-addiction/mc32")
    ap.add_argument("--apply", action="store_true", help="move rejects; otherwise report only")
    args = ap.parse_args()

    root = Path(args.dir)
    quarantine = root / "QUARANTINE"
    kept = moved = 0
    for path in sorted(root.glob("final_*.json")):
        keep, reason = inspect(path)
        if keep:
            kept += 1
            continue
        moved += 1
        print(f"REJECT {path.name}\n       {reason}")
        if args.apply:
            quarantine.mkdir(exist_ok=True)
            shutil.move(str(path), str(quarantine / path.name))

    verb = "moved" if args.apply else "would move"
    print(f"\n{kept} cells pass, {moved} {verb} to {quarantine}")
    if moved and not args.apply:
        print("re-run with --apply to move them")


if __name__ == "__main__":
    main()
