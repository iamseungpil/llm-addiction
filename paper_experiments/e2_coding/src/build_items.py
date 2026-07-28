"""Build the human-coding item set.

Sampling rules, fixed before any item was drawn:

  * Source: two arms, so that the human-labelled variable-minus-fixed contrast the
    third decision rule names is actually computable. The variable arm is drawn from
    the matched-cap cells, which the D5 fixed-bet defect never touched. The fixed arm
    is drawn from the 18-cell re-run executed after that defect was corrected, so no
    item comes from a cell that executed the wrong stake. An earlier version of this
    file sampled the variable arm only; the fixed stratum was added on 2026-07-28,
    before any label was collected, and the variable draw is unchanged because it uses
    its own seeded generator.
    Only responses that are NOT exactly 500 characters are eligible, because a
    500-character response is the signature of the known truncation and a coder cannot
    judge a mutilated trace.
  * Frames: the four frozen constructs of `convergent_codebook.FROZEN.py`. The
    quantitative claims rest on those four, so the human validation has to target
    the same four.
  * Within each frame, half the items are ones the regex flagged and half are ones
    it did not. Coding only the flagged ones would estimate precision and leave
    the false-negative rate — the quantity the regex undercoverage actually calls
    into question — unmeasured.
  * Balanced across models by round-robin, so no single model dominates a frame.
  * Coders see the trace and the frame name only. Model, condition, flag status and
    the matched span are all withheld, so the label cannot be reconstructed from
    anything but the text.

The draw is seeded, so the item set is reproducible from this file alone.
"""

from __future__ import annotations

import glob
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from convergent_codebook import COMPILED, matches  # noqa: E402

SOURCE_GLOBS = {
    # arm -> glob. The fixed arm comes from the post-correction re-run, never from the
    # cells that executed the wrong stake.
    "variable": "/home/v-seungplee/data/llm-addiction/track0_w3/final_*_variable_*.json",
    "fixed": "/home/v-seungplee/data/llm-addiction/track0_rerun/final_*_fixed_*.json",
}
OUT_ITEMS = Path("/home/v-seungplee/llm-addiction/paper_experiments/e2_coding/site/public/items.json")
OUT_KEY = Path("/home/v-seungplee/llm-addiction/paper_experiments/e2_coding/items_key.json")

SEED = 24231  # the submission number; fixed here so the draw is reproducible
N_PER_FRAME = 25
TRUNCATION_LENGTH = 500
MIN_CHARS = 200  # a trace shorter than this carries too little to code


def load_pool(source_glob: str) -> list[dict]:
    pool = []
    for path in sorted(glob.glob(source_glob)):
        payload = json.load(open(path))
        model = payload["model"]
        cap = payload["cap"]
        for game in payload["results"]:
            for rnd in game.get("rounds", []):
                text = (rnd.get("response") or "").strip()
                if len(text) == TRUNCATION_LENGTH or len(text) < MIN_CHARS:
                    continue
                pool.append(
                    {
                        "model": model,
                        "cap": cap,
                        "mode": payload["mode"],
                        "game_id": game.get("game_id"),
                        "round": rnd.get("round"),
                        "text": text,
                    }
                )
    return pool


def draw_arm(arm: str, source_glob: str, seed: int, start_counter: int) -> tuple[list, list, int]:
    """Draw N_PER_FRAME items per frame from one arm.

    Each arm gets its own generator so that adding the fixed arm leaves the variable
    draw exactly as it was.
    """
    rng = random.Random(seed)
    pool = load_pool(source_glob)
    print(f"[{arm}] eligible responses: {len(pool)}")

    items, key = [], []
    counter = start_counter
    for frame in sorted(COMPILED):
        # Bucket by (flagged, model) so the round-robin can balance both at once.
        buckets: dict[tuple[bool, str], list[dict]] = defaultdict(list)
        for row in pool:
            buckets[(matches(row["text"], frame), row["model"])].append(row)
        for bucket in buckets.values():
            rng.shuffle(bucket)

        for flagged in (True, False):
            models = sorted({m for (f, m) in buckets if f == flagged and buckets[(f, m)]})
            if not models:
                print(f"  !! {frame} flagged={flagged}: no candidates")
                continue
            target = N_PER_FRAME // 2 if flagged else N_PER_FRAME - N_PER_FRAME // 2
            taken, i = 0, 0
            while taken < target:
                model = models[i % len(models)]
                bucket = buckets[(flagged, model)]
                i += 1
                if not bucket:
                    if all(not buckets[(flagged, m)] for m in models):
                        print(f"  !! [{arm}] {frame} flagged={flagged}: pool exhausted at {taken}/{target}")
                        break
                    continue
                row = bucket.pop()
                counter += 1
                item_id = f"E2-{counter:03d}"
                items.append({"id": item_id, "frame": frame, "text": row["text"]})
                key.append(
                    {
                        "id": item_id,
                        "frame": frame,
                        "arm": arm,
                        "regex_flagged": flagged,
                        "model": row["model"],
                        "cap": row["cap"],
                        "mode": row["mode"],
                        "game_id": row["game_id"],
                        "round": row["round"],
                    }
                )
                taken += 1

    return items, key, counter


def drop_duplicate_texts(items: list[dict], key: list[dict]) -> tuple[list[dict], list[dict]]:
    """Remove items whose trace text already appears in the set.

    Models emit byte-identical stock responses (refusals in particular), and the same
    text reaching a coder twice would inflate agreement and waste a slot. Kept for the
    first occurrence, so the arm drawn first is not penalised.
    """
    seen: set[str] = set()
    keep_ids: set[str] = set()
    out_items = []
    for item in items:
        if item["text"] in seen:
            continue
        seen.add(item["text"])
        keep_ids.add(item["id"])
        out_items.append(item)
    dropped = len(items) - len(out_items)
    if dropped:
        print(f"dropped {dropped} item(s) whose trace text was already in the set")
    return out_items, [k for k in key if k["id"] in keep_ids]


def main() -> None:
    items: list[dict] = []
    key: list[dict] = []
    counter = 0
    # The variable arm keeps SEED so its draw is byte-identical to the pre-extension
    # version; the fixed arm gets its own seed.
    arm_seeds = {"variable": SEED, "fixed": SEED + 1}
    for arm, source_glob in sorted(SOURCE_GLOBS.items()):
        arm_items, arm_key, counter = draw_arm(arm, source_glob, arm_seeds[arm], counter)
        items.extend(arm_items)
        key.extend(arm_key)

    items, key = drop_duplicate_texts(items, key)

    # Presentation order is shuffled so a coder cannot infer the frame, the arm or the
    # flag status from position. Seeded separately from either draw.
    random.Random(SEED + 100).shuffle(items)

    OUT_ITEMS.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    OUT_KEY.write_text(json.dumps({"seed": SEED, "source_globs": SOURCE_GLOBS, "key": key},
                                  ensure_ascii=False, indent=2))
    print(f"\nwrote {len(items)} items -> {OUT_ITEMS}")
    print(f"wrote key            -> {OUT_KEY}")
    for arm in sorted(SOURCE_GLOBS):
        print(f"  [{arm}]")
        for frame in sorted(COMPILED):
            sel = [k for k in key if k["frame"] == frame and k["arm"] == arm]
            nf = sum(1 for k in sel if k["regex_flagged"])
            print(f"    {frame:24s} {len(sel):3d} items ({nf} flagged / {len(sel) - nf} unflagged)")


if __name__ == "__main__":
    main()
