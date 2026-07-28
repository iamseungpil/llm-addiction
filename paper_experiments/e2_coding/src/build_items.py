"""Build the human-coding item set.

Sampling rules, fixed before any item was drawn:

  * Source: two arms, so that the human-labelled variable-minus-fixed contrast the
    third decision rule names is actually computable. The variable arm is drawn from
    the matched-cap cells, which the D5 fixed-bet defect never touched. The fixed arm
    is drawn from the 18-cell re-run executed after that defect was corrected, so no
    item comes from a cell that executed the wrong stake. An earlier version of this
    file sampled the variable arm only; the fixed stratum was added on 2026-07-28,
    before any label was collected.
    Only responses that are NOT exactly 500 characters are eligible, because a
    500-character response is the signature of the known truncation and a coder cannot
    judge a mutilated trace.

  * Length confound, found and fixed 2026-07-28, before any label was collected.
    The two source corpora do not store responses the same way. In `track0_w3` five of
    the six models are stored hard-capped at 500 characters -- not one response of
    claude-haiku, gemini-flash, gemma, gpt-4.1-mini or gpt-4o-mini exceeds 500 there --
    and llama is capped in its cap-$10 cells only. `track0_rerun` stores everything in
    full. Excluding responses of exactly 500 characters therefore acted as a filter on
    one arm and not the other: it left the variable arm holding only the replies that
    happened to end below the cap, while the fixed arm kept its long ones. Measured on
    the pools as they were drawn, the eligible variable arm had median 622 / mean 636.4
    characters against the fixed arm's median 772 / mean 772.3 (Cohen's d = +0.648,
    Mann-Whitney p = 6.2e-259), and per model the gap was far worse than that aggregate
    -- claude-haiku 499 vs 1036, gpt-4o-mini 499 vs 1027, gpt-4.1-mini 499 vs 748,
    gemini-flash 498 vs 822. Length tracks how much reasoning is written and therefore
    how often any expression appears: within the variable pool the impaired-control flag
    rate runs 15.2% / 23.4% / 27.7% across length terciles, and within the fixed pool
    illusion-of-control runs 3.6% / 4.9% / 21.0%. A variable-minus-fixed contrast drawn
    that way measures storage, not behaviour.

    Fix: draw both arms from cells that store responses in full. Inside track0 that is
    exactly llama at caps $30/$50/$70, so both arms are restricted to those cells by
    ARM_FILTER below. The two alternatives were checked and rejected on the numbers.
    Imposing a common length window is arithmetically impossible: a window both arms can
    populate must lie under 500 characters, and there the fixed arm holds zero
    self-serving-bias-flagged responses, zero claude-haiku and zero gpt-4o-mini, so four
    of the sixteen frame-by-flag buckets cannot be filled. Drawing from `mc32` or
    `e7_factorial`, which do store both arms in full, was rejected because both corpora
    were still being written while this was written, and an item set drawn from a moving
    corpus is not reproducible from this file -- the property the whole instrument rests
    on. The residual llama length difference that survives (median 717 variable vs 666
    fixed) is left alone deliberately: with storage equalised it is a real property of
    the arms, and discretion buys longer deliberation, so conditioning on it would be
    conditioning on a mediator.

    The restriction also repairs a second mismatch. The fixed re-run covers caps
    $30/$50/$70 only, so the previous draw put 28 variable items at cap $10 against no
    fixed counterpart whatever; both arms now span the same three caps.

    The cost is stated rather than hidden: the contrast is now within-model on llama.
    Five models leave the item set, because track0 holds no full-text variable response
    for any of them. Precision and kappa are therefore estimated on llama as well, and
    no claim about the other five models' traces can rest on this instrument.
  * Frames: the four frozen constructs of `convergent_codebook.FROZEN.py`. The
    quantitative claims rest on those four, so the human validation has to target
    the same four.
  * Within each frame, half the items are ones the regex flagged and half are ones
    it did not. Coding only the flagged ones would estimate precision and leave
    the false-negative rate — the quantity the regex undercoverage actually calls
    into question — unmeasured.
  * Balanced across models by round-robin, so no single model dominates a frame. Since
    the length fix above leaves one model in both arms, the round-robin is currently a
    no-op; it is kept because it is what restores model balance the moment a full-text
    variable corpus exists for the other five models.
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
# Both arms are restricted to the cells that store responses in full, so that the
# 500-character exclusion cannot filter one arm harder than the other. See the module
# docstring for the measurement that forced this. These are the only track0 cells where
# both arms are stored uncapped, and they also put both arms on the same three caps.
ARM_FILTER = {"models": {"llama"}, "caps": {30, 50, 70}}
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
        if model not in ARM_FILTER["models"] or cap not in ARM_FILTER["caps"]:
            continue
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


def length_summary(lengths: list[int]) -> dict:
    """Length distribution of a set of traces, recorded so the confound stays auditable."""
    if not lengths:
        return {"n": 0}
    s = sorted(lengths)

    def pct(p: float) -> int:
        return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]

    return {
        "n": len(s),
        "mean": round(sum(s) / len(s), 1),
        "min": s[0],
        "p10": pct(10),
        "p25": pct(25),
        "median": pct(50),
        "p75": pct(75),
        "p90": pct(90),
        "max": s[-1],
        "n_exactly_500": sum(1 for x in s if x == TRUNCATION_LENGTH),
    }


def draw_arm(arm: str, source_glob: str, seed: int, start_counter: int) -> tuple[list, list, int]:
    """Draw N_PER_FRAME items per frame from one arm.

    Each arm gets its own generator so that adding the fixed arm leaves the variable
    draw exactly as it was.
    """
    rng = random.Random(seed)
    pool = load_pool(source_glob)
    print(f"[{arm}] eligible responses: {len(pool)}")
    print(f"[{arm}] eligible length: {length_summary([len(r['text']) for r in pool])}")

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
                        "n_chars": len(row["text"]),
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

    drawn_lengths = {arm: length_summary([k["n_chars"] for k in key if k["arm"] == arm])
                     for arm in sorted(SOURCE_GLOBS)}

    OUT_ITEMS.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    OUT_KEY.write_text(json.dumps({"seed": SEED,
                                   "source_globs": SOURCE_GLOBS,
                                   "arm_filter": {"models": sorted(ARM_FILTER["models"]),
                                                  "caps": sorted(ARM_FILTER["caps"])},
                                   "drawn_length_summary": drawn_lengths,
                                   "key": key},
                                  ensure_ascii=False, indent=2))
    print(f"\nwrote {len(items)} items -> {OUT_ITEMS}")
    print(f"wrote key            -> {OUT_KEY}")
    for arm in sorted(SOURCE_GLOBS):
        print(f"  [{arm}]")
        for frame in sorted(COMPILED):
            sel = [k for k in key if k["frame"] == frame and k["arm"] == arm]
            nf = sum(1 for k in sel if k["regex_flagged"])
            print(f"    {frame:24s} {len(sel):3d} items ({nf} flagged / {len(sel) - nf} unflagged)")
        print(f"    drawn length: {drawn_lengths[arm]}")
        caps = sorted({k["cap"] for k in key if k["arm"] == arm})
        models = sorted({k["model"] for k in key if k["arm"] == arm})
        print(f"    caps: {caps}   models: {models}")


if __name__ == "__main__":
    main()
