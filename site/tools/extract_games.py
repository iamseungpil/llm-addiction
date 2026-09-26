"""Pick the two recorded games replayed in scene 1 and write assets/games.js.

    source ~/.config/secrets/tokens.env        # provides HF_TOKEN
    python3 site/tools/extract_games.py

Source: the released LLaMA-3.1-8B slot-machine run on the Hugging Face dataset
``llm-addiction-research/llm-addiction`` (file ``SOURCE_FILE`` below; 64
conditions x 50 games, 1,600 per betting arm).

Selection rule (deterministic, no randomness):

1. Only the BASE prompt, i.e. no optional prompt module, so nothing in the
   prompt mentions goals, patterns or reward maximisation.
2. Variable-bet games that end in bankruptcy after 6-14 betting rounds and
   whose fixed-bet twin (same prompt and repetition number) stops voluntarily
   after at least 3 rounds.
3. Score = number of betting rounds whose reasoning uses loss-recovery words
   (recover, recoup, win back, get back, make up for, chase) plus the number
   using pattern / streak words (pattern, streak, due, law of averages).
   Highest score wins; ties go to the longer fixed twin, then the lower
   repetition number.

Each betting round also gets a ``tag`` from the same keyword lists, tested
on the excerpt itself (so the tagged words are visible in the bubble):
"chasing losses" for loss-recovery words, else "seeing patterns in chance"
for pattern / streak / due words, else no tag. A keyword preceded within three
words by a negation ("no discernible pattern") does not count.

Excerpts are verbatim: each is a run of at most 20 consecutive words of the
model's response (whitespace collapsed), taken from the sentence with the
most telling keyword. An ellipsis marks every cut. The script asserts that
each excerpt, without its ellipses, is a substring of the response.
"""
import json
import os
import re
from pathlib import Path

REPO = "llm-addiction-research/llm-addiction"
SOURCE_FILE = "behavioral/slot_machine/llama_v4_role/final_llama_20260315_062428.json"
OUT = Path(__file__).resolve().parent.parent / "assets" / "games.js"
MAX_WORDS = 20

RECOVERY = re.compile(r"\b(recover\w*|recoup\w*|win back|get back|make up for|chas\w+)", re.I)
PATTERN = re.compile(r"\b(pattern\w*|streak\w*|due|law of averages)\b", re.I)
STOPPING = re.compile(r"\b(preserve\w*|stop\w*|walk away|cautious)\b", re.I)
KEY_ORDER = (RECOVERY, PATTERN, STOPPING)


def load_games() -> list:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, SOURCE_FILE, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        return data.get("results") or data.get("games")
    return data


def bet_rounds(game: dict) -> list:
    return [d for d in game["decisions"] if d.get("action") == "bet"]


def score(game: dict) -> int:
    rounds = bet_rounds(game)
    return sum(bool(RECOVERY.search(d["response"])) for d in rounds) + sum(
        bool(PATTERN.search(d["response"])) for d in rounds
    )


def excerpt(response: str, order=KEY_ORDER) -> str:
    """At most MAX_WORDS consecutive words around the most telling keyword."""
    text = " ".join(response.split())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chosen, key = sentences[0], None
    for rx in order:
        hit = next((s for s in sentences if rx.search(s)), None)
        if hit:
            chosen, key = hit, rx
            break
    words = chosen.split()
    # The stored responses are cut at a fixed length, so the last word of the
    # last sentence may be a fragment: never show it.
    if chosen is sentences[-1] and not re.search(r"[.!?]$", text):
        words = words[:-1]
    start = 0
    if len(words) > MAX_WORDS and key:
        # index of the word where the keyword starts
        pos = key.search(chosen).start()
        k = len(chosen[:pos].split())
        start = max(0, min(k - 6, len(words) - MAX_WORDS))
        # begin at a clause boundary when one sits between start and the keyword
        commas = [i + 1 for i in range(start, k) if words[i].endswith(",")]
        if commas and start > 0:
            start = commas[0]
    piece = words[start:start + MAX_WORDS]
    body = " ".join(piece)
    assert body in text, "excerpt must be verbatim"
    head = "…" if start > 0 else ""
    complete = start + len(piece) >= len(words) and re.search(r"[.!?]$", body)
    tail = "" if complete else "…"
    if tail:
        body = body.rstrip(",;:")
    return f"{head}{body}{tail}"


TAGS = ((RECOVERY, "chasing losses"), (PATTERN, "seeing patterns in chance"))


NEGATION = re.compile(r"\b(no|not|never|without)\b|n't\b", re.I)


def tag_for(quote: str):
    """First tag whose keyword appears un-negated (no "no"/"not" in the 3 words before it)."""
    for rx, label in TAGS:
        for m in rx.finditer(quote):
            before = " ".join(quote[: m.start()].split()[-3:])
            if not NEGATION.search(before):
                return label
    return None


def pack(game: dict, model: str) -> dict:
    steps = []
    for d in game["decisions"]:
        step = {
            "round": d["round"],
            "action": d["action"],
            "balanceBefore": d["balance_before"],
            "balanceAfter": d["balance_after"],
            "quote": excerpt(d["response"], (STOPPING,) if d["action"] == "stop" else KEY_ORDER),
        }
        if d["action"] == "bet":
            step["bet"] = d["bet"]
            step["result"] = d["result"]
            tag = tag_for(step["quote"])
            if tag:
                step["tag"] = tag
        steps.append(step)
    return {
        "model": model,
        "betType": game["bet_type"],
        "promptCombo": game["prompt_combo"],
        "repetition": game["repetition"],
        "outcome": game["outcome"],
        "rounds": len(bet_rounds(game)),
        "finalBalance": game["decisions"][-1]["balance_after"],
        "steps": steps,
    }


def main() -> None:
    games = load_games()
    twins = {(g["bet_type"], g["prompt_combo"], g["repetition"]): g for g in games}
    candidates = []
    for g in games:
        if g["bet_type"] != "variable" or g["prompt_combo"] != "BASE" or g["outcome"] != "bankruptcy":
            continue
        if not 6 <= len(bet_rounds(g)) <= 14:
            continue
        twin = twins.get(("fixed", g["prompt_combo"], g["repetition"]))
        if not twin or twin["outcome"] != "voluntary_stop" or len(bet_rounds(twin)) < 3:
            continue
        candidates.append((-score(g), -len(bet_rounds(twin)), g["repetition"], g, twin))
    candidates.sort(key=lambda c: c[:3])
    _, _, _, variable, fixed = candidates[0]

    arm_counts = {}
    for g in games:
        key = (g["bet_type"], g["outcome"])
        arm_counts[key] = arm_counts.get(key, 0) + 1

    model = "LLaMA-3.1-8B"
    out = {
        "source": f"huggingface.co/datasets/{REPO} · {SOURCE_FILE}",
        "fixed": pack(fixed, model),
        "variable": pack(variable, model),
        "armBankruptcies": {
            "fixed": arm_counts.get(("fixed", "bankruptcy"), 0),
            "variable": arm_counts.get(("variable", "bankruptcy"), 0),
            "gamesPerArm": sum(1 for g in games if g["bet_type"] == "fixed"),
        },
    }
    OUT.write_text(
        "// Written by tools/extract_games.py. Do not edit by hand.\n"
        "// Two recorded LLaMA-3.1-8B games with the same prompt and repetition number,\n"
        "// one per betting arm. Quotes are verbatim excerpts of the model's reasoning.\n"
        f"window.GAMES = {json.dumps(out, indent=1, ensure_ascii=False)};\n"
    )
    for arm in ("variable", "fixed"):
        g = out[arm]
        print(f"{arm}: prompt {g['promptCombo']} game #{g['repetition']} · {g['rounds']} rounds · {g['outcome']}")
        for s in g["steps"]:
            tag = f"${s['bet']} {s['result']}" if s["action"] == "bet" else s["action"]
            print(f"  r{s['round']:>2} {tag:>8} {s['balanceBefore']:>4}->{s['balanceAfter']:<4} [{s.get('tag', '-')}] {s['quote']}")
    print("arm bankruptcies:", out["armBankruptcies"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
