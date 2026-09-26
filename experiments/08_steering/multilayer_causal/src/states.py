"""−G slot-machine state pool (frozen, M3'' semantics) + HF catalog bootstrap."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

SEED_BASE = 42
HF_REPO = "llm-addiction-research/llm-addiction"


def behavioral_root() -> Path:
    return Path(os.environ.get("LLM_ADDICTION_BEHAVIORAL_ROOT",
                               "/home/v-seungplee/data/llm-addiction/behavioral"))


def ensure_sm_catalog(model: str = "gemma") -> Path:
    """Download behavioral/slot_machine/{model}_v4_role/*.json from HF if absent.

    Both §3 corpora follow the same prefix: gemma_v4_role (1 file) and
    llama_v4_role (1 file, final_llama_20260315_062428.json — verified on the
    HF dataset 2026-06-11; llama −G variable pool = 3636 states, which covers
    the power-sized w3L anchors: n=878 + offset 0 + 50 runner slack = 928.
    Gate POWER is a separate question — out/llama_gap_estimate.json found
    n=50 power 0.106, hence the 878 anchors in configs/arms.yaml).
    """
    dest = behavioral_root() / "slot_machine" / f"{model}_v4_role"
    if dest.exists() and any(dest.glob("*.json")):
        return dest
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    prefix = f"behavioral/slot_machine/{model}_v4_role/"
    files = [f for f in api.list_repo_files(HF_REPO, repo_type="dataset")
             if f.startswith(prefix) and f.endswith(".json")]
    assert files, f"no catalog files under {prefix} on {HF_REPO}"
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        p = hf_hub_download(HF_REPO, f, repo_type="dataset", token=token)
        # Atomic write: concurrent arm processes glob *.json — a half-written
        # .json must never be visible (caused JSONDecodeError race in e1 v1).
        tmp = dest / (Path(f).name + ".tmp")
        tmp.write_bytes(Path(p).read_bytes())
        os.replace(tmp, dest / Path(f).name)
    print(f"[states] downloaded {len(files)} catalog files → {dest}", flush=True)
    return dest


def exclude_combo_states(states: list, exclude: str) -> list:
    """Drop (game, round_idx) states whose prompt_combo shares ANY char with `exclude`.

    Used by sec4_w14 to build a G∩M-free eval pool so that BOTH the +G and +M
    twins (prompts.twin_combo) are real prompt changes on the SAME matched pool:
    twin_combo is idempotent when the component is already in the base combo, so
    an M-bearing base state silently no-op'd the +M twin (~52% of the W4/W10b
    plusM eval states). Returns states unchanged when `exclude` is falsy.
    """
    if not exclude:
        return states
    return [s for s in states
            if not any(c in s[0].get("prompt_combo", "") for c in exclude)]


def load_minusG_states(model: str, n: int = 200) -> list:
    """§3 −G variable (game, round_idx) pairs.

    PROVENANCE: logic identical to run_m3pp_strong_patching.load_minusG_states
    (rounds 2–6, variable betting, no 'G' in combo, random.Random(42) shuffle).
    Model-agnostic: the llama_v4_role catalog stores the same schema
    (bet_type / prompt_combo / decisions / history), so the gemma filter
    applies unchanged ('BASE' combos have no 'G' and stay in the −G pool).
    """
    path = behavioral_root() / "slot_machine" / f"{model}_v4_role"
    states = []
    for game_file in sorted(path.glob("*.json")):
        d = json.load(open(game_file))
        games = d.get("results", d.get("games", []))
        if isinstance(games, dict):
            games = list(games.values())
        for game in games:
            if game.get("bet_type") != "variable":
                continue
            if "G" in game.get("prompt_combo", ""):
                continue
            n_decisions = len(game.get("decisions", []))
            for round_idx in range(2, min(7, n_decisions)):
                states.append((game, round_idx))
    rng = random.Random(SEED_BASE)
    rng.shuffle(states)
    return states[:n]
