"""Build the I_BA BEHAVIOUR axis from phase_a hidden states (offline, CPU).

Unlike the G-axis (prompt contrast, 50 pairs), this axis contrasts decision
states by the BEHAVIOUR they produced — high vs low bet_ratio terciles within
the −G variable subset (identical prompts condition-wise → no instruction
leak). Estimated from ~thousands of rounds, killing the 50-pair noise caveat.

Row alignment: phase_a rows follow extract_all_rounds.SMAdapter order —
single catalog file, games in file order, non-skip decisions in order.
Three sanity asserts (n_rounds, n_games, n_bankruptcy vs extraction metadata)
guard against misalignment.

Usage:
  python -m multilayer_causal.src.behavior_axis --dest multilayer_causal/assets/directions_iba.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HF_REPO = "llm-addiction-research/llm-addiction"
PHASE_A = "sae_features_v3/slot_machine/gemma/checkpoint/phase_a_hidden_states.npz"
CATALOG = "behavioral/slot_machine/gemma_v4_role/final_gemma_20260227_002507.json"
N_LAYERS, D_MODEL = 42, 3584
EXPECT = {"n_rounds": 21421, "n_games": 3200, "n_bankruptcy": 87}


def load_round_labels(catalog_path):
    """Replicate extract_all_rounds.SMAdapter ordering (PROVENANCE:
    sae_v3_analysis/src/extract_all_rounds.py SMAdapter.load_rounds)."""
    data = json.load(open(catalog_path))
    results = data.get("results", data if isinstance(data, list) else [])
    rows, game_counter = [], 0
    for game in results:
        game_counter += 1
        bet_type = game.get("bet_type", "fixed")
        combo = game.get("prompt_combo", "BASE")
        outcome = game.get("outcome", "")
        valid = [d for d in game.get("decisions", []) if d.get("action") != "skip"]
        for dec in valid:
            action = dec.get("action", "")
            balance = dec.get("balance_before", 100)
            bet = dec.get("bet") if action == "bet" else None
            rows.append({
                "game_id": game_counter, "bet_type": bet_type, "combo": combo,
                "outcome": outcome, "action": action,
                "bet_ratio": (bet / max(balance, 1)) if (action == "bet" and bet) else 0.0,
            })
    n_bk = len({r["game_id"] for r in rows if r["outcome"] == "bankruptcy"})
    assert len(rows) == EXPECT["n_rounds"], f"rounds {len(rows)} != {EXPECT['n_rounds']}"
    assert game_counter == EXPECT["n_games"], f"games {game_counter}"
    assert n_bk == EXPECT["n_bankruptcy"], f"bankruptcies {n_bk}"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    from huggingface_hub import hf_hub_download
    cat = hf_hub_download(HF_REPO, CATALOG, repo_type="dataset", token=token)
    pa = hf_hub_download(HF_REPO, PHASE_A, repo_type="dataset", token=token)
    rows = load_round_labels(cat)
    z = np.load(pa, mmap_mode="r")
    hs = z["hidden_states"]
    assert hs.shape == (EXPECT["n_rounds"], N_LAYERS, D_MODEL), hs.shape

    # −G variable subset: same distribution as the intervention state pool.
    idx = np.array([i for i, r in enumerate(rows)
                    if r["bet_type"] == "variable" and "G" not in r["combo"]])
    br = np.array([rows[i]["bet_ratio"] for i in idx])
    lo_cut, hi_cut = np.quantile(br, [1 / 3, 2 / 3])
    top, bot = idx[br >= hi_cut], idx[br <= lo_cut]
    print(f"subset n={len(idx)} | terciles: low<= {lo_cut:.3f}, high>= {hi_cut:.3f} "
          f"| n_top={len(top)} n_bot={len(bot)}")

    dirs = np.zeros((N_LAYERS, D_MODEL), np.float32)
    scales = np.zeros(N_LAYERS, np.float32)
    for l in range(N_LAYERS):
        X = np.asarray(hs[:, l, :], dtype=np.float32)
        delta = X[top].mean(axis=0) - X[bot].mean(axis=0)
        dirs[l] = delta / (np.linalg.norm(delta) + 1e-9)
        scales[l] = 0.03 * np.median(np.linalg.norm(X[idx], axis=1))
    np.savez(args.dest, directions=dirs, scales=scales,
             n_top=len(top), n_bot=len(bot), lo_cut=lo_cut, hi_cut=hi_cut)
    print(f"I_BA axis -> {args.dest}")

    # Cos similarity with the G-axis (interpretability cross-check)
    g = np.load(os.path.join(os.path.dirname(args.dest), "directions.npz"))
    cos = (dirs * g["directions"]).sum(axis=1)
    print("cos(I_BA axis, G axis) L18-23:", np.round(cos[18:24], 3))

    if not args.no_upload and token:
        from huggingface_hub import HfApi
        HfApi(token=token).upload_file(
            path_or_fileobj=args.dest,
            path_in_repo="experiments/multilayer_causal/assets/directions_iba.npz",
            repo_id=HF_REPO, repo_type="dataset",
            commit_message="I_BA behaviour axis (tercile contrast, -G variable subset)")


if __name__ == "__main__":
    main()
