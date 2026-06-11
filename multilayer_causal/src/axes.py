"""Build the W0 direction assets from phase_a hidden states (offline, CPU).

Five axes (spec 2026-06-10-w1-section-interleaved-causal-design.md §2), all in
the runner npz format — directions (42, 3584) float32 UNIT rows + scales (42,)
= 0.03 * median natural hidden norm of the estimation subset per layer:

  iba_v2   −G variable rounds, bet_ratio top vs bottom tercile mean diff
           (behavior_axis.py recipe + eval-game exclusion).
  ilc      rounds whose DISPLAYED previous round is a loss; escalated
           (r_t > r_prev) vs de-escalated mean diff over bet rows.
  iec      r >= 0.5 vs r < 0.5 bet rows, contrast WITHIN each prompt_condition
           with >= 20 rows of each class, per-condition deltas averaged.
  readout  per-layer Ridge(alpha=100) on standardized X, target min(r, 1) over
           −G variable bet rounds — "the axis a linear reader uses".
  ic       IC phase_a, risky choice (3, 4) vs safe (1, 2) mean diff.

Eval-game exclusion (iba_v2 / ilc / iec / readout): the W1 eval states are the
first 300 entries of the frozen −G pool (states.load_minusG_states order); all
phase_a rows from those GAMES are dropped from axis estimation so eval states
never leak into the axes (rulebook R1).

Usage:
  python -m multilayer_causal.src.axes --which all --dest-dir multilayer_causal/assets
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

from .behavior_axis import CATALOG, EXPECT, HF_REPO, PHASE_A, load_round_labels
from .states import SEED_BASE, ensure_sm_catalog, load_minusG_states

N_LAYERS, D_MODEL = 42, 3584
N_EVAL = 300  # frozen W1 eval-pool size whose games are excluded from axes
SCALE_FRAC = 0.03
IC_PHASE_A = "sae_features_v3/investment_choice/gemma/checkpoint/phase_a_hidden_states.npz"
IC_META = "sae_features_v3/investment_choice/gemma/checkpoint/phase_a_metadata.json"
# PROVENANCE: extract_all_rounds.PARADIGM_CONFIG["ic"]["files"] — EXACT order;
# the HF copies of that config's data_dir live under v2_role_gemma/.
IC_FILES = [
    "behavioral/investment_choice/v2_role_gemma/gemma_investment_c10_20260225_122319.json",
    "behavioral/investment_choice/v2_role_gemma/gemma_investment_c30_20260225_184458.json",
    "behavioral/investment_choice/v2_role_gemma/gemma_investment_c50_20260226_020821.json",
    "behavioral/investment_choice/v2_role_gemma/gemma_investment_c70_20260226_082029.json",
]
COS_ASSETS = [  # name -> filename, for the L18-23 cos-similarity report
    ("iba_v2", "directions_iba_v2.npz"), ("ilc", "directions_ilc.npz"),
    ("iec", "directions_iec.npz"), ("readout", "directions_readout.npz"),
    ("ic", "directions_ic.npz"), ("G", "directions.npz"),
    ("iba_v1", "directions_iba.npz"),
]


# ---------------------------------------------------------------- pure math

def tercile_contrast(X, values):
    """mean(rows with values in top tercile) − mean(bottom tercile)."""
    values = np.asarray(values, dtype=np.float64)
    assert len(X) == len(values) and len(values) >= 3, "need >=3 aligned rows"
    lo_cut, hi_cut = np.quantile(values, [1 / 3, 2 / 3])
    top, bot = values >= hi_cut, values <= lo_cut
    assert top.any() and bot.any(), "empty tercile"
    return X[top].mean(axis=0) - X[bot].mean(axis=0)


def binary_contrast(X, pos):
    """mean(X[pos]) − mean(X[~pos])."""
    pos = np.asarray(pos, dtype=bool)
    assert len(X) == len(pos) and pos.any() and (~pos).any(), "need both classes"
    return X[pos].mean(axis=0) - X[~pos].mean(axis=0)


def within_group_contrast(X, labels, groups, min_n=20):
    """Average per-group binary contrast over groups with >= min_n of EACH
    class. Returns (delta, n_groups_used). Condition-matched: each group
    contributes equally, killing between-condition mean shifts."""
    labels = np.asarray(labels, dtype=bool)
    groups = np.asarray(groups)
    deltas = []
    for g in sorted(set(groups.tolist())):
        m = groups == g
        pos, neg = (labels & m), (~labels & m)
        if pos.sum() >= min_n and neg.sum() >= min_n:
            deltas.append(X[pos].mean(axis=0) - X[neg].mean(axis=0))
    assert deltas, f"no group has >={min_n} rows of each class"
    return np.mean(deltas, axis=0), len(deltas)


def qualifying_groups(labels, groups, min_n=20):
    """Groups with >= min_n rows of each class (layer-independent, so computed
    once and reused for the scales subset)."""
    labels = np.asarray(labels, dtype=bool)
    groups = np.asarray(groups)
    return sorted(g for g in set(groups.tolist())
                  if (labels & (groups == g)).sum() >= min_n
                  and (~labels & (groups == g)).sum() >= min_n)


def ridge_axis(X, y, alpha=100.0):
    """Unit direction of a standardized Ridge readout, mapped to RAW space.

    The reader fits y ≈ z @ w with z = (x − mu) / sd, i.e.
    y ≈ x @ (w / sd) + const — so the raw-activation-space direction the
    linear reader uses is w / sd (per-feature), then unit-normalized.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    ridge = Ridge(alpha=alpha).fit(scaler.transform(X), y)
    d = ridge.coef_ / scaler.scale_
    return d / (np.linalg.norm(d) + 1e-9)


# ------------------------------------------------- eval-game exclusion pool

def minusG_pool_with_game_ids(catalog_path):
    """(game_id, round_idx) pool replicating states.load_minusG_states EXACTLY
    (same single-file iteration, filters, rounds 2..6, Random(42) shuffle —
    Fisher-Yates depends only on length, so the permutation is identical) but
    carrying the 1-based catalog game index used by load_round_labels."""
    data = json.load(open(catalog_path))
    games = data.get("results", data.get("games", []))
    if isinstance(games, dict):
        games = list(games.values())
    pool, game_counter = [], 0
    for game in games:
        game_counter += 1
        if game.get("bet_type") != "variable":
            continue
        if "G" in game.get("prompt_combo", ""):
            continue
        n_decisions = len(game.get("decisions", []))
        for round_idx in range(2, min(7, n_decisions)):
            pool.append((game_counter, round_idx))
    random.Random(SEED_BASE).shuffle(pool)
    return pool


def excluded_game_ids(pool, n_eval=N_EVAL):
    """Games whose rounds appear in the first n_eval eval-pool entries."""
    return {gid for gid, _ in pool[:n_eval]}


# --------------------------------------------------------------- ILC labels

def ilc_fields_for_game(game):
    """Per valid (non-skip) decision: (displayed_prev_loss, r_prev).

    PROVENANCE: history alignment mirrors extract_all_rounds.SMAdapter
    (bet decisions map 1:1 to history entries; hist_idx advances after bets);
    win flag read exactly like prompts.build_prompt:
    h.get("win", str(h.get("result", "")) == "W").
    """
    history = game.get("history", [])
    valid = [d for d in game.get("decisions", []) if d.get("action") != "skip"]
    out, hist_idx, last_bet_r = [], 0, 0.0
    for dec in valid:
        if 0 < hist_idx <= len(history):
            h = history[hist_idx - 1]
            prev_loss = not h.get("win", str(h.get("result", "")) == "W")
        else:
            prev_loss = False  # round 1: no displayed previous round
        out.append((prev_loss, last_bet_r))
        if dec.get("action") == "bet":
            bet = dec.get("bet")
            bal = dec.get("balance_before", 100)
            last_bet_r = (bet / max(bal, 1)) if bet else 0.0
            hist_idx += 1
    return out


def load_ilc_fields(catalog_path):
    """Row-aligned (load_round_labels order) prev_loss + r_prev arrays."""
    data = json.load(open(catalog_path))
    results = data.get("results", data if isinstance(data, list) else [])
    prev_loss, r_prev = [], []
    for game in results:
        for pl, rp in ilc_fields_for_game(game):
            prev_loss.append(pl)
            r_prev.append(rp)
    assert len(prev_loss) == EXPECT["n_rounds"], f"ilc rows {len(prev_loss)}"
    return np.array(prev_loss, dtype=bool), np.array(r_prev, dtype=np.float64)


# ---------------------------------------------------------------- IC labels

def load_ic_choices(filepaths):
    """Per-row choice list, replicating extract_all_rounds.ICAdapter.load_rounds
    ordering EXACTLY (files in config order, games in file order, decisions in
    order; skip rows with dec['skipped'] or empty full_prompt/actual_prompt)."""
    choices = []
    for fp in filepaths:
        data = json.load(open(fp))
        for game in data["results"]:
            for dec in game.get("decisions", []):
                if dec.get("skipped"):
                    continue
                if not (dec.get("full_prompt") or dec.get("actual_prompt", "")):
                    continue
                choices.append(dec.get("choice"))
    return choices


# ----------------------------------------------------------------- builders

def _per_layer(hs, fn, scale_idx):
    """Apply fn(X_layer) -> delta per layer; unit rows + subset-norm scales."""
    dirs = np.zeros((N_LAYERS, D_MODEL), np.float32)
    scales = np.zeros(N_LAYERS, np.float32)
    for l in range(N_LAYERS):
        X = np.asarray(hs[:, l, :], dtype=np.float32)
        d = fn(X)
        dirs[l] = d / (np.linalg.norm(d) + 1e-9)
        scales[l] = SCALE_FRAC * np.median(np.linalg.norm(X[scale_idx], axis=1))
        print(f"  L{l} done", flush=True)
    return dirs, scales


def build_iba_v2(hs, rows, excluded):
    idx = np.array([i for i, r in enumerate(rows)
                    if r["bet_type"] == "variable" and "G" not in r["combo"]
                    and r["game_id"] not in excluded])
    br = np.array([rows[i]["bet_ratio"] for i in idx])
    lo_cut, hi_cut = np.quantile(br, [1 / 3, 2 / 3])
    print(f"[iba_v2] n={len(idx)} (excl {len(excluded)} games) "
          f"terciles low<={lo_cut:.3f} high>={hi_cut:.3f}")
    dirs, scales = _per_layer(hs, lambda X: tercile_contrast(X[idx], br), idx)
    return dirs, scales, {"n_rows": len(idx), "n_top": int((br >= hi_cut).sum()),
                          "n_bot": int((br <= lo_cut).sum()),
                          "lo_cut": lo_cut, "hi_cut": hi_cut}


def build_ilc(hs, rows, excluded, catalog_path):
    # VARIABLE games only: with fixed $10 bets a loss mechanically raises
    # r (10/(B-10) > 10/B), flooding the escalated class with bet_type
    # signal instead of voluntary loss-chasing (caught in adversarial review).
    prev_loss, r_prev = load_ilc_fields(catalog_path)
    idx = np.array([i for i, r in enumerate(rows)
                    if prev_loss[i] and r["action"] == "bet"
                    and r["bet_type"] == "variable"
                    and r["game_id"] not in excluded])
    esc = np.array([rows[i]["bet_ratio"] > r_prev[i] for i in idx])
    print(f"[ilc] n={len(idx)} displayed-loss bet rows | "
          f"escalated={int(esc.sum())} de-escalated={int((~esc).sum())}")
    dirs, scales = _per_layer(hs, lambda X: binary_contrast(X[idx], esc), idx)
    return dirs, scales, {"n_rows": len(idx), "n_escalated": int(esc.sum()),
                          "n_deescalated": int((~esc).sum())}


def build_iec(hs, rows, excluded):
    idx = np.array([i for i, r in enumerate(rows)
                    if r["action"] == "bet" and r["game_id"] not in excluded])
    labels = np.array([rows[i]["bet_ratio"] >= 0.5 for i in idx])
    groups = np.array([rows[i]["combo"] for i in idx])
    ok = qualifying_groups(labels, groups, min_n=20)
    used = np.isin(groups, ok)
    print(f"[iec] n={len(idx)} bet rows | {len(ok)} conditions qualify "
          f"(>=20/class): {ok} | n_used={int(used.sum())}")
    def fn(X):
        delta, n_g = within_group_contrast(X[idx][used], labels[used],
                                           groups[used], min_n=20)
        assert n_g == len(ok), f"group count drifted: {n_g} != {len(ok)}"
        return delta
    dirs, scales = _per_layer(hs, fn, idx[used])
    return dirs, scales, {"n_rows": int(used.sum()), "n_conditions": len(ok)}


def build_readout(hs, rows, excluded):
    idx = np.array([i for i, r in enumerate(rows)
                    if r["bet_type"] == "variable" and "G" not in r["combo"]
                    and r["action"] == "bet" and r["game_id"] not in excluded])
    y = np.minimum(np.array([rows[i]["bet_ratio"] for i in idx]), 1.0)
    print(f"[readout] n={len(idx)} -G variable bet rows | Ridge(alpha=100)")
    dirs, scales = _per_layer(hs, lambda X: ridge_axis(X[idx], y), idx)
    return dirs, scales, {"n_rows": len(idx)}


def build_ic(token):
    from huggingface_hub import hf_hub_download
    pa = hf_hub_download(HF_REPO, IC_PHASE_A, repo_type="dataset", token=token)
    meta = json.load(open(hf_hub_download(HF_REPO, IC_META,
                                          repo_type="dataset", token=token)))
    assert meta["n_layers"] == N_LAYERS and meta["hidden_dim"] == D_MODEL, meta
    paths = [hf_hub_download(HF_REPO, f, repo_type="dataset", token=token)
             for f in IC_FILES]
    choices = load_ic_choices(paths)
    assert len(choices) == meta["n_rounds"], \
        f"IC rows {len(choices)} != metadata n_rounds {meta['n_rounds']}"
    z = np.load(pa, mmap_mode="r")
    hs = z["hidden_states"]
    assert hs.shape == (meta["n_rounds"], N_LAYERS, D_MODEL), hs.shape
    risky = np.array([c in (3, 4) for c in choices])
    safe = np.array([c in (1, 2) for c in choices])
    used = np.where(risky | safe)[0]
    print(f"[ic] n={len(choices)} rows | risky={int(risky.sum())} "
          f"safe={int(safe.sum())} skipped_none={len(choices) - len(used)}")
    dirs, scales = _per_layer(
        hs, lambda X: X[risky].mean(axis=0) - X[safe].mean(axis=0), used)
    return dirs, scales, {"n_rows": len(used), "n_risky": int(risky.sum()),
                          "n_safe": int(safe.sum())}


# ------------------------------------------------------------------ report

def print_cos_matrix(dest_dir):
    """Pairwise cos similarity at L18-23 among every asset present."""
    loaded = [(name, np.load(Path(dest_dir) / fn)["directions"])
              for name, fn in COS_ASSETS if (Path(dest_dir) / fn).exists()]
    assert loaded, f"no direction assets in {dest_dir}"
    names = [n for n, _ in loaded]
    for layer in range(18, 24):
        print(f"\ncos similarity @ L{layer}  ({' '.join(names)})")
        for ni, di in loaded:
            row = " ".join(f"{float(di[layer] @ dj[layer]):+.3f}"
                           for _, dj in loaded)
            print(f"  {ni:>8}: {row}")


# -------------------------------------------------------------------- main

def main():
    sm_builders = {"iba_v2": build_iba_v2, "ilc": build_ilc,
                   "iec": build_iec, "readout": build_readout}
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True,
                    choices=sorted(sm_builders) + ["ic", "all"])
    ap.add_argument("--dest-dir", required=True)
    ap.add_argument("--skip-ic", action="store_true")
    ap.add_argument("--n-eval", type=int, default=N_EVAL,
                    help="exclude games of pool[0:n_eval] from axis fits "
                         "(W1 discovery=300; W2 confirmatory at offset 300 "
                         "evaluates pool[300:500+] so needs 500)")
    args = ap.parse_args()
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    todo = sorted(sm_builders) + ["ic"] if args.which == "all" else [args.which]
    if args.skip_ic and "ic" in todo:
        todo.remove("ic")

    if any(w in sm_builders for w in todo):
        from huggingface_hub import hf_hub_download
        cat = hf_hub_download(HF_REPO, CATALOG, repo_type="dataset", token=token)
        rows = load_round_labels(cat)
        pool = minusG_pool_with_game_ids(cat)
        # Cross-check the rebuilt pool against the frozen eval-pool loader on
        # the env-root copy of the SAME catalog (rulebook: no silent drift).
        ensure_sm_catalog("gemma")
        ref = load_minusG_states("gemma", n=args.n_eval)
        assert [ri for _, ri in pool[:args.n_eval]] == [ri for _, ri in ref], \
            "rebuilt eval pool diverges from states.load_minusG_states"
        excluded = excluded_game_ids(pool, n_eval=args.n_eval)
        print(f"[axes] eval pool first {args.n_eval} -> "
              f"{len(excluded)} excluded games")
        pa = hf_hub_download(HF_REPO, PHASE_A, repo_type="dataset", token=token)
        z = np.load(pa, mmap_mode="r")
        hs = z["hidden_states"]
        assert hs.shape == (EXPECT["n_rounds"], N_LAYERS, D_MODEL), hs.shape
        for which in [w for w in todo if w in sm_builders]:
            print(f"[axes] building {which} ...", flush=True)
            if which == "ilc":
                dirs, scales, meta = build_ilc(hs, rows, excluded, cat)
            else:
                dirs, scales, meta = sm_builders[which](hs, rows, excluded)
            dest = dest_dir / f"directions_{which}.npz"
            np.savez(dest, directions=dirs, scales=scales,
                     n_excluded_games=len(excluded), **meta)
            print(f"[axes] {which} -> {dest}")

    if "ic" in todo:
        print("[axes] building ic ...", flush=True)
        dirs, scales, meta = build_ic(token)
        dest = dest_dir / "directions_ic.npz"
        np.savez(dest, directions=dirs, scales=scales, **meta)
        print(f"[axes] ic -> {dest}")

    print_cos_matrix(dest_dir)


if __name__ == "__main__":
    main()
