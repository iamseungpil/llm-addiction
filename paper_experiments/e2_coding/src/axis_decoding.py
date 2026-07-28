"""Does the direction that writes also read?

The paper reports two directions and finds them dissociated. The readout is fitted to predict the
betting indicator from SAE features at layer 22; the behavioural axis is defined from the model's
own betting and is the one that moves behaviour when steered. The paper's write-band readout table
already shows the readout still decodes inside the write band, so the null on writing is not an
artefact of testing at the wrong depth. What no table yet shows is the converse: how well the
direction that writes decodes.

That is the question this script answers, because it decides which of two stories the paper can
tell. If the behavioural axis decodes about as well as the readout, then one direction both reads
and writes, and the read and write halves of the section describe the same object. If it decodes
much worse, the dissociation stands as the paper states it.

Method, kept as close to the paper's read pipeline as a one-dimensional direction allows: the
target is the betting indicator, non-linearly deconfounded against balance and round by a
RandomForest fitted inside each training fold, exactly as the published pipeline does. The
predictor is the projection of the decision-time hidden state onto the direction under test. Folds
are grouped by game. A direction is scored at every layer of the write band and at layer 22, so
the comparison runs on the same grid as the paper's own table.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

SNAP = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--llm-addiction-research--llm-addiction/snapshots"
    "/b4ec4c173164d5dcadb02818847b2dad5e2f98cc"))
BEHAV = SNAP / "behavioral/slot_machine/gemma_v4_role/final_gemma_20260227_002507.json"
HIDDEN = SNAP / "sae_features_v3/slot_machine/gemma/checkpoint/phase_a_hidden_states.npz"
ASSETS = Path("/home/v-seungplee/llm-addiction/multilayer_causal/assets")

BAND = [16, 17, 18, 19, 20, 21]
REFERENCE_LAYER = 22


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (target, balance, round, game) for variable-bet decisions, plus hidden states."""
    payload = json.load(open(BEHAV))
    rows = []
    for game_index, game in enumerate(payload["results"]):
        for dec in game["decisions"]:
            if dec.get("action") == "skip":
                continue
            rows.append((game_index, game["bet_type"], dec.get("action"), dec.get("parsed_bet"),
                         dec.get("balance_before"), dec.get("round")))

    z = np.load(HIDDEN, allow_pickle=True)
    hidden = z["hidden_states"]
    if len(rows) != hidden.shape[0]:
        raise SystemExit(f"join mismatch: {len(rows)} decisions against {hidden.shape[0]} states")

    idx = [i for i, r in enumerate(rows)
           if r[1] == "variable" and r[2] == "bet" and r[3] is not None and r[4]]
    idx = np.asarray(idx)
    y = np.asarray([rows[i][3] / rows[i][4] for i in idx], dtype=float)
    bal = np.asarray([rows[i][4] for i in idx], dtype=float)
    rnd = np.asarray([rows[i][5] for i in idx], dtype=float)
    game = np.asarray([rows[i][0] for i in idx])
    return y, bal, rnd, game, hidden[idx]


def deconfound(y_tr, bal_tr, rn_tr, y_te, bal_te, rn_te):
    """The published pipeline's in-fold RandomForest deconfound, same covariate set."""
    def cov(b, r):
        return np.column_stack([b, r, b ** 2, np.log1p(b), b * r])
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(cov(bal_tr, rn_tr), y_tr)
    return y_tr - rf.predict(cov(bal_tr, rn_tr)), y_te - rf.predict(cov(bal_te, rn_te))


def score(direction, states, y, bal, rnd, game, n_folds=5, seed=42):
    """Grouped-CV R^2 of the deconfounded target on the one-dimensional projection."""
    proj = states @ direction
    games = np.unique(game)
    fold_of = {g: i % n_folds for i, g in enumerate(np.random.default_rng(seed).permutation(games))}
    fold = np.asarray([fold_of[g] for g in game])
    scores = []
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        res_tr, res_te = deconfound(y[tr], bal[tr], rnd[tr], y[te], bal[te], rnd[te])
        x_tr, x_te = proj[tr].reshape(-1, 1), proj[te].reshape(-1, 1)
        mu, sd = x_tr.mean(), x_tr.std() or 1.0
        pred = Ridge(alpha=100.0).fit((x_tr - mu) / sd, res_tr).predict((x_te - mu) / sd)
        scores.append(r2_score(res_te, pred))
    return float(np.mean(scores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/v-seungplee/llm-addiction/paper_experiments/e2_coding/axis_decoding.json")
    args = ap.parse_args()

    y, bal, rnd, game, states = load()
    print(f"variable-bet decisions {len(y)}  games {len(np.unique(game))}")

    axes = {}
    for name, fn in (("behavioural axis", "directions_iba_v2.npz"),
                     ("readout direction", "directions_readout.npz")):
        path = ASSETS / fn
        if path.exists():
            axes[name] = np.load(path)["directions"]
        else:
            print(f"  missing {path}")

    report = {}
    layers = BAND + [REFERENCE_LAYER]
    print(f"\n{'direction':<20}" + "".join(f"{'L'+str(l):>9}" for l in layers))
    for name, mat in axes.items():
        row = []
        for layer in layers:
            d = mat[layer].astype(np.float64)
            n = np.linalg.norm(d)
            row.append(score(d / n, states[:, layer, :].astype(np.float64), y, bal, rnd, game)
                       if n > 0 else float("nan"))
        print(f"{name:<20}" + "".join(f"{v:>9.3f}" for v in row))
        report[name] = dict(zip([f"L{l}" for l in layers], row))

    # A random direction gives the floor this comparison should be read against.
    rng = np.random.default_rng(0)
    floor = []
    for layer in layers:
        vals = []
        for _ in range(5):
            d = rng.normal(size=states.shape[2])
            vals.append(score(d / np.linalg.norm(d), states[:, layer, :].astype(np.float64),
                              y, bal, rnd, game))
        floor.append(float(np.mean(vals)))
    print(f"{'random (5 draws)':<20}" + "".join(f"{v:>9.3f}" for v in floor))
    report["random"] = dict(zip([f"L{l}" for l in layers], floor))

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    print("The paper's own band table gives the SAE readout at 0.105-0.143 across L16-21 and "
          "0.167 at L22 for this indicator; read these single-direction numbers against that.")


if __name__ == "__main__":
    main()
