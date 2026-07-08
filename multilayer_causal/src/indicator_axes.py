"""§4 causal-wave direction library: three deconfounded steering axes.

For an indicator k (I_BA / I_LC / I_EC) on a task t we build three candidate
"risk directions" in the residual stream, ALL deconfounded identically
(indicator residualised against balance+round within-fold) so any read-vs-write
difference is not a deconfounding artefact:

  readout      SAE-decoder projection of the §4.1 Ridge readout — the direction
               that best PREDICTS k (the "thermometer"). Reuses the paper_axes
               §4.1 primitives UNMODIFIED (rf_deconfound_split / spearman_topk /
               fit_full_ridge / decoder_map).
  behavioural  mean(top residual-quantile states) - mean(bottom) per layer — the
               direction along which k BEHAVIOUR varies (the candidate "heater").
               The SAME within-fold RF deconfound is applied to the indicator
               BEFORE the top/bottom split.
  confound     OLS(hidden ~ balance+round) coefficient direction per layer — a
               control proving a positive write is not merely "the model was
               told it has more money".

Three array-level cores (pure numpy + sklearn, unit-testable on synthetic
arrays) plus a cached-HF loader (``load_task_arrays``) and a ``main`` CLI that
saves ``assets/sec4/{model}_{task}_{indicator}_{axis}.npz`` with provenance,
decoding AUC and cos_read_write. The loader / CLI reference HF paths and are
exercised only at run time (Task 8), never in the unit tests.

Usage:
  python -m multilayer_causal.src.indicator_axes \
      --model gemma --task slot_machine --indicators i_ba \
      --axes readout,behavioural,confound --layers 16 21 --dest assets/sec4/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import paper_axes as pa

# §4.1 pipeline constants (reuse the paper_axes values so the readout axis is
# built in the SAME feature/ridge geometry as Table 1).
TOP_K = pa.TOP_K
RIDGE_ALPHA = pa.RIDGE_ALPHA
SCALE_FRAC = pa.SCALE_FRAC
SCHEMA_VERSION = pa.SCHEMA_VERSION
N_LAYERS, D_MODEL = pa.N_LAYERS, pa.D_MODEL
EC_THRESHOLD = 0.99  # extreme-choice: bet ratio at/above this = max bet
DEFAULT_SPLITS = 5
# Prompt-set label the axes are built on (the §3 addiction ROLE+G+M corpus).
# Stamped into every axis npz so runner.sec4_prompt_set_hash can assert an arm
# replays the SAME distribution — MUST equal the arms' `prompt_set` default.
BUILD_PROMPT_SET = "addiction_role_gm"

TASK_DIR = {  # SAE-feature / phase_a namespaces on HF (mirror paper_axes)
    "slot_machine": "slot_machine",
    "investment_choice": "investment_choice",
    "mystery_wheel": "mystery_wheel",
}


# --------------------------------------------------------------- helpers

def _n_splits(groups, cap=DEFAULT_SPLITS):
    """GroupKFold splits, clamped to the number of distinct groups (a synthetic
    array with few groups must not ask for more folds than groups)."""
    return int(max(2, min(cap, len(np.unique(np.asarray(groups))))))


def _oriented(v, positive_axis):
    """Sign a unit vector so it points the +indicator way (v @ ref >= 0)."""
    v = pa.unit(v)
    return v if float(v @ pa.unit(positive_axis)) >= 0 else -v


# --------------------------------------------------------- array cores

def build_readout_axis_from_arrays(feats, indicator, balance, rounds, groups,
                                   decoder, layers):
    """Deconfounded §4.1 readout axis from arrays.

    feats (N,K) SAE features, indicator (N,), balance (N,), rounds (N,),
    groups (N,) game ids, decoder (K,d) SAE-decoder rows (or (L,K,d) per-layer),
    layers list of length L. Within each GroupKFold fold: RF-deconfound the
    indicator vs balance+round (paper_axes.rf_deconfound_split), select the
    top-200 features by |Spearman| with the residual (spearman_topk), Ridge-fit
    and predict the held-out fold — the pooled held-out prediction gives the
    decoding AUC. The saved direction is the full-data refit (fit_full_ridge on
    the median-R^2 fold's features) mapped through the decoder (decoder_map),
    unit-normed and replicated to every requested layer.

    Returns {directions (L,d) unit f32, scales (L,) f32, auc float,
    provenance str, n_selected, fold_r2}.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    if decoder is None:
        raise NotImplementedError(
            "readout axis requires an SAE decoder (GemmaScope L22 for gemma; "
            "the LlamaScope per-layer W_dec via build_llama_readout_from_arrays "
            "for llama). A None decoder means the caller built no decoder — the "
            "W8/W9 behavioural hidden mean-diff axis needs none, but the readout "
            "axis cannot be built without one")
    feats = np.asarray(feats, dtype=np.float64)
    indicator = np.asarray(indicator, dtype=np.float64)
    balance = np.asarray(balance, dtype=np.float64)
    rounds = np.asarray(rounds, dtype=np.float64)
    decoder = np.asarray(decoder, dtype=np.float64)
    n_feat = feats.shape[1]
    layers = list(layers)

    fold_idx, fold_r2, preds, labs = [], [], [], []
    gkf = GroupKFold(n_splits=_n_splits(groups))
    for tr, te in gkf.split(feats, groups=groups):
        res_tr, res_te = pa.rf_deconfound_split(
            indicator[tr], balance[tr], rounds[tr],
            indicator[te], balance[te], rounds[te])
        idx = pa.spearman_topk(feats[tr], res_tr, TOP_K)
        sc = StandardScaler()
        Xtr = sc.fit_transform(feats[tr][:, idx])
        Xte = sc.transform(feats[te][:, idx])
        ridge = Ridge(alpha=RIDGE_ALPHA).fit(Xtr, res_tr)
        p = ridge.predict(Xte)
        fold_idx.append(idx)
        fold_r2.append(float(r2_score(res_te, p)))
        preds.append(p)
        labs.append(res_te)

    preds = np.concatenate(preds)
    labs = np.concatenate(labs)
    # held-out decoding AUC: does the readout rank the high-residual half above
    # the low-residual half on genuinely held-out rows.
    bin_lab = (labs > np.median(labs)).astype(int)
    auc = (float(roc_auc_score(bin_lab, preds))
           if bin_lab.min() != bin_lab.max() else float("nan"))

    median_fold = int(np.argsort(fold_r2)[len(fold_r2) // 2])
    sel = fold_idx[median_fold]
    w, scale = pa.fit_full_ridge(feats, indicator, balance, rounds, sel)
    dec_sel = decoder[sel] if decoder.ndim == 2 else None

    directions = np.zeros((len(layers), decoder.shape[-1]), dtype=np.float32)
    for li in range(len(layers)):
        rows = dec_sel if dec_sel is not None else decoder[li][sel]
        directions[li] = pa.decoder_map(w, scale, rows).astype(np.float32)

    return {
        "directions": directions,
        "scales": np.ones(len(layers), dtype=np.float32),  # CLI overrides w/ hidden
        "auc": auc,
        "n_selected": int(len(sel)),
        "fold_r2": np.asarray(fold_r2, dtype=np.float64),
        "provenance": (
            "readout axis: §4.1 top-200 SAE-feature Ridge readout of the "
            "indicator deconfounded vs balance+round within-fold "
            "(rf_deconfound_split/spearman_topk/fit_full_ridge), full-data "
            "refit on the median-R^2 fold's features, mapped through the SAE "
            "decoder rows (decoder_map), unit-normed and replicated to the "
            "requested layers; held-out decoding AUC pooled over GroupKFold."),
    }


def build_llama_readout_from_arrays(feats_by_layer, indicator, balance, rounds,
                                    groups, decoder_by_layer, layers):
    """W10 §4.1 LLaMA readout axis — one readout PER window layer.

    LLaMA has a DISTINCT LlamaScope SAE + decoder at every layer (unlike gemma's
    single replicated L22 readout), and feature index j means different things
    across independently-trained SAEs — so the readout is built INDEPENDENTLY at
    each layer from that layer's OWN feats + W_dec (a single-layer
    build_readout_axis_from_arrays call, reused UNMODIFIED), then the per-layer
    directions are stacked. feats_by_layer / decoder_by_layer are aligned lists
    (one (N,K_l) feats + (K_l,d) decoder per requested layer). Returns
    {directions (L,d) unit f32, scales (L,) ones (CLI overrides), auc (mean
    per-layer held-out decoding AUC), aucs_by_layer (L,), n_selected, provenance}.
    """
    assert len(feats_by_layer) == len(decoder_by_layer) == len(layers), \
        "feats/decoder/layers length mismatch"
    d = np.asarray(decoder_by_layer[0]).shape[1]
    directions = np.zeros((len(layers), d), dtype=np.float32)
    aucs, nsel = [], []
    for li, l in enumerate(layers):
        r = build_readout_axis_from_arrays(
            feats_by_layer[li], indicator, balance, rounds, groups,
            decoder_by_layer[li], [l])
        directions[li] = r["directions"][0]
        aucs.append(r["auc"])
        nsel.append(r["n_selected"])
    return {
        "directions": directions,
        "scales": np.ones(len(layers), dtype=np.float32),  # CLI overrides
        "auc": float(np.nanmean(aucs)) if aucs else float("nan"),
        "aucs_by_layer": np.asarray(aucs, dtype=np.float64),
        "n_selected": int(np.mean(nsel)) if nsel else 0,
        "provenance": (
            "llama per-layer readout axis: §4.1 top-200 SAE-feature Ridge "
            "readout of the indicator deconfounded vs balance+round within-fold "
            "(rf_deconfound_split/spearman_topk/fit_full_ridge), built "
            "INDEPENDENTLY at EACH window layer's own LlamaScope SAE + decoder "
            "(fnlp Llama3_1-8B-Base-L{l}R-8x, decoder.weight.T active subset), "
            "unit-normed; held-out decoding AUC pooled per layer "
            "(aucs_by_layer)."),
    }


def build_behavioural_axis_from_arrays(hidden, indicator, balance, rounds,
                                       groups, layers, q=0.25):
    """Deconfounded behavioural axis: mean(top-q) - mean(bottom-q) per layer.

    hidden (N,L,d) — axis 1 indexes the ``layers`` list positionally.
    The indicator is FIRST residualised against balance+round with the SAME
    within-fold RF as the readout axis (paper_axes.rf_deconfound_split, one
    held-out residual per row via GroupKFold), THEN the top-q / bottom-q rows of
    that residual define the two centroids whose difference is the axis at each
    layer. Per-layer scales = 0.03 * median hidden norm (scales_from_phase_a
    convention). Returns {directions (L,d) unit f32, scales (L,) f32,
    provenance str, q, n_top, n_bot, auc(=nan)}.
    """
    from sklearn.model_selection import GroupKFold

    hidden = np.asarray(hidden, dtype=np.float64)
    indicator = np.asarray(indicator, dtype=np.float64)
    balance = np.asarray(balance, dtype=np.float64)
    rounds = np.asarray(rounds, dtype=np.float64)
    layers = list(layers)
    n, _, d = hidden.shape

    # Held-out RF residual per row — the SAME deconfound the readout applies,
    # so read/write differ by construction, not by preprocessing.
    resid = np.empty(n, dtype=np.float64)
    gkf = GroupKFold(n_splits=_n_splits(groups))
    for tr, te in gkf.split(indicator, groups=groups):
        _, res_te = pa.rf_deconfound_split(
            indicator[tr], balance[tr], rounds[tr],
            indicator[te], balance[te], rounds[te])
        resid[te] = res_te

    lo, hi = np.quantile(resid, [q, 1.0 - q])
    top = resid >= hi
    bot = resid <= lo
    assert top.any() and bot.any(), "empty top/bottom residual quantile"

    directions = np.zeros((len(layers), d), dtype=np.float32)
    scales = np.zeros(len(layers), dtype=np.float32)
    for li in range(len(layers)):
        X = hidden[:, li, :]
        delta = X[top].mean(axis=0) - X[bot].mean(axis=0)
        directions[li] = pa.unit(delta).astype(np.float32)
        scales[li] = SCALE_FRAC * float(np.median(np.linalg.norm(X, axis=1)))

    return {
        "directions": directions,
        "scales": scales,
        "auc": float("nan"),  # behavioural axis is a contrast, not a decoder
        "q": float(q),
        "n_top": int(top.sum()),
        "n_bot": int(bot.sum()),
        "provenance": (
            "behavioural axis: indicator residualised vs balance+round with the "
            "same within-fold RF as the readout (held-out residual per row), "
            "then per layer mean(top-q residual states) - mean(bottom-q); "
            "unit-normed; scales = 0.03*median per-layer hidden norm."),
    }


def build_confound_axis_from_arrays(hidden, balance, rounds, layers):
    """Confound axis: OLS(hidden ~ balance+round) coefficient direction / layer.

    balance and round are standardised so their slopes are comparable; the
    per-layer axis is the leading right-singular vector of the (2,d) slope
    matrix (the principal balance+round direction), oriented balance-increasing.
    Returns {directions (L,d) unit f32, scales (L,) f32, provenance str}.
    """
    hidden = np.asarray(hidden, dtype=np.float64)
    b = np.asarray(balance, dtype=np.float64)
    r = np.asarray(rounds, dtype=np.float64)
    layers = list(layers)
    d = hidden.shape[2]

    def _z(x):
        s = x.std()
        return (x - x.mean()) / (s if s > 1e-12 else 1.0)

    A = np.column_stack([_z(b), _z(r), np.ones_like(b)])
    directions = np.zeros((len(layers), d), dtype=np.float32)
    scales = np.zeros(len(layers), dtype=np.float32)
    for li in range(len(layers)):
        X = hidden[:, li, :]
        coef, *_ = np.linalg.lstsq(A, X, rcond=None)  # (3, d)
        slopes = coef[:2]                              # balance, round slopes
        _, _, vt = np.linalg.svd(slopes, full_matrices=False)
        axis = pa.unit(vt[0])
        if float(slopes[0] @ axis) < 0:  # orient balance-increasing
            axis = -axis
        directions[li] = axis.astype(np.float32)
        scales[li] = SCALE_FRAC * float(np.median(np.linalg.norm(X, axis=1)))

    return {
        "directions": directions,
        "scales": scales,
        "auc": float("nan"),
        "provenance": (
            "confound axis: per layer, leading right-singular vector of the "
            "OLS(hidden ~ standardised balance+round) slope matrix, oriented "
            "balance-increasing; scales = 0.03*median per-layer hidden norm."),
    }


# ------------------------------------------------- shared (common) axis

# Right singular values within this relative tolerance are treated as tied; a
# tie means no single indicator direction dominates, so the shared axis falls
# back to the centroid (equal-weight mean) rather than an arbitrary degenerate
# pick — this is what makes orthogonal indicators yield an intermediate axis.
SHARED_TIE_TOL = 1e-6


def build_shared_axis_from_axes(axes_list, layers):
    """SVD top-1 SHARED component per layer across per-indicator behavioural axes.

    axes_list is a list of per-indicator behavioural axis stacks, each (L, d)
    unit (e.g. [i_ba_behavioural['directions'], i_ec_behavioural['directions']]).
    Per layer: orient every indicator row to a common reference (their centroid),
    stack into an (n_ind, d) matrix, SVD, and take right singular vector 1 — the
    one common direction the indicators most share. When the top two singular
    values are tied (no dominant shared component, e.g. orthogonal indicators)
    the axis falls back to the oriented centroid so it stays equidistant from
    each indicator instead of collapsing onto one. Unit-normed per layer.

    Returns {directions (L,d) unit f32, singular_energy (L,) f64 = s1^2/sum(s^2)
    fraction of variance the shared axis captures, provenance str}.
    """
    stacks = [np.asarray(a, dtype=np.float64) for a in axes_list]
    assert len(stacks) >= 2, "shared axis needs >= 2 indicator axes"
    layers = list(layers)
    n_layer = len(layers)
    d = stacks[0].shape[1]

    directions = np.zeros((n_layer, d), dtype=np.float32)
    singular_energy = np.zeros(n_layer, dtype=np.float64)
    for li in range(n_layer):
        rows = np.stack([pa.unit(s[li]) for s in stacks])  # (n_ind, d)
        centroid = rows.mean(axis=0)
        ref = pa.unit(centroid) if np.linalg.norm(centroid) > 1e-9 else rows[0]
        rows = np.stack([r if float(r @ ref) >= 0 else -r for r in rows])
        centroid = rows.mean(axis=0)  # re-centroid after orientation

        _, s, vt = np.linalg.svd(rows, full_matrices=False)
        s2 = s ** 2
        singular_energy[li] = float(s2[0] / max(float(s2.sum()), 1e-12))
        tied = len(s) > 1 and (s[0] - s[1]) <= SHARED_TIE_TOL * max(s[0], 1e-12)
        v = pa.unit(centroid) if tied else vt[0]
        if float(v @ centroid) < 0:  # orient the shared axis the +indicator way
            v = -v
        directions[li] = pa.unit(v).astype(np.float32)

    return {
        "directions": directions,
        "singular_energy": singular_energy,
        "provenance": (
            "shared axis: per layer, indicator behavioural rows oriented to their "
            "centroid then SVD; right singular vector 1 (the common component), "
            "falling back to the oriented centroid on a singular-value tie; "
            "unit-normed. singular_energy = s1^2/sum(s^2) shared-ness fraction."),
    }


# ------------------------------------------------------- indicators

def _lc_ec_from_iba(bet_ratio, balances, game_ids, rounds):
    """Derive I_LC (loss-chasing) and I_EC (extreme-choice) from the per-row
    I_BA bet ratio and the game/round bookkeeping.

    I_EC = 1 when the bet ratio is at/above EC_THRESHOLD (a max bet), else 0.
    I_LC = relative bet increase over the previous round OF THE SAME GAME when
    that previous round was a loss (balance fell), else 0 — the loss-chasing
    slope. Rows without a valid predecessor get 0 (no chase defined).
    """
    bet_ratio = np.asarray(bet_ratio, dtype=np.float64)
    balances = np.asarray(balances, dtype=np.float64)
    game_ids = np.asarray(game_ids)
    rounds = np.asarray(rounds, dtype=np.float64)
    i_ec = (bet_ratio >= EC_THRESHOLD).astype(np.float64)
    i_lc = np.zeros_like(bet_ratio)
    # index previous round within game by (game_id, round-1)
    key = {(int(g), int(rn)): i for i, (g, rn) in enumerate(zip(game_ids, rounds))}
    for i, (g, rn) in enumerate(zip(game_ids, rounds)):
        j = key.get((int(g), int(rn) - 1))
        if j is None:
            continue
        loss = balances[i] < balances[j]
        r_prev = bet_ratio[j]
        if loss and r_prev > 0 and np.isfinite(bet_ratio[i]):
            i_lc[i] = max(0.0, (bet_ratio[i] - r_prev) / r_prev)
    return i_lc, i_ec


def _discovery_mask(game_ids, held_out=False):
    """DISCOVERY / HELD-OUT split by a stable hash of the game id (last hex
    digit even = discovery, odd = held-out). Deterministic, prompt-independent."""
    def _bit(g):
        h = hashlib.md5(str(int(g)).encode()).hexdigest()
        return int(h[-1], 16) % 2
    bits = np.array([_bit(g) for g in np.asarray(game_ids)])
    return (bits == 1) if held_out else (bits == 0)


def _iba_from_catalog(meta, task, model):
    """Per-row I_BA for ic/mw by joining the behavioural catalog decisions —
    the paper pipeline's exact rule (run_comprehensive_robustness.compute_iba):
    sequential game ids from extraction order (i+1), non-skip decisions,
    bet = parsed_bet|bet|bet_amount, bal = balance_before|balance (fallback:
    meta balances), ratio = min(bet/bal, 1).
    Returns (i_ba, balances, choices, bets); bets is the raw parsed bet per row
    (NaN when the catalog carries none) — the Wave-6 MW spin/stop labels need
    it because a stop row has no bet, so i_ba alone cannot tell 'stop' apart
    from 'unjoinable'."""
    import json as _json
    if task == "investment_choice":
        from .ic import ensure_ic_catalog
        cdir = ensure_ic_catalog(model)
        files = sorted(cdir.glob("*.json"))
    elif task == "mystery_wheel":
        from .mw import ensure_mw_catalog
        cdir = ensure_mw_catalog(model)
        files = sorted(cdir.glob(f"{model}_mysterywheel_*.json"))
    else:
        raise ValueError(f"_iba_from_catalog: unexpected task {task}")
    games = []
    for f in files:
        d = _json.load(open(f))
        res = d.get("results", d.get("games", []))
        games.extend(res.values() if isinstance(res, dict) else res)
    assert games, f"empty {task} catalog under {cdir}"
    game_map = {i + 1: g for i, g in enumerate(games)}

    n = len(meta["game_ids"])
    i_ba = np.full(n, np.nan)
    balances = (np.asarray(meta["balances"], dtype=np.float64).copy()
                if meta.get("balances") is not None else np.full(n, np.nan))
    choices = np.full(n, np.nan)
    bets = np.full(n, np.nan)
    for i in range(n):
        g = game_map.get(int(meta["game_ids"][i]))
        if g is None:
            continue
        raw = g.get("decisions", g.get("history", g.get("rounds", [])))
        decs = [d for d in raw
                if d.get("action") != "skip" and not d.get("skipped", False)]
        rn = int(meta["round_nums"][i]) - 1
        if rn >= len(decs):
            continue
        dec = decs[rn]
        if dec.get("choice") is not None:
            choices[i] = float(dec["choice"])
        bet = dec.get("parsed_bet") or dec.get("bet") or dec.get("bet_amount")
        bal = dec.get("balance_before") or dec.get("balance")
        try:
            bet = float(bet)
            bal = float(bal) if bal is not None else float(balances[i])
        except (TypeError, ValueError):
            continue
        bets[i] = bet
        if bal > 0 and bet > 0:
            i_ba[i] = min(bet / bal, 1.0)
            balances[i] = bal
    return i_ba, balances, choices, bets


def _mw_rc_labels(choices, bets):
    """Wave-6 MW spin-vs-stop contrast labels (the catalog's verified choice
    semantics: choice 2 WITH bet>0 = SPIN n=8948 == paper Table-1 MW n,
    choice 1 = STOP n=3146, choice 2 without a bet = parse edge -> no label).

    i_rc = 1.0 spin (choice==2 and bet>0), 0.0 stop (choice==1), NaN otherwise.
    A NaN bet never labels a spin (NaN>0 is False), so parse-edge rows stay
    unlabeled instead of leaking into either class."""
    choices = np.asarray(choices, dtype=np.float64)
    bets = np.asarray(bets, dtype=np.float64)
    rc = np.full(len(choices), np.nan)
    rc[choices == 1.0] = 0.0
    with np.errstate(invalid="ignore"):
        rc[(choices == 2.0) & (bets > 0)] = 1.0
    return rc


def _keep_mask(split, i_ba, i_rc, rc_keep=False):
    """Row-keep rule for load_task_arrays. Default (frozen Wave-1..5 rule):
    split & isfinite(i_ba). rc_keep (Wave-6 MW build ONLY): additionally keep
    rows with a finite spin/stop label — stop rows carry no bet, so i_ba is
    NaN and the default rule drops the entire stop class the mw_rc contrast
    needs. The default path is bit-identical to the pre-Wave-6 mask, so every
    earlier wave build sees exactly the same rows."""
    split = np.asarray(split, dtype=bool)
    if rc_keep:
        return split & (np.isfinite(i_ba) | np.isfinite(i_rc))
    return split & np.isfinite(i_ba)


def _build_wave5_axes(task_data, layers):
    """Wave-5 Q2 refinement (goals-v2 ladder rung-3, expression-matched AXIS
    BUILD): IC's amount-ratio I_BA is ~29% constraint-variance (c10-c70 sets
    the invested fraction), so the Wave-3/4 IC axis partly encodes the
    CONSTRAINT, not decision-level risk appetite. Rebuild the IC axis on the
    risky-OPTION contrast (i_rc = choice>=3, constraint-free, and
    expression-matched to the measured IC outcome), then recompute the shared
    component: shared3b = SVD-top1 of (sm_iba, ic_rc, mw_iba). Emits sm/ic/mw
    sigma-unit variants of shared3b and prints the cosines that decide whether
    the choice-level IC axis aligns with the other tasks' risk axes.
    Returns ({name: built}, cos_pairs)."""
    sm = build_behavioural_axis_from_arrays(
        task_data["slot_machine"]["hidden"],
        task_data["slot_machine"]["indicators"]["i_ba"],
        task_data["slot_machine"]["balance"], task_data["slot_machine"]["rounds"],
        task_data["slot_machine"]["groups"], layers)
    mw = build_behavioural_axis_from_arrays(
        task_data["mystery_wheel"]["hidden"],
        task_data["mystery_wheel"]["indicators"]["i_ba"],
        task_data["mystery_wheel"]["balance"], task_data["mystery_wheel"]["rounds"],
        task_data["mystery_wheel"]["groups"], layers)
    ic_d = task_data["investment_choice"]
    rc = np.asarray(ic_d["indicators"]["i_rc"], dtype=np.float64)
    m = np.isfinite(rc)
    assert m.sum() >= 500, f"too few IC rows with a choice label ({m.sum()})"
    ic_rc = build_behavioural_axis_from_arrays(
        ic_d["hidden"][m], rc[m], ic_d["balance"][m], ic_d["rounds"][m],
        ic_d["groups"][m], layers)
    # the old amount-ratio IC axis, for the diagnostic cosine only
    ic_iba = build_behavioural_axis_from_arrays(
        ic_d["hidden"], ic_d["indicators"]["i_ba"], ic_d["balance"],
        ic_d["rounds"], ic_d["groups"], layers)
    cos_pairs = {}
    for name_a, a, name_b, b in (("ic_rc", ic_rc, "sm_iba", sm),
                                 ("ic_rc", ic_rc, "mw_iba", mw),
                                 ("ic_rc", ic_rc, "ic_iba", ic_iba)):
        cs = [float(pa.unit(a["directions"][li]) @ pa.unit(b["directions"][li]))
              for li in range(len(layers))]
        cos_pairs[f"{name_a}~{name_b}"] = float(np.mean(cs))
        print(f"[sec4-axes/wave5] cos({name_a}, {name_b}) = {np.mean(cs):.4f}",
              flush=True)
    shared = build_shared_axis_from_axes(
        [sm["directions"], ic_rc["directions"], mw["directions"]], layers)
    shared["auc"] = float("nan")
    built = {"ic_rc": dict(ic_rc)}
    for tag, src in (("shared3b", sm), ("shared3b_icscale", ic_rc),
                     ("shared3b_mwscale", mw)):
        v = dict(shared)
        v["scales"] = src["scales"].copy()
        v["provenance"] = (shared["provenance"]
                           + f" {tag} sigma-units (Wave-5 choice-level IC).")
        built[tag] = v
    return built, cos_pairs


def _build_wave6_axes(task_data, layers):
    """Wave-6 Q2 rung (MW object fix): the W4+5 adjudication showed MW's
    amount-ratio axis is the WRONG object (same anomaly signature IC had
    before its Wave-5 choice-level fix: -3 raises betting, parse craters at
    +3). MW's second behavioural expression is CONTINUE-vs-STOP, so rebuild
    the MW axis on the spin/stop contrast (i_rc: choice2+bet>0=1, choice1=0 —
    requires the rc_keep loader branch because stop rows have no bet) and
    recompute the shared component shared3c = SVD-top1(sm_iba, ic_rc, mw_rc).

    task_data must hold slot_machine/investment_choice loaded with the DEFAULT
    keep rule and mystery_wheel loaded with rc_keep=True. Prints (a) the
    mw_rc diagnostic cosines vs mw_iba (rebuilt on the iba_finite subset =
    exactly the Wave-3/5 MW rows), sm_iba, ic_rc, and (b) the shared3c
    loadings on each of its three inputs — the W7 pre-registration numbers
    that predict the per-task steering signs. Returns ({name: built},
    cos_pairs) with mw/ic/sm sigma-unit shared3c variants (mirrors wave5)."""
    sm = build_behavioural_axis_from_arrays(
        task_data["slot_machine"]["hidden"],
        task_data["slot_machine"]["indicators"]["i_ba"],
        task_data["slot_machine"]["balance"], task_data["slot_machine"]["rounds"],
        task_data["slot_machine"]["groups"], layers)
    ic_d = task_data["investment_choice"]
    rc = np.asarray(ic_d["indicators"]["i_rc"], dtype=np.float64)
    m = np.isfinite(rc)
    assert m.sum() >= 500, f"too few IC rows with a choice label ({m.sum()})"
    ic_rc = build_behavioural_axis_from_arrays(
        ic_d["hidden"][m], rc[m], ic_d["balance"][m], ic_d["rounds"][m],
        ic_d["groups"][m], layers)
    mw_d = task_data["mystery_wheel"]
    mrc = np.asarray(mw_d["indicators"]["i_rc"], dtype=np.float64)
    mm = np.isfinite(mrc)
    assert mm.sum() >= 500, f"too few MW rows with a spin/stop label ({mm.sum()})"
    assert (mrc[mm] == 0.0).any() and (mrc[mm] == 1.0).any(), \
        "MW spin/stop contrast needs BOTH classes (rc_keep loader not used?)"
    mw_rc = build_behavioural_axis_from_arrays(
        mw_d["hidden"][mm], mrc[mm], mw_d["balance"][mm], mw_d["rounds"][mm],
        mw_d["groups"][mm], layers)
    # the old amount-ratio MW axis for the diagnostic cosine only, rebuilt on
    # the iba_finite subset (identical rows to the Wave-3/5 MW build).
    fb = np.asarray(mw_d["iba_finite"], dtype=bool)
    mw_iba = build_behavioural_axis_from_arrays(
        mw_d["hidden"][fb], mw_d["indicators"]["i_ba"][fb], mw_d["balance"][fb],
        mw_d["rounds"][fb], mw_d["groups"][fb], layers)
    shared = build_shared_axis_from_axes(
        [sm["directions"], ic_rc["directions"], mw_rc["directions"]], layers)
    shared["auc"] = float("nan")
    cos_pairs = {}
    for name_a, a, name_b, b in (("mw_rc", mw_rc, "mw_iba", mw_iba),
                                 ("mw_rc", mw_rc, "sm_iba", sm),
                                 ("mw_rc", mw_rc, "ic_rc", ic_rc),
                                 ("shared3c", shared, "sm_iba", sm),
                                 ("shared3c", shared, "ic_rc", ic_rc),
                                 ("shared3c", shared, "mw_rc", mw_rc)):
        cs = [float(pa.unit(a["directions"][li]) @ pa.unit(b["directions"][li]))
              for li in range(len(layers))]
        cos_pairs[f"{name_a}~{name_b}"] = float(np.mean(cs))
        print(f"[sec4-axes/wave6] cos({name_a}, {name_b}) = {np.mean(cs):.4f}",
              flush=True)
    built = {"mw_rc": dict(mw_rc)}
    for tag, src in (("shared3c", sm), ("shared3c_icscale", ic_rc),
                     ("shared3c_mwscale", mw_rc)):
        v = dict(shared)
        v["scales"] = src["scales"].copy()
        v["provenance"] = (shared["provenance"]
                           + f" {tag} sigma-units (Wave-6 spin/stop MW).")
        built[tag] = v
    return built, cos_pairs


def _build_wave7_axes(task_data, layers):
    """W7 confirmatory grid: the FROZEN objects (INDEX.md object-freeze) —
    sm_iba, ic_rc, mw_rc, shared3c — each emitted at EVERY target task's
    sigma-units, so all 12 pre-registered matrix cells steer norm-matched to
    their target. Construction is byte-identical to _build_wave6_axes (same
    deconfound, same subsets); this function only re-scales. Returns
    ({(src, tgt): built}, cos_pairs) with cos_pairs re-printed for the launch
    log (must match the frozen pre-registration numbers)."""
    built6, cos_pairs = _build_wave6_axes(task_data, layers)
    # rebuild the two inputs wave6 does not return, identically to its body
    sm = build_behavioural_axis_from_arrays(
        task_data["slot_machine"]["hidden"],
        task_data["slot_machine"]["indicators"]["i_ba"],
        task_data["slot_machine"]["balance"], task_data["slot_machine"]["rounds"],
        task_data["slot_machine"]["groups"], layers)
    ic_d = task_data["investment_choice"]
    rc = np.asarray(ic_d["indicators"]["i_rc"], dtype=np.float64)
    m = np.isfinite(rc)
    ic_rc = build_behavioural_axis_from_arrays(
        ic_d["hidden"][m], rc[m], ic_d["balance"][m], ic_d["rounds"][m],
        ic_d["groups"][m], layers)
    sources = {"smiba": sm, "icrc": ic_rc, "mwrc": built6["mw_rc"],
               "sh3c": built6["shared3c"]}
    tgt_scales = {"sm": sm["scales"], "ic": ic_rc["scales"],
                  "mw": built6["mw_rc"]["scales"]}
    grid = {}
    for sk, src in sources.items():
        for tk, sc in tgt_scales.items():
            v = dict(src)
            v["scales"] = np.asarray(sc).copy()
            v["provenance"] = (src.get("provenance", "") +
                               f" W7 grid: source {sk} at {tk} sigma-units.")
            grid[(sk, tk)] = v
    return grid, cos_pairs


def load_task_arrays(model, task, layers, held_out=False, rc_keep=False,
                     want_readout=False):
    """Pull cached HF SAE features + all-layer hidden states + indicators for a
    (model, task), restricted to the DISCOVERY (or held-out) game split.

    Returns a dict with feats (N,K), decoder (K,d) [L22 canonical], hidden
    (N,L,d) sliced to ``layers``, indicators {i_ba,i_lc,i_ec,i_rc} (N,),
    balance (N,), rounds (N,), groups (N,) game ids, an ``iba_finite`` row mask
    (which kept rows carry a real bet — all True on the default keep rule), and
    the raw layer list. ``rc_keep`` (Wave-6, mystery_wheel ONLY) widens the
    keep rule to choice-validity so the spin/stop contrast keeps its stop rows
    (see _keep_mask); the default path is bit-identical to the Wave-1..5
    loader. Referenced only at run time (Task 8); the unit tests use synthetic
    arrays directly.
    """
    if model == "llama":
        return _load_llama_task_arrays(task, layers, held_out=held_out,
                                       rc_keep=rc_keep, want_readout=want_readout)
    assert model == "gemma", f"sec4 Phase-1 supports gemma/llama (got {model})"
    # gemma always carries the GemmaScope decoder, so want_readout is a no-op
    # here (the readout axis is always buildable) — kept for a symmetric API.
    token = os.environ.get("HF_TOKEN")
    tdir = TASK_DIR[task]
    sae_rel = f"sae_features_v3/{tdir}/{model}/sae_features_L{pa.LAYER}.npz"
    phase_rel = (pa.PHASE_A["sm"] if task == "slot_machine"
                 else f"sae_features_v3/{tdir}/{model}/checkpoint/"
                      "phase_a_hidden_states.npz")

    sp, meta = _load_sae(sae_rel, token)
    feats = sp.toarray().astype(np.float64)
    decoder = pa.load_gemmascope_l22_wdec(token)  # (K_full, d)

    z = np.load(pa._hf_path(phase_rel, token), mmap_mode="r")
    hs = z["hidden_states"]
    assert hs.shape[0] == feats.shape[0], "SAE/phase_a row mismatch"
    hidden = np.stack([np.asarray(hs[:, l, :], dtype=np.float64) for l in layers],
                      axis=1)

    game_ids = np.asarray(meta["game_ids"])
    rounds = np.asarray(meta["round_nums"], dtype=np.float64)
    if task == "slot_machine":
        cat = pa._hf_path(pa.SM_CATALOG, token)
        i_ba, balances = pa.compute_iba_sm(meta, cat)
    else:
        # ic/mw: the SAE meta has NO bet_ratios field (confirmed on HF) — join
        # the behavioural catalogs exactly like the paper pipeline
        # (run_comprehensive_robustness.compute_iba): sequential game_map,
        # non-skip decisions, bet from parsed_bet|bet|bet_amount over
        # balance_before|balance, ratio = min(bet/bal, 1).
        i_ba, balances, choices, bets = _iba_from_catalog(meta, task, model)
    i_lc, i_ec = _lc_ec_from_iba(np.nan_to_num(i_ba), balances, game_ids, rounds)
    # per-row decision-level risk label, computed BEFORE the keep mask so
    # rc_keep can retain choice-valid rows the i_ba rule would drop:
    #   sm  — no choice object, all NaN;
    #   ic  — risky OPTION (choice>=3), the Wave-5 constraint-free indicator;
    #   mw  — spin-vs-stop (choice 2 with bet>0 = 1, choice 1 = 0), Wave-6.
    if task == "slot_machine":
        i_rc = np.full(len(game_ids), np.nan)
    elif task == "investment_choice":
        i_rc = ((choices >= 3).astype(np.float64)
                * np.where(np.isfinite(choices), 1.0, np.nan))
    else:
        i_rc = _mw_rc_labels(choices, bets)

    if task == "slot_machine":
        # Disjointness with the runner's replay: the SM runner steers the
        # offset-300 −G slice (state_offset 300, n<=200 => pool[300:500]), so
        # EXCLUDE every game in that window from the axis build. Reuses the
        # frozen W2/W3 convention (axes.excluded_game_ids over the SAME
        # Random(42) −G pool order load_minusG_states uses; paper_axes uses
        # EVAL_EXCLUDE_N=500), matched to the runner's positional replay rather
        # than a parallel hash split that would only overlap it by chance.
        from .axes import excluded_game_ids, minusG_pool_with_game_ids
        replay_games = excluded_game_ids(
            minusG_pool_with_game_ids(cat), n_eval=pa.EVAL_EXCLUDE_N)
        split = ~np.isin(game_ids, sorted(replay_games))
    elif task == "investment_choice":
        # Same positional-replay convention for IC: the sec4_w3 IC arms replay
        # the first ic.IC_REPLAY_EXCLUDE_N entries of the SAME Random(42)
        # load_ic_states order (game_counter shares the behavior_axis
        # global-counter id space of meta['game_ids']), so EXCLUDE every game
        # in that window from the axis build — a hash-parity split would leak
        # ~half the replayed games into the build set (train-on-test).
        from .ic import ensure_ic_catalog, replay_game_ids
        ensure_ic_catalog(model)
        replayed = replay_game_ids(model)
        split = ~np.isin(game_ids, sorted(replayed))
        print(f"[sec4-axes] ic: excluded {len(replayed)} replayed games from "
              f"the axis build ({int(split.sum())}/{len(split)} rows kept)",
              flush=True)
    else:
        # mw: sec4_w4 arms replay the frozen MW pool (offset 0) — exclude that
        # window from the axis build (mirrors the IC branch; a hash-parity
        # split would leak ~half the replayed games into the build set).
        from .mw import ensure_mw_catalog, replay_game_ids as mw_replay_ids
        ensure_mw_catalog(model)
        replayed = mw_replay_ids(model)
        split = ~np.isin(game_ids, sorted(replayed))
        print(f"[sec4-axes] mw: excluded {len(replayed)} replayed games from "
              f"the axis build ({int(split.sum())}/{len(split)} rows kept)",
              flush=True)
    assert not (rc_keep and task != "mystery_wheel"), \
        "rc_keep is the Wave-6 MW-only keep rule"
    keep = _keep_mask(split, i_ba, i_rc, rc_keep=rc_keep)
    # decoder is the FULL (K_full,d) matrix; the SAE feature columns are the
    # active subset, so restrict decoder rows to the active feature ids.
    active = np.asarray(meta.get("feature_indices",
                                 np.arange(feats.shape[1])))
    return {
        "feats": feats[keep],
        "decoder": decoder[active],
        "hidden": hidden[keep],
        "indicators": {"i_ba": np.nan_to_num(i_ba)[keep],
                       "i_lc": i_lc[keep], "i_ec": i_ec[keep],
                       # ic/mw only: decision-level risk label (see above) —
                       # NaN where the catalog row has no valid choice (or sm).
                       "i_rc": i_rc[keep]},
        # which kept rows carry a real bet: under the default rule this is all
        # True; under rc_keep it lets the Wave-6 diagnostics rebuild the
        # i_ba-based MW axis on EXACTLY the rows the Wave-3/5 builds saw.
        "iba_finite": np.isfinite(i_ba)[keep],
        "balance": balances[keep],
        "rounds": rounds[keep],
        "groups": game_ids[keep],
        "build_game_ids": np.unique(game_ids[keep]),
        "layers": list(layers),
    }


# LLaMA full-layer hidden dumps per task (W8 SM window scan + W9 reduced matrix).
LLAMA_HIDDEN = {
    "slot_machine": pa.LLAMA_SM_HIDDEN,
    "investment_choice": pa.LLAMA_IC_HIDDEN,
    "mystery_wheel": pa.LLAMA_MW_HIDDEN,
}


def _llama_catalog_meta(z, task, n_rows):
    """(game_ids, round_nums, balances) meta the gemma catalog join
    (_iba_from_catalog) consumes for a llama IC/MW hidden dump.

    REQUIRES the dump's own provenance — SM and the W8-extract MW dump carry
    game_ids/round_nums/balances (verify_llama_fullhidden asserts them), which
    is the SAME id space (1-based game_counter) _iba_from_catalog / the
    ic|mw.replay_game_ids exclusion expect, so every task joins the catalog BY
    ID exactly like SM/MW/gemma. A dump that lacks game_ids (the old IC
    checkpoint shape: hidden_states [+ valid_mask] only) carries NO per-row
    anchor at all — no game_ids, no balances, no choices — so an order-only
    join could be validated no further than a row COUNT and would SILENTLY
    mislabel every row on any reorder / filter mismatch between the extractor's
    actual row set and the catalog non-skip walk (the FATAL case the brief
    flagged). Refuse that dump LOUD and re-extract IC with provenance (as
    W8-extract did for SM/MW) before building IC arms — a count-guarded order
    join is not a safe substitute for real ids."""
    files = set(z.files)
    assert "game_ids" in files and "round_nums" in files, (
        f"llama {task} hidden dump lacks game_ids/round_nums provenance "
        f"(keys={sorted(files)}); the by-id catalog join needs them. The old "
        "IC checkpoint shape (hidden_states[+valid_mask] only) has no per-row "
        "anchor to validate an order join against, so it cannot be joined "
        "safely — re-extract IC with provenance as W8-extract did for SM/MW.")
    return {"game_ids": np.asarray(z["game_ids"]),
            "round_nums": np.asarray(z["round_nums"]),
            "balances": (np.asarray(z["balances"], dtype=np.float64)
                         if "balances" in files
                         else np.full(n_rows, np.nan))}


def _load_llama_task_arrays(task, layers, held_out=False, rc_keep=False,
                            want_readout=False):
    """W8/W9/W10 LLaMA loader — BEHAVIOURAL-axis inputs for SM/IC/MW (symmetry),
    plus the W10 §4.1 per-layer READOUT inputs when want_readout.

    LLaMA's causal write axis is the BEHAVIOURAL hidden mean-diff, which needs
    NO SAE decoder, so by default this returns feats=None / decoder=None (the
    readout builder raises on a None decoder). W10 §4.1 sets want_readout=True
    (SM only): it ADDITIONALLY loads the per-layer LlamaScope SAE features
    (sae_features_v3/{task}/llama/sae_features_L{l}.npz, active nnz-subset) and
    the per-layer decoder (pa.load_llamascope_wdec) restricted to that active
    subset, returned as feats_by_layer / decoder_by_layer for
    build_llama_readout_from_arrays. Hidden states come
    from the 32-layer llama dump per task (pa.LLAMA_{SM,IC,MW}_HIDDEN). The
    indicator labeling mirrors the gemma _load_task_arrays branches EXACTLY,
    reusing the SAME helpers (pa.compute_iba_sm for SM; _iba_from_catalog for
    the ic/mw choice+bet join; i_rc = choice>=3 for IC; _mw_rc_labels for MW;
    _keep_mask incl. rc_keep for MW stop rows). Held-out disjointness reuses the
    per-task replay-window exclusion (SM excluded_game_ids; IC ic.replay_game_ids;
    MW mw.replay_game_ids). Referenced only at run time (Task 8/W9); the unit
    tests exercise the array cores + labeling on synthetic data.
    """
    assert task in LLAMA_HIDDEN, \
        f"llama sec4 loader: unexpected task {task}"
    assert not (rc_keep and task != "mystery_wheel"), \
        "rc_keep is the Wave-6 MW-only keep rule"
    token = os.environ.get("HF_TOKEN")
    n_layers_full, d_model = pa.MODEL_DIMS["llama"]
    z = np.load(pa._hf_path(LLAMA_HIDDEN[task], token), mmap_mode="r")
    hs = z["hidden_states"]
    assert hs.shape[1] == n_layers_full and hs.shape[2] == d_model, hs.shape
    # The dump carries an explicit `layers` array mapping each layer-axis ROW to
    # its true decoder-layer NUMBER (the canonical
    # run_rq2_aligned_hidden_transfer.load_hidden_states_dp convention). A
    # count-only guard (shape[1]==32) cannot tell a post-layer dump [L0..L31]
    # from an embedding-row dump [embed,L0..L30] — both are exactly 32 rows —
    # so a naive hs[:, l, :] would silently estimate every window axis one layer
    # off from where MultiLayerSteerer WRITES it (model.model.layers[l]).
    # Resolve each requested layer number to its stored row so an off-by-one /
    # missing-layer dump fails LOUD instead of mis-targeting the whole build.
    assert "layers" in z.files, (
        "llama phase_a dump lacks the `layers` provenance array; cannot verify "
        "the layer-axis convention (embedding-row/off-by-one undetectable)")
    dp_layers = np.asarray(z["layers"]).tolist()

    def _row_for(layer):
        try:
            return dp_layers.index(layer)
        except ValueError:
            raise ValueError(
                f"llama phase_a dump has no layer {layer} "
                f"(stored layers: {dp_layers})")

    hidden = np.stack(
        [np.asarray(hs[:, _row_for(l), :], dtype=np.float64) for l in layers],
        axis=1)
    assert hs.shape[0] == hidden.shape[0], "llama phase_a row mismatch"
    n_rows = hs.shape[0]

    if task == "slot_machine":
        # SM path — byte-identical to the W8 window scan.
        meta = {"game_ids": np.asarray(z["game_ids"]),
                "round_nums": np.asarray(z["round_nums"]),
                "balances": np.asarray(z["balances"], dtype=np.float64)}
        cat = pa._hf_path(pa.sm_catalog_for("llama"), token)
        i_ba, balances = pa.compute_iba_sm(meta, cat)
        game_ids = meta["game_ids"]
        rounds = np.asarray(meta["round_nums"], dtype=np.float64)
        i_rc = np.full(n_rows, np.nan)
        from .axes import excluded_game_ids, minusG_pool_with_game_ids
        replay = excluded_game_ids(
            minusG_pool_with_game_ids(cat), n_eval=pa.EVAL_EXCLUDE_N)
        split = ~np.isin(game_ids, sorted(replay))
        print(f"[sec4-axes] llama sm: excluded {len(replay)} replay-window "
              f"games from the axis build ({int(split.sum())}/{len(split)} "
              f"rows kept)", flush=True)
    else:
        # IC/MW: join the llama behavioural catalog EXACTLY like gemma's
        # _iba_from_catalog (sequential game_map, non-skip decisions, choices +
        # bets), then the per-task decision-level risk label.
        meta = _llama_catalog_meta(z, task, n_rows)
        i_ba, balances, choices, bets = _iba_from_catalog(meta, task, "llama")
        game_ids = np.asarray(meta["game_ids"])
        rounds = np.asarray(meta["round_nums"], dtype=np.float64)
        if task == "investment_choice":
            # risky OPTION (choice>=3), the Wave-5 constraint-free indicator.
            i_rc = ((choices >= 3).astype(np.float64)
                    * np.where(np.isfinite(choices), 1.0, np.nan))
            from .ic import ensure_ic_catalog, replay_game_ids
            ensure_ic_catalog("llama")
            replayed = replay_game_ids("llama")
        else:
            # spin-vs-stop (choice 2 with bet>0 = 1, choice 1 = 0), Wave-6.
            i_rc = _mw_rc_labels(choices, bets)
            from .mw import ensure_mw_catalog, replay_game_ids as mw_replay_ids
            ensure_mw_catalog("llama")
            replayed = mw_replay_ids("llama")
        split = ~np.isin(game_ids, sorted(replayed))
        print(f"[sec4-axes] llama {task}: excluded {len(replayed)} replayed "
              f"games from the axis build ({int(split.sum())}/{len(split)} "
              f"rows kept)", flush=True)

    i_lc, i_ec = _lc_ec_from_iba(np.nan_to_num(i_ba), balances, game_ids, rounds)
    keep = _keep_mask(split, i_ba, i_rc, rc_keep=rc_keep)
    out = {
        "feats": None,     # behavioural axis uses no SAE features ...
        "decoder": None,   # ... W10 readout loads per-layer feats/decoder below
        "hidden": hidden[keep],
        "indicators": {"i_ba": np.nan_to_num(i_ba)[keep],
                       "i_lc": i_lc[keep], "i_ec": i_ec[keep],
                       "i_rc": i_rc[keep]},
        "iba_finite": np.isfinite(i_ba)[keep],
        "balance": balances[keep],
        "rounds": rounds[keep],
        "groups": game_ids[keep],
        "build_game_ids": np.unique(game_ids[keep]),
        "layers": list(layers),
    }
    if want_readout:
        # W10 §4.1: per-layer LlamaScope feats + decoder for the readout axis.
        # Each layer's cached SAE-feature npz (col-index space == d_sae 32768)
        # must align 1:1 with the hidden-dump rows; the active nnz-subset keeps
        # the ridge tractable and the decoder rows aligned (mirrors the gemma
        # build_saerd active-feature convention). SM only for W10 (the §4.1
        # write-window readout task); the fnlp SAE holds both encoder+decoder.
        feats_by_layer, decoder_by_layer, active_by_layer = [], [], []
        for l in layers:
            rel = (f"sae_features_v3/{TASK_DIR[task]}/llama/"
                   f"sae_features_L{l}.npz")
            sp, sae_meta = _load_sae(rel, token)
            assert sp.shape[0] == hs.shape[0], (
                f"llama {task} L{l} SAE rows {sp.shape[0]} != hidden rows "
                f"{hs.shape[0]}")
            # ROW-ORDER provenance (not just count): the per-layer SAE npz was
            # encoded in a SEPARATE pass from the W8 hidden dump that supplies
            # the labels/groups/keep mask, so a count-only guard would silently
            # mis-pair feature rows with labels -> a mismatched Ridge -> an
            # inert readout -> a FALSE 'monitor != controller' win. Verify the
            # joint (game_id, round) key element-wise (the co-provenance the
            # gemma path gets for free by taking feats AND labels from ONE npz)
            # and fail LOUD on any drift.
            assert np.array_equal(np.asarray(sae_meta["game_ids"]), game_ids), (
                f"llama {task} L{l}: SAE npz game_ids differ from the hidden "
                f"dump order — feats/labels would be mis-paired")
            assert np.array_equal(
                np.asarray(sae_meta["round_nums"], dtype=np.float64), rounds), (
                f"llama {task} L{l}: SAE npz round_nums differ from the hidden "
                f"dump order — feats/labels would be mis-paired")
            nnz = np.diff(sp.tocsc().indptr)
            active = np.where(nnz > pa.ACTIVE_MIN_NNZ)[0]
            assert len(active) >= TOP_K, (
                f"llama {task} L{l}: only {len(active)} active features "
                f"(< top-K {TOP_K}); readout cannot select")
            Wdec = pa.load_llamascope_wdec(l, token)  # (d_sae, 4096)
            assert Wdec.shape[0] == sp.shape[1], (
                f"L{l} decoder rows {Wdec.shape[0]} != SAE feature cols "
                f"{sp.shape[1]}")
            feats_by_layer.append(sp[keep][:, active].toarray().astype(np.float64))
            decoder_by_layer.append(Wdec[active])
            active_by_layer.append(active)
        out["feats_by_layer"] = feats_by_layer
        out["decoder_by_layer"] = decoder_by_layer
        out["active_by_layer"] = active_by_layer
    return out


def _load_sae(rel, token):
    """sae_features npz -> (csr, meta) — mirrors paper_axes.load_sae_l22 for an
    arbitrary task path."""
    from scipy import sparse
    z = np.load(pa._hf_path(rel, token), allow_pickle=False)
    sp = sparse.csr_matrix((z["values"], (z["row_indices"], z["col_indices"])),
                           shape=tuple(z["shape"]), dtype=np.float32)
    meta = {k: z[k] for k in z.files
            if k not in ("row_indices", "col_indices", "values", "shape")}
    return sp, meta


# ------------------------------------------------------------- CLI

def _build_all_axes(data, indicator_key, layers):
    """Build the requested axes for one indicator from a loaded task dict,
    fixing the behavioural/confound scales onto the readout axis too (so all
    three arms steer at matched σ-units)."""
    ind = data["indicators"][indicator_key]
    hidden = data["hidden"]
    out = {}
    out["behavioural"] = build_behavioural_axis_from_arrays(
        hidden, ind, data["balance"], data["rounds"], data["groups"], layers)
    out["confound"] = build_confound_axis_from_arrays(
        hidden, data["balance"], data["rounds"], layers)
    readout = build_readout_axis_from_arrays(
        data["feats"], ind, data["balance"], data["rounds"], data["groups"],
        data["decoder"], layers)
    readout["scales"] = out["behavioural"]["scales"].copy()  # matched σ-units
    out["readout"] = readout
    return out


def _build_wave2_axes(data, layers):
    """Wave-2 COMMON-AXIS build: the I_BA and I_EC behavioural axes plus the
    SHARED (SVD top-1) axis across them, and the (indicator-agnostic) confound.

    Returns {asset_name: built} where asset_name is the ``{indicator}_{axis}``
    stem the CLI saves. The shared axis borrows the I_BA behavioural scales so it
    steers at the same sigma-units as the per-indicator behavioural arms.
    """
    hidden = data["hidden"]
    bal, rnd, grp = data["balance"], data["rounds"], data["groups"]
    iba = build_behavioural_axis_from_arrays(
        hidden, data["indicators"]["i_ba"], bal, rnd, grp, layers)
    iec = build_behavioural_axis_from_arrays(
        hidden, data["indicators"]["i_ec"], bal, rnd, grp, layers)
    conf = build_confound_axis_from_arrays(hidden, bal, rnd, layers)
    shared = build_shared_axis_from_axes([iba["directions"], iec["directions"]],
                                         layers)
    shared["scales"] = iba["scales"].copy()  # matched sigma-units
    shared["auc"] = float("nan")
    return {
        "i_ba_behavioural": iba,
        "i_ec_behavioural": iec,
        "i_ba_confound": conf,
        "shared_iba_iec_behavioural": shared,
    }


# Wave-3 (goals-v2 Q2) cross-task build: the tasks whose OWN behavioural I_BA
# axes feed the 3-task shared component, in fixed order.
W3_TASKS = ("slot_machine", "investment_choice", "mystery_wheel")
# scales_from_phase_a task keys (paper_axes.PHASE_A namespaces).
W3_TASK_KEY = {"slot_machine": "sm", "investment_choice": "ic",
               "mystery_wheel": "mw"}


def _build_wave3_axes(task_data, layers):
    """Wave-3 CROSS-TASK build (goals-v2 Q2 rung-1, expression-matched):
    per-task behavioural I_BA axes for the three tasks plus the 3-task
    SVD-top1 SHARED axis across them (build_shared_axis_from_axes reused).

    task_data: {task in W3_TASKS: load_task_arrays dict}. Prints the pairwise
    cross-task cosines of the behavioural axes over the write window
    (reconnaissance: how aligned are the tasks' own risk directions BEFORE any
    steering). The shared axis is emitted TWICE with identical directions:
    'shared3' borrows the SM behavioural scales (the sec4_w3 SM arms steer it
    at the same sigma-units as the Wave-1/2 behavioural arms) and
    'shared3_icscale' borrows the IC behavioural scales — the sh3_ic arms
    must be dosed at the SAME sigma-magnitude as the ic_own positive control
    and the IC null band they are adjudicated against (SM/IC hidden norms can
    differ in the write window; the per-layer scale ratio is printed).
    Returns ({task | 'shared3' | 'shared3_icscale': built}, cos_pairs).
    """
    per_task = {}
    for task in W3_TASKS:
        data = task_data[task]
        per_task[task] = build_behavioural_axis_from_arrays(
            data["hidden"], data["indicators"]["i_ba"], data["balance"],
            data["rounds"], data["groups"], layers)
    cos_pairs = {}
    for i, a in enumerate(W3_TASKS):
        for b in W3_TASKS[i + 1:]:
            cs = [float(pa.unit(per_task[a]["directions"][li])
                        @ pa.unit(per_task[b]["directions"][li]))
                  for li in range(len(layers))]
            cos_pairs[f"{a}~{b}"] = float(np.mean(cs))
            print(f"[sec4-axes/wave3] cos({a}, {b}) mean over window = "
                  f"{np.mean(cs):.4f}", flush=True)
    shared = build_shared_axis_from_axes(
        [per_task[t]["directions"] for t in W3_TASKS], layers)
    shared["scales"] = per_task["slot_machine"]["scales"].copy()  # SM sigma-units
    shared["auc"] = float("nan")
    # IC-scale twin asset: SAME directions, IC write-window sigma-units, so the
    # sh3_ic dose ladder is norm-matched to the ic_own control / IC null band.
    shared_ic = dict(shared)
    shared_ic["scales"] = per_task["investment_choice"]["scales"].copy()
    shared_ic["provenance"] = (shared["provenance"]
                               + " IC write-window scales (sh3_ic dosing "
                                 "norm-matched to the IC-own control/null).")
    ratio = shared["scales"] / np.maximum(shared_ic["scales"], 1e-12)
    print(f"[sec4-axes/wave3] SM/IC write-window scale ratio per layer = "
          f"{np.round(ratio, 4).tolist()}", flush=True)
    # MW-scale twin (Wave-4 sh3_mw arms): SAME directions, MW write-window
    # sigma-units, norm-matched to the mw_own control / MW null band.
    shared_mw = dict(shared)
    shared_mw["scales"] = per_task["mystery_wheel"]["scales"].copy()
    shared_mw["provenance"] = (shared["provenance"]
                               + " MW write-window scales (sh3_mw dosing "
                                 "norm-matched to the MW-own control/null).")
    built = dict(per_task)
    built["shared3"] = shared
    built["shared3_icscale"] = shared_ic
    built["shared3_mwscale"] = shared_mw
    return built, cos_pairs


def _replicate_to_full(directions, layers, n_layers=N_LAYERS):
    """Expand a (L,d) per-window direction stack to the (n_layers,d) runner npz
    row schema: the L requested layers carry the built rows, all others hold the
    first row (placement is the arm's ``layers`` field, like replicate_rows).
    n_layers defaults to the gemma 42 so every existing caller is byte-identical;
    W8 llama passes 32 (pa.MODEL_DIMS)."""
    full = np.tile(pa.unit(directions[0]).astype(np.float32)[None, :],
                   (n_layers, 1))
    for li, l in enumerate(layers):
        full[l] = pa.unit(directions[li]).astype(np.float32)
    return full


def _save_axis(dest_dir, model, task, indicator, axis, built, layers,
               scales_full, cos_read_write, build_game_ids):
    n_layers, _ = pa.MODEL_DIMS[model]  # gemma -> 42 (byte-identical), llama 32
    directions = _replicate_to_full(built["directions"], layers, n_layers)
    scales = scales_full if scales_full is not None else np.zeros(n_layers, np.float32)
    for li, l in enumerate(layers):
        if li < len(built["scales"]):
            scales[l] = built["scales"][li]
    dest = Path(dest_dir) / f"{model}_{task}_{indicator}_{axis}.npz"
    np.savez(
        dest, directions=directions, scales=scales.astype(np.float32),
        schema_version=SCHEMA_VERSION, gate_passed=True,
        axis=axis, indicator=indicator, task=task, model=model,
        layers=np.asarray(layers), auc=float(built.get("auc", np.nan)),
        cos_read_write=float(cos_read_write),
        # Replay-guard provenance (runner._assert_replay_ok): the prompt set the
        # axis was built on + the exact games that informed it, so a mis-paired
        # or leaky arm fails loudly at run time.
        build_prompt_set=BUILD_PROMPT_SET,
        build_game_ids=np.asarray(build_game_ids),
        provenance=built["provenance"])
    return dest


def _build_llama_window(dest_dir, model, layers):
    """W8 STEP 1: llama SM I_BA BEHAVIOURAL axis for ONE candidate window.

    Saves {model}_slot_machine_i_ba_behavioural_L{lo}_{hi}.npz (window in the
    stem — the scan builds several to pick llama's causal write window before
    the reduced symmetric matrix). Behavioural axis only: NO SAE decoder is
    loaded (load_task_arrays returns decoder=None for llama). The npz row/scale
    schema and replay-guard stamps match _save_axis so the runner replays it
    exactly like a gemma behavioural axis, but at llama's 32-layer geometry.
    """
    lo, hi = layers[0], layers[-1]
    data = load_task_arrays(model, "slot_machine", layers)
    built = build_behavioural_axis_from_arrays(
        data["hidden"], data["indicators"]["i_ba"], data["balance"],
        data["rounds"], data["groups"], layers)
    n_layers, _ = pa.MODEL_DIMS[model]
    directions = _replicate_to_full(built["directions"], layers, n_layers)
    scales = np.zeros(n_layers, np.float32)
    for li, l in enumerate(layers):
        scales[l] = built["scales"][li]
    dest = (Path(dest_dir)
            / f"{model}_slot_machine_i_ba_behavioural_L{lo}_{hi}.npz")
    np.savez(
        dest, directions=directions, scales=scales.astype(np.float32),
        schema_version=SCHEMA_VERSION, gate_passed=True,
        axis="behavioural", indicator="i_ba", task="slot_machine", model=model,
        layers=np.asarray(layers), auc=float("nan"), cos_read_write=float("nan"),
        build_prompt_set=BUILD_PROMPT_SET,
        build_game_ids=np.asarray(data["build_game_ids"]),
        provenance=(built["provenance"]
                    + f" W8 llama SM window scan (L{lo}-{hi})."))
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma")
    ap.add_argument("--task", default="slot_machine")
    ap.add_argument("--indicators", default="i_ba",
                    help="comma list: i_ba,i_lc,i_ec")
    ap.add_argument("--axes", default="readout,behavioural,confound")
    ap.add_argument("--layers", nargs=2, type=int, default=[16, 21],
                    help="inclusive write window low high")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--wave2", action="store_true",
                    help="build the COMMON-AXIS set (i_ba+i_ec behavioural, "
                         "shared, confound) instead of the per-indicator Wave-1 "
                         "readout/behavioural/confound axes")
    ap.add_argument("--wave5", action="store_true",
                    help="Q2 rung-3 build: IC risky-CHOICE axis (i_rc) + "
                         "shared3b = SVD(sm_iba, ic_rc, mw_iba) with per-task "
                         "sigma-unit variants")
    ap.add_argument("--wave7", action="store_true",
                    help="W7 confirmatory build: the FROZEN objects (sm_iba, "
                         "ic_rc, mw_rc, shared3c) saved at EVERY target's "
                         "sigma-units — the full source x target-scale grid "
                         "the 12-cell pre-registered matrix steers")
    ap.add_argument("--wave7-llama", action="store_true",
                    help="W9 model-symmetry build: the SAME W7 frozen-object x "
                         "target-scale grid on LLaMA task_data at --layers "
                         "(L14-19), saving llama_w7_{src}_{tgt}scale.npz and "
                         "printing the llama loadings vs the gemma W7 "
                         "pre-registered sign table")
    ap.add_argument("--wave6", action="store_true",
                    help="Q2 MW object fix: MW spin-vs-stop axis (mw_rc, "
                         "rc_keep loader) + shared3c = SVD(sm_iba, ic_rc, "
                         "mw_rc) with per-task sigma-unit variants")
    ap.add_argument("--wave3", action="store_true",
                    help="build the CROSS-TASK set (per-task behavioural I_BA "
                         "axes for slot_machine/investment_choice/mystery_wheel "
                         "+ their SVD-top1 shared3 axis) — goals-v2 Q2 rung-1")
    ap.add_argument("--build-llama-windows", action="store_true",
                    help="W8 STEP 1: build ONE llama SM I_BA behavioural axis "
                         "for the --layers window (llama_slot_machine_i_ba_"
                         "behavioural_L{lo}_{hi}.npz); no SAE decoder needed")
    ap.add_argument("--wave10-llama", action="store_true",
                    help="W10 §4.1 model symmetry: build the LLaMA SM "
                         "readout/behavioural/confound i_ba axes at --layers "
                         "(L14-19) PLUS i_ec/i_lc behavioural specificity axes. "
                         "The readout maps a per-layer LlamaScope Ridge readout "
                         "through decoder.weight.T (build_llama_readout_from_"
                         "arrays); read/write are compared at the SAME layers")
    args = ap.parse_args()
    assert sum(map(bool, (args.wave2, args.wave3, args.wave5, args.wave6,
                          args.wave7, args.wave7_llama, args.wave10_llama))) <= 1, \
        "--wave2/3/5/6/7/7-llama/10-llama are exclusive"

    lo, hi = args.layers
    layers = list(range(lo, hi + 1))
    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    indicators = [i.strip() for i in args.indicators.split(",") if i.strip()]

    if args.build_llama_windows:
        dest = _build_llama_window(dest_dir, args.model, layers)
        print(f"[sec4-axes/w8scan] {args.model} SM behavioural L{lo}-{hi} -> "
              f"{dest}", flush=True)
        print(json.dumps({"dest": str(dest)}, default=str))
        return

    if args.wave10_llama:
        # W10 §4.1 model symmetry: LLaMA SM readout/behavioural/confound i_ba
        # axes at the L14-19 write window, plus i_ec/i_lc behavioural specificity
        # axes. The readout is built PER LAYER from that layer's own LlamaScope
        # SAE + decoder (build_llama_readout_from_arrays) so read and write are
        # compared at IDENTICAL layers; its scales borrow the behavioural axis
        # so all three arms steer at matched sigma-units.
        data = load_task_arrays("llama", "slot_machine", layers,
                                want_readout=True)
        bal, rnd, grp = data["balance"], data["rounds"], data["groups"]
        ind = data["indicators"]["i_ba"]
        behav = build_behavioural_axis_from_arrays(
            data["hidden"], ind, bal, rnd, grp, layers)
        conf = build_confound_axis_from_arrays(data["hidden"], bal, rnd, layers)
        readout = build_llama_readout_from_arrays(
            data["feats_by_layer"], ind, bal, rnd, grp,
            data["decoder_by_layer"], layers)
        readout["scales"] = behav["scales"].copy()      # matched sigma-units
        cos_rw = float(np.mean([
            pa.unit(readout["directions"][li]) @ pa.unit(behav["directions"][li])
            for li in range(len(layers))]))
        built = {("i_ba", "readout"): readout,
                 ("i_ba", "behavioural"): behav,
                 ("i_ba", "confound"): conf}
        # indicator specificity (cheap: same hidden, different indicator): the
        # i_ec (extreme-choice) and i_lc (loss-chasing) behavioural axes.
        for spec in ("i_ec", "i_lc"):
            built[(spec, "behavioural")] = build_behavioural_axis_from_arrays(
                data["hidden"], data["indicators"][spec], bal, rnd, grp, layers)
        summary = {"cos_read_write": cos_rw,
                   "readout_auc": float(readout["auc"]),
                   "aucs_by_layer": readout["aucs_by_layer"].tolist()}
        for (indicator, axis), b in built.items():
            crw = cos_rw if indicator == "i_ba" else float("nan")
            dest = _save_axis(dest_dir, "llama", "slot_machine", indicator,
                              axis, b, layers, None, crw,
                              data["build_game_ids"])
            summary[f"{indicator}_{axis}"] = {
                "dest": str(dest), "auc": float(b.get("auc", np.nan))}
            print(f"[sec4-axes/wave10-llama] {indicator}/{axis}: auc="
                  f"{b.get('auc', float('nan'))} -> {dest}", flush=True)
        print(f"[sec4-axes/wave10-llama] cos(read,write) window-mean = "
              f"{cos_rw:.4f}", flush=True)
        print(json.dumps(summary, indent=2, default=str))
        return

    if args.wave7:
        token = os.environ.get("HF_TOKEN")
        task_data = {t: load_task_arrays(args.model, t, layers,
                                         rc_keep=(t == "mystery_wheel"))
                     for t in W3_TASKS}
        grid, cos_pairs = _build_wave7_axes(task_data, layers)
        summary = {"w7_cos": cos_pairs}
        SCALE_TASK = {"sm": "slot_machine", "ic": "investment_choice",
                      "mw": "mystery_wheel"}
        for (sk, tk), b in grid.items():
            src_task = SCALE_TASK[tk]
            sf = pa.scales_from_phase_a(W3_TASK_KEY[src_task],
                                        np.arange(len(task_data[src_task]["groups"])),
                                        token)
            dest = _save_axis(dest_dir, args.model, "w7", sk,
                              f"{tk}scale", b, layers, sf, float("nan"),
                              task_data[src_task]["build_game_ids"])
            summary[f"{sk}_{tk}"] = {"dest": str(dest)}
            print(f"[sec4-axes/wave7] {sk}@{tk} -> {dest}", flush=True)
        print(json.dumps(summary, indent=2, default=str))
        return

    if args.wave7_llama:
        # W9: the SAME W7 grid on LLaMA (L14-19). Construction reuses
        # _build_wave7_axes verbatim (model-agnostic array cores); only the
        # loaded task_data (llama hidden), the saved model tag (32-row npz) and
        # the sigma-unit source differ. Per-window target scales come from each
        # grid entry's built["scales"] (computed on llama hidden by the
        # behavioural builder), so no gemma scales_from_phase_a is needed —
        # _save_axis fills the window rows from built["scales"] and leaves the
        # unused non-window rows zero (arms only steer the window).
        task_data = {t: load_task_arrays("llama", t, layers,
                                         rc_keep=(t == "mystery_wheel"))
                     for t in W3_TASKS}
        grid, cos_pairs = _build_wave7_axes(task_data, layers)
        # Pre-registration comparison: gemma W7 loadings (INDEX.md object-freeze)
        # are the model-INDEPENDENT prediction; print llama vs gemma per pair.
        GEMMA_W7_LOADINGS = {
            "mw_rc~mw_iba": -0.29, "mw_rc~sm_iba": +0.10, "mw_rc~ic_rc": +0.24,
            "shared3c~sm_iba": -0.80, "shared3c~ic_rc": +0.89,
            "shared3c~mw_rc": +0.26}
        print("[sec4-axes/wave7-llama] loadings: llama vs gemma W7 "
              "pre-registration (sign match = model symmetry)", flush=True)
        for name, gv in GEMMA_W7_LOADINGS.items():
            lv = cos_pairs.get(name, float("nan"))
            match = "SIGN_MATCH" if (np.sign(lv) == np.sign(gv)) else "SIGN_DIFF"
            print(f"  cos({name}) llama={lv:+.4f} gemma={gv:+.4f} {match}",
                  flush=True)
        summary = {"w9_llama_cos": cos_pairs, "gemma_w7_loadings":
                   GEMMA_W7_LOADINGS}
        SCALE_TASK = {"sm": "slot_machine", "ic": "investment_choice",
                      "mw": "mystery_wheel"}
        for (sk, tk), b in grid.items():
            src_task = SCALE_TASK[tk]
            dest = _save_axis(dest_dir, "llama", "w7", sk,
                              f"{tk}scale", b, layers, None, float("nan"),
                              task_data[src_task]["build_game_ids"])
            summary[f"{sk}_{tk}"] = {"dest": str(dest)}
            print(f"[sec4-axes/wave7-llama] {sk}@{tk} -> {dest}", flush=True)
        print(json.dumps(summary, indent=2, default=str))
        return

    if args.wave6:
        token = os.environ.get("HF_TOKEN")
        # MW is loaded with the Wave-6 choice-validity keep rule so its stop
        # rows survive; SM/IC keep the frozen default rule (byte-identical to
        # the Wave-3/5 builds).
        task_data = {t: load_task_arrays(args.model, t, layers,
                                         rc_keep=(t == "mystery_wheel"))
                     for t in W3_TASKS}
        built, cos_pairs = _build_wave6_axes(task_data, layers)
        summary = {"wave6_cos": cos_pairs}
        SRC = {"mw_rc": ("mystery_wheel", "mw_rc", "behavioural"),
               "shared3c": ("shared3c", "rc", "behavioural"),
               "shared3c_icscale": ("shared3c", "rc", "behavioural_icscale"),
               "shared3c_mwscale": ("shared3c", "rc", "behavioural_mwscale")}
        SCALE_SRC = {"mw_rc": "mystery_wheel",
                     "shared3c": "slot_machine",
                     "shared3c_icscale": "investment_choice",
                     "shared3c_mwscale": "mystery_wheel"}
        for name, b in built.items():
            task_label, indicator, axis = SRC[name]
            src = SCALE_SRC[name]
            sf = pa.scales_from_phase_a(W3_TASK_KEY[src],
                                        np.arange(len(task_data[src]["groups"])),
                                        token)
            dest = _save_axis(dest_dir, args.model, task_label, indicator,
                              axis, b, layers, sf, float("nan"),
                              task_data[src]["build_game_ids"])
            summary[name] = {"dest": str(dest)}
            print(f"[sec4-axes/wave6] {name} -> {dest}", flush=True)
        print(json.dumps(summary, indent=2, default=str))
        return

    if args.wave5:
        token = os.environ.get("HF_TOKEN")
        task_data = {t: load_task_arrays(args.model, t, layers)
                     for t in W3_TASKS}
        built, cos_pairs = _build_wave5_axes(task_data, layers)
        summary = {"wave5_cos": cos_pairs}
        SRC = {"ic_rc": ("investment_choice", "i_rc", "behavioural"),
               "shared3b": ("shared3b", "irc", "behavioural"),
               "shared3b_icscale": ("shared3b", "irc", "behavioural_icscale"),
               "shared3b_mwscale": ("shared3b", "irc", "behavioural_mwscale")}
        SCALE_SRC = {"ic_rc": "investment_choice",
                     "shared3b": "slot_machine",
                     "shared3b_icscale": "investment_choice",
                     "shared3b_mwscale": "mystery_wheel"}
        for name, b in built.items():
            task_label, indicator, axis = SRC[name]
            src = SCALE_SRC[name]
            sf = pa.scales_from_phase_a(W3_TASK_KEY[src],
                                        np.arange(len(task_data[src]["groups"])),
                                        token)
            dest = _save_axis(dest_dir, args.model, task_label, indicator,
                              axis, b, layers, sf, float("nan"),
                              task_data[src]["build_game_ids"])
            summary[name] = {"dest": str(dest)}
            print(f"[sec4-axes/wave5] {name} -> {dest}", flush=True)
        print(json.dumps(summary, indent=2, default=str))
        return

    if args.wave3:
        token = os.environ.get("HF_TOKEN")
        task_data = {t: load_task_arrays(args.model, t, layers)
                     for t in W3_TASKS}
        built, cos_pairs = _build_wave3_axes(task_data, layers)
        summary = {"cross_task_cos": cos_pairs,
                   # dosing provenance: how far apart the SM and IC sigma-units
                   # are in the write window (the icscale asset exists so the
                   # sh3_ic ladder is dosed in the IC units regardless).
                   "sm_ic_scale_ratio": (
                       built["shared3"]["scales"]
                       / np.maximum(built["shared3_icscale"]["scales"], 1e-12)
                   ).round(4).tolist()}
        for name, b in built.items():
            axis = "behavioural"
            if name == "shared3":
                # shared3 steers the SM held-out replay, so it carries the SM
                # target scales and the SM build-game ids for the runner's
                # twin=G disjointness guard.
                task_label, indicator, src = "shared3", "iba", "slot_machine"
            elif name == "shared3_icscale":
                # same directions at IC sigma-units for the sh3_ic ladder
                # (norm-matched to ic_own / the IC null band); carries the IC
                # build-game ids — the IC-component build already excludes the
                # ic.replay_game_ids window, and the runner's IC guard only
                # id-checks npz stamped task 'investment_choice'.
                task_label, indicator, src = "shared3", "iba", "investment_choice"
                axis = "behavioural_icscale"
            elif name == "shared3_mwscale":
                # same directions at MW sigma-units for the Wave-4 sh3_mw
                # ladder (norm-matched to mw_own / the MW null band).
                task_label, indicator, src = "shared3", "iba", "mystery_wheel"
                axis = "behavioural_mwscale"
            else:
                task_label, indicator, src = name, "i_ba", name
            # target-task scales (PHASE_A covers sm/ic/mw), same subset
            # convention as the Wave-1/2 SM path.
            scale_key = W3_TASK_KEY[src]
            n_rows = len(task_data[src]["groups"])
            game_ids = task_data[src]["build_game_ids"]
            sf = pa.scales_from_phase_a(scale_key, np.arange(n_rows), token)
            dest = _save_axis(dest_dir, args.model, task_label, indicator,
                              axis, b, layers, sf, float("nan"),
                              game_ids)
            summary[name] = {"dest": str(dest)}
            print(f"[sec4-axes/wave3] {name} -> {dest}", flush=True)
        print(json.dumps(summary, indent=2))
        return

    data = load_task_arrays(args.model, args.task, layers)
    # per-layer scales over ALL discovery rows for the full 42-row npz.
    token = os.environ.get("HF_TOKEN")
    scales_full = pa.scales_from_phase_a(
        "sm" if args.task == "slot_machine" else args.task,
        np.arange(len(data["groups"])), token) if args.task == "slot_machine" \
        else None

    if args.wave2:
        built = _build_wave2_axes(data, layers)
        summary = {}
        for name, b in built.items():
            indicator, axis = name.rsplit("_", 1)
            dest = _save_axis(dest_dir, args.model, args.task, indicator, axis,
                              b, layers, scales_full, float("nan"),
                              data["build_game_ids"])
            summary[name] = {"dest": str(dest), "axis": axis,
                             "indicator": indicator}
            print(f"[sec4-axes/wave2] {name} -> {dest}", flush=True)
        print(json.dumps(summary, indent=2))
        return

    summary = {}
    for indicator in indicators:
        built = _build_all_axes(data, indicator, layers)
        r = built.get("readout"); b = built.get("behavioural")
        cos_rw = (float(pa.unit(r["directions"][0]) @ pa.unit(b["directions"][0]))
                  if r is not None and b is not None else float("nan"))
        for axis in axes:
            dest = _save_axis(dest_dir, args.model, args.task, indicator, axis,
                              built[axis], layers, scales_full, cos_rw,
                              data["build_game_ids"])
            summary[f"{indicator}_{axis}"] = {
                "dest": str(dest), "auc": float(built[axis].get("auc", np.nan)),
                "cos_read_write": cos_rw}
            print(f"[sec4-axes] {indicator}/{axis}: auc="
                  f"{built[axis].get('auc', float('nan'))} cos_read_write="
                  f"{cos_rw:.4f} -> {dest}", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
