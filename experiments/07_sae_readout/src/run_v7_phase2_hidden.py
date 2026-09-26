#!/usr/bin/env python3
"""
V7 Phase 2: Comprehensive hidden-state-primary classification across ALL 42 layers.

Covers:
  1. BK classification at all 42 layers (DP + R1) for each paradigm
  2. Cross-domain transfer at all 42 layers (6 pairs)
  3. Condition encoding at all 42 layers (bet_type, bet_constraint, prompt_condition)

Hidden states: Gemma-2-9B-IT, 3584-dim, 42 layers (L0-L41)
  IC: 1600 games, 172 BK
  SM: 3200 games, 87 BK
  MW: 3200 games, 54 BK

OPTIMIZATION:
  - Loads each paradigm's full hidden state NPZ once into RAM, slices layers in-memory
  - PCA(50) with randomized SVD for fast 3584 -> 50 dim reduction (unsupervised, pre-CV)
  - LogisticRegression on 50 features: ~0.01s per fold
  - Total expected time: ~10-15 minutes

Usage:
    cd /home/jovyan/llm-addiction/sae_v3_analysis/src
    python run_v7_phase2_hidden.py
"""

import gc
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PARADIGMS, N_LAYERS, HIDDEN_DIM, RANDOM_SEED,
    PARADIGM_LABELS, JSON_DIR, LOG_DIR,
)
from data_loader import load_sparse_npz, get_metadata, get_labels

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

# PCA components for dimensionality reduction
PCA_COMPONENTS = 50


# ===================================================================
# JSON encoder for numpy types
# ===================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ===================================================================
# Flushing logger
# ===================================================================

def get_logger(log_file):
    """Create a logger that flushes after every message."""
    logger = logging.getLogger("v7_phase2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def log(logger, msg):
    """Log and flush immediately."""
    logger.info(msg)
    for h in logger.handlers:
        h.flush()
    sys.stdout.flush()


# ===================================================================
# Bulk hidden-state loader
# ===================================================================

class HiddenStateCache:
    """Load hidden states + metadata once per paradigm."""

    def __init__(self, paradigm, logger):
        self.paradigm = paradigm
        sae_dir = PARADIGMS[paradigm]["sae_dir"]
        ckpt_path = sae_dir / "checkpoint" / "phase_a_hidden_states.npz"

        t0 = time.time()
        log(logger, f"  Loading {ckpt_path.name} for {PARADIGM_LABELS[paradigm]}...")

        ckpt = np.load(ckpt_path, allow_pickle=False)
        self.hidden_all = ckpt["hidden_states"]  # (n_rounds, n_layers, hidden_dim)
        valid_mask = ckpt["valid_mask"].astype(bool) if "valid_mask" in ckpt else None

        # Metadata from any SAE feature file
        any_sae = sorted(sae_dir.glob("sae_features_L*.npz"))
        raw = load_sparse_npz(any_sae[0])
        self.meta = get_metadata(raw)

        if valid_mask is not None:
            self.hidden_all = self.hidden_all[valid_mask]
            self.meta = {k: v[valid_mask] for k, v in self.meta.items()}

        self.is_last_round = self.meta["is_last_round"]
        self.r1_mask = self.meta["round_nums"] == 1

        self.dp_meta = {k: v[self.is_last_round] for k, v in self.meta.items()}
        self.r1_meta = {k: v[self.r1_mask] for k, v in self.meta.items()}

        self.dp_labels = get_labels(self.dp_meta)
        self.r1_labels = get_labels(self.r1_meta)

        n_rounds, n_layers, hidden_dim = self.hidden_all.shape
        dt = time.time() - t0
        log(logger, f"    shape=({n_rounds}, {n_layers}, {hidden_dim}), "
                     f"DP={self.is_last_round.sum()}, R1={self.r1_mask.sum()}, "
                     f"BK(DP)={int(self.dp_labels.sum())}, BK(R1)={int(self.r1_labels.sum())} [{dt:.1f}s]")

    def get_dp(self, layer):
        return self.hidden_all[self.is_last_round, layer, :]

    def get_r1(self, layer):
        return self.hidden_all[self.r1_mask, layer, :]

    def free(self):
        del self.hidden_all
        gc.collect()


# ===================================================================
# Fast classification: PCA outside CV, LR inside CV
# ===================================================================

def reduce_dim(X, n_comp=PCA_COMPONENTS):
    """StandardScaler + PCA(randomized SVD) -- unsupervised, applied to all data."""
    n_comp = min(n_comp, X.shape[0] - 1, X.shape[1])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=RANDOM_SEED)
    X_reduced = pca.fit_transform(X_scaled)
    return X_reduced, scaler, pca


def classify_cv(X, labels, n_folds=5):
    """PCA(unsupervised) + 5-fold stratified CV LR."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    effective_folds = min(n_folds, n_pos, n_neg)

    if effective_folds < 2:
        return {
            "auc": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0,
            "n_pos": n_pos, "n_neg": n_neg, "skipped": True,
        }

    X_r, _, _ = reduce_dim(X)

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=RANDOM_SEED)
    aucs, f1s, precs, recs = [], [], [], []

    for tr, te in skf.split(X_r, labels):
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                  class_weight="balanced", random_state=RANDOM_SEED)
        clf.fit(X_r[tr], labels[tr])
        yp = clf.predict_proba(X_r[te])[:, 1]
        yh = clf.predict(X_r[te])
        aucs.append(roc_auc_score(labels[te], yp))
        f1s.append(f1_score(labels[te], yh, zero_division=0))
        precs.append(precision_score(labels[te], yh, zero_division=0))
        recs.append(recall_score(labels[te], yh, zero_division=0))

    return {
        "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "f1": float(np.mean(f1s)),
        "precision": float(np.mean(precs)), "recall": float(np.mean(recs)),
        "n_pos": n_pos, "n_neg": n_neg, "skipped": False,
    }


def classify_transfer(X_train, y_train, X_test, y_test):
    """PCA fit on train, transform test, then LR."""
    n_pos_tr = int(y_train.sum())
    n_neg_tr = len(y_train) - n_pos_tr
    n_pos_te = int(y_test.sum())
    n_neg_te = len(y_test) - n_pos_te

    if n_pos_tr < 2 or n_neg_tr < 2 or n_pos_te < 2 or n_neg_te < 2:
        return {"auc": 0.5, "skipped": True,
                "n_train": len(y_train), "n_test": len(y_test)}

    n_comp = min(PCA_COMPONENTS, X_train.shape[0] - 1, X_train.shape[1])
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=RANDOM_SEED)
    X_tr = pca.fit_transform(X_tr)
    X_te = pca.transform(X_te)

    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                              class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(X_tr, y_train)
    yp = clf.predict_proba(X_te)[:, 1]

    return {
        "auc": float(roc_auc_score(y_test, yp)),
        "n_train": len(y_train), "n_test": len(y_test),
        "n_pos_train": n_pos_tr, "n_pos_test": n_pos_te,
        "skipped": False,
    }


def classify_multiclass_cv(X, labels, n_folds=5):
    """PCA + multi-class OVR AUC (macro)."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    if n_classes < 2:
        return {"auc": 0.5, "n_classes": n_classes, "n_samples": len(y), "skipped": True}

    cc = np.bincount(y)
    eff = min(n_folds, cc.min())
    if eff < 2:
        return {"auc": 0.5, "n_classes": n_classes, "n_samples": len(y),
                "class_counts": cc.tolist(), "skipped": True}

    X_r, _, _ = reduce_dim(X)

    skf = StratifiedKFold(n_splits=eff, shuffle=True, random_state=RANDOM_SEED)
    aucs = []
    for tr, te in skf.split(X_r, y):
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                  class_weight="balanced", random_state=RANDOM_SEED)
        clf.fit(X_r[tr], y[tr])
        yp = clf.predict_proba(X_r[te])
        try:
            aucs.append(roc_auc_score(y[te], yp, multi_class="ovr", average="macro"))
        except ValueError:
            pass

    if not aucs:
        return {"auc": 0.5, "n_classes": n_classes, "n_samples": len(y), "skipped": True}

    return {
        "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "n_classes": n_classes, "n_samples": len(y),
        "class_counts": cc.tolist(), "classes": le.classes_.tolist(),
        "skipped": False,
    }


# ===================================================================
# Part 1: BK Classification
# ===================================================================

def run_bk_classification(logger):
    log(logger, "=" * 70)
    log(logger, "PART 1: BK Classification -- ALL 42 Layers (DP + R1)")
    log(logger, "=" * 70)

    results = {}
    for paradigm in PARADIGMS:
        pname = PARADIGM_LABELS[paradigm]
        log(logger, f"\n  === {pname} ===")
        cache = HiddenStateCache(paradigm, logger)
        results[paradigm] = {"dp": [], "r1": []}

        # DP
        log(logger, f"  [DP] {int(cache.dp_labels.sum())} BK / {len(cache.dp_labels)} total")
        t0 = time.time()
        for layer in range(N_LAYERS):
            feat = cache.get_dp(layer)
            n_bk = int(cache.dp_labels.sum())
            if n_bk < 5:
                results[paradigm]["dp"].append({"layer": layer, "auc": 0.5, "f1": 0.0,
                    "precision": 0.0, "recall": 0.0, "n_pos": n_bk,
                    "n_neg": len(cache.dp_labels)-n_bk, "skipped": True})
                continue
            res = classify_cv(feat, cache.dp_labels)
            res["layer"] = layer
            results[paradigm]["dp"].append(res)
            if layer % 7 == 0 or layer == N_LAYERS - 1:
                log(logger, f"    L{layer:2d} DP: AUC={res['auc']:.3f} F1={res['f1']:.3f} "
                             f"P={res['precision']:.3f} R={res['recall']:.3f}")
        dt = time.time() - t0
        valid = [r for r in results[paradigm]["dp"] if not r.get("skipped")]
        if valid:
            best = max(valid, key=lambda r: r["auc"])
            log(logger, f"  >> Best DP: L{best['layer']} AUC={best['auc']:.3f} F1={best['f1']:.3f} [{dt:.1f}s]")

        # R1
        log(logger, f"  [R1] {int(cache.r1_labels.sum())} BK / {len(cache.r1_labels)} total")
        t0 = time.time()
        for layer in range(N_LAYERS):
            feat = cache.get_r1(layer)
            n_bk = int(cache.r1_labels.sum())
            if n_bk < 5:
                results[paradigm]["r1"].append({"layer": layer, "auc": 0.5, "f1": 0.0,
                    "precision": 0.0, "recall": 0.0, "n_pos": n_bk,
                    "n_neg": len(cache.r1_labels)-n_bk, "skipped": True})
                continue
            res = classify_cv(feat, cache.r1_labels)
            res["layer"] = layer
            results[paradigm]["r1"].append(res)
            if layer % 7 == 0 or layer == N_LAYERS - 1:
                log(logger, f"    L{layer:2d} R1: AUC={res['auc']:.3f} F1={res['f1']:.3f}")
        dt = time.time() - t0
        valid = [r for r in results[paradigm]["r1"] if not r.get("skipped")]
        if valid:
            best = max(valid, key=lambda r: r["auc"])
            log(logger, f"  >> Best R1: L{best['layer']} AUC={best['auc']:.3f} F1={best['f1']:.3f} [{dt:.1f}s]")

        cache.free()

    return results


# ===================================================================
# Part 2: Cross-Domain Transfer
# ===================================================================

def run_cross_domain_transfer(logger):
    log(logger, "\n" + "=" * 70)
    log(logger, "PART 2: Cross-Domain Transfer -- ALL 42 Layers")
    log(logger, "=" * 70)

    pairs = [("ic","sm"),("ic","mw"),("sm","ic"),("sm","mw"),("mw","ic"),("mw","sm")]

    # Pre-cache DP features
    log(logger, "  Pre-caching DP features...")
    dp_feat = {}
    dp_lab = {}
    for paradigm in PARADIGMS:
        cache = HiddenStateCache(paradigm, logger)
        dp_feat[paradigm] = {L: cache.get_dp(L).copy() for L in range(N_LAYERS)}
        dp_lab[paradigm] = cache.dp_labels.copy()
        cache.free()
    log(logger, "  Cached.\n")

    results = {}
    for src, tgt in pairs:
        pk = f"{src}_to_{tgt}"
        results[pk] = []
        log(logger, f"  {PARADIGM_LABELS[src]} -> {PARADIGM_LABELS[tgt]}")
        t0 = time.time()
        for layer in range(N_LAYERS):
            res = classify_transfer(dp_feat[src][layer], dp_lab[src],
                                     dp_feat[tgt][layer], dp_lab[tgt])
            res["layer"] = layer
            results[pk].append(res)
            if layer % 7 == 0 or layer == N_LAYERS - 1:
                log(logger, f"    L{layer:2d}: AUC={res['auc']:.3f}")
        dt = time.time() - t0
        valid = [e for e in results[pk] if not e.get("skipped")]
        if valid:
            best = max(valid, key=lambda r: r["auc"])
            log(logger, f"    >> Best: L{best['layer']} AUC={best['auc']:.3f} [{dt:.1f}s]")

    del dp_feat
    gc.collect()
    return results


# ===================================================================
# Part 3: Condition Encoding
# ===================================================================

def run_condition_encoding(logger):
    log(logger, "\n" + "=" * 70)
    log(logger, "PART 3: Condition Encoding -- ALL 42 Layers")
    log(logger, "=" * 70)

    results = {"bet_type": {}, "bet_constraint_ic": [], "prompt_condition_ic": []}

    # 3a: Bet type
    log(logger, "\n  [3a] Bet Type (binary)")
    for paradigm in PARADIGMS:
        results["bet_type"][paradigm] = []
        pname = PARADIGM_LABELS[paradigm]
        log(logger, f"\n  --- {pname} ---")
        cache = HiddenStateCache(paradigm, logger)
        bt = (cache.dp_meta["bet_types"] == "variable").astype(np.int32)
        nv, nf = int(bt.sum()), len(bt) - int(bt.sum())
        log(logger, f"    var={nv}, fix={nf}")
        t0 = time.time()
        for layer in range(N_LAYERS):
            feat = cache.get_dp(layer)
            if nv < 5 or nf < 5:
                results["bet_type"][paradigm].append({"layer": layer, "auc": 0.5, "skipped": True,
                    "n_variable": nv, "n_fixed": nf})
                continue
            res = classify_cv(feat, bt)
            res["layer"] = layer
            res["n_variable"] = nv
            res["n_fixed"] = nf
            results["bet_type"][paradigm].append(res)
            if layer % 7 == 0 or layer == N_LAYERS - 1:
                log(logger, f"    L{layer:2d}: AUC={res['auc']:.3f}")
        dt = time.time() - t0
        valid = [r for r in results["bet_type"][paradigm] if not r.get("skipped")]
        if valid:
            best = max(valid, key=lambda r: r["auc"])
            log(logger, f"    >> Best: L{best['layer']} AUC={best['auc']:.3f} [{dt:.1f}s]")
        cache.free()

    # 3b: Bet constraint (IC only)
    log(logger, "\n  [3b] Bet Constraint (IC, 4-class)")
    cache_ic = HiddenStateCache("ic", logger)
    if "bet_constraints" in cache_ic.dp_meta:
        bc = cache_ic.dp_meta["bet_constraints"]
        bc_mask = np.isin(bc, ["10", "30", "50", "70"])
        bc_v = bc[bc_mask]
        log(logger, f"    classes={np.unique(bc_v)}, n={bc_mask.sum()}")
        t0 = time.time()
        for layer in range(N_LAYERS):
            feat = cache_ic.get_dp(layer)[bc_mask]
            if bc_mask.sum() < 20:
                results["bet_constraint_ic"].append({"layer": layer, "auc": 0.5, "skipped": True})
                continue
            res = classify_multiclass_cv(feat, bc_v)
            res["layer"] = layer
            results["bet_constraint_ic"].append(res)
            if layer % 7 == 0 or layer == N_LAYERS - 1:
                log(logger, f"    L{layer:2d}: AUC={res['auc']:.3f}")
        dt = time.time() - t0
        valid = [r for r in results["bet_constraint_ic"] if not r.get("skipped")]
        if valid:
            best = max(valid, key=lambda r: r["auc"])
            log(logger, f"    >> Best: L{best['layer']} AUC={best['auc']:.3f} [{dt:.1f}s]")

    # 3c: Prompt condition (IC only)
    log(logger, "\n  [3c] Prompt Condition (IC, 4-class)")
    if "prompt_conditions" in cache_ic.dp_meta:
        pc = cache_ic.dp_meta["prompt_conditions"]
        upc = np.unique(pc)
        log(logger, f"    classes={upc}, n={len(pc)}")
        t0 = time.time()
        for layer in range(N_LAYERS):
            feat = cache_ic.get_dp(layer)
            if len(upc) < 2:
                results["prompt_condition_ic"].append({"layer": layer, "auc": 0.5, "skipped": True})
                continue
            res = classify_multiclass_cv(feat, pc)
            res["layer"] = layer
            results["prompt_condition_ic"].append(res)
            if layer % 7 == 0 or layer == N_LAYERS - 1:
                log(logger, f"    L{layer:2d}: AUC={res['auc']:.3f}")
        dt = time.time() - t0
        valid = [r for r in results["prompt_condition_ic"] if not r.get("skipped")]
        if valid:
            best = max(valid, key=lambda r: r["auc"])
            log(logger, f"    >> Best: L{best['layer']} AUC={best['auc']:.3f} [{dt:.1f}s]")

    cache_ic.free()
    return results


# ===================================================================
# Summary
# ===================================================================

def print_summary(logger, bk, xd, cond):
    log(logger, "\n" + "=" * 70)
    log(logger, "FINAL SUMMARY")
    log(logger, "=" * 70)

    log(logger, "\n--- BK Classification ---")
    log(logger, f"{'Paradigm':<22} {'Mode':<6} {'Best L':<8} {'AUC':<8} {'F1':<8} {'P':<8} {'R':<8}")
    log(logger, "-" * 68)
    for p in PARADIGMS:
        for m in ["dp", "r1"]:
            v = [e for e in bk.get(p,{}).get(m,[]) if not e.get("skipped")]
            if v:
                b = max(v, key=lambda r: r["auc"])
                log(logger, f"{PARADIGM_LABELS[p]:<22} {m.upper():<6} L{b['layer']:<6} "
                             f"{b['auc']:<8.3f} {b['f1']:<8.3f} "
                             f"{b.get('precision',0):<8.3f} {b.get('recall',0):<8.3f}")

    log(logger, "\n--- Cross-Domain Transfer ---")
    log(logger, f"{'Pair':<20} {'Best L':<8} {'AUC':<8}")
    log(logger, "-" * 36)
    for pk in sorted(xd.keys()):
        v = [e for e in xd[pk] if not e.get("skipped")]
        if v:
            b = max(v, key=lambda r: r["auc"])
            log(logger, f"{pk:<20} L{b['layer']:<6} {b['auc']:<8.3f}")

    log(logger, "\n--- Condition Encoding ---")
    log(logger, f"{'Type':<40} {'Best L':<8} {'AUC':<8}")
    log(logger, "-" * 56)
    for p in PARADIGMS:
        v = [e for e in cond.get("bet_type",{}).get(p,[]) if not e.get("skipped")]
        if v:
            b = max(v, key=lambda r: r["auc"])
            log(logger, f"bet_type ({PARADIGM_LABELS[p]}){'':<15} L{b['layer']:<6} {b['auc']:<8.3f}")
    for k, label in [("bet_constraint_ic","bet_constraint (IC, 4-class)"),
                      ("prompt_condition_ic","prompt_condition (IC, 4-class)")]:
        v = [e for e in cond.get(k,[]) if not e.get("skipped")]
        if v:
            b = max(v, key=lambda r: r["auc"])
            log(logger, f"{label:<40} L{b['layer']:<6} {b['auc']:<8.3f}")

    # Top-10 layers
    log(logger, "\n--- Top-10 Layers (DP AUC) ---")
    for p in PARADIGMS:
        v = sorted([e for e in bk.get(p,{}).get("dp",[]) if not e.get("skipped")],
                   key=lambda r: r["auc"], reverse=True)[:10]
        if v:
            s = ", ".join(f"L{e['layer']}={e['auc']:.3f}" for e in v)
            log(logger, f"  {PARADIGM_LABELS[p]}: {s}")

    log(logger, "\n--- Top-10 Layers (R1 AUC) ---")
    for p in PARADIGMS:
        v = sorted([e for e in bk.get(p,{}).get("r1",[]) if not e.get("skipped")],
                   key=lambda r: r["auc"], reverse=True)[:10]
        if v:
            s = ", ".join(f"L{e['layer']}={e['auc']:.3f}" for e in v)
            log(logger, f"  {PARADIGM_LABELS[p]}: {s}")


# ===================================================================
# Main
# ===================================================================

def main():
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"v7_phase2_hidden_{ts}.log"
    logger = get_logger(log_file)

    log(logger, "V7 Phase 2: Comprehensive Hidden-State Classification")
    log(logger, f"  Layers: ALL {N_LAYERS}, Hidden dim: {HIDDEN_DIM}")
    log(logger, f"  PCA: {PCA_COMPONENTS} components (randomized SVD)")
    log(logger, f"  Paradigms: {list(PARADIGMS.keys())}, Seed: {RANDOM_SEED}")

    t_total = time.time()

    bk = run_bk_classification(logger)
    xd = run_cross_domain_transfer(logger)
    cond = run_condition_encoding(logger)

    print_summary(logger, bk, xd, cond)

    output = {
        "metadata": {
            "timestamp": ts,
            "n_layers": N_LAYERS,
            "hidden_dim": HIDDEN_DIM,
            "pca_components": PCA_COMPONENTS,
            "paradigms": {k: v["name"] for k, v in PARADIGMS.items()},
            "classification": {
                "cv_folds": 5, "C": 1.0, "class_weight": "balanced",
                "pipeline": f"StandardScaler -> PCA({PCA_COMPONENTS}, randomized) -> LogReg(lbfgs)",
            },
            "total_time_seconds": time.time() - t_total,
        },
        "hidden_bk_all_layers": bk,
        "hidden_cross_domain_all_layers": xd,
        "hidden_condition_encoding": cond,
    }

    json_file = JSON_DIR / "v7_phase2_hidden.json"
    with open(json_file, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    dt = time.time() - t_total
    log(logger, f"\nTotal time: {dt:.1f}s ({dt/60:.1f} min)")
    log(logger, f"Results saved: {json_file}")
    log(logger, f"Log saved: {log_file}")


if __name__ == "__main__":
    main()
