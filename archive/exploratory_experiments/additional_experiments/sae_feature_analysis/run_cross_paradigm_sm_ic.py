#!/usr/bin/env python3
"""
Cross-Paradigm SAE Analysis: Slot Machine vs Investment Choice (Gemma)
======================================================================

Compares SAE features between two gambling paradigms using the same model (Gemma-2-9b-it)
and same SAE (GemmaScope 131K). Both use V2 decision-point-fixed features.

Analyses:
1. Per-layer discriminative feature overlap (bankruptcy vs safe)
2. Cross-paradigm transfer classification (train on SM → test on IC, and vice versa)
3. Shared significant features with consistent direction
4. Layer-wise summary figures

Output: results/cross_paradigm_sm_ic/
"""

import json
import logging
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===========================================================================
# Paths
# ===========================================================================
SM_DIR = Path("/home/jovyan/beomi/llm-addiction-data/sae_features_v2/gemma")
IC_DIR = Path("/home/jovyan/beomi/llm-addiction-data/sae_features_v2/investment_choice/gemma")
OUT_DIR = Path(__file__).parent / "results" / "cross_paradigm_sm_ic"

LAYERS = list(range(42))
FDR_ALPHA = 0.05
TOP_K = 500  # Top features per paradigm for overlap analysis


def load_layer(paradigm_dir: Path, layer: int):
    """Load features and binary outcome for a layer."""
    f = np.load(paradigm_dir / f"layer_{layer}_features.npz", allow_pickle=True)
    features = f["features"]  # (N, 131072)
    outcomes = f["outcomes"]   # (N,) str
    y = (outcomes == "bankruptcy").astype(int)
    return features, y


def fdr_correction(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns boolean mask of significant."""
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    thresholds = np.arange(1, n + 1) / n * alpha
    reject = sorted_p <= thresholds
    # Find largest k where p_k <= k/n * alpha
    if reject.any():
        max_k = np.max(np.where(reject)[0])
        significant = np.zeros(n, dtype=bool)
        significant[sorted_idx[:max_k + 1]] = True
        return significant
    return np.zeros(n, dtype=bool)


def compute_cohens_d(x1, x2):
    """Cohen's d effect size."""
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(x1) - np.mean(x2)) / pooled_std


# ===========================================================================
# Analysis 1: Per-layer discriminative feature identification
# ===========================================================================
def find_discriminative_features(features, y, top_k=500, min_active_frac=0.01):
    """Find top-K features discriminating bankruptcy vs safe.
    Returns: dict with feature indices, t-stats, p-values, Cohen's d, directions.
    """
    n_games, n_feat = features.shape
    bk_mask = y == 1
    safe_mask = y == 0
    n_bk, n_safe = bk_mask.sum(), safe_mask.sum()

    if n_bk < 5 or n_safe < 5:
        return None

    # Filter to features active in at least min_active_frac of games
    active_frac = np.mean(features != 0, axis=0)
    active_mask = active_frac >= min_active_frac
    active_indices = np.where(active_mask)[0]

    if len(active_indices) == 0:
        return None

    # T-test on active features only
    feat_active = features[:, active_indices]
    t_stats = np.zeros(len(active_indices))
    p_vals = np.ones(len(active_indices))

    for j in range(len(active_indices)):
        col = feat_active[:, j]
        bk_vals = col[bk_mask]
        safe_vals = col[safe_mask]
        if bk_vals.std() == 0 and safe_vals.std() == 0:
            continue
        t, p = stats.ttest_ind(bk_vals, safe_vals, equal_var=False)
        t_stats[j] = t
        p_vals[j] = p

    # FDR correction
    sig_mask = fdr_correction(p_vals, FDR_ALPHA)

    # Rank by absolute t-stat, take top-K
    ranked = np.argsort(-np.abs(t_stats))
    top_indices = ranked[:top_k]

    result_indices = active_indices[top_indices]
    result_t = t_stats[top_indices]
    result_p = p_vals[top_indices]
    result_sig = sig_mask[top_indices]

    # Cohen's d for top features
    result_d = np.zeros(top_k)
    for i, idx in enumerate(top_indices):
        col = feat_active[:, idx]
        result_d[i] = compute_cohens_d(col[bk_mask], col[safe_mask])

    # Direction: positive t = higher in bankruptcy
    directions = np.sign(result_t)

    return {
        "feature_indices": result_indices,
        "t_stats": result_t,
        "p_values": result_p,
        "significant": result_sig,
        "cohens_d": result_d,
        "directions": directions,
        "n_active": len(active_indices),
        "n_significant_total": sig_mask.sum(),
        "n_bk": int(n_bk),
        "n_safe": int(n_safe),
    }


# ===========================================================================
# Analysis 2: Cross-paradigm transfer classification
# ===========================================================================
def transfer_classification(feat_train, y_train, feat_test, y_test, max_features=5000):
    """Train LR on one paradigm, test on another.
    Uses top features from training set by variance.
    """
    # Select features with highest variance in training set
    var = np.var(feat_train, axis=0)
    top_idx = np.argsort(-var)[:max_features]

    X_train = feat_train[:, top_idx]
    X_test = feat_test[:, top_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=0.1, solver="saga", penalty="l1", random_state=42)
    clf.fit(X_train_s, y_train)

    # Within-paradigm AUC
    try:
        train_auc = roc_auc_score(y_train, clf.predict_proba(X_train_s)[:, 1])
    except:
        train_auc = 0.5

    # Cross-paradigm AUC
    try:
        test_auc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
    except:
        test_auc = 0.5

    return {"train_auc": train_auc, "test_auc": test_auc, "n_features": len(top_idx)}


# ===========================================================================
# Main analysis
# ===========================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("Cross-Paradigm SAE Analysis: Slot Machine vs Investment Choice (Gemma)")
    logger.info("=" * 70)

    # Results storage
    layer_results = []

    for layer in tqdm(LAYERS, desc="Layers"):
        sm_file = SM_DIR / f"layer_{layer}_features.npz"
        ic_file = IC_DIR / f"layer_{layer}_features.npz"
        if not sm_file.exists() or not ic_file.exists():
            logger.warning(f"Layer {layer}: missing file, skipping")
            continue

        sm_feat, sm_y = load_layer(SM_DIR, layer)
        ic_feat, ic_y = load_layer(IC_DIR, layer)

        # 1. Discriminative features per paradigm
        sm_disc = find_discriminative_features(sm_feat, sm_y, TOP_K)
        ic_disc = find_discriminative_features(ic_feat, ic_y, TOP_K)

        # 2. Feature overlap
        overlap_info = {}
        if sm_disc and ic_disc:
            sm_set = set(sm_disc["feature_indices"])
            ic_set = set(ic_disc["feature_indices"])
            intersection = sm_set & ic_set
            union = sm_set | ic_set
            jaccard = len(intersection) / len(union) if union else 0

            # Among shared features, how many have consistent direction?
            consistent = 0
            shared_indices = sorted(intersection)
            for idx in shared_indices:
                sm_pos = np.where(sm_disc["feature_indices"] == idx)[0]
                ic_pos = np.where(ic_disc["feature_indices"] == idx)[0]
                if len(sm_pos) > 0 and len(ic_pos) > 0:
                    if sm_disc["directions"][sm_pos[0]] == ic_disc["directions"][ic_pos[0]]:
                        consistent += 1

            # Significant overlap (features significant in BOTH)
            sm_sig_set = set(sm_disc["feature_indices"][sm_disc["significant"]])
            ic_sig_set = set(ic_disc["feature_indices"][ic_disc["significant"]])
            sig_intersection = sm_sig_set & ic_sig_set

            overlap_info = {
                "top_k_overlap": len(intersection),
                "top_k_jaccard": jaccard,
                "consistent_direction": consistent,
                "consistent_ratio": consistent / len(intersection) if intersection else 0,
                "sm_n_significant": len(sm_sig_set),
                "ic_n_significant": len(ic_sig_set),
                "both_significant": len(sig_intersection),
                "shared_features": [int(x) for x in sorted(intersection)[:50]],  # Store top 50
            }

        # 3. Transfer classification
        transfer_sm_to_ic = {"train_auc": 0.5, "test_auc": 0.5}
        transfer_ic_to_sm = {"train_auc": 0.5, "test_auc": 0.5}

        if sm_y.sum() >= 5 and ic_y.sum() >= 5:
            try:
                transfer_sm_to_ic = transfer_classification(sm_feat, sm_y, ic_feat, ic_y)
            except Exception as e:
                logger.warning(f"Layer {layer} SM→IC transfer failed: {e}")

            try:
                transfer_ic_to_sm = transfer_classification(ic_feat, ic_y, sm_feat, sm_y)
            except Exception as e:
                logger.warning(f"Layer {layer} IC→SM transfer failed: {e}")

        result = {
            "layer": layer,
            "sm_n_games": len(sm_y),
            "sm_n_bk": int(sm_y.sum()),
            "ic_n_games": len(ic_y),
            "ic_n_bk": int(ic_y.sum()),
            "sm_n_active": sm_disc["n_active"] if sm_disc else 0,
            "ic_n_active": ic_disc["n_active"] if ic_disc else 0,
            "sm_n_significant": sm_disc["n_significant_total"] if sm_disc else 0,
            "ic_n_significant": ic_disc["n_significant_total"] if ic_disc else 0,
            "overlap": overlap_info,
            "transfer_sm_to_ic": transfer_sm_to_ic,
            "transfer_ic_to_sm": transfer_ic_to_sm,
        }

        layer_results.append(result)
        logger.info(
            f"L{layer}: overlap={overlap_info.get('top_k_overlap', 0)}/{TOP_K}, "
            f"jaccard={overlap_info.get('top_k_jaccard', 0):.3f}, "
            f"consistent={overlap_info.get('consistent_direction', 0)}, "
            f"SM→IC={transfer_sm_to_ic.get('test_auc', 0.5):.3f}, "
            f"IC→SM={transfer_ic_to_sm.get('test_auc', 0.5):.3f}"
        )

    # Save results
    results_file = OUT_DIR / f"cross_paradigm_results_{ts}.json"

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_file, "w") as f:
        json.dump(layer_results, f, indent=2, default=convert)
    logger.info(f"Results saved: {results_file}")

    # === Generate figures ===
    generate_figures(layer_results, OUT_DIR, ts)

    logger.info("Done!")


def generate_figures(layer_results, out_dir, ts):
    """Generate summary figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    layers = [r["layer"] for r in layer_results]

    # --- Fig 1: Feature Overlap by Layer ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    overlaps = [r["overlap"].get("top_k_overlap", 0) for r in layer_results]
    jaccards = [r["overlap"].get("top_k_jaccard", 0) for r in layer_results]
    consistent = [r["overlap"].get("consistent_ratio", 0) for r in layer_results]

    axes[0].bar(layers, overlaps, color="steelblue", alpha=0.8)
    axes[0].set_ylabel(f"Shared Features\n(top-{TOP_K} overlap)")
    axes[0].set_title("Cross-Paradigm Feature Overlap: Slot Machine vs Investment Choice (Gemma)")
    ax0_twin = axes[0].twinx()
    ax0_twin.plot(layers, jaccards, "r-o", markersize=4, label="Jaccard index")
    ax0_twin.set_ylabel("Jaccard Index", color="red")
    ax0_twin.tick_params(axis="y", labelcolor="red")
    ax0_twin.legend(loc="upper right")

    axes[1].plot(layers, consistent, "g-o", markersize=4)
    axes[1].set_ylabel("Consistent Direction\nRatio")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_feature_overlap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig 2: Transfer Classification AUC ---
    fig, ax = plt.subplots(figsize=(14, 5))

    sm_to_ic = [r["transfer_sm_to_ic"].get("test_auc", 0.5) for r in layer_results]
    ic_to_sm = [r["transfer_ic_to_sm"].get("test_auc", 0.5) for r in layer_results]
    sm_within = [r["transfer_sm_to_ic"].get("train_auc", 0.5) for r in layer_results]
    ic_within = [r["transfer_ic_to_sm"].get("train_auc", 0.5) for r in layer_results]

    ax.plot(layers, sm_to_ic, "b-o", markersize=4, label="SM→IC (cross-paradigm)")
    ax.plot(layers, ic_to_sm, "r-o", markersize=4, label="IC→SM (cross-paradigm)")
    ax.plot(layers, sm_within, "b--", alpha=0.4, label="SM within (train)")
    ax.plot(layers, ic_within, "r--", alpha=0.4, label="IC within (train)")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC")
    ax.set_title("Cross-Paradigm Transfer: Bankruptcy Classification")
    ax.legend()
    ax.set_ylim(0.4, 1.0)

    # Annotate best transfer layers
    best_sm_ic = max(range(len(sm_to_ic)), key=lambda i: sm_to_ic[i])
    best_ic_sm = max(range(len(ic_to_sm)), key=lambda i: ic_to_sm[i])
    ax.annotate(f"L{layers[best_sm_ic]}: {sm_to_ic[best_sm_ic]:.3f}",
                (layers[best_sm_ic], sm_to_ic[best_sm_ic]),
                textcoords="offset points", xytext=(10, 10), fontsize=9, color="blue")
    ax.annotate(f"L{layers[best_ic_sm]}: {ic_to_sm[best_ic_sm]:.3f}",
                (layers[best_ic_sm], ic_to_sm[best_ic_sm]),
                textcoords="offset points", xytext=(10, -15), fontsize=9, color="red")

    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_transfer_auc.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig 3: Significant Features Count ---
    fig, ax = plt.subplots(figsize=(14, 5))

    sm_sig = [r["sm_n_significant"] for r in layer_results]
    ic_sig = [r["ic_n_significant"] for r in layer_results]
    both_sig = [r["overlap"].get("both_significant", 0) for r in layer_results]

    ax.plot(layers, sm_sig, "b-o", markersize=4, label="SM significant")
    ax.plot(layers, ic_sig, "r-o", markersize=4, label="IC significant")
    ax.fill_between(layers, both_sig, alpha=0.3, color="purple", label="Both significant")
    ax.set_xlabel("Layer")
    ax.set_ylabel("# Significant Features (FDR < 0.05)")
    ax.set_title("Discriminative Features: Slot Machine vs Investment Choice")
    ax.legend()

    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_significant_features.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig 4: Combined Summary ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 4a: Overlap count
    axes[0, 0].bar(layers, overlaps, color="steelblue", alpha=0.8)
    axes[0, 0].set_title(f"Top-{TOP_K} Feature Overlap")
    axes[0, 0].set_ylabel("# Shared Features")

    # 4b: Transfer AUC
    axes[0, 1].plot(layers, sm_to_ic, "b-o", markersize=3, label="SM→IC")
    axes[0, 1].plot(layers, ic_to_sm, "r-o", markersize=3, label="IC→SM")
    axes[0, 1].axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    axes[0, 1].set_title("Cross-Paradigm Transfer AUC")
    axes[0, 1].set_ylabel("AUC")
    axes[0, 1].legend(fontsize=8)

    # 4c: Consistent direction
    axes[1, 0].plot(layers, consistent, "g-o", markersize=3)
    axes[1, 0].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1, 0].set_title("Consistent Direction Ratio")
    axes[1, 0].set_ylabel("Ratio")
    axes[1, 0].set_xlabel("Layer")

    # 4d: Both significant
    axes[1, 1].bar(layers, both_sig, color="purple", alpha=0.7)
    axes[1, 1].set_title("Features Significant in Both Paradigms")
    axes[1, 1].set_ylabel("# Features")
    axes[1, 1].set_xlabel("Layer")

    fig.suptitle("Cross-Paradigm SAE Analysis: Slot Machine vs Investment Choice (Gemma V2)", fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Figures saved to {fig_dir}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    best_overlap_layer = layers[np.argmax(overlaps)]
    print(f"Best feature overlap: Layer {best_overlap_layer} ({max(overlaps)} shared features, Jaccard={jaccards[np.argmax(overlaps)]:.3f})")
    print(f"Best SM→IC transfer: Layer {layers[best_sm_ic]} (AUC={sm_to_ic[best_sm_ic]:.3f})")
    print(f"Best IC→SM transfer: Layer {layers[best_ic_sm]} (AUC={ic_to_sm[best_ic_sm]:.3f})")
    print(f"Max both-significant features: Layer {layers[np.argmax(both_sig)]} ({max(both_sig)} features)")

    # Average across layers
    avg_overlap = np.mean(overlaps)
    avg_jaccard = np.mean(jaccards)
    avg_consistent = np.mean(consistent)
    avg_sm_ic = np.mean(sm_to_ic)
    avg_ic_sm = np.mean(ic_to_sm)
    print(f"\nAverages across all layers:")
    print(f"  Overlap: {avg_overlap:.1f}/{TOP_K}")
    print(f"  Jaccard: {avg_jaccard:.3f}")
    print(f"  Consistent direction: {avg_consistent:.3f}")
    print(f"  SM→IC AUC: {avg_sm_ic:.3f}")
    print(f"  IC→SM AUC: {avg_ic_sm:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
