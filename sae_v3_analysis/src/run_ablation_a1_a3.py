#!/usr/bin/env python3
"""
A1: PCA dimension sensitivity + A3: Nonlinear classifier comparison.
CPU-only, uses existing extracted data.
"""
import sys, json, numpy as np, time, warnings
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
from config import PARADIGMS, LLAMA_PARADIGMS

np.random.seed(42)
OUT = Path(__file__).parent.parent / "results" / "json"


def load_sae_dp(cfg, layer):
    npz = Path(cfg['sae_dir']) / f'sae_features_L{layer}.npz'
    if not npz.exists(): return None, None
    d = np.load(npz, allow_pickle=False)
    is_last = d['is_last_round'].astype(bool)
    dp = np.where(is_last)[0]
    shape = tuple(d['shape'])
    row, col, val = d['row_indices'], d['col_indices'], d['values']
    mask = np.isin(row, dp)
    rmap = {old: new for new, old in enumerate(dp)}
    rnew = np.array([rmap[r] for r in row[mask]])
    dense = np.zeros((len(dp), shape[1]), dtype=np.float32)
    dense[rnew, col[mask]] = val[mask]
    active = (dense > 0).mean(0) >= 0.01
    labels = (d['game_outcomes'][dp] == 'bankruptcy').astype(int)
    return dense[:, active], labels


def classify(X, y, n_pca, clf_name='logreg'):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    nc = min(n_pca, Xs.shape[1], Xs.shape[0]-1) if n_pca > 0 else Xs.shape[1]
    if n_pca > 0:
        pca = PCA(n_components=nc)
        Xs = pca.fit_transform(Xs)
        ev = pca.explained_variance_ratio_.sum()
    else:
        ev = 1.0

    if clf_name == 'logreg':
        clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)
    elif clf_name == 'mlp':
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    elif clf_name == 'svm':
        clf = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(Xs, y):
        clf.fit(Xs[tr], y[tr])
        proba = clf.predict_proba(Xs[te])[:, 1]
        if len(np.unique(y[te])) >= 2:
            aucs.append(roc_auc_score(y[te], proba))
    return np.mean(aucs) if aucs else 0, ev, nc


def main():
    start = time.time()
    results = {'timestamp': datetime.now().isoformat()}

    # Datasets
    datasets = [
        ('gemma_sm', PARADIGMS['sm'], 22),
        ('gemma_ic', PARADIGMS['ic'], 22),
        ('gemma_mw', PARADIGMS['mw'], 30),
        ('llama_sm', LLAMA_PARADIGMS['sm'], 22),
        ('llama_ic', LLAMA_PARADIGMS['ic'], 22),
        ('llama_mw', LLAMA_PARADIGMS['mw'], 22),
    ]

    # === A1: PCA sensitivity ===
    print("=" * 60)
    print("A1: PCA DIMENSION SENSITIVITY")
    print("=" * 60)
    pca_dims = [10, 20, 50, 100, 200, 0]  # 0 = no PCA
    a1_results = {}

    for name, cfg, layer in datasets:
        X, y = load_sae_dp(cfg, layer)
        if X is None:
            print(f"  {name}: data not found")
            continue
        print(f"\n  {name} (n={len(y)}, BK={y.sum()}, features={X.shape[1]})")
        row = {}
        for dim in pca_dims:
            dim_label = str(dim) if dim > 0 else 'full'
            auc, ev, actual_nc = classify(X, y, dim, 'logreg')
            row[dim_label] = {'auc': round(auc, 4), 'explained_var': round(ev, 4), 'actual_nc': actual_nc}
            print(f"    PCA={dim_label:>4}: AUC={auc:.4f}, EV={ev:.3f}, nc={actual_nc}")
        a1_results[name] = row

    results['a1_pca_sensitivity'] = a1_results

    # === A3: Classifier comparison ===
    print("\n" + "=" * 60)
    print("A3: CLASSIFIER COMPARISON (PCA=50)")
    print("=" * 60)
    classifiers = ['logreg', 'mlp', 'svm']
    a3_results = {}

    for name, cfg, layer in datasets:
        X, y = load_sae_dp(cfg, layer)
        if X is None: continue
        print(f"\n  {name} (n={len(y)}, BK={y.sum()})")
        row = {}
        for clf_name in classifiers:
            auc, _, _ = classify(X, y, 50, clf_name)
            row[clf_name] = round(auc, 4)
            print(f"    {clf_name:>6}: AUC={auc:.4f}")
        a3_results[name] = row

    results['a3_classifier_comparison'] = a3_results

    # Summary
    print("\n" + "=" * 60)
    print("A1 SUMMARY: Is PCA=50 robust?")
    print("=" * 60)
    for name, row in a1_results.items():
        aucs = {k: v['auc'] for k, v in row.items()}
        auc50 = aucs.get('50', 0)
        auc_full = aucs.get('full', 0)
        saturated = all(abs(aucs.get(str(d), 0) - auc50) < 0.02 for d in [100, 200] if str(d) in aucs)
        print(f"  {name}: PCA50={auc50:.4f}, Full={auc_full:.4f}, Saturated at 50: {'YES' if saturated else 'NO'}")

    print("\n" + "=" * 60)
    print("A3 SUMMARY: Is LogReg sufficient?")
    print("=" * 60)
    for name, row in a3_results.items():
        lr = row.get('logreg', 0)
        mlp = row.get('mlp', 0)
        svm = row.get('svm', 0)
        linear_ok = max(mlp, svm) - lr < 0.02
        print(f"  {name}: LogReg={lr:.4f}, MLP={mlp:.4f}, SVM={svm:.4f}, Linear sufficient: {'YES' if linear_ok else 'NO'}")

    elapsed = time.time() - start
    results['elapsed_seconds'] = round(elapsed)
    print(f"\nTotal time: {elapsed/60:.1f} min")

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert(i) for i in obj]
        return obj

    out_file = OUT / f"ablation_a1_a3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"Saved to {out_file}")


if __name__ == '__main__':
    main()
