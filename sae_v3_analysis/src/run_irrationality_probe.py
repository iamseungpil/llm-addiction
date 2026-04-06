"""
Per-Turn Irrationality Probe: Full Layer Sweep
================================================
의도: 각 턴에서 모델의 선택이 얼마나 비합리적인지를 hidden state에서 예측.
      전체 layer에 걸쳐 어디에서 비합리성 신호가 가장 강한지 찾는다.

타깃 (Section 2의 지표와 직접 대응):
  - bet_ratio: 연속, bet/balance (I_BA의 per-turn 버전)
  - loss_chasing: 이진, 손실 후 bet_ratio 증가 여부 (I_LC의 per-turn 버전)
  - extreme_bet: 이진, bet/balance >= 0.5 (I_EC의 per-turn 버전)

통제 변수: round_number, balance, prompt_combo
핵심 질문: 통제 변수를 넘어서는 hidden state 신호가 있는가?
"""

import numpy as np, json, sys
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def load_and_label(hs_path, games_path):
    """Load hidden states and compute per-turn irrationality labels."""
    d = np.load(hs_path, allow_pickle=True)
    games = json.load(open(games_path))['results']

    hs = d['hidden_states']  # (n, n_layers, hidden_dim)
    n_layers = hs.shape[1]

    # Build per-turn labels from metadata
    labels = {
        'bet_ratio': d['bet_ratios'].astype(float) if 'bet_ratios' in d else None,
        'round_num': d['round_nums'].astype(float),
        'balance': d['balances'].astype(float),
        'combo': d['prompt_conditions'],
        'bt': d['bet_types'],
        'outcome': d['game_outcomes'],
        'action': d['actions'] if 'actions' in d else None,
    }

    # Compute loss_chasing and extreme_bet from game logs
    gids = d['game_ids']
    loss_chasing = np.zeros(len(gids))
    extreme_bet = np.zeros(len(gids))

    for i, gid in enumerate(gids):
        g = games[gid]
        rn = int(labels['round_num'][i]) - 1  # 0-indexed
        decs = g['decisions']
        hist = g.get('history', [])

        # Current bet ratio
        br = float(labels['bet_ratio'][i]) if labels['bet_ratio'] is not None else 0

        # Extreme betting
        extreme_bet[i] = 1 if br >= 0.5 else 0

        # Loss chasing: did bet_ratio increase after a loss?
        if rn > 0 and rn - 1 < len(hist):
            prev_loss = not hist[rn-1].get('win', str(hist[rn-1].get('result','')) == 'W')
            if prev_loss and rn - 1 < len(decs):
                prev_bet = float(decs[rn-1].get('parsed_bet', decs[rn-1].get('bet', 10)))
                prev_bal = float(decs[rn-1].get('balance_before', 100))
                prev_ratio = min(prev_bet / prev_bal, 1.0) if prev_bal > 0 else 0
                if br > prev_ratio:
                    loss_chasing[i] = 1

    labels['loss_chasing'] = loss_chasing
    labels['extreme_bet'] = extreme_bet

    return hs, labels, n_layers


def probe_layer(hs_layer, target, covariates, is_binary=False, n_perm=30):
    """Probe a single layer for irrationality, controlling for covariates."""
    n = len(target)

    sc = StandardScaler()
    X_hs = sc.fit_transform(hs_layer)
    X_full = np.column_stack([X_hs, covariates])

    if is_binary:
        if target.sum() < 10 or (1-target).sum() < 10:
            return {'metric': 0.5, 'perm_mean': 0.5, 'p': 1.0, 'n': n}
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Full model (HS + covariates)
        aucs_full = []
        for tr, te in kf.split(X_full, target):
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', random_state=42)
            clf.fit(X_full[tr], target[tr])
            yp = clf.predict_proba(X_full[te])[:, 1]
            if len(np.unique(target[te])) == 2:
                aucs_full.append(roc_auc_score(target[te], yp))
        # Covariates-only baseline
        aucs_base = []
        for tr, te in kf.split(covariates, target):
            clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', random_state=42)
            clf.fit(covariates[tr], target[tr])
            yp = clf.predict_proba(covariates[te])[:, 1]
            if len(np.unique(target[te])) == 2:
                aucs_base.append(roc_auc_score(target[te], yp))

        metric_full = np.mean(aucs_full) if aucs_full else 0.5
        metric_base = np.mean(aucs_base) if aucs_base else 0.5
        metric_name = 'AUC'
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_full, r2_base = [], []
        for tr, te in kf.split(X_full):
            r2_full.append(r2_score(target[te], Ridge(100.).fit(X_full[tr], target[tr]).predict(X_full[te])))
            r2_base.append(r2_score(target[te], Ridge(100.).fit(covariates[tr], target[tr]).predict(covariates[te])))
        metric_full = np.mean(r2_full)
        metric_base = np.mean(r2_base)
        metric_name = 'R²'

    # Within-combo permutation for the GAIN (full - base)
    gain = metric_full - metric_base
    perm_gains = []
    combos = covariates[:, -1] if covariates.shape[1] > 3 else np.zeros(n)  # last col might be combo
    for _ in range(n_perm):
        tgt_p = target.copy()
        # Simple permutation (not within-combo for speed)
        tgt_p = np.random.permutation(tgt_p)
        if is_binary:
            aucs_p = []
            for tr, te in kf.split(X_full, tgt_p):
                if len(np.unique(tgt_p[te])) < 2: continue
                clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', random_state=42)
                clf.fit(X_full[tr], tgt_p[tr])
                yp = clf.predict_proba(X_full[te])[:, 1]
                if len(np.unique(tgt_p[te])) == 2:
                    aucs_p.append(roc_auc_score(tgt_p[te], yp))
            mp = np.mean(aucs_p) if aucs_p else 0.5
        else:
            r2p = [r2_score(tgt_p[te], Ridge(100.).fit(X_full[tr], tgt_p[tr]).predict(X_full[te])) for tr, te in kf.split(X_full)]
            mp = np.mean(r2p)
        perm_gains.append(mp - metric_base)

    p = (sum(1 for pg in perm_gains if pg >= gain) + 1) / (n_perm + 1)

    return {
        'full': round(metric_full, 4),
        'base': round(metric_base, 4),
        'gain': round(gain, 4),
        'perm_mean_gain': round(np.mean(perm_gains), 4),
        'p': round(p, 4),
        'n': n,
        'metric_name': metric_name,
    }


def main():
    # Check if all-layers file exists
    all_layers_path = "/home/v-seungplee/data/llm-addiction/sae_features_v3/slot_machine/llama/hidden_states_dp_all_layers.npz"
    five_layers_path = "/home/v-seungplee/data/llm-addiction/sae_features_v3/slot_machine/llama/hidden_states_dp.npz"
    games_path = "/home/v-seungplee/data/llm-addiction/behavioral/slot_machine/llama_v4_role/final_llama_20260315_062428.json"

    from pathlib import Path
    if Path(all_layers_path).exists():
        print("Using ALL LAYERS data")
        hs_path = all_layers_path
    else:
        print("Using 5-layer data (all-layers not yet available)")
        hs_path = five_layers_path

    hs, labels, n_layers = load_and_label(hs_path, games_path)

    # Filter: Variable games only, betting decisions only
    var_bet = (labels['bt'] == 'variable') & (labels['bet_ratio'] > 0)

    print(f"\nData: {hs.shape[0]} total, {var_bet.sum()} variable+betting")
    print(f"Layers: {n_layers}")
    print(f"Targets: bet_ratio mean={labels['bet_ratio'][var_bet].mean():.3f}, "
          f"extreme_bet={labels['extreme_bet'][var_bet].mean():.1%}, "
          f"loss_chasing={labels['loss_chasing'][var_bet].mean():.1%}")

    # Covariates: round, balance, combo (one-hot)
    enc = OneHotEncoder(sparse_output=False, drop='first')
    combo_oh = enc.fit_transform(labels['combo'][var_bet].reshape(-1, 1))
    covariates = np.column_stack([
        labels['round_num'][var_bet],
        labels['balance'][var_bet],
        combo_oh,
    ])

    # Run sweep
    targets_config = [
        ('bet_ratio', labels['bet_ratio'][var_bet], False),
        ('extreme_bet', labels['extreme_bet'][var_bet], True),
        ('loss_chasing', labels['loss_chasing'][var_bet], True),
    ]

    print(f"\n{'='*80}")
    print(f"LAYER SWEEP: Per-Turn Irrationality Probe")
    print(f"{'='*80}")

    for tgt_name, tgt_vals, is_bin in targets_config:
        print(f"\n--- {tgt_name} ({'binary' if is_bin else 'continuous'}) ---")
        print(f"  {'Layer':>6} {'Full':>8} {'Base':>8} {'Gain':>8} {'p':>6}")

        best_gain, best_layer = -999, -1
        for li in range(n_layers):
            result = probe_layer(hs[var_bet, li, :], tgt_vals, covariates, is_bin)
            sig = '***' if result['p'] < 0.01 else ('**' if result['p'] < 0.05 else ('*' if result['p'] < 0.1 else ''))
            print(f"  L{li:>3d}: {result['full']:>8.4f} {result['base']:>8.4f} {result['gain']:>+8.4f} {result['p']:>6.3f} {sig}")

            if result['gain'] > best_gain:
                best_gain = result['gain']
                best_layer = li

        print(f"  Best: L{best_layer} (gain={best_gain:+.4f})")


if __name__ == "__main__":
    main()
