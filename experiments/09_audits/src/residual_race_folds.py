"""residual_race with per-fold R2 kept, so the increment's own fold spread can be read.

Additive: `residual_race.py` is untouched. The loop below is copied from it verbatim
except that run() returns the per-fold list instead of only the mean, which lets the
three feature sets be differenced fold by fold. The folds are identical across the
three runs because GroupKFold is deterministic on the same groups, so fold i of one
run is the same rows as fold i of another. The RF deconfound is seeded
(random_state=42), so this reproduces the published means or the pipeline is not
deterministic -- either way that is worth knowing before any +- goes into a letter.
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '07_sae_readout' / 'src'))
import nested_baseline as nb
from run_perm_null_ilc import nl_deconfound_split, TOP_K, RIDGE_ALPHA
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

d = np.load('design_v2.npz', allow_pickle=True)
y, observed, raw, game = d['y'], d['observed'], d['raw'].astype(np.float64), d['game']
balance, rnd, idx = d['balance'], d['rnd'], d['idx']
sae, meta, active = nb.load_sae_block(idx, 22)
state = nb.state_hashes(raw)
target = np.minimum(y, 1.0)


def run(X, groups, label, topk=True):
    gkf = GroupKFold(n_splits=5)
    r2s = []
    for tr, te in gkf.split(X, groups=groups):
        res_tr, res_te = nl_deconfound_split(target[tr], balance[tr], rnd[tr],
                                             target[te], balance[te], rnd[te])
        if topk and X.shape[1] > TOP_K:
            c = np.array([abs(spearmanr(X[tr, j], res_tr)[0]) if X[tr, j].std() > 0 else 0
                          for j in range(X.shape[1])])
            sel = np.argsort(np.nan_to_num(c))[-TOP_K:]
        else:
            sel = np.arange(X.shape[1])
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr][:, sel])
        Xte = sc.transform(X[te][:, sel])
        r2s.append(r2_score(res_te, Ridge(alpha=RIDGE_ALPHA).fit(Xtr, res_tr).predict(Xte)))
    r2s = np.array(r2s)
    print(f'  {label:34s} R2 {r2s.mean():+.4f} +- {r2s.std(ddof=1):.4f}   '
          f'folds {" ".join(f"{v:+.4f}" for v in r2s)}', flush=True)
    return r2s


both = np.concatenate([observed, sae], axis=1)
for gname, groups in (('game', game), ('state', state)):
    print(f'=== paper pipeline, deconfounded residual target, folds grouped by {gname} ===')
    r_sae = run(sae, groups, 'SAE features (the paper cell)')
    r_obs = run(observed, groups, 'game-log observables only')
    r_both = run(both, groups, 'observables + SAE')
    # The quantity the letter needs: the increment measured inside each fold, so the
    # fold-to-fold variation the two models share cancels instead of adding.
    diff = r_both - r_obs
    print(f'  {"increment (both - observables), paired":34s} '
          f'{diff.mean():+.4f} +- {diff.std(ddof=1):.4f}   '
          f'folds {" ".join(f"{v:+.4f}" for v in diff)}')
    print(f'  {"  unpaired SD for comparison":34s} '
          f'{np.hypot(r_both.std(ddof=1), r_obs.std(ddof=1)):.4f}')
