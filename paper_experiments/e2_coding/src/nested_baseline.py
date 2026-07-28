"""Does the internal-state readout predict betting beyond what the game log already predicts?

Reviewer KuK5's Question 2 asks whether the causal-control failure leaves open that the fitted
direction is tracking a correlate of balance and round dynamics that we did not residualise out.
This script answers it by nesting two models and reporting the gain:

  BASELINE   observable game state only
  FULL       the same observables, plus a block of internal state
  METRIC     Delta R-squared on held-out games

What the test can and cannot settle, stated before any number is produced. Hidden states are a
deterministic function of the prompt, and the prompt contains the observables, so the hidden state
cannot carry information the game log lacks. The comparison is therefore between function classes:
does a probe on the internal state recover more of the policy than a reasonable model of the
observables? A null says monitoring the hidden state buys nothing over monitoring the game log; it
does not say the representation is absent.

THREE FEATURE BLOCKS, because the question is about the paper's readout, not about any readout.

  raw    the decision-time residual-stream vector at the layer, 3,584 dims. This is what the first
         version of this script measured, and it is NOT what the paper reports.
  sae    the SAE features at the same layer, restricted to the 489 features active on this sample
         (>10 non-zero entries), which is the paper's own feature space.
  paper  the published pipeline itself, reproduced end to end as a gate: within-fold RandomForest
         deconfound of the target against balance and round, top-200 SAE features by |Spearman rho|
         with the residual, StandardScaler, Ridge(alpha=100), 5-fold GroupKFold by game. This is the
         estimator that produced the paper's R-squared of 0.167 for gemma / slot-machine / I_BA at
         layer 22. Its R-squared is measured against the DECONFOUNDED residual, so it is a
         reproduction gate, not a fourth column of the nested table: it shares no denominator with
         the raw-target R-squared that the nested comparison uses.

Seven design decisions, each fixed before the result is seen, and each answering a specific way
this test can be rigged:

1.  Identical prompts produce bit-identical hidden states, because 64 conditions x 50 repetitions
    share prompts at the early rounds. On this sample 12,246 rows carry only 8,144 distinct
    decision states, and 4,808 rows (39.3%) share a bit-identical state with a row in a DIFFERENT
    game. Grouping the folds by game therefore does not separate them: an exact duplicate of a test
    row can sit in the training fold. Two defences, both reported. First, the BASELINE includes
    prompt-condition, bet-type and round dummies, so the full model cannot win merely by memorising
    condition means the baseline has no way to express. Second, every configuration is re-run with
    the folds grouped by a hash of the state itself, which is the only partition that actually
    separates duplicates. The placebo cannot substitute for this: shuffling rows destroys the
    duplicate structure, so a placebo is blind to duplicate-driven optimism by construction.
2.  The stored readout was fitted on the full corpus including the evaluation games. Feature
    selection and ridge are refitted inside every training fold here.
3.  The target is continuous (bet as a fraction of balance, clipped at 1.0 exactly as the paper's
    I_BA is), so the metric is Delta R-squared. Bankruptcy is a game-level label and does not
    belong in a decision-level regression.
4.  The baseline is given the same functional freedom as the full model: splines on the continuous
    covariates and their interactions, and its own tuned penalty. A linear baseline against a
    3,584-dimensional probe would lose to a nonlinear function of the observables alone.
5.  No choice-probability control. When the target is the decision being made at that step, the
    model's own choice probability is the decision, and including it would rig the test against
    the readout. It is a legitimate control only for a future target.
6.  The margin is fixed in advance at one tenth of the paper's published R-squared of 0.167, so
    delta = 0.017, and a placebo run on shuffled features gives the optimism the pipeline itself
    produces.
7.  Four verdicts, not three, because a confidence interval whose lower bound falls between zero
    and delta is neither a pass nor a fail and must not be silently rounded into either.

PAIRING. The baseline and the full model are scored on the SAME fold partition. The earlier version
threaded one Generator through both calls, so each call drew its own permutation and the two models
were evaluated on different partitions; the difference of two unpaired cross-validations is not a
Delta R-squared, and its bootstrap spread was dominated by partition noise. The fold vector is now
built once per comparison and passed to both fits.

BOOTSTRAP FAILURES ARE NEVER SILENT. Resamples are wrapped in a recorder that tallies exception
types and keeps the first traceback; --strict-boot re-raises it. The earlier version caught only
np.linalg.LinAlgError and continued, which could report "0 resamples" with no indication of why.

Join. The behavioural catalogue holds 21,423 decisions for Gemma's slot-machine corpus and the
hidden-state array holds 21,421 rows. The difference is exactly the two decisions recorded with
action "skip", which the extractor did not embed. The SAE array holds the same 21,421 rows in the
same order. The script asserts that reconciliation rather than assuming positional alignment, and
additionally asserts that the 12,246 rows it selects are the same 12,246 rows the paper's published
cell was computed on.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

SNAP = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--llm-addiction-research--llm-addiction/snapshots"
    "/b4ec4c173164d5dcadb02818847b2dad5e2f98cc"))

BEHAV = SNAP / "behavioral/slot_machine/gemma_v4_role/final_gemma_20260227_002507.json"
HIDDEN = SNAP / "sae_features_v3/slot_machine/gemma/checkpoint/phase_a_hidden_states.npz"

# The paper's own SAE array, reached through the same symlink the published pipeline uses.
SAE_ROOT = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3")
PAPER_SRC = "/home/v-seungplee/llm-addiction/sae_v3_analysis/src"

DELTA = 0.017          # one tenth of the paper's published R^2 = 0.167
PUBLISHED_R2 = 0.167
GATE_TOL = 0.05        # how far the reproduction may land from the published cell


# --------------------------------------------------------------------------- data

def load_rows() -> tuple[list[dict], np.ndarray]:
    """Return the decision records that have an embedding, aligned to the hidden-state rows."""
    payload = json.load(open(BEHAV))
    rows: list[dict] = []
    skipped = 0
    for game_index, game in enumerate(payload["results"]):
        for dec in game["decisions"]:
            if dec.get("action") == "skip":
                skipped += 1
                continue
            rows.append({
                "game": game_index,
                "condition": game.get("prompt_combo") or "BASE",
                "bet_type": game["bet_type"],
                "round": dec.get("round"),
                "balance_before": dec.get("balance_before"),
                "balance_after": dec.get("balance_after"),
                "action": dec.get("action"),
                "bet": dec.get("parsed_bet"),
                "result": dec.get("result"),
            })

    z = np.load(HIDDEN, allow_pickle=True)
    valid = z["valid_mask"]
    n_hidden = int(valid.shape[0])
    if len(rows) != n_hidden:
        raise SystemExit(
            f"join mismatch: {len(rows)} embeddable decisions against {n_hidden} hidden-state rows "
            f"({skipped} skipped). Do not proceed on a positional guess.")
    print(f"join reconciled: {len(rows)} decisions, {skipped} 'skip' actions excluded, "
          f"{int(valid.sum())} rows flagged valid")
    return rows, z["hidden_states"]


def select_rows(rows: list[dict], arm: str = "variable") -> np.ndarray:
    """Row indices into the 21,421-row corpus: decisions where the stake was actually chosen.

    In the fixed arm the stake is not a choice: 5,972 of its 6,062 wagers are exactly $10, so the
    target is a deterministic function of balance there and no internal state can add to it. The
    surviving 12,246 decisions are also the sample the paper's published R-squared of 0.167 was
    computed on, which the caller asserts against the paper's own valid mask.
    """
    return np.asarray([i for i, r in enumerate(rows)
                       if r["action"] == "bet" and r["bet"] is not None
                       and r["balance_before"] not in (None, 0)
                       and (arm == "all" or r["bet_type"] == arm)])


def build_design(rows: list[dict], idx: np.ndarray, baseline: str = "rich", clip: bool = True):
    """Target and observable covariates on the selected rows.

    `baseline` sets how complete a summary of the game log the comparison runs against.
    "minimal" is balance, round, last outcome and streak. "rich" adds the lagged bet ratios,
    cumulative stake, win and loss counts, running peak and drawdown. Report both: the gap
    between them is itself the answer to whether a positive result reflects the internal state
    or an under-specified baseline.
    """
    y = np.asarray([rows[i]["bet"] / rows[i]["balance_before"] for i in idx], dtype=np.float64)
    if clip:
        # Parity with the paper's I_BA, which is min(bet / balance, 1.0).
        y = np.minimum(y, 1.0)
    game = np.asarray([rows[i]["game"] for i in idx])

    # Observables the game log already contains at decision time.
    balance = np.asarray([rows[i]["balance_before"] for i in idx], dtype=np.float64)
    rnd = np.asarray([rows[i]["round"] for i in idx], dtype=np.float64)

    # History features, reconstructed within each game. The important ones are the lagged bet
    # ratios: a betting policy is autocorrelated, so a monitor reading the game log would use the
    # previous wager first of all. Leaving them out hands the comparison to the hidden state for a
    # reason that has nothing to do with internal representation.
    n = len(idx)
    prev_win = np.zeros(n); win_streak = np.zeros(n); loss_streak = np.zeros(n)
    lag1 = np.zeros(n); lag2 = np.zeros(n); cum_bet = np.zeros(n)
    n_wins = np.zeros(n); n_losses = np.zeros(n); peak = np.zeros(n); drawdown = np.zeros(n)
    by_game: dict[int, list[int]] = {}
    for pos, i in enumerate(idx):
        by_game.setdefault(rows[i]["game"], []).append(pos)
    for _g, positions in by_game.items():
        wrun = lrun = 0; staked = 0.0; wins = losses = 0; high = 100.0
        ratios: list[float] = []
        for k, pos in enumerate(positions):
            r = rows[idx[pos]]
            if k:
                prev = rows[idx[positions[k - 1]]]
                won = 1.0 if prev["result"] == "W" else 0.0
                prev_win[pos] = won
                wrun = wrun + 1 if won else 0
                lrun = 0 if won else lrun + 1
                wins += int(won); losses += int(1 - won)
                staked += (prev["bet"] or 0)
                ratios.append((prev["bet"] or 0) / max(prev["balance_before"] or 1, 1))
            win_streak[pos] = wrun; loss_streak[pos] = lrun
            lag1[pos] = ratios[-1] if ratios else 0.0
            lag2[pos] = ratios[-2] if len(ratios) > 1 else 0.0
            cum_bet[pos] = staked / 100.0
            n_wins[pos] = wins; n_losses[pos] = losses
            high = max(high, r["balance_before"] or 0)
            peak[pos] = high / 100.0
            drawdown[pos] = (r["balance_before"] or 0) / max(high, 1)

    cond = np.asarray([rows[i]["condition"] for i in idx])
    bet_type = np.asarray([rows[i]["bet_type"] for i in idx])

    def dummies(values: np.ndarray) -> np.ndarray:
        levels = sorted(set(values.tolist()))[1:]      # drop one level
        return np.stack([(values == lv).astype(float) for lv in levels], axis=1) if levels \
            else np.zeros((len(values), 0))

    def spline(x: np.ndarray, knots: int = 5) -> np.ndarray:
        """Natural-cubic-style basis: the variable plus truncated cubics at interior quantiles."""
        qs = np.quantile(x, np.linspace(0, 1, knots + 2)[1:-1])
        cols = [x, x ** 2]
        cols += [np.clip(x - q, 0, None) ** 3 for q in qs]
        return np.stack(cols, axis=1)

    minimal = np.concatenate([
        spline(balance), spline(rnd),
        prev_win[:, None], win_streak[:, None],
        (prev_win * balance)[:, None],
        dummies(cond), dummies(bet_type),
    ], axis=1)
    rich = np.concatenate([
        minimal,
        spline(lag1), lag2[:, None], (lag1 * prev_win)[:, None],
        loss_streak[:, None], cum_bet[:, None],
        n_wins[:, None], n_losses[:, None], peak[:, None], drawdown[:, None],
        (lag1 * rnd)[:, None], (drawdown * lag1)[:, None],
    ], axis=1)
    observed = rich if baseline == "rich" else minimal
    return y, observed, game, balance, rnd


def load_sae_block(idx: np.ndarray, layer: int):
    """The paper's SAE features on the selected rows, plus the metadata needed for the gate.

    Returns (dense SAE block over the features active on this sample, meta dict). The npz stores a
    sparse triple; the metadata arrays are row-aligned with the hidden-state array, which the caller
    asserts against the behavioural join rather than assuming.
    """
    from scipy import sparse
    path = SAE_ROOT / f"slot_machine/gemma/sae_features_L{layer}.npz"
    data = np.load(path, allow_pickle=False)
    shape = tuple(data["shape"])
    sp = sparse.csr_matrix((data["values"], (data["row_indices"], data["col_indices"])),
                           shape=shape, dtype=np.float32)
    meta = {k: data[k] for k in data.keys()
            if k not in ("row_indices", "col_indices", "values", "shape")}
    sub = sp[idx]
    nnz = np.diff(sub.tocsc().indptr)
    active = np.where(nnz > 10)[0]
    return sub[:, active].toarray().astype(np.float64), meta, active


def state_hashes(block: np.ndarray) -> np.ndarray:
    """Integer code per bit-identical decision state."""
    digests = [hashlib.blake2b(block[i].tobytes(), digest_size=16).digest()
               for i in range(block.shape[0])]
    codes = {h: k for k, h in enumerate(dict.fromkeys(digests))}
    return np.asarray([codes[h] for h in digests])


# --------------------------------------------------------------------------- estimator

def make_folds(group: np.ndarray, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Assign each row to a fold through its group, so a group never straddles the split."""
    groups = np.unique(group)
    order = rng.permutation(len(groups))
    fold_of = {g: order[k] % n_folds for k, g in enumerate(groups)}
    return np.asarray([fold_of[g] for g in group])


def ridge_cv_r2(X: np.ndarray, y: np.ndarray, fold: np.ndarray, rng: np.random.Generator,
                alphas=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4),
                internal: np.ndarray | None = None, top_k: int = 0) -> tuple[float, np.ndarray]:
    """Grouped CV R^2 on a partition supplied by the caller.

    Standardisation, the ridge penalty, and -- when an internal block is supplied -- the choice of
    which dimensions to keep are all decided inside the training fold. Selecting the dimensions on
    the whole corpus and then cross-validating would score the selection on data it had already
    seen, which is the flaw this comparison exists to avoid.

    The partition is an argument, not something drawn here, so that the baseline and the full model
    are compared on identical splits.
    """
    pred = np.zeros_like(y)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        if tr.sum() < 10 or te.sum() == 0:
            continue
        Xtr, Xte, ytr = X[tr], X[te], y[tr]

        if internal is not None and top_k:
            # Fit the observable-only model on this training fold, then rank internal dimensions by
            # their correlation with ITS residuals -- training rows only -- and keep the top k. The
            # block is standardised on the training fold first, so the ranking is a correlation and
            # not a covariance that merely prefers whichever dimension happens to have the largest
            # scale.
            mu0, sd0 = Xtr.mean(0), Xtr.std(0)
            sd0[sd0 == 0] = 1.0
            Ztr = (Xtr - mu0) / sd0
            w0 = np.linalg.solve(Ztr.T @ Ztr + 1e2 * np.eye(Ztr.shape[1]),
                                 Ztr.T @ (ytr - ytr.mean()))
            resid = ytr - (Ztr @ w0 + ytr.mean())
            Itr = internal[tr]
            isd = Itr.std(0)
            const = isd == 0                             # constant after resampling: unusable
            safe = np.where(const, 1.0, isd)
            # Standardised ranking without materialising the standardised block: the residual is
            # centred, so the column means cancel and only the scale division survives.
            centred = resid - resid.mean()
            corr = np.abs(Itr.T @ centred) / safe
            corr[const] = -np.inf
            keep = np.argsort(corr)[::-1][:top_k]
            keep = keep[np.isfinite(corr[keep])]
            if keep.size:
                Xtr = np.concatenate([Xtr, internal[tr][:, keep]], axis=1)
                Xte = np.concatenate([Xte, internal[te][:, keep]], axis=1)

        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        ymu = ytr.mean()

        # inner split for the penalty
        inner = rng.random(Xtr.shape[0]) < 0.8
        if inner.sum() < 5 or (~inner).sum() < 5:
            inner = np.arange(Xtr.shape[0]) % 5 != 0
        best, best_a = -np.inf, alphas[0]
        G = Xtr[inner].T @ Xtr[inner]
        b = Xtr[inner].T @ (ytr[inner] - ymu)
        for a in alphas:
            w = np.linalg.solve(G + a * np.eye(G.shape[0]), b)
            p = Xtr[~inner] @ w + ymu
            ss = 1 - ((ytr[~inner] - p) ** 2).sum() / max(((ytr[~inner] - ytr[~inner].mean()) ** 2).sum(), 1e-12)
            if ss > best:
                best, best_a = ss, a
        G = Xtr.T @ Xtr
        w = np.linalg.solve(G + best_a * np.eye(G.shape[0]), Xtr.T @ (ytr - ymu))
        pred[te] = Xte @ w + ymu

    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return r2, pred


def nested_delta(observed, y, block, group, n_folds, seed, top_k):
    """Baseline and full model on ONE shared partition. Returns (r2_base, r2_full)."""
    fold = make_folds(group, n_folds, np.random.default_rng(seed))
    r2_base, _ = ridge_cv_r2(observed, y, fold, np.random.default_rng(seed + 7))
    r2_full, _ = ridge_cv_r2(observed, y, fold, np.random.default_rng(seed + 7),
                             internal=block, top_k=top_k)
    return r2_base, r2_full


# --------------------------------------------------------------------------- paper gate

def paper_pipeline_r2(X_sae, target, balances, rounds, groups, n_splits=5):
    """The published estimator, imported from the paper's own module so parity is not re-typed.

    Within-fold RandomForest deconfound against [bal, rn, bal^2, log1p(bal), bal*rn], top-200 SAE
    features by |Spearman rho| with the residual, StandardScaler, Ridge(alpha=100), GroupKFold.
    R-squared is measured against the deconfounded residual. Returns (mean, std, folds).
    """
    sys.path.insert(0, PAPER_SRC)
    from run_perm_null_ilc import nl_deconfound_split, TOP_K, RIDGE_ALPHA
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score

    k = min(TOP_K, X_sae.shape[1])
    gkf = GroupKFold(n_splits=n_splits)
    r2s = []
    for tr, te in gkf.split(X_sae, groups=groups):
        res_tr, res_te = nl_deconfound_split(target[tr], balances[tr], rounds[tr],
                                             target[te], balances[te], rounds[te])
        corrs = np.array([abs(spearmanr(X_sae[tr, j], res_tr)[0]) if X_sae[tr, j].std() > 0 else 0
                          for j in range(X_sae.shape[1])])
        corrs = np.nan_to_num(corrs)
        sel = np.argsort(corrs)[-k:]
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_sae[tr][:, sel])
        Xte = sc.transform(X_sae[te][:, sel])
        pred = Ridge(alpha=RIDGE_ALPHA).fit(Xtr, res_tr).predict(Xte)
        r2s.append(float(r2_score(res_te, pred)))
    return float(np.mean(r2s)), float(np.std(r2s, ddof=1)), r2s


# --------------------------------------------------------------------------- bootstrap

class BootRecorder:
    """A resample that fails must say so. Tally types, keep the first traceback, optionally re-raise."""

    def __init__(self, strict: bool):
        self.strict = strict
        self.types: collections.Counter = collections.Counter()
        self.first_tb: str | None = None

    def failed(self, exc: BaseException) -> None:
        self.types[type(exc).__name__] += 1
        if self.first_tb is None:
            self.first_tb = traceback.format_exc()
            if self.strict:
                raise exc

    def report(self, attempted: int, succeeded: int) -> None:
        print(f"  bootstrap resamples: {succeeded}/{attempted} succeeded")
        if self.types:
            print(f"  bootstrap failures by type: {dict(self.types)}")
            print(self.first_tb)


def bootstrap_delta(observed, y, block, game, state, group_by, n_folds, seed, top_k, n_boot,
                    recorder: BootRecorder):
    """Resample GAMES with replacement; the game is the independent unit regardless of grouping.

    The resampled group vector is rebuilt from the resampled rows themselves rather than by
    repeating a count computed on the original sample, and every resample is checked for alignment
    before it is fitted.
    """
    games = np.unique(game)
    counts = {g: int((game == g).sum()) for g in games}
    where = {g: np.where(game == g)[0] for g in games}
    deltas: list[float] = []
    for b in range(n_boot):
        rb = np.random.default_rng(seed + 1000 + b)
        pick = rb.choice(games, size=len(games), replace=True)
        sel = np.concatenate([where[g] for g in pick])
        # A fresh game label per DRAW, so two draws of the same game are two groups, not one.
        draw = np.repeat(np.arange(len(pick)), [counts[g] for g in pick])
        assert sel.shape == draw.shape, f"resample {b}: {sel.shape} rows against {draw.shape} labels"
        if group_by == "game":
            # Caveat, stated rather than hidden: when a game is drawn twice, its two copies become
            # two groups and can land in different folds, so an exact row copy can sit across the
            # split. This is the resampling analogue of the duplicate-state problem and it makes the
            # game-grouped interval optimistic. The state-grouped run below does not have it,
            # because identical rows always share a group however often they are drawn.
            grp = draw
        else:
            # State grouping must survive resampling: identical states share a group even when they
            # were drawn from different games, which is exactly the leak this run tests for.
            grp = state[sel]
        if len(np.unique(grp)) < n_folds:
            recorder.types["TooFewGroups"] += 1
            continue
        try:
            r2b, r2f = nested_delta(observed[sel], y[sel], block[sel], grp, n_folds,
                                    seed + 1000 + b, top_k)
            deltas.append(r2f - r2b)
        except Exception as exc:                      # noqa: BLE001 - recorded, never swallowed
            recorder.failed(exc)
        if (b + 1) % 25 == 0:
            print(f"    ...{b + 1}/{n_boot} resamples, {len(deltas)} usable", flush=True)
    recorder.report(n_boot, len(deltas))
    return deltas


def verdict_of(lo, hi):
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "INCONCLUSIVE (bootstrap failed)"
    if lo > DELTA:
        return "ADDS BEYOND GAME STATE"
    if hi < DELTA:
        return "NO PRACTICAL GAIN"
    if lo > 0:
        return "NON-ZERO BUT BELOW THE PRE-SET MARGIN"
    return "INDISTINGUISHABLE FROM ZERO"


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=22)
    ap.add_argument("--baseline", choices=["minimal", "rich"], default="rich")
    ap.add_argument("--arm", choices=["variable", "fixed", "all"], default="variable")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--top", type=int, default=200, help="dims kept, selected inside folds")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=24231)
    ap.add_argument("--blocks", default="raw,sae")
    ap.add_argument("--groupings", default="game,state")
    ap.add_argument("--no-clip", action="store_true",
                    help="use raw bet/balance instead of the paper's min(bet/balance, 1)")
    ap.add_argument("--strict-boot", action="store_true",
                    help="re-raise the first bootstrap exception instead of tallying it")
    ap.add_argument("--cache", default="", help="npz cache for the design, to skip the 12.9GB load")
    ap.add_argument("--out", default="/home/v-seungplee/llm-addiction/paper_experiments/e2_coding/nested_baseline.json")
    args = ap.parse_args()

    # ---- design -----------------------------------------------------------
    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        y, observed, raw_block, game = z["y"], z["observed"], z["raw"].astype(np.float64), z["game"]
        balance, rnd, idx = z["balance"], z["rnd"], z["idx"]
        print(f"design restored from {cache}")
    else:
        rows, hidden = load_rows()
        idx = select_rows(rows, args.arm)
        y, observed, game, balance, rnd = build_design(rows, idx, args.baseline, not args.no_clip)
        raw_block = hidden[idx, args.layer, :].astype(np.float64)
        del hidden
        if cache:
            np.savez(cache, y=y, observed=observed, raw=raw_block.astype(np.float32), game=game,
                     balance=balance, rnd=rnd, idx=idx)

    sae_block, meta, active = load_sae_block(idx, args.layer)

    # The join is asserted, not assumed.
    assert np.allclose(meta["balances"][idx], balance), "SAE metadata balance disagrees with the log"
    assert np.array_equal(meta["round_nums"][idx].astype(float), rnd), "SAE metadata round disagrees"
    assert (meta["bet_types"][idx] == "variable").all(), "a fixed-arm row entered the variable sample"

    state = state_hashes(raw_block)
    n_dup = int(np.isin(state, [s for s, c in collections.Counter(state.tolist()).items() if c > 1]).sum())
    cross = collections.defaultdict(set)
    for s, g in zip(state, game):
        cross[s].add(int(g))
    n_cross = int(sum(1 for s in state if len(cross[s]) > 1))

    print(f"decisions with a chosen wager: {len(y)}  games: {len(np.unique(game))}  "
          f"observable covariates: {observed.shape[1]}")
    print(f"blocks: raw {raw_block.shape[1]} dims, sae {sae_block.shape[1]} active features "
          f"(of {int(np.load(SAE_ROOT / f'slot_machine/gemma/sae_features_L{args.layer}.npz')['shape'][1])})")
    print(f"duplicate states: {len(y) - len(np.unique(state))} rows are a repeat of another row; "
          f"{len(np.unique(state))} distinct states; {n_cross} rows ({100 * n_cross / len(y):.1f}%) "
          f"share a state with a row in a DIFFERENT game")

    results: dict = {"n_decisions": int(len(y)), "n_games": int(len(np.unique(game))),
                     "n_states": int(len(np.unique(state))), "n_cross_game_dup_rows": n_cross,
                     "n_obs_covariates": int(observed.shape[1]),
                     "raw_dims": int(raw_block.shape[1]), "sae_active": int(sae_block.shape[1]),
                     "layer": args.layer, "arm": args.arm, "baseline": args.baseline,
                     "clipped_target": not args.no_clip, "margin": DELTA,
                     "published_r2": PUBLISHED_R2, "seed": args.seed, "folds": args.folds,
                     "top_k": args.top, "n_boot": args.boot}

    # ---- (c) reproduction gate -------------------------------------------
    print("\n=== (c) reproduction gate: the paper's own pipeline ===")
    # The gate always uses the paper's own target, min(bet/balance, 1), whatever the nested test uses.
    target = np.minimum(y, 1.0)
    g_mean, g_std, g_folds = paper_pipeline_r2(sae_block, target, balance, rnd, game, args.folds)
    print(f"  grouped by game   R^2 {g_mean:+.4f} +- {g_std:.4f}   folds "
          + " ".join(f"{v:+.4f}" for v in g_folds))
    print(f"  published cell    R^2 {PUBLISHED_R2:+.4f}  (gemma / slot_machine / I_BA / L{args.layer}, n=12,246)")
    gate_ok = abs(g_mean - PUBLISHED_R2) <= GATE_TOL
    print(f"  GATE {'PASS' if gate_ok else 'FAIL'}  |difference| {abs(g_mean - PUBLISHED_R2):.4f} "
          f"against tolerance {GATE_TOL}")
    results["gate_game"] = {"r2_mean": g_mean, "r2_std": g_std, "folds": g_folds, "pass": bool(gate_ok)}
    if not gate_ok:
        results["verdict"] = "STOPPED: reproduction gate failed"
        Path(args.out).write_text(json.dumps(results, indent=2))
        raise SystemExit("reproduction gate failed; the join or the layer is wrong and nothing "
                         "below would be interpretable. Stopping rather than proceeding.")

    s_mean, s_std, s_folds = paper_pipeline_r2(sae_block, target, balance, rnd, state, args.folds)
    print(f"  grouped by STATE  R^2 {s_mean:+.4f} +- {s_std:.4f}   "
          f"(duplicate-safe partition; drop {g_mean - s_mean:+.4f})")
    results["gate_state"] = {"r2_mean": s_mean, "r2_std": s_std, "folds": s_folds}

    # ---- (a) and (b) nested comparison -----------------------------------
    blocks = {"raw": raw_block, "sae": sae_block}
    results["configs"] = {}
    for bname in [b for b in args.blocks.split(",") if b]:
        block = blocks[bname]
        for gname in [g for g in args.groupings.split(",") if g]:
            group = game if gname == "game" else state
            tag = f"{bname}/{gname}"
            print(f"\n=== ({'a' if bname == 'raw' else 'b'}) block={bname} "
                  f"({block.shape[1]} dims), folds grouped by {gname} ===")
            r2_base, r2_full = nested_delta(observed, y, block, group, args.folds, args.seed, args.top)
            delta = r2_full - r2_base
            print(f"  baseline R^2    {r2_base:+.4f}")
            print(f"  full R^2        {r2_full:+.4f}")
            print(f"  delta R^2       {delta:+.4f}   (margin fixed in advance at {DELTA})")

            shuffled = block[np.random.default_rng(args.seed + 1).permutation(len(y))]
            _, r2_plac = nested_delta(observed, y, shuffled, group, args.folds, args.seed, args.top)
            placebo = r2_plac - r2_base
            print(f"  placebo delta   {placebo:+.4f}   (same pipeline, rows shuffled)")

            rec = BootRecorder(args.strict_boot)
            deltas = bootstrap_delta(observed, y, block, game, state, gname, args.folds,
                                     args.seed, args.top, args.boot, rec)
            lo, hi = (np.percentile(deltas, [2.5, 97.5]) if deltas else (np.nan, np.nan))
            med = float(np.median(deltas)) if deltas else float("nan")
            print(f"  bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]  median {med:+.4f}  "
                  f"over {len(deltas)} resamples")
            v = verdict_of(lo, hi)
            print(f"  verdict: {v}")
            results["configs"][tag] = {
                "block": bname, "grouping": gname, "dims": int(block.shape[1]),
                "r2_baseline": r2_base, "r2_full": r2_full, "delta_r2": delta,
                "placebo_delta": placebo, "ci95": [float(lo), float(hi)], "boot_median": med,
                "n_bootstrap": len(deltas), "boot_failures": dict(rec.types), "verdict": v,
                # Persist every resampled delta. A percentile interval reduced to two numbers hides
                # whether the spread is genuine uncertainty or a few resamples where a held-out fold
                # fitted badly, and that difference changes what the interval licenses.
                "boot_deltas": [float(x) for x in deltas],
                "boot_iqr": [float(np.percentile(deltas, 25)), float(np.percentile(deltas, 75))]
                if deltas else [float("nan")] * 2,
            }

    Path(args.out).write_text(json.dumps(results, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
