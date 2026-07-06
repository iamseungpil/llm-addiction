"""Q1 rung-1: FREE post-loss re-analysis of the existing Wave-1/2 SM rollouts.

The §4.1 third indicator I_LC (loss-chasing) is post-loss escalation — so the
FIRST, zero-GPU test of "does ONE axis drive I_BA *and* I_LC" (goals-v2 Q1) is
a re-slice of rollouts we already have: join every Wave-1 (sec4_p0) and Wave-2
(sec4_w2) steer trial back to the exact −G pool state it replayed, label that
state post-loss vs post-win from the PREVIOUS displayed round's outcome, and
ask whether steering the axis raises betting MORE on post-loss states (an
LC-like dose x post-loss INTERACTION).

Three pieces, all reused by later rungs:
  join_trials_to_states  seed -> pool-state join, VALIDATED per row against the
                         recorded source_state (round_idx / balance / combo) so
                         a wrong offset / seed_base / pool size fails loudly
                         instead of silently mislabeling every trial.
  label_postloss         single source of truth for the post-loss label — the
                         runner's Q1 rung-2 state_filter imports THIS function,
                         so rung-1 relabeling and rung-2 conditioning can never
                         drift apart. Reads the previous round exactly as
                         prompts.build_prompt displays it (hist[round_idx-1],
                         win flag h.get('win', result=='W')).
  analyze_postloss       per axis (behavioural / shared / readout): mean
                         bet_ratio by (dose x postloss/postwin) cell, Spearman
                         per condition, OLS dose x post-loss interaction, and a
                         Q1_RUNG1 verdict in {LC_LIKE_INTERACTION,
                         NO_INTERACTION, UNDERPOWERED}.

CLI (run-time only; unit tests are synthetic):
  python -m multilayer_causal.src.postloss_analysis \
      [--results-dirs multilayer_causal/results/sec4_p0 \
                      multilayer_causal/results/sec4_w2] [--from-hf]
writes multilayer_causal/results/sec4_w3/postloss_rung1.json + .png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_MLC = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIRS = (_MLC / "results" / "sec4_p0", _MLC / "results" / "sec4_w2")
OUT_JSON = _MLC / "results" / "sec4_w3" / "postloss_rung1.json"
OUT_PNG = _MLC / "results" / "sec4_w3" / "postloss_rung1.png"

# runner.run_arm trial seeding: seed = seed_base + i * 997 (alpha-independent).
SEED_STEP = 997
# Wave-1/2 SM arm defaults (configs/arms_sec4_p0.yaml / arms_sec4_w2.yaml).
W12_SEED_BASE = 2000042
W12_OFFSET = 300
# runner.run_arm pool sizing for those arms: n(200) + state_offset(300) + the
# twin=="G" pad(50) — the join MUST rebuild a pool of this exact length or the
# (i + offset) % len(pool) mapping differs from the rollout's.
W12_POOL_N = 200 + W12_OFFSET + 50

MAX_MISMATCH_FRAC = 0.02   # silent-misalignment guard: abort the join above this
MIN_CELL_N = 30            # UNDERPOWERED below this per (dose x condition) cell
INTERACTION_T = 1.96       # two-sided ~5% gate on the OLS interaction term

# Wave-1/2 steer-arm families carrying each axis (dose ladders only; nulls,
# confound and baselines have no per-axis dose ladder to interact). The W1
# sec4_behavioural_* and W2 sec4_w2_behav_iba_* arms are CONDITION-IDENTICAL
# (same npz asset, seed_base, offset, alphas, prompts) with seeded generation,
# so their rows are (near-)duplicates, not independent samples — analyze_
# postloss dedupes per axis on (alpha, seed) so pooling them cannot halve the
# OLS interaction SE (pseudoreplication).
AXIS_ARM_PREFIXES = {
    "behavioural": ("sec4_behavioural_a", "sec4_w2_behav_iba_a"),
    "shared": ("sec4_w2_shared_a",),
    "readout": ("sec4_readout_a",),
}


# ------------------------------------------------------------- labeling

def label_postloss(pool_entry) -> Optional[bool]:
    """True = post-loss, False = post-win, None = no previous displayed round.

    pool_entry is a states.load_minusG_states (game, round_idx) pair — the game
    dict IS the catalog record, carrying history/decisions. The previous round
    the model actually SAW at round_idx is history[round_idx - 1] (the last row
    prompts.build_prompt renders), with the win flag read exactly as build_prompt
    reads it: h.get('win', str(h.get('result','')) == 'W'). States whose game
    has no bet before round_idx (empty/short history) are unlabeled -> None.
    """
    game, round_idx = pool_entry
    hist = game.get("history", [])
    if round_idx <= 0 or round_idx - 1 >= len(hist):
        return None
    h = hist[round_idx - 1]
    win = h.get("win", str(h.get("result", "")) == "W")
    return not bool(win)


# ------------------------------------------------------------------ join

def join_trials_to_states(rows: List[dict], pool: list, offset: int,
                          seed_base: int,
                          max_mismatch_frac: float = MAX_MISMATCH_FRAC
                          ) -> List[Tuple[dict, tuple]]:
    """Map each rollout row back to its −G pool state and VALIDATE the join.

    The runner replays states[(i + offset) % len(pool)] with seed =
    seed_base + i * SEED_STEP, so i is recovered from the recorded seed. Every
    row is then cross-checked against its recorded source_state (round_idx,
    balance when present, prompt_combo when present); rows that disagree are
    counted as mismatches and dropped. If more than max_mismatch_frac of the
    rows mismatch the whole join ABORTS (ValueError) — a wrong pool size /
    offset / seed_base must fail loudly, never silently mislabel the wave.

    Returns [(row, (game, round_idx)), ...] for the validated rows.
    """
    assert pool, "empty state pool"
    joined, mismatches = [], 0
    for row in rows:
        seed = int(row["seed"])
        i, rem = divmod(seed - seed_base, SEED_STEP)
        if rem != 0 or i < 0:
            mismatches += 1
            continue
        game, round_idx = pool[(i + offset) % len(pool)]
        src = row.get("source_state") or {}
        ok = src.get("round_idx") == round_idx
        if ok and src.get("balance") is not None:
            bal = game["decisions"][round_idx].get("balance_before")
            ok = bal is not None and int(float(bal)) == int(src["balance"])
        if ok and src.get("prompt_combo") is not None:
            ok = src["prompt_combo"] == game.get("prompt_combo", "")
        if not ok:
            mismatches += 1
            continue
        joined.append((row, (game, round_idx)))
    if rows and mismatches / len(rows) > max_mismatch_frac:
        raise ValueError(
            f"state join misaligned: {mismatches}/{len(rows)} rows mismatch "
            f"(> {max_mismatch_frac:.0%}) — pool/offset/seed_base do not "
            f"reproduce the rollout mapping; refusing to relabel")
    return joined


# -------------------------------------------------------------- analysis

def _ols_interaction(doses, is_loss, bets):
    """OLS bet ~ 1 + dose + loss + dose*loss; returns (coef, t) of the
    interaction term — coef > 0 means steering raises bets MORE post-loss."""
    x = np.asarray(doses, dtype=np.float64)
    g = np.asarray(is_loss, dtype=np.float64)
    y = np.asarray(bets, dtype=np.float64)
    X = np.column_stack([np.ones_like(x), x, g, x * g])
    n, p = X.shape
    if n <= p or float(x.std()) < 1e-12 or g.min() == g.max():
        return float("nan"), float("nan")
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    sigma2 = float(resid @ resid) / (n - p)
    se = float(np.sqrt(max(sigma2 * xtx_inv[3, 3], 1e-30)))
    return float(beta[3]), float(beta[3] / se)


def _spearman(xs, ys):
    if len(set(xs)) < 2 or len(set(ys)) < 2 or len(xs) < 3:
        return float("nan")
    from scipy.stats import spearmanr
    return float(spearmanr(xs, ys)[0])


def _axis_of_arm(arm_id: str) -> Optional[str]:
    for axis, prefixes in AXIS_ARM_PREFIXES.items():
        if any(arm_id.startswith(p) for p in prefixes):
            return axis
    return None


def analyze_postloss(arm_rows: Dict[str, List[dict]], pool: list,
                     offset: int = W12_OFFSET, seed_base: int = W12_SEED_BASE,
                     out_json: Optional[Path] = None,
                     out_png: Optional[Path] = None) -> dict:
    """Q1 rung-1 over already-recorded rollouts.

    arm_rows: {arm_id: jsonl rows} for the Wave-1/2 arms (nulls/baselines are
    ignored via AXIS_ARM_PREFIXES). Per axis: join + post-loss label every
    parse-ok steer trial, then mean bet_ratio and n per (dose x condition)
    cell, Spearman(dose, bet) per condition, and the OLS dose x post-loss
    interaction. Verdict per axis (and overall, from the behavioural axis — the
    Wave-1-confirmed writer): UNDERPOWERED if any populated cell has n <
    MIN_CELL_N (or no cells), LC_LIKE_INTERACTION if the interaction is
    positive with |t| > INTERACTION_T, else NO_INTERACTION.

    Pseudoreplication guard: trials are deduped per axis on (alpha, seed)
    BEFORE any cell/OLS computation — the W1/W2 behavioural ladders replay the
    same seeds at the same alphas on the same states (see AXIS_ARM_PREFIXES),
    and counting those rows twice would deflate the interaction SE by ~sqrt(2)
    and inflate the pre-registered t > INTERACTION_T gate. Sorted arm order
    keeps the Wave-1 row of each duplicate pair.
    """
    join_stats = {"n_rows": 0, "n_joined": 0, "n_unlabeled": 0, "n_dedup": 0}
    trials: Dict[str, list] = {a: [] for a in AXIS_ARM_PREFIXES}
    seen: Dict[str, set] = {a: set() for a in AXIS_ARM_PREFIXES}
    for arm_id, rows in sorted(arm_rows.items()):
        axis = _axis_of_arm(arm_id)
        if axis is None:
            continue
        join_stats["n_rows"] += len(rows)
        for row, entry in join_trials_to_states(rows, pool, offset, seed_base):
            join_stats["n_joined"] += 1
            if not row.get("parse_ok") or row.get("bet_ratio") is None \
                    or row.get("alpha") is None:
                continue
            key = (float(row["alpha"]), int(row["seed"]))
            if key in seen[axis]:
                join_stats["n_dedup"] += 1
                continue
            seen[axis].add(key)
            lab = label_postloss(entry)
            if lab is None:
                join_stats["n_unlabeled"] += 1
                continue
            trials[axis].append((float(row["alpha"]), bool(lab),
                                 float(row["bet_ratio"])))

    axes_out = {}
    for axis, ts in trials.items():
        cells: Dict[tuple, list] = {}
        for dose, loss, bet in ts:
            cells.setdefault((dose, loss), []).append(bet)
        cell_out = {
            f"{dose}|{'postloss' if loss else 'postwin'}":
                {"mean_bet": float(np.mean(v)), "n": len(v)}
            for (dose, loss), v in sorted(cells.items())}
        spear = {}
        for name, want in (("postloss", True), ("postwin", False)):
            xs = [d for d, l, _ in ts if l is want]
            ys = [b for _, l, b in ts if l is want]
            spear[name] = _spearman(xs, ys)
        coef, t = _ols_interaction([d for d, _, _ in ts],
                                   [l for _, l, _ in ts],
                                   [b for _, _, b in ts])
        min_n = min((c["n"] for c in cell_out.values()), default=0)
        if min_n < MIN_CELL_N:
            verdict = "UNDERPOWERED"
        elif np.isfinite(t) and coef > 0 and t > INTERACTION_T:
            verdict = "LC_LIKE_INTERACTION"
        else:
            verdict = "NO_INTERACTION"
        axes_out[axis] = {"cells": cell_out, "spearman": spear,
                          "interaction": {"coef": coef, "t": t, "n": len(ts)},
                          "min_cell_n": min_n, "verdict": verdict}

    primary = axes_out.get("behavioural") or next(iter(axes_out.values()))
    result = {"verdict": primary["verdict"], "axes": axes_out,
              "join": join_stats,
              "params": {"offset": offset, "seed_base": seed_base,
                         "pool_len": len(pool), "min_cell_n": MIN_CELL_N}}
    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(result, indent=1))
    if out_png is not None:
        try:
            make_postloss_figure(result, Path(out_png))
        except Exception as e:  # figure is a nicety, never fail analysis on it
            print(f"[postloss] figure skipped: {e}")
    return result


def make_postloss_figure(res: dict, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes = [a for a in AXIS_ARM_PREFIXES
            if res["axes"].get(a, {}).get("cells")]
    fig, axs = plt.subplots(1, max(len(axes), 1), figsize=(4.2 * len(axes), 4),
                            squeeze=False)
    for ax, axis in zip(axs[0], axes):
        st = res["axes"][axis]
        for cond, color in (("postloss", "C3"), ("postwin", "C0")):
            pts = sorted((float(k.split("|")[0]), v["mean_bet"])
                         for k, v in st["cells"].items()
                         if k.endswith(cond))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                        color=color, label=cond)
        it = st["interaction"]
        ax.set_title(f"{axis} — {st['verdict']}\n"
                     f"interaction {it['coef']:.4f} (t={it['t']:.2f})",
                     fontsize=9)
        ax.set_xlabel("steering alpha (sigma-units)")
        ax.set_ylabel("mean bet ratio")
        ax.legend(fontsize=8)
    fig.suptitle(f"Q1 rung-1 post-loss re-analysis — verdict: {res['verdict']}")
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    fig.savefig(str(out_png).replace(".png", ".pdf"))
    plt.close(fig)


# ------------------------------------------------------------------- CLI

def _load_arm_rows(results_dirs) -> Dict[str, List[dict]]:
    out = {}
    for d in results_dirs:
        for fp in sorted(Path(d).glob("sec4_*.jsonl")):
            out[fp.stem] = [json.loads(l) for l in open(fp) if l.strip()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dirs", nargs="+",
                    default=[str(d) for d in DEFAULT_RESULTS_DIRS],
                    help="dirs holding the Wave-1/2 sec4_*.jsonl rollouts")
    ap.add_argument("--model", default="gemma")
    ap.add_argument("--offset", type=int, default=W12_OFFSET)
    ap.add_argument("--seed-base", type=int, default=W12_SEED_BASE)
    ap.add_argument("--pool-n", type=int, default=W12_POOL_N,
                    help="pool length the arms ran with (n + offset + pad)")
    ap.add_argument("--from-hf", action="store_true",
                    help="download the sec4_p0/sec4_w2 rollouts from the HF "
                         "checkpoint tree into the results dirs first")
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--out-png", default=str(OUT_PNG))
    args = ap.parse_args()

    if args.from_hf:
        from .sec4_stats import fetch_hf_rollouts
        for phase, d in zip(("sec4_p0", "sec4_w2"), args.results_dirs):
            n = fetch_hf_rollouts(phase, Path(d))
            print(f"[postloss] fetched {n} rollouts for {phase} -> {d}",
                  flush=True)

    from .states import ensure_sm_catalog, load_minusG_states
    ensure_sm_catalog(args.model)
    pool = load_minusG_states(args.model, n=args.pool_n)
    arm_rows = _load_arm_rows(args.results_dirs)
    res = analyze_postloss(arm_rows, pool, args.offset, args.seed_base,
                           Path(args.out_json), Path(args.out_png))
    print(json.dumps({"verdict": res["verdict"],
                      "per_axis": {a: s["verdict"]
                                   for a, s in res["axes"].items()},
                      "join": res["join"]}, indent=2))


if __name__ == "__main__":
    main()
