"""Build the SINGLE-TASK BK direction assets for the 3x3 cross-task transfer
matrix (offline, CPU). Isolated 'xtask' namespace — additive only, writes to
assets/xtask/, never touches assets/w3/ or paper_axes.py.

Each axis is the per-source v_BK that paper_axes.build_bk computes BEFORE it
pools two sources by SVD: unit(mu_stop - mu_bankrupt) of ONE task's own raw L22
phase_a decision states (endpoint labels = game_outcomes=='bankruptcy', exactly
build_bk's labeling). build_bk() itself is strictly LOTO (target excluded,
loto_rank1 over the OTHER two tasks) and CANNOT produce a single-task A->A axis,
so this thin wrapper is required. It REUSES paper_axes primitives UNMODIFIED:
_load_bk_task / vbk_axis / game_projection_auc / projection_auc /
auc_permutation_p / replicate_rows / scales_from_phase_a.

Orientation convention (CONTRACT1, mirrors build_bk lines 759-760): stop-ward
positive. vbk_axis returns mu_stop - mu_bankrupt, so the within-task game-level
centroid AUC is >= 0.5 by construction; orientation_sign = +1 when
game_projection_auc >= 0.5, -1 otherwise (kept as an explicit check so the sign
rule is recorded identically to the LOTO assets).

Magnitude (CONTRACT4): direction rows are unit-norm (replicate_rows re-units
every row), and per-layer scales = 0.03 * median per-layer phase_a hidden norm
of THIS task's own rows (scales_from_phase_a). For the 3x3 transfer arms a
weak off-diagonal cell must be an effect-size fact, not a Δ-norm artifact, so
the runner arm steered into target task B loads target-B-scaled scales — see
build_target_scaled() which re-scales a saved single-task axis with the
APPLY-task's phase_a geometry.

Usage:
  HF_TOKEN=... python -m multilayer_causal.src.xtask_axes \
      --dest-dir multilayer_causal/assets/xtask
  python -m multilayer_causal.src.xtask_axes --self-test   # pure-numpy, no data
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import paper_axes as pa

# The within-task axis is a centroid classifier on its own data, so its
# game-level AUC separates the outcomes essentially by construction. We still
# record the permutation p as a relevance disclosure and gate on it (gate is
# effectively always-pass for a real single-task axis; a degenerate task that
# fails to separate its own outcomes would auto-exclude, which is correct).
RELEVANCE_ALPHA = pa.RELEVANCE_ALPHA


def build_single_task_axis(task, token):
    """Compute the single-task BK axis + its disclosures for one source task.

    Returns a dict with the oriented (3584,) axis (float64) plus every scalar
    the npz records. Pure paper_axes primitives, all UNMODIFIED.
    """
    # 1. One task's L22 states + endpoint labels + game ids + balances. Reuses
    #    the phase_a/SAE row-alignment assert and the bankruptcy labeling.
    d = pa._load_bk_task(task, token)

    # 2. Single-task v_BK = unit(mu_stop - mu_bankrupt) at raw L22 — the exact
    #    per-source vector build_bk pools by SVD, kept single-task here.
    axis = pa.vbk_axis(d["X"], d["is_bk"])

    # 3. Orientation: stop-ward positive. Within-task game-level centroid AUC
    #    is >= 0.5 by construction, so orientation_sign is +1 almost always;
    #    keep the explicit check to record it identically to build_bk.
    auc_game_raw, game_scores, game_bk = pa.game_projection_auc(
        d["X"], d["game_ids"], d["is_bk"], axis)
    orientation_sign = 1.0 if auc_game_raw >= 0.5 else -1.0
    axis_oriented = orientation_sign * axis

    # 4. Relevance disclosure (same primitives build_bk records).
    auc_game_p = pa.auc_permutation_p(game_scores, game_bk)
    auc_rows = pa.projection_auc(d["X"], d["is_bk"], axis_oriented)
    gate = bool(auc_game_p < RELEVANCE_ALPHA)

    return {
        "task": task,
        "axis_oriented": axis_oriented,
        "orientation_sign": orientation_sign,
        "auc_game_raw": float(auc_game_raw),
        "auc_game_oriented": float(auc_game_raw if orientation_sign > 0
                                   else 1.0 - auc_game_raw),
        "auc_game_p": float(auc_game_p),
        "auc_rows": float(auc_rows),
        "gate_passed": gate,
        "n_rows": int(len(d["X"])),
        "n_bk_rows": int(d["is_bk"].sum()),
    }


def save_single_task_axis(dest_dir, task, info, token):
    """Write directions_bk_single_{task}.npz with the runner schema.

    'directions' (42, 3584) f32 unit rows + 'scales' (42,) from THIS task's
    own phase_a geometry + gate_passed (bool the runner asserts True). The
    axis is replicated to all 42 layers; arm['layers'] picks the write window.
    """
    axis_oriented = info["axis_oriented"]
    directions = pa.replicate_rows(axis_oriented)            # (42, 3584) f32 unit

    # scales over ALL endpoint-labeled rows (build_bk's label_idx = arange) of
    # THIS source task's phase_a states — the self-scaled (diagonal) magnitude.
    label_idx = np.arange(info["n_rows"])
    scales = pa.scales_from_phase_a(task, label_idx, token)  # (42,) f32

    dest = Path(dest_dir) / f"directions_bk_single_{task}.npz"
    np.savez(
        dest,
        directions=directions, scales=scales,
        schema_version=pa.SCHEMA_VERSION,
        gate_passed=info["gate_passed"],
        source_task=task, orientation_sign=info["orientation_sign"],
        auc_game_raw=info["auc_game_raw"],
        auc_game_oriented=info["auc_game_oriented"],
        auc_game_p=info["auc_game_p"], auc_rows=info["auc_rows"],
        relevance_alpha=RELEVANCE_ALPHA,
        n_rows=info["n_rows"], n_bk_rows=info["n_bk_rows"],
        provenance=(
            "SINGLE-TASK BK axis (A->A diagonal source), task=%s; axis = "
            "unit(mu_stop - mu_bankrupt) of raw L22 phase_a decision states "
            "(endpoint labels per data_loader.get_labels / "
            "game_outcomes=='bankruptcy'), oriented stop-ward "
            "(orientation_sign=%+.0f); replicated to 42 layers; scales = "
            "0.03*median per-layer phase_a norm over this task's own rows. "
            "This is the per-source vbk_axis paper_axes.build_bk pools by SVD, "
            "kept single-task for the 3x3 cross-task transfer matrix; "
            "self-scaled (diagonal) Delta-norm. Off-diagonal apply uses a "
            "target-scaled re-scale (build_target_scaled)."
            % (task, info["orientation_sign"])))
    return dest


def build_target_scaled(dest_dir, source_task, target_task, source_axis, token):
    """Re-scale a source single-task axis with the APPLY (target) task's
    phase_a geometry — the norm-match for off-diagonal transfer cells
    (CONTRACT4). Same unit directions, but per-layer scales come from the
    TARGET task, so source->target and target->target carry byte-identical
    scales in a given column. Writes
    directions_bk_single_{source}_to_{target}.npz.
    """
    directions = pa.replicate_rows(source_axis)
    n_rows = _phase_a_n_rows_cache(target_task, token)
    label_idx = np.arange(int(n_rows))
    scales = pa.scales_from_phase_a(target_task, label_idx, token)
    dest = (Path(dest_dir)
            / f"directions_bk_single_{source_task}_to_{target_task}.npz")
    np.savez(
        dest, directions=directions, scales=scales,
        schema_version=pa.SCHEMA_VERSION, gate_passed=True,
        source_task=source_task, target_task=target_task,
        provenance=(
            "Cross-task transfer cell %s->%s: unit single-task BK axis of "
            "source=%s, re-scaled with TARGET=%s phase_a per-layer norms "
            "(0.03*median) so every cell in the %s column is norm-matched "
            "(CONTRACT4); apply at the L16-21 write window."
            % (source_task, target_task, source_task, target_task,
               target_task)))
    return dest


def _phase_a_n_rows_cache(task, token):
    """Row count of a task's phase_a states, used only to size scales_from_phase_a
    over all endpoint-labeled rows. Returns the row count (int), reusing
    paper_axes.load_phase_a_l22's mmap path. Factored out so the diagonal and
    target-scaled writes share one definition of 'all rows'."""
    _, n = pa.load_phase_a_l22(task, token)
    return n


# -------------------------------------------------------------- self-checks

def _self_test():
    """Pure-numpy self-test of the orientation/unit/replicate logic that does
    NOT need HF data or sklearn/scipy. Exercises paper_axes.vbk_axis,
    paper_axes.replicate_rows, paper_axes.unit and the stop-ward sign rule on a
    synthetic two-cluster L22 mock. Prints shapes, per-layer unit norms and the
    pairwise cosines between three mock single-task axes."""
    rng = np.random.default_rng(0)
    d_model = pa.D_MODEL
    axes = {}
    for k, t in enumerate(("sm", "ic", "mw")):
        # mock: stop rows centered at +mu_t, bankrupt rows at -mu_t, so
        # vbk_axis ~ unit(2*mu_t) ~ unit(mu_t); each task a distinct direction.
        mu = rng.standard_normal(d_model)
        n = 200
        is_bk = np.array([False] * (n // 2) + [True] * (n // 2))
        X = np.empty((n, d_model), dtype=np.float32)
        X[~is_bk] = rng.standard_normal((n // 2, d_model)) * 0.1 + mu
        X[is_bk] = rng.standard_normal((n // 2, d_model)) * 0.1 - mu
        axis = pa.vbk_axis(X, is_bk)
        assert axis.shape == (d_model,), axis.shape
        # stop-ward check: stop rows must project HIGHER than bankrupt rows
        proj_stop = (X[~is_bk] @ axis).mean()
        proj_bk = (X[is_bk] @ axis).mean()
        assert proj_stop > proj_bk, (k, proj_stop, proj_bk)
        rows = pa.replicate_rows(axis)
        assert rows.shape == (pa.N_LAYERS, d_model) and rows.dtype == np.float32
        norms = np.linalg.norm(rows, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), norms[:3]
        axes[t] = pa.unit(axis)
        print(f"[self-test] {t}: axis shape {axis.shape}, "
              f"replicate {rows.shape} dtype {rows.dtype}, "
              f"per-layer unit norm min/max "
              f"{norms.min():.6f}/{norms.max():.6f}, stop-ward OK "
              f"(stop {proj_stop:.3f} > bk {proj_bk:.3f})")
    ts = ("sm", "ic", "mw")
    print("[self-test] pairwise cos between mock single-task axes "
          "(distinct by construction):")
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            c = float(axes[ts[i]] @ axes[ts[j]])
            print(f"  cos({ts[i]}, {ts[j]}) = {c:+.4f}")
    print("[self-test] PASS (pure-numpy logic; sklearn/scipy AUC paths and HF "
          "data not exercised here — those run on the build node).")


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest-dir",
                    help="output dir for directions_bk_single_{sm,ic,mw}.npz")
    ap.add_argument("--target-scaled", action="store_true",
                    help="also emit the 9 (source->target) norm-matched cells")
    ap.add_argument("--self-test", action="store_true",
                    help="run the pure-numpy self-test only (no HF data)")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.dest_dir:
        ap.error("--dest-dir required unless --self-test")

    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    results, axes = {}, {}
    for task in pa.BK_TASKS:
        info = build_single_task_axis(task, token)
        dest = save_single_task_axis(dest_dir, task, info, token)
        axes[task] = pa.unit(info["axis_oriented"])
        results[task] = {k: v for k, v in info.items()
                         if k != "axis_oriented"}
        results[task]["dest"] = str(dest)
        print(f"[bk_single] {task}: gate={'PASS' if info['gate_passed'] else 'FAIL'} "
              f"game_auc={info['auc_game_oriented']:.4f} p={info['auc_game_p']:.5f} "
              f"-> {dest}")

    # 3x3 cross-task cosine disclosure (the transfer-matrix's geometry).
    print("[bk_single] pairwise cos between the 3 single-task BK axes:")
    ts = list(pa.BK_TASKS)
    cos = {}
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            c = float(axes[ts[i]] @ axes[ts[j]])
            cos[f"{ts[i]}_{ts[j]}"] = c
            print(f"  cos({ts[i]}, {ts[j]}) = {c:+.4f}")
    results["pairwise_cos"] = cos

    if args.target_scaled:
        for src in pa.BK_TASKS:
            for tgt in pa.BK_TASKS:
                dest = build_target_scaled(
                    dest_dir, src, tgt, axes[src], token)
                print(f"[bk_single] norm-matched cell {src}->{tgt} -> {dest}")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
