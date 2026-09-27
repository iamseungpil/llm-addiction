"""Emitters for the four body tables of §4, plus a diff against the typed .tex.

Why this file exists
--------------------
``neurips_content_en/4.neural.tex`` carries four tables and none of them has a
generator: every cell was typed by hand.  This script recomputes each cell from
the released corpora and writes one LaTeX *fragment* per table containing only
the body rows, so the existing float in the .tex can ``\\input`` it without any
other change.  Nothing here is a literal copied out of the paper: every printed
number is derived, and the few pieces of *markup* the paper adds (which cell is
green, which is bold) are derived from explicit rules that are stated next to
the code that applies them.

The four tables
---------------
1. ``tab:neurips-sae-results``    read-side readout R^2, 6 rows x 3 indicators
2. ``tab:causal-battery-suffnec`` sufficiency (steer) + necessity (remove)
3. ``tab:sharing-transfer``       cross-task read sharing on Gemma L22
4. ``tab:condition-modulation``   slot-machine readout by prompt condition

Sources
-------
``sae_v3_analysis/results/table1_groupkfold_L22.json``            table 1
``experiments/sec4_causal/checkpoints/sec4_p0/*.jsonl``           table 2 steer
``experiments/sec4_causal/analysis/sec4_w2_analysis.json``        table 2 cross-check
``experiments/sec4_causal/checkpoints/sec4_w2/sec4_w2_null*.jsonl`` table 2 null band
``experiments/sec4_causal/checkpoints/sec4_w13/*.jsonl``          table 2 removal
``sae_v3_analysis/results/shared_subspace_hidden_audit_*.json``   table 3 (i)
``sae_v3_analysis/results/iba_cross_task_transfer.json``          table 3 (ii)
``.../rq2_aligned_hidden_transfer_gemma_centroid_pca_L22_r{1,2}`` table 3 (iii)
``sae_v3_analysis/results/condition_modulation_groupkfold_L22``   table 4

Each source is looked up in the local mirror first and pulled from the public
dataset ``llm-addiction-research/llm-addiction`` otherwise.  **A source that is
absent from both raises.**  It never degrades to a skipped check or an omitted
row -- that failure mode is exactly what ``scripts/verify_section4_numbers.py``
does today, printing "0 FAIL" while nine checks silently [SKIP] on a missing
file.

Outputs
-------
``paper_data/tables/body/table1_sae_readout_r2.tex``
``paper_data/tables/body/table2_causal_suffnec.tex``
``paper_data/tables/body/table3_sharing_transfer.tex``
``paper_data/tables/body/table4_condition_modulation.tex``

Run
---
``cd <repo> && HF_HUB_DISABLE_XET=1 python3 scripts/tables/body_tables.py``
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "figures"))

# The dataset id and the interval helper the rest of the camera-ready uses:
# imported, not re-typed.
from build_figure_data import REPO_ID  # noqa: E402

# The §4 rollout readers already written for body Figure 4.  Table 2 is the
# tabular view of that figure, so it must be the same arithmetic on the same
# rollouts; duplicating the loaders here would let the two drift apart.
import fig04_causal_battery as CB  # noqa: E402

# Optional local mirrors of the analysis tree.  Point ``LLM_ADDICTION_ANALYSIS``
# at that checkout to read from disk; ``load_json`` falls back to the same
# files on the public dataset when a path is missing.
ANALYSIS_ROOT = Path(os.environ.get("LLM_ADDICTION_ANALYSIS", Path.home() / "llm-addiction"))
LOCAL_SAE = ANALYSIS_ROOT / "experiments" / "07_sae_readout" / "results"
LOCAL_CAUSAL = ANALYSIS_ROOT / "experiments" / "08_steering" / "multilayer_causal" / "results"
HF_TABLES = "paper_neurips_2026/tables"
CACHE = Path("/tmp/hfcache_bodytables")

OUT_DIR = REPO_ROOT / "paper_data" / "tables" / "body"
NEURAL_TEX = REPO_ROOT / "neurips_content_en" / "4.neural.tex"

TASKS = ("sm", "ic", "mw")
INDICATORS = ("i_lc", "i_ba", "i_ec")
MODELS = ("gemma", "llama")
MODEL_LABEL = {"gemma": "Gemma", "llama": "LLaMA"}
TASK_LABEL = {"sm": "SM", "ic": "IC", "mw": "MW"}
GREEN = r"\cellcolor{green!12}"


class MissingSource(FileNotFoundError):
    """A source file this emitter needs is in neither the mirror nor the hub."""


# ------------------------------------------------------------------ sources


def resolve(local: Path, hf_path: str) -> Path:
    """Local mirror first, public dataset otherwise.  Raise when neither has it.

    The raise is the point.  A table cell whose source is gone must stop the
    build, because the alternative -- emitting the cell anyway -- means printing
    a number nobody can trace.
    """
    if local.exists():
        return local
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(REPO_ID, hf_path, repo_type="dataset",
                                    cache_dir=str(CACHE)))
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed failure
        raise MissingSource(
            f"source missing from both the local mirror ({local}) and "
            f"{REPO_ID}:{hf_path} -- {type(exc).__name__}: {exc}") from exc


def load_json(local: Path, hf_path: str) -> dict:
    with open(resolve(local, hf_path)) as fh:
        return json.load(fh)


def require(value, what: str):
    """Every cell must come from a value that is actually present and finite."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        raise MissingSource(f"no usable value for {what} (got {value!r})")
    return value


# --------------------------------------------------------------- formatting


def r3(x: float) -> str:
    """Three decimals, negatives in math mode -- the rounding the paper uses."""
    s = f"{x:.3f}"
    return f"${s}$" if x < 0 else s


def signed3(x: float) -> str:
    return f"${x:+.3f}$"


def signed2(x: float) -> str:
    return f"${x:+.2f}$"


def pfmt(p: float) -> str:
    """APA-style p: ``<.001``, ``1.0`` at the ceiling, else three decimals."""
    if p < 0.001:
        return r"$<.001$"
    if p >= 0.9995:
        return "$1.0$"
    return "$" + f"{p:.3f}".lstrip("0") + "$"


def header(title: str, label: str, sources: list[str], rules: list[str]) -> str:
    lines = [f"% {title}", f"% paper label: {label}",
             "% AUTO-GENERATED by scripts/tables/body_tables.py -- do not edit.",
             "% Body rows only: \\input this inside the existing float.",
             "% sources:"]
    lines += [f"%   {s}" for s in sources]
    if rules:
        lines.append("% markup rules (derived, not copied from the paper):")
        lines += [f"%   {r}" for r in rules]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------- table 1: read side


def table1():
    """R^2 for decoding each indicator from L22 SAE features, 6 rows x 3 cols.

    Markup, both derived rather than copied:

    * green = the *binding* indicator of the task, taken as the indicator with
      the highest R^2 averaged over the two models within that task.  This
      reproduces the paper's green pattern (SM->I_BA, IC->I_BA, MW->I_LC)
      without assuming it, and it is what the caption describes: "the primary
      binding indicator per row ... not the largest cell".
    * bold = green, or the largest cell in the row.  (LLaMA IC I_EC is bold in
      the paper without being green; it is that row's maximum.)
    """
    t1 = load_json(LOCAL_SAE / "table1_groupkfold_L22.json",
                   "sae_v3_analysis/results/table1_groupkfold_L22.json")

    def r2(model, task, ind):
        key = f"{model}_{task}_{ind}_L22"
        if key not in t1:
            raise MissingSource(f"table1_groupkfold_L22.json has no {key}")
        return float(require(t1[key].get("r2_mean"), key))

    grid = {(m, t, i): r2(m, t, i)
            for m in MODELS for t in TASKS for i in INDICATORS}

    binding = {}
    for t in TASKS:
        means = {i: float(np.mean([grid[(m, t, i)] for m in MODELS]))
                 for i in INDICATORS}
        binding[t] = max(means, key=means.get)

    rows = []
    for m in MODELS:
        for t in TASKS:
            vals = [grid[(m, t, i)] for i in INDICATORS]
            top = max(range(3), key=lambda k: vals[k])
            cells = []
            for k, i in enumerate(INDICATORS):
                txt = r3(vals[k])
                is_green = i == binding[t]
                if is_green or k == top:
                    txt = r"\textbf{" + txt + "}"
                if is_green:
                    txt = GREEN + txt
                cells.append(txt)
            rows.append(f"{MODEL_LABEL[m]} & {TASK_LABEL[t]} & "
                        + " & ".join(cells) + r" \\")

    note = ("% NOTE: the paper suppresses five cells behind '---' (inside the "
            "control band)\n%   and 'n/c' (below the reporting threshold).  No "
            "per-cell L22 permutation p\n%   exists anywhere in the released "
            "corpus -- headline_robustness.json carries a\n%   game-block null "
            "for two cells only, and robustness/permutation_null.json is at\n"
            "%   L24/L16 -- so the suppression decision is not reproducible and "
            "is NOT applied\n%   here.  The measured value is printed for every "
            "cell instead.  Screening flag\n%   below is the fold-level check "
            "(mean > 2 standard errors over the 5 folds),\n%   offered as "
            "information, not used to blank anything:\n")
    for m in MODELS:
        for t in TASKS:
            for i in INDICATORS:
                rec = t1[f"{m}_{t}_{i}_L22"]
                se = float(rec["r2_std"]) / math.sqrt(5.0)
                flag = "above" if rec["r2_mean"] - 2 * se > 0 else "INSIDE-BAND"
                note += (f"%   {m}_{t}_{i}: r2={rec['r2_mean']:+.4f} "
                         f"2se={2 * se:.4f} -> {flag}\n")

    body = header(
        "Table 1 -- read-side readout R^2 at L22",
        "tab:neurips-sae-results",
        ["sae_v3_analysis/results/table1_groupkfold_L22.json"],
        ["green = argmax over indicators of the model-averaged R^2 within the task",
         "bold  = green cell, or the row maximum"],
    ) + note + "\n".join(rows) + "\n"
    return body, grid, binding


# ------------------------------------------------------ table 2: write side


def table2():
    """Sufficiency (steering slope + z) and necessity (removal delta + paired p).

    Sufficiency is the parse-gated trial-level OLS slope of bet ratio on dose,
    scored as z against the twenty random-direction slopes of ``sec4_w2``.
    ``sec4_w2_analysis.json`` is loaded as a cross-check: it carries the same
    two arms under the names ``behav_iba`` (behavioural) and ``confound``
    (balance), and the emitter raises if the recomputation disagrees.  It has no
    readout arm, which is why the slopes are recomputed from the ``sec4_p0``
    rollouts, where all three directions were run.

    Necessity is the mean within-seed change in bet ratio when the direction is
    projected out, with an exact two-sided sign test on the nonzero pairs.

    Markup: a row is green when removal significantly *lowers* betting
    (delta < 0 and p < .05) -- the caption's "passes both tests" is the
    necessity leg, which is the leg both models have.
    """
    p0 = CB.phase_dir("sec4_p0")
    w2 = CB.phase_dir("sec4_w2")
    w13 = CB.phase_dir("sec4_w13")

    ladder = CB.ladder("gemma", p0, "sec4_")
    band = CB.gemma_null_slopes(w2)
    if band["n_directions"] == 0 or not math.isfinite(band["sd"]):
        raise MissingSource("no random-direction slope band in sec4_w2")

    w2a = load_json(LOCAL_CAUSAL / "sec4_w2" / "sec4_w2_analysis.json",
                    "experiments/sec4_causal/checkpoints/sec4_w2/"
                    "sec4_w2_analysis.json")
    crosscheck = {"behavioural": "behav_iba", "confound": "confound"}
    for axis, key in crosscheck.items():
        want = float(require(w2a["axes"][key]["i_ba"]["slope"], f"w2 {key} slope"))
        got = ladder[axis]["slope_parse_gated"]
        if abs(want - got) > 1e-6:
            raise MissingSource(
                f"sec4_w2_analysis.json {key} slope {want:.6f} disagrees with "
                f"the sec4_p0 recomputation {got:.6f}")

    steer = {}
    for axis in CB.AXIS_ORDER:
        slope = float(require(ladder[axis]["slope_parse_gated"], f"{axis} slope"))
        steer[axis] = {"slope": slope,
                       "z": (slope - band["mean"]) / band["sd"]}

    remove = {m: CB.removal(m, w13) for m in MODELS}

    rows, computed = [], {}
    for m in MODELS:
        for axis in CB.AXIS_ORDER:
            e = remove[m]["axes"][axis]
            d = float(require(e["delta_paired"], f"{m} {axis} delta"))
            p = float(require(e["p_paired_sign_test"], f"{m} {axis} p"))
            label = CB.AXIS_SHORT[axis]
            if m == "gemma":
                s, z = steer[axis]["slope"], steer[axis]["z"]
                suff = [signed3(s), signed2(z)]
            else:
                # LLaMA steering picked its own write window, so the paper does
                # not score sufficiency there; the dagger footnote already in
                # the float explains the dash.  Nothing is computed into these
                # cells, so nothing is printed into them.
                suff = ["---$^\\dagger$" if axis == "behavioural" else "---",
                        "---"]
            if d < 0 and p < 0.05:
                rows.append(r"\rowcolor{green!12}")
            rows.append(f"{MODEL_LABEL[m]} & {label} & " + " & ".join(suff)
                        + f" & {signed3(d)} & {pfmt(p)}" + r" \\")
            computed[(m, label)] = {
                "slope": steer[axis]["slope"] if m == "gemma" else None,
                "z": steer[axis]["z"] if m == "gemma" else None,
                "delta": d, "p": p, "n_pairs": e["n_pairs"],
                "n_up": e["n_pairs_up"], "n_down": e["n_pairs_down"]}

    note = ("% null slope band: n={} directions, mean={:.6f}, sd={:.6f} "
            "(sec4_w2_null1..20 at alpha=-3,+3)\n".format(
                band["n_directions"], band["mean"], band["sd"]))
    note += ("% p is the exact two-sided sign test on the nonzero within-seed "
             "differences.\n")
    note += ("% (the p column typed in the .tex equals min(1, 2 x this p) in "
             "all six cells, i.e. a\n%  second two-sided doubling applied on top "
             "of an already two-sided test.)\n")
    for m in MODELS:
        note += ("%   {} baseline bet ratio {:.5f} on {}/{} parse-ok\n".format(
            m, remove[m]["baseline_mean_bet_ratio"],
            remove[m]["baseline_n_parse_ok"], remove[m]["baseline_n_run"]))
        for axis in CB.AXIS_ORDER:
            e = remove[m]["axes"][axis]
            note += ("%   {} {}: {} down / {} up of {} pairs\n".format(
                m, CB.AXIS_SHORT[axis], e["n_pairs_down"], e["n_pairs_up"],
                e["n_pairs"]))

    body = header(
        "Table 2 -- write-side sufficiency and necessity",
        "tab:causal-battery-suffnec",
        ["experiments/sec4_causal/checkpoints/sec4_p0/sec4_{behavioural,readout,confound}_a*.jsonl",
         "experiments/sec4_causal/checkpoints/sec4_w2/sec4_w2_null{1..20}_a{m3,p3}.jsonl",
         "experiments/sec4_causal/analysis/sec4_w2_analysis.json (cross-check)",
         "experiments/sec4_causal/checkpoints/sec4_w13/sec4_w13_*_{base,behavioural,readout,confound}.jsonl"],
        ["green = removal lowers betting significantly (delta < 0 and p < .05)"],
    ) + note + "\n".join(rows) + "\n"
    return body, computed, steer, band


# ------------------------------------------------- table 3: cross-task read


#: This table's own column order (IC, MW, SM) -- distinct from the
#: module-level ``TASKS`` (SM, IC, MW), which is what the *rest* of the file
#: uses.  Matches the printed header ``Audit & IC & MW & SM``.
RQ2_TASKS = ("ic", "mw", "sm")
RQ2_TASK_DIR = {"sm": "slot_machine", "ic": "investment_choice", "mw": "mystery_wheel"}


def table3():
    """Cross-task read sharing on Gemma L22: n, cosine, transfer, shared AUC.

    As of the camera-ready pass this table prints point estimates and row
    counts only; the standard errors behind rows (iii)/(iv) moved to an
    appendix sentence (``appendix.tex``, the paragraph after the LOTO PCA
    transfer discussion).  Those SDs are still computed here -- they are the
    ``se`` values folded into the emitted fragment's comments -- so anyone
    regenerating that appendix sentence has one source of truth instead of a
    second recompute; nothing about the emitted table fragment changes.

    Rows, top to bottom:
      n rows / bankrupt  -- raw BK-valid decision counts (bankruptcy vs.
        voluntary-stop rounds with a valid balance), straight off the same
        ``hidden_states_dp.npz`` the BK direction and the LOTO PCA audit below
        are built from.  Cross-checked against the bankruptcy counts
        ``scripts/figures/fig5b_pca_appendix.py`` already commits to
        (BK = 87/172/54 for SM/IC/MW -- the same numbers in this table's own
        IC/MW/SM column order).
      (i)   the largest absolute cosine between a task's BK direction and the
            other two, from the L22 shared-subspace audit.
      (ii)  the best (least negative) off-diagonal sparse-feature transfer R^2
            into the task, reported as ``$<0$`` when that best case is still
            negative.  The released matrix (``iba_cross_task_transfer.json``)
            covers SM<->MW at L18/L24 only; it has no entry touching IC at any
            layer, so the IC cell is emitted as ``n/c`` here.  The .tex prints
            ``$<0$`` for IC too, from a recompute
            (``scripts/tables/appendix_neural_tables.py::_feature_transfer_r2``)
            that needs the private ``sae_v3_analysis/src`` checkout this
            environment does not have -- see the note in the emitted fragment.
      (iii) shared-only AUC of the leave-one-task-out PCA readout
            decomposition, at rank 1 and rank 2.  Green+bold at AUC >= 0.7,
            the caption's own threshold; row (iv) is never coloured, matching
            the caption's explicit carve-out.
      (iv)  the other two slices of the same decomposition: residual-only
            (rank-1 file) and the full/combined AUC (rank-2 file).
    """
    n_rows, bankrupt = {}, {}
    for t in RQ2_TASKS:
        p = resolve(
            ANALYSIS_ROOT / "sae_features_v3" / RQ2_TASK_DIR[t] / "gemma"
            / "hidden_states_dp.npz",
            f"sae_features_v3/{RQ2_TASK_DIR[t]}/gemma/hidden_states_dp.npz",
        )
        d = np.load(p, allow_pickle=False)
        out = d["game_outcomes"]
        bal = d["balances"].astype(np.float32)
        valid = ((out == "bankruptcy") | (out == "voluntary_stop")) & ~np.isnan(bal)
        n_rows[t] = int(valid.sum())
        bankrupt[t] = int((out[valid] == "bankruptcy").sum())

    audit = load_json(LOCAL_SAE / "shared_subspace_hidden_audit_20260410.json",
                      "sae_v3_analysis/results/"
                      "shared_subspace_hidden_audit_20260410.json")
    g = audit["gemma_ic_sm_mw_hidden"]
    if int(g.get("layer", -1)) != 22:
        raise MissingSource(f"BK cosine audit is at layer {g.get('layer')}, not 22")
    cos = g["shared_subspace"]["weight_cosines"]
    cos_of = {}
    for t in RQ2_TASKS:
        vals = [abs(float(v)) for k, v in cos.items() if t in k.split("_")]
        if not vals:
            raise MissingSource(f"no BK cosine entry involving {t}")
        cos_of[t] = max(vals)

    xfer = load_json(LOCAL_SAE / "iba_cross_task_transfer.json",
                     "sae_v3_analysis/results/iba_cross_task_transfer.json")
    if "gemma" not in xfer:
        raise MissingSource("iba_cross_task_transfer.json has no gemma block")
    best_in = {}
    for layer, rec in xfer["gemma"].items():
        for key, val in rec.items():
            mo = re.fullmatch(r"([a-z]{2})_to_([a-z]{2})", key)
            if not mo:
                continue
            src, dst = mo.groups()
            if src == dst:
                continue
            best_in.setdefault(dst, []).append((float(val), layer, src))
    xfer_layers = sorted(xfer["gemma"])

    auc, resid, comb = {}, {}, {}
    se = {}
    for rank in (1, 2):
        name = (f"rq2_aligned_hidden_transfer_gemma_centroid_pca_L22_r{rank}"
                f"_e8g1_L22_r{rank}.json")
        d = load_json(LOCAL_SAE / "robustness" / name,
                      f"{HF_TABLES}/body/table2_rq2_audits/data/{name}")
        if int(d["config"]["layer"]) != 22 or int(d["config"]["rank"]) != rank:
            raise MissingSource(f"{name}: config is not L22 rank-{rank}")
        dec = d["summary"]["readout_decomposition"]
        for t in RQ2_TASKS:
            if t not in dec:
                raise MissingSource(f"{name}: no readout_decomposition for {t}")
            auc[(t, rank)] = float(require(dec[t].get("shared_only_auc_mean"),
                                           f"{name} {t} shared_only_auc"))
            se[(t, rank, "shared")] = float(dec[t].get("shared_only_auc_std", float("nan")))
            if rank == 1:
                resid[t] = float(require(dec[t].get("residual_only_auc_mean"),
                                         f"{name} {t} residual_only_auc"))
                se[(t, "residual")] = float(dec[t].get("residual_only_auc_std", float("nan")))
            else:
                comb[t] = float(require(dec[t].get("full_auc_mean"),
                                        f"{name} {t} full_auc"))
                se[(t, "combined")] = float(dec[t].get("full_auc_std", float("nan")))

    def green(v: float, s: str) -> str:
        return r"\cellcolor{green!12}\textbf{" + s + "}" if v >= 0.7 else s

    cells_n = [f"{n_rows[t]:,}".replace(",", "{,}") for t in RQ2_TASKS]
    cells_bk = [f"{bankrupt[t]:,}".replace(",", "{,}") for t in RQ2_TASKS]
    cells_i = [f"{cos_of[t]:.2f}" for t in RQ2_TASKS]

    cells_ii = []
    missing_transfer = []
    for t in RQ2_TASKS:
        cands = best_in.get(t)
        if not cands:
            cells_ii.append("n/c")
            missing_transfer.append(t)
        else:
            best = max(cands)[0]
            cells_ii.append(r"$<\!0$" if best < 0 else f"{best:.2f}")

    cells_rank1 = [green(auc[(t, 1)], f"{auc[(t, 1)]:.2f}") for t in RQ2_TASKS]
    cells_rank2 = [green(auc[(t, 2)], f"{auc[(t, 2)]:.2f}") for t in RQ2_TASKS]
    cells_resid = [f"{resid[t]:.2f}" for t in RQ2_TASKS]
    cells_comb = [f"{comb[t]:.2f}" for t in RQ2_TASKS]

    note = "% n rows / bankrupt (hidden_states_dp.npz, BK-valid rounds): " + ", ".join(
        f"{t}=n{n_rows[t]}/bk{bankrupt[t]}" for t in RQ2_TASKS) + "\n"
    note += "% BK weight cosines (gemma L22): " + ", ".join(
        f"{k}={float(v):+.4f}" for k, v in cos.items()) + "\n"
    note += ("% sparse-feature transfer available at gemma layers {} only "
             "(not L22); best off-diagonal into each task:\n".format(
                 ", ".join(xfer_layers)))
    for t in RQ2_TASKS:
        if t in best_in:
            v, layer, src = max(best_in[t])
            note += f"%   into {t}: {v:+.4f} (from {src}, {layer})\n"
        else:
            note += (f"%   into {t}: NO OFF-DIAGONAL TRANSFER IN THE RELEASED MATRIX -- "
                     "the .tex prints '$<0$' here from a recompute "
                     "(appendix_neural_tables.py::_feature_transfer_r2) that needs the "
                     "private sae_v3_analysis/src checkout; not reproducible in this "
                     "environment, so this one cell is left 'n/c' rather than guessed\n")
    note += ("% rank-1/rank-2/residual/combined AUC, +- SD over the 5 CV splits "
             "(this is the interval the appendix sentence after the LOTO PCA "
             "discussion carries; the table itself prints the point estimate only):\n")
    for t in RQ2_TASKS:
        note += (f"%   {t}: rank1={auc[(t,1)]:.4f}+-{se[(t,1,'shared')]:.4f}  "
                 f"rank2={auc[(t,2)]:.4f}+-{se[(t,2,'shared')]:.4f}  "
                 f"residual={resid[t]:.4f}+-{se[(t,'residual')]:.4f}  "
                 f"combined={comb[t]:.4f}+-{se[(t,'combined')]:.4f}\n")

    rows = [
        f"$n$ rows & " + " & ".join(cells_n) + r" \\",
        r"\quad bankrupt & " + " & ".join(cells_bk) + r" \\",
        r"\midrule",
        f"$|\\cos|$ vs others & " + " & ".join(cells_i) + r" \\",
        r"\midrule",
        "best from other & " + " & ".join(cells_ii) + r" \\",
        r"\midrule",
        "rank-1 & " + " & ".join(cells_rank1) + r" \\",
        "rank-2 & " + " & ".join(cells_rank2) + r" \\",
        r"\midrule",
        "Residual-only & " + " & ".join(cells_resid) + r" \\",
        "Combined (r=2) & " + " & ".join(cells_comb) + r" \\",
    ]

    body = header(
        "Table 3 -- cross-task read sharing on Gemma (L22)",
        "tab:sharing-transfer",
        ["sae_features_v3/{investment_choice,mystery_wheel,slot_machine}/gemma/"
         "hidden_states_dp.npz",
         "sae_v3_analysis/results/shared_subspace_hidden_audit_20260410.json",
         "sae_v3_analysis/results/iba_cross_task_transfer.json",
         f"{HF_TABLES}/body/table2_rq2_audits/data/"
         "rq2_aligned_hidden_transfer_gemma_centroid_pca_L22_r{1,2}_e8g1_L22_r{1,2}.json"],
        ["row (i) = max |cosine| of the task's BK direction against the other two",
         "row (ii) = best off-diagonal transfer into the task; '$<0$' when it is negative",
         "row (iii)/(iv) bold+green = AUC >= 0.7 (caption's threshold); colours row (iii) "
         "only, never row (iv)",
         "no '+-' in the printed cells -- SEs moved to the appendix sentence after the "
         "LOTO PCA transfer discussion; carried in this fragment's comments instead"],
    ) + note + "\n".join(rows) + "\n"
    return body, cos_of, best_in, auc, missing_transfer


# ------------------------------------------- table 4: condition modulation


def _pct(plus: float | None, minus: float | None) -> str:
    """Percent change of ``plus`` over ``minus``, the ``\\Delta`` rows' cells.

    Blanked (``---``) when either endpoint is non-positive: a ratio through
    zero is not interpretable, so the paper prints a dash there instead of a
    number.  Same rule ``appendix_neural_tables.py``'s condition tables use.
    """
    if plus is None or minus is None or minus <= 0 or plus <= 0:
        return "---"
    v = (plus - minus) / minus * 100
    return f"$-{abs(v):.0f}$" if v < 0 else f"$+{v:.0f}$"


def _bold(txt: str) -> str:
    """Wrap an already-formatted cell in bold, keeping it in math mode if it
    already is one (``$-0.018$`` -> ``$\\mathbf{-0.018}$``, not
    ``\\textbf{$-0.018$}``, which is what the .tex itself does)."""
    if txt.startswith("$") and txt.endswith("$"):
        return r"$\mathbf{" + txt[1:-1] + "}$"
    return r"\textbf{" + txt + "}"


def table4(with_se: bool = False):
    """Slot-machine readout R^2 by prompt condition, L22.

    As of the camera-ready pass the body table prints point estimates and $n$
    only; the fold-level SEs behind the six condition cells (All
    variable/-G/+G/-M/+M) moved to appendix Table tab:appendix-condition-se.
    ``with_se=True`` switches those six rows' cells back to the ``v {\\pm} se``
    form that appendix table prints (its own layout has no $n$ column and no
    Delta/Fixed rows, which this flag does not add or remove -- it only
    changes how the six condition-cell values above are formatted), so
    regenerating that appendix table has one source of truth for the numbers
    instead of a second recompute.

    Markup, all derived rather than copied:
    * in the ``+G`` row a cell is bold when the goal module raises the
      readout above its ``-G`` counterpart (reproduces the paper's bolding).
    * in the ``$\\Delta_G\\%$`` row, the I_BA cell (both models) is bold --
      the row's headline number, matching the caption's own boldface claim
      ("Goal-setting most clearly sharpens the betting-aggression readout").
      ``$\\Delta_M\\%$`` is never bold, matching the .tex.
    * the *Fixed* row comes from a different pipeline (continuous-I_LC, plain
      cross-validation) than the six rows above it (GroupKFold by game id) --
      a known, documented mismatch (see ``appendix_neural_tables.py``'s
      ``table_condition_full``, "DEFECT 1"), reproduced here rather than
      papered over.  One Fixed cell (LLaMA I_BA) is -0.0007, which would
      round to -0.001 at 3dp; since the row is an explicitly non-comparator
      boundary case (footnoted in the caption -- label variance collapses
      with the wager locked), any Fixed-row cell under 0.001 in magnitude is
      printed as an approximate zero (``$\\sim$0``, the .tex's own notation)
      instead of adding false precision.
    """
    cm = load_json(LOCAL_SAE / "condition_modulation_groupkfold_L22.json",
                   "sae_v3_analysis/results/"
                   "condition_modulation_groupkfold_L22.json")
    fixed_src = load_json(LOCAL_SAE / "condition_modulation_continuous_ilc_L22.json",
                          "sae_v3_analysis/results/"
                          "condition_modulation_continuous_ilc_L22.json")
    # The "All variable" row is Table 1's slot-machine row re-printed, so its
    # standard error is carried over from the sidecar Table 1 prints from
    # rather than recomputed; the released fold sds agree to <=0.0007 but would
    # differ in the third decimal on two LLaMA cells, and one quantity must not
    # print two different intervals in one section.
    t1se = json.loads(
        (REPO_ROOT / "paper_data" / "table1_se_N200.json").read_text())["cells"]

    def _rec(model, ind, subset):
        key = f"{model}_sm_{ind}_L22"
        if key not in cm:
            raise MissingSource(f"condition_modulation_groupkfold_L22: no {key}")
        subs = cm[key].get("subsets", {})
        if subset not in subs:
            raise MissingSource(f"condition_modulation_groupkfold_L22: "
                                f"{key} has no subset {subset}")
        return key, subs[subset]

    def sub(model, ind, subset):
        key, rec = _rec(model, ind, subset)
        return float(require(rec.get("r2_mean"), f"{key}/{subset}"))

    def sub_se(model, ind, subset):
        """fold sd / sqrt(5) -- the definition Table 1 uses."""
        key, rec = _rec(model, ind, subset)
        if subset == "all_variable":
            return float(require(t1se[key]["se"], f"table1_se_N200/{key}"))
        sd = float(require(rec.get("r2_std"), f"{key}/{subset} r2_std"))
        return sd / math.sqrt(5.0)

    def fixed(model, ind):
        """Fixed-bet cell: the continuous-I_LC pipeline, not GroupKFold --
        the same pipeline the .tex's printed Fixed row matches (DEFECT 1)."""
        key = f"{model}_sm_{ind}_L22"
        rec = fixed_src.get(key, {}).get("subsets", {}).get("fixed_all")
        if rec is None:
            raise MissingSource(
                f"condition_modulation_continuous_ilc_L22: no {key}/fixed_all")
        return float(require(rec.get("r2"), f"{key}/fixed_all"))

    order = [("All variable", "all_variable"), ("$-G$", "minus_G"),
             ("$+G$", "plus_G"), ("$-M$", "minus_M"), ("$+M$", "plus_M")]
    cols = [(m, i) for m in MODELS for i in INDICATORS]
    vals = {(label, m, i): sub(m, i, key)
            for label, key in order for m, i in cols}
    ses = {(label, m, i): sub_se(m, i, key)
           for label, key in order for m, i in cols}
    fixed_vals = {(m, i): fixed(m, i) for m, i in cols}

    def cellfmt(v, se_):
        if with_se:
            # ``{\pm}`` and not ``\pm``: the spaced binary operator pushes the
            # nine-column tabular past \textwidth.
            return f"${v:.3f}{{\\pm}}{se_:.3f}$"
        return r3(v)

    rows, ns = [], {}
    for label, key in order:
        if label in ("$-G$", "$-M$"):
            rows.append(r"\midrule")
        if label == "$+G$":
            rows.append(r"\rowcolor{green!12}")
        cells = []
        for m, i in cols:
            v = vals[(label, m, i)]
            txt = cellfmt(v, ses[(label, m, i)])
            if label == "$+G$" and v > vals[("$-G$", m, i)]:
                txt = _bold(txt)
            cells.append(txt)
        ns[label] = {f"{m}_{i}": cm[f"{m}_sm_{i}_L22"]["subsets"][key]["n"]
                     for m, i in cols}
        # column 2 and column 6 are the row's decision count, the n behind
        # I_BA/I_EC; the I_LC subsets are smaller and are given in the caption.
        def _n(model):
            return f"{ns[label][model + '_i_ba']:,}".replace(",", "{,}")
        rows.append(f"{label:<22} & {_n('gemma')} & "
                    + " & ".join(cells[:3])
                    + f" & {_n('llama')} & " + " & ".join(cells[3:]) + r" \\")

        if label == "$+G$" and not with_se:
            dg = []
            for m, i in cols:
                c = _pct(vals[("$+G$", m, i)], vals[("$-G$", m, i)])
                if i == "i_ba" and c != "---":
                    c = _bold(c)
                dg.append(c)
            rows.append(r"$\Delta_G\%$" + " & & " + " & ".join(dg[:3])
                        + " & & " + " & ".join(dg[3:]) + r" \\")
        if label == "$+M$" and not with_se:
            dm = [_pct(vals[("$+M$", m, i)], vals[("$-M$", m, i)]) for m, i in cols]
            rows.append(r"$\Delta_M\%$" + " & & " + " & ".join(dm[:3])
                        + " & & " + " & ".join(dm[3:]) + r" \\")

    if not with_se:
        rows.append(r"\midrule")
        rows.append(r"\rowcolor{red!8}")

        def fixedfmt(v):
            return r"$\sim$0" if abs(v) < 0.001 else r3(v)

        fcells = [fixedfmt(fixed_vals[(m, i)]) for m, i in cols]
        rows.append("Fixed" + " & & " + " & ".join(fcells[:3])
                    + " & & " + " & ".join(fcells[3:]) + r" \\")

    note = "% subset sizes (rounds):\n"
    for label, _ in order:
        note += f"%   {label}: " + ", ".join(
            f"{k}={v}" for k, v in ns[label].items()) + "\n"
    note += "% Fixed row (continuous-I_LC pipeline, fixed_all subset):\n"
    for m, i in cols:
        note += f"%   {m}_sm_{i}: r2={fixed_vals[(m, i)]:+.6f}\n"

    body = header(
        "Table 4 -- slot-machine readout R^2 by prompt condition (L22)",
        "tab:condition-modulation",
        ["sae_v3_analysis/results/condition_modulation_groupkfold_L22.json",
         "sae_v3_analysis/results/condition_modulation_continuous_ilc_L22.json "
         "(Fixed row only -- DEFECT 1, a different pipeline from the rows above it)"],
        ["bold in the +G row = the cell is higher than its -G counterpart",
         "bold in the Delta_G%% row = the I_BA column, the row's headline number",
         "no '+-' in the printed cells -- SEs moved to appendix "
         "Table tab:appendix-condition-se; call table4(with_se=True) for that form",
         "Fixed-row cells under 0.001 in magnitude print as '~0', matching the .tex "
         "(the row is an explicit non-comparator, footnoted in the caption)"],
    ) + note + "\n".join(rows) + "\n"
    return body, vals


# ----------------------------------------------------------- diff vs the tex


def tex_tabular(label: str) -> list[str]:
    """The *body* rows of the tabular carrying ``label`` in 4.neural.tex.

    Body = everything after the first ``\\midrule`` that is not a spanning
    (``\\multicolumn``) row, which is exactly the set of rows a fragment would
    replace.  Header rows sit above that first rule and stay in the float.
    """
    if not NEURAL_TEX.exists():
        raise MissingSource(f"{NEURAL_TEX} is missing")
    text = NEURAL_TEX.read_text()
    at = text.find("\\label{" + label + "}")
    if at < 0:
        raise MissingSource(f"{NEURAL_TEX} has no \\label{{{label}}}")
    start = text.find(r"\begin{tabular}", at)
    end = text.find(r"\end{tabular}", start)
    if start < 0 or end < 0:
        raise MissingSource(f"no tabular after \\label{{{label}}}")
    inner = text[text.find("}", text.find("}", start) + 1) + 1:end]

    out, in_body = [], False
    for raw in inner.split(r"\\"):
        line = re.sub(r"%.*", "", raw).strip()
        if r"\midrule" in line:
            in_body = True
        for cmd in (r"\toprule", r"\midrule", r"\bottomrule", r"\addlinespace"):
            line = line.replace(cmd, "")
        line = re.sub(r"\\cmidrule\(lr\)\{[^}]*\}", "", line)
        line = re.sub(r"\\rowcolor\{[^}]*\}", "", line)
        line = line.strip()
        if in_body and line and "&" in line and r"\multicolumn" not in line:
            out.append(line)
    if not out:
        raise MissingSource(f"no body rows found in the tabular for {label}")
    return out


def cells(row: str) -> list[str]:
    return [c.strip() for c in row.split("&")]


def norm(cell: str) -> str:
    """Compare what the reader sees: strip colour, bold, blue and whitespace."""
    c = re.sub(r"\\cellcolor\{[^}]*\}", "", cell)
    c = re.sub(r"\\(?:textbf|blue|emph|text)\{", "{", c)
    c = c.replace("{", "").replace("}", "")
    c = c.replace("\\,", "").replace("$", "").replace("\\!", "")
    c = re.sub(r"\\[a-zA-Z]+", "", c)
    c = c.replace("^", "").replace("---", "--")
    return re.sub(r"\s+", "", c)


def diff_table(name: str, label: str, emitted: str, key_cols: int = 1) -> list[dict]:
    """Cell-by-cell comparison of the fragment with the typed .tex.

    Rows are matched on their leading label cells, not on position, so an
    inserted rule or a reordered row cannot masquerade as a column of value
    changes.  A row that exists on one side only is itself reported.
    """
    def keyed(rows):
        out = {}
        for r in rows:
            c = cells(r)
            out["|".join(norm(x) for x in c[:key_cols])] = c
        return out

    typed = keyed(tex_tabular(label))
    mine = keyed([r.strip().rstrip("\\").strip()
                  for r in emitted.splitlines()
                  if r.strip() and not r.startswith("%")
                  and not r.strip().startswith("\\midrule")
                  and not r.strip().startswith("\\rowcolor")])

    found = []
    for key in typed:
        if key not in mine:
            found.append({"element": f"{name} row [{key}]", "paper": "present",
                          "computed": "absent",
                          "note": "row is in the .tex but the emitter produced none"})
    for key in mine:
        if key not in typed:
            found.append({"element": f"{name} row [{key}]", "paper": "absent",
                          "computed": "present",
                          "note": "emitter produced a row the .tex does not have"})
    for key, tc in typed.items():
        mc = mine.get(key)
        if mc is None:
            continue
        if len(tc) != len(mc):
            found.append({"element": f"{name} [{key}] column count",
                          "paper": str(len(tc)), "computed": str(len(mc)),
                          "note": ""})
        for k in range(min(len(tc), len(mc))):
            if norm(tc[k]) != norm(mc[k]):
                found.append({"element": f"{name} [{key}] col {k + 1}",
                              "paper": tc[k].strip(),
                              "computed": mc[k].strip(),
                              "note": ""})
    return found


# ------------------------------------------------------------------- main


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t1_body, t1_grid, binding = table1()
    t2_body, t2_vals, steer, band = table2()
    t3_body, cos_of, best_in, auc, missing_transfer = table3()
    t4_body, t4_vals = table4()

    written = []
    for fname, body in (("table1_sae_readout_r2.tex", t1_body),
                        ("table2_causal_suffnec.tex", t2_body),
                        ("table3_sharing_transfer.tex", t3_body),
                        ("table4_condition_modulation.tex", t4_body)):
        p = OUT_DIR / fname
        p.write_text(body)
        written.append(str(p))
        print(f"wrote {p}")

    print()
    print("=== emitted fragments ===")
    for fname in ("table1_sae_readout_r2.tex", "table2_causal_suffnec.tex",
                  "table3_sharing_transfer.tex", "table4_condition_modulation.tex"):
        print(f"--- {fname} ---")
        for line in (OUT_DIR / fname).read_text().splitlines():
            if not line.startswith("%"):
                print("   " + line)

    print()
    print("=== diff against neurips_content_en/4.neural.tex ===")
    diffs = []
    diffs += diff_table("Table 1", "tab:neurips-sae-results", t1_body, key_cols=2)
    diffs += diff_table("Table 2", "tab:causal-battery-suffnec", t2_body, key_cols=2)
    diffs += diff_table("Table 3", "tab:sharing-transfer", t3_body, key_cols=1)
    diffs += diff_table("Table 4", "tab:condition-modulation", t4_body, key_cols=1)
    if not diffs:
        print("  no differences")
    for d in diffs:
        print(f"  {d['element']}: paper={d['paper']!r} computed={d['computed']!r}"
              + (f"  [{d['note']}]" if d["note"] else ""))

    print()
    print("=== provenance ===")
    print(f"  null slope band: n={band['n_directions']} mean={band['mean']:.6f} "
          f"sd={band['sd']:.6f}")
    for axis, e in steer.items():
        print(f"  gemma steer {axis}: slope={e['slope']:+.6f} z={e['z']:+.4f}")
    for k, e in t2_vals.items():
        print(f"  removal {k[0]} {k[1]}: delta={e['delta']:+.5f} p={e['p']:.6f} "
              f"({e['n_down']} down / {e['n_up']} up of {e['n_pairs']} pairs)")
    print(f"  BK max|cos|: " + ", ".join(f"{t}={v:.4f}" for t, v in cos_of.items()))
    print(f"  shared-only AUC: " + ", ".join(
        f"{t} r{r}={auc[(t, r)]:.4f}" for t in TASKS for r in (1, 2)))
    if missing_transfer:
        print(f"  NO off-diagonal feature transfer released for: "
              f"{', '.join(missing_transfer)}")
    print(f"  task binding (green in Table 1): {binding}")
    print(f"  wrote {len(written)} fragments")
    return 0


if __name__ == "__main__":
    sys.exit(main())
