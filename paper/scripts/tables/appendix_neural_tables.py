"""Regenerate every neural table of ``neurips_content_en/appendix.tex`` from data.

Scope
-----
This is the neural half of the appendix tables.  Walking ``appendix.tex`` plus
its two ``\\input`` files, the neural tables are:

  tab:appendix-sae-full            full SAE readout table (n, R^2, p)
  tab:appendix-groupkfold-sweep    layer sweep          [already auto-generated]
  tab:appendix-band-readout        write-band readout   [already auto-generated]
  tab:neurips-selectivity-l2       prompt-condition exclusion cross-validation
  tab:appendix-selectivity-controls selectivity vs noise-only controls
  tab:rq2-sharing                  cross-task representation-sharing audit
  tab:appendix-condition-multi     MW/IC +-G/+-M condition modulation
  tab:appendix-condition-full      SM +-G/+-M condition modulation + fixed row
  tab:causal-transfer-matrix       12-cell cross-task steering matrix
  tab:causal-condition-writability condition writability of the behavioural axis

The two already-generated ones are *wired in*, not rewritten: their scripts
live in the analysis repo and their stdout was verified byte-identical to the
committed ``.tex``.  We import them, repoint their results directory at a
staging tree built from the public HF dataset, and capture stdout.

Everything else is computed here, from the public dataset
``llm-addiction-research/llm-addiction``, the local behavioural mirror, and the
local causal raw records.  No number in any emitted fragment is a literal.

Emitted fragments contain **tabular body rows only** (no ``\\begin{table}``, no
caption) so they can be ``\\input`` into the existing floats, with the existing
column order and rounding.  Intervals that the existing column layout has no
room for are carried as LaTeX comments (``%``) inside the fragment, so the
information is preserved without changing the float.

Three known defects are surfaced rather than papered over:

  D1  ``tab:appendix-condition-full``'s fixed-bet row is computed by a
      *different* pipeline (continuous-I_LC, plain 5-fold) from the six other
      rows (GroupKFold by game id), with no note in the paper.  Both pipelines'
      values for that row are emitted and labelled.

  D2  ``tab:appendix-band-readout`` evaluates the LLaMA IC and MW rows at
      L14--19, but the appendix's own causal protocol names L12--17 (IC) and
      L16--21 (MW) as those tasks' write bands.  The rows are re-emitted at the
      protocol layers alongside L14--19 and the difference is reported.

  D3  The 12-cell steering matrix exists only as literals inside
      ``scripts/gen_fig_cross_context_write.py``, and the project's own
      ``INDEX.md`` disagrees with the paper on two cells.  All twelve cells are
      recomputed from ``experiments/sec4_causal/checkpoints/sec4_w7/*.jsonl`` and all
      three versions are reported.

Run with::

    from the repository root, \
        HF_HUB_DISABLE_XET=1 python3 scripts/tables/appendix_neural_tables.py
"""

from __future__ import annotations

import glob
import io
import json
import os
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import hf_hub_download  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ID = "llm-addiction-research/llm-addiction"
HERE = Path(__file__).resolve()
PAPER_ROOT = HERE.parent.parent.parent
OUT_DIR = PAPER_ROOT / "paper_data" / "tables" / "appendix"

# The two analysis checkouts this module shells out to.  They are separate
# repositories, not part of the paper repo and not on the dataset; point
# ``LLM_ADDICTION_ANALYSIS`` at the directory that holds them.
ANALYSIS_ROOT = Path(os.environ.get("LLM_ADDICTION_ANALYSIS", Path.home() / "llm-addiction"))
SAE_REPO = ANALYSIS_ROOT / "experiments" / "07_sae_readout"
MLC_REPO = ANALYSIS_ROOT / "experiments" / "08_steering" / "multilayer_causal"
W7_DIR = MLC_REPO / "results" / "sec4_w7"
W14_DIR = MLC_REPO / "results" / "sec4_w14"

STAGE = Path("/tmp/appendix_neural_stage")
TASK_DIR = {"sm": "slot_machine", "ic": "investment_choice", "mw": "mystery_wheel"}
MODEL_TEX = {"gemma": "Gemma", "llama": "LLaMA"}
TASK_TEX = {"sm": "SM", "ic": "IC", "mw": "MW"}
IND_TEX = {"i_lc": r"$I_\text{LC}$", "i_ba": r"$I_\text{BA}$", "i_ec": r"$I_\text{EC}$"}

# The reporting floor the two already-generated tables use ("below null floor").
R2_FLOOR = 0.01

DISCREPANCIES: list[dict] = []
NOTES: list[str] = []


def note(msg: str) -> None:
    NOTES.append(msg)
    print(f"[note] {msg}")


def flag(element: str, paper: str, computed: str, source: str, comment: str) -> None:
    DISCREPANCIES.append(
        {
            "element": element,
            "paper_value": paper,
            "computed_value": computed,
            "source": source,
            "note": comment,
        }
    )
    print(f"[DIFF] {element}: paper={paper} computed={computed}  ({comment})")


# --------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------

def fetch(path: str) -> Path:
    return Path(hf_hub_download(REPO_ID, path, repo_type="dataset"))


def load_json(hf_path: str):
    return json.load(open(fetch(hf_path)))


def stage(hf_path: str, rel: str) -> Path:
    """Symlink an HF file into the staging tree under ``rel``."""
    src = fetch(hf_path)
    dst = STAGE / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)
    return dst


# --------------------------------------------------------------------------
# emission helpers
# --------------------------------------------------------------------------

def write_tex(name: str, header_comment: str, body: list[str]) -> Path:
    path = OUT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"% {ln}" for ln in header_comment.strip().splitlines()]
    lines.append("% Generated by scripts/tables/appendix_neural_tables.py -- do not hand-edit.")
    lines.append("% Body rows only: \\input into the existing float.")
    lines += body
    path.write_text("\n".join(lines) + "\n")
    print(f"[emit] {path}  ({len(body)} lines)")
    return path


def f3(x, floor: float | None = None, dash: str = "---") -> str:
    if x is None:
        return dash
    if floor is not None and x < floor:
        return dash
    return f"{x:.3f}"


def signed3(x) -> str:
    if x is None:
        return "---"
    return f"$-${abs(x):.3f}" if x < 0 else f"{x:.3f}"


def thousands(n: int) -> str:
    s = f"{n:,}"
    return s.replace(",", "{,}")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def boot_ci(vals: np.ndarray, n_boot: int = 2000, seed: int = 24231) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[idx].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


# ==========================================================================
# T1  tab:appendix-sae-full
# ==========================================================================

SAE_FULL_PAPER = {
    # (model, task): (n, {ind: (r2_str, p_str)})
    ("gemma", "sm"): ("12{,}246", {"i_lc": ("0.059", "<.05"), "i_ba": ("0.167", "<.05"), "i_ec": ("0.051", "<.05")}),
    ("gemma", "ic"): ("5{,}735", {"i_lc": ("---", "n.s."), "i_ba": ("0.203", "<.05"), "i_ec": ("---", "n.s.")}),
    ("gemma", "mw"): ("8{,}948", {"i_lc": ("0.107", "<.05"), "i_ba": ("0.072", "<.05"), "i_ec": ("n/c", "---")}),
    ("llama", "sm"): ("45{,}551", {"i_lc": ("0.106", "<.05"), "i_ba": ("0.109", "<.05"), "i_ec": ("0.036", "<.05")}),
    ("llama", "ic"): ("2{,}064", {"i_lc": ("---", "n.s."), "i_ba": ("0.268", "<.05"), "i_ec": ("0.307", "<.05")}),
    ("llama", "mw"): ("57{,}220", {"i_lc": ("0.145", "<.05"), "i_ba": ("0.070", "<.05"), "i_ec": ("n/c", "---")}),
}
# Which cell carries the green "dominant predictor" mark in the paper.
SAE_FULL_PAPER_DOMINANT = {
    ("gemma", "sm"): "i_ba", ("gemma", "ic"): "i_ba", ("gemma", "mw"): "i_lc",
    ("llama", "sm"): "i_ba", ("llama", "ic"): "i_ba", ("llama", "mw"): "i_lc",
}


def table_sae_full() -> None:
    l22 = load_json("sae_v3_analysis/results/table1_groupkfold_L22.json")
    perm = load_json("sae_v3_analysis/results/table1_perm_null.json")

    body: list[str] = []
    comments: list[str] = []
    for model in ("gemma", "llama"):
        for task in ("sm", "ic", "mw"):
            rec = {ind: l22.get(f"{model}_{task}_{ind}_L22", {}) for ind in IND_TEX}
            r2 = {ind: rec[ind].get("r2_mean") for ind in IND_TEX}
            # n column: the I_BA/I_EC all-variable subset size (paper's convention).
            n_ba = rec["i_ba"].get("n")
            n_lc = rec["i_lc"].get("n")
            reported = {ind: (r2[ind] is not None and r2[ind] >= R2_FLOOR) for ind in IND_TEX}
            dominant = max(
                (ind for ind in IND_TEX if reported[ind]),
                key=lambda i: r2[i],
                default=None,
            )
            cells: list[str] = []
            for ind in ("i_lc", "i_ba", "i_ec"):
                pk = f"{model}_{task}_{ind}_L22"
                prec = perm.get(pk)
                if not reported[ind]:
                    val, pstr = "---", "n.s."
                elif prec is None:
                    val, pstr = f"{r2[ind]:.3f}", "---"
                    comments.append(
                        f"{model} {task} {ind}: R2={r2[ind]:.4f} but NO permutation record "
                        f"in table1_perm_null.json -- p left as ---"
                    )
                else:
                    val, pstr = f"{r2[ind]:.3f}", (r"$<\!.05$" if prec["perm_p"] < 0.05 else f"{prec['perm_p']:.3f}")
                if ind == dominant and reported[ind]:
                    val = r"\cellcolor{green!12}\textbf{" + val + "}"
                cells += [val, pstr]
                # interval carried as a comment: fold-to-fold spread
                sd = rec[ind].get("r2_std")
                if r2[ind] is not None and sd is not None:
                    comments.append(
                        f"{model} {task} {ind}: R2={r2[ind]:+.4f} +- {sd:.4f} (5-fold SD), "
                        f"n={rec[ind].get('n')}, groups={rec[ind].get('n_groups')}"
                    )
            n_str = thousands(int(n_ba)) if n_ba else "---"
            body.append(
                f"{MODEL_TEX[model]} & {TASK_TEX[task]} & {n_str} & " + " & ".join(cells) + r" \\"
            )

            # ---- compare against the typed table -------------------------
            p_n, p_cells = SAE_FULL_PAPER[(model, task)]
            if p_n != n_str:
                flag(
                    f"tab:appendix-sae-full {MODEL_TEX[model]}/{TASK_TEX[task]} n",
                    p_n, n_str, "table1_groupkfold_L22.json",
                    "n column is the I_BA all-variable subset size",
                )
            for ind in ("i_lc", "i_ba", "i_ec"):
                got_r2 = f"{r2[ind]:.3f}" if reported[ind] else "---"
                want_r2 = p_cells[ind][0]
                if want_r2 == "n/c":
                    if reported[ind]:
                        flag(
                            f"tab:appendix-sae-full {MODEL_TEX[model]}/{TASK_TEX[task]} {ind} R2",
                            "n/c (footnote: R2~0)", got_r2, "table1_groupkfold_L22.json",
                            "cell is above the 0.01 reporting floor, so 'n/c' understates it",
                        )
                elif want_r2 != got_r2:
                    flag(
                        f"tab:appendix-sae-full {MODEL_TEX[model]}/{TASK_TEX[task]} {ind} R2",
                        want_r2, got_r2, "table1_groupkfold_L22.json",
                        f"raw R2={r2[ind]:+.4f}; floor {R2_FLOOR}",
                    )
                want_p = p_cells[ind][1]
                has_perm = f"{model}_{task}_{ind}_L22" in perm
                if want_p == "<.05" and not has_perm:
                    flag(
                        f"tab:appendix-sae-full {MODEL_TEX[model]}/{TASK_TEX[task]} {ind} p",
                        "$<.05$", "unavailable", "table1_perm_null.json",
                        "no permutation record for this cell in the released null file",
                    )
            if SAE_FULL_PAPER_DOMINANT[(model, task)] != dominant:
                flag(
                    f"tab:appendix-sae-full {MODEL_TEX[model]}/{TASK_TEX[task]} dominant indicator",
                    SAE_FULL_PAPER_DOMINANT[(model, task)], str(dominant),
                    "table1_groupkfold_L22.json", "green/bold mark",
                )
            # I_LC uses the post-loss subset -> its own n
            if n_lc and n_ba and n_lc != n_ba:
                comments.append(
                    f"{model} {task}: I_LC post-loss subset n*={n_lc} vs all-variable n={n_ba}"
                )

    write_tex(
        "A01_sae_full.tex",
        "tab:appendix-sae-full -- L22 SAE readout R^2 with permutation p.\n"
        "Source: HF sae_v3_analysis/results/table1_groupkfold_L22.json (R^2, n)\n"
        "        HF sae_v3_analysis/results/table1_perm_null.json (perm p).\n"
        f"Reporting floor R^2 >= {R2_FLOOR} (same floor as the layer-sweep table).\n"
        "Columns: Model & Task & n & R2(I_LC) & p & R2(I_BA) & p & R2(I_EC) & p\n"
        "Intervals (5-fold SD) below:\n" + "\n".join(comments),
        body,
    )


# ==========================================================================
# T2/T3  wired-in already-generated tables
# ==========================================================================

def stage_groupkfold_results() -> Path:
    """Staging dir holding every table1_groupkfold_* json the two scripts read."""
    root = STAGE / "gk_results"
    root.mkdir(parents=True, exist_ok=True)
    for layer in (8, 12, 22, 25, 30):
        stage(
            f"sae_v3_analysis/results/table1_groupkfold_L{layer}.json",
            f"gk_results/table1_groupkfold_L{layer}.json",
        )
    for model in ("gemma", "llama"):
        stage(
            f"sae_v3_analysis/results/table1_groupkfold_band_{model}.json",
            f"gk_results/table1_groupkfold_band_{model}.json",
        )
    return root


def _run_wired(module_name: str, script_path: Path, results_dir: Path) -> str:
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.RESULTS = results_dir  # repoint at the staged HF copies
    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.main()
    return buf.getvalue()


def _tabular_body(stdout: str) -> list[str]:
    """Keep only the row/rule lines of a full-table stdout, drop the float."""
    drop_prefixes = (
        r"\begin{table}", r"\end{table}", r"\centering", r"\footnotesize",
        r"\caption{", r"\label{", r"\begin{tabular}", r"\end{tabular}",
        r"\toprule", r"\bottomrule",
    )
    out: list[str] = []
    for raw in stdout.splitlines():
        ln = raw.rstrip()
        if not ln.strip():
            continue
        if any(ln.lstrip().startswith(p) for p in drop_prefixes):
            continue
        out.append(ln)
    return out


_NUM_RE = None


def _numeric_cells(text: str) -> list[str]:
    """Every numeric cell of a LaTeX table body, in reading order."""
    import re

    global _NUM_RE
    if _NUM_RE is None:
        _NUM_RE = re.compile(r"-?\d+\.\d+|---")
    cells: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("%") or s.startswith(r"\caption"):
            continue
        if not (s.endswith(r"\\") and "&" in s):
            continue
        cells += _NUM_RE.findall(s)
    return cells


def _parity_check(label: str, generated: str, committed: Path) -> None:
    """Numbers of the generator's stdout vs the committed .tex."""
    a = _numeric_cells(generated)
    b = _numeric_cells(committed.read_text())
    if a == b:
        note(f"{label}: all {len(a)} numeric cells identical to {committed.name}")
    else:
        flag(
            f"{label} numeric parity with {committed.name}",
            f"{len(b)} cells", f"{len(a)} cells",
            str(committed),
            "generator stdout and the committed .tex disagree numerically",
        )
        for i, (x, y) in enumerate(zip(b, a)):
            if x != y:
                print(f"    cell {i}: committed={x} generated={y}")
    gen_rows = [ln for ln in generated.splitlines()
                if ln.strip().endswith(r"\\") and "&" in ln]
    com_rows = [ln for ln in committed.read_text().splitlines()
                if ln.strip().endswith(r"\\") and "&" in ln]
    if [r.strip() for r in gen_rows] != [r.strip() for r in com_rows]:
        flag(
            f"{label} formatting parity with {committed.name}",
            "committed layout", "generator stdout layout",
            str(committed),
            "the committed .tex is NOT byte-identical to its generator's stdout: it "
            "was re-laid-out by hand after generation (numbers unchanged)",
        )


def table_layer_sweep(results_dir: Path) -> None:
    script = SAE_REPO / "scripts" / "build_appendix_layer_sweep_table.py"
    txt = _run_wired("build_appendix_layer_sweep_table", script, results_dir)
    body = _tabular_body(txt)
    # the header row is the first non-comment line after the drops
    write_tex(
        "A02_layer_sweep.tex",
        "tab:appendix-groupkfold-sweep -- WIRED IN, not rewritten.\n"
        f"Generator: HF sae_v3_analysis/scripts/{script.name}\n"
        "Source: HF sae_v3_analysis/results/table1_groupkfold_L{8,12,22,25,30}.json\n"
        "Emitted verbatim from the generator's stdout with the float wrapper stripped.",
        body,
    )
    _parity_check(
        "tab:appendix-groupkfold-sweep", txt,
        PAPER_ROOT / "neurips_content_en" / "_appendix_layer_sweep_table.tex",
    )


def table_band_readout(results_dir: Path) -> None:
    script = SAE_REPO / "scripts" / "build_appendix_band_readout_table.py"
    txt = _run_wired("build_appendix_band_readout_table", script, results_dir)
    body = _tabular_body(txt)
    write_tex(
        "A03_band_readout.tex",
        "tab:appendix-band-readout -- WIRED IN, not rewritten.\n"
        f"Generator: HF sae_v3_analysis/scripts/{script.name}\n"
        "Source: HF sae_v3_analysis/results/table1_groupkfold_band_{gemma,llama}.json\n"
        "        + table1_groupkfold_L22.json for the reference column.\n"
        "Emitted verbatim from the generator's stdout with the float wrapper stripped.",
        body,
    )
    _parity_check(
        "tab:appendix-band-readout", txt,
        PAPER_ROOT / "neurips_content_en" / "_appendix_band_readout_table.tex",
    )


# ==========================================================================
# T3b  DEFECT 2 -- band readout at the layers the causal protocol names
# ==========================================================================

PROTOCOL_BANDS = {
    # appendix:causal-protocol: "the cross-task investment-choice and
    # mystery-wheel arms use their own task-specific windows (L12--17 and L16--21)"
    ("llama", "ic"): list(range(12, 18)),
    ("llama", "mw"): list(range(16, 22)),
}
PUBLISHED_BAND = {"gemma": list(range(16, 22)), "llama": list(range(14, 20))}
BAND_CELLS = {("llama", "ic"): ["i_ba", "i_ec"], ("llama", "mw"): ["i_lc", "i_ba"]}


def _gk_pipeline():
    """Import the paper-canonical GroupKFold recompute, repointed at staged SAE."""
    sys.path.insert(0, str(SAE_REPO / "src"))
    import run_perm_null_ilc as rpn
    import run_groupkfold_recompute as gk

    rpn.DATA_ROOT = STAGE / "sae_features_v3"
    return rpn, gk


def table_band_protocol_layers() -> None:
    published = {m: load_json(f"sae_v3_analysis/results/table1_groupkfold_band_{m}.json")
                 for m in ("gemma", "llama")}
    sweep12 = load_json("sae_v3_analysis/results/table1_groupkfold_L12.json")

    def known(model, task, ind, layer):
        key = f"{model}_{task}_{ind}_L{layer}"
        if layer in PUBLISHED_BAND[model] and key in published[model]:
            return published[model][key].get("r2_mean")
        if layer == 12 and key in sweep12:
            return sweep12[key].get("r2_mean")
        return None

    # which (model, task, ind, layer) still need computing
    todo = []
    for (model, task), layers in PROTOCOL_BANDS.items():
        for ind in BAND_CELLS[(model, task)]:
            for L in layers:
                if known(model, task, ind, L) is None:
                    todo.append((model, task, ind, L))
    need_layers = sorted({(m, t, L) for m, t, _, L in todo})

    computed: dict[tuple, float | None] = {}
    if need_layers:
        rpn, gk = _gk_pipeline()
        for model, task, L in need_layers:
            stage(
                f"sae_features_v3/{TASK_DIR[task]}/{model}/sae_features_L{L}.npz",
                f"sae_features_v3/{TASK_DIR[task]}/{model}/sae_features_L{L}.npz",
            )
            sp, meta = rpn.load_sae_and_meta(model, task, L)
            if sp is None:
                note(f"band-protocol: SAE features missing for {model}/{task}/L{L}")
                continue
            for ind in BAND_CELLS[(model, task)]:
                res = gk.fit_one_subset(meta, sp, model, task, ind)
                computed[(model, task, ind, L)] = res.get("r2_mean")
                print(
                    f"  [recompute] {model}/{task}/{ind}/L{L}: "
                    f"R2={res.get('r2_mean')} n={res.get('n')}"
                )

    def val(model, task, ind, layer):
        v = known(model, task, ind, layer)
        return v if v is not None else computed.get((model, task, ind, layer))

    body: list[str] = []
    comments: list[str] = []
    for (model, task), layers in PROTOCOL_BANDS.items():
        body.append(
            r"\multicolumn{2}{l}{\emph{" + MODEL_TEX[model] + " " + TASK_TEX[task]
            + f" (protocol band L{layers[0]}--L{layers[-1]})" + r"}} & "
            + " & ".join(f"L{L}" for L in layers) + r" & L14--19 mean & $\Delta$ \\"
        )
        for ind in BAND_CELLS[(model, task)]:
            proto = [val(model, task, ind, L) for L in layers]
            pub = [val(model, task, ind, L) for L in PUBLISHED_BAND[model]]
            proto_ok = [v for v in proto if v is not None]
            pub_ok = [v for v in pub if v is not None]
            pm = float(np.mean(proto_ok)) if proto_ok else None
            um = float(np.mean(pub_ok)) if pub_ok else None
            d = (pm - um) if (pm is not None and um is not None) else None
            cells = [f3(v) for v in proto]
            body.append(
                r"\quad " + IND_TEX[ind] + " & & " + " & ".join(cells)
                + f" & {f3(um)} & {signed3(d)}" + r" \\"
            )
            lo, hi = (min(proto_ok), max(proto_ok)) if proto_ok else (None, None)
            comments.append(
                f"{model}/{task}/{ind}: protocol band L{layers[0]}-{layers[-1]} "
                f"mean={pm:.4f} range=[{lo:.4f},{hi:.4f}] ; "
                f"published band L14-19 mean={um:.4f} ; delta={d:+.4f}"
            )
            flag(
                f"tab:appendix-band-readout {MODEL_TEX[model]}/{TASK_TEX[task]}/{ind} band layers",
                f"L14--19 (mean {um:.3f})",
                f"L{layers[0]}--L{layers[-1]} (mean {pm:.3f})",
                "appendix:causal-protocol names L12-17 (IC) and L16-21 (MW)",
                f"D2: readout band and causal write band disagree; delta {d:+.4f}",
            )

    write_tex(
        "A03b_band_readout_protocol_layers.tex",
        "DEFECT 2 -- tab:appendix-band-readout evaluates the LLaMA IC/MW rows at\n"
        "L14--19, but appendix:causal-protocol names L12--17 (IC) and L16--21 (MW)\n"
        "as those tasks' write bands.  Rows re-emitted at the protocol layers, with\n"
        "the published L14--19 band mean and the difference.\n"
        "Source: HF sae_v3_analysis/results/table1_groupkfold_band_llama.json + table1_groupkfold_L12.json,\n"
        "        plus a same-pipeline recompute (run_groupkfold_recompute.fit_one_subset)\n"
        "        for the layers no released file covers.\n" + "\n".join(comments),
        body,
    )


# ==========================================================================
# T4  tab:neurips-selectivity-l2
# ==========================================================================

SELECT_L2_PAPER = {
    "gemma_sm_L24_i_lc": ("0.228", "-0.019", "0.032"),
    "gemma_ic_L24_i_lc": ("0.203", "-0.031", "0.032"),
    "gemma_mw_L24_i_lc": ("0.126", "-0.017", "0.032"),
    "llama_sm_L16_i_lc": ("0.326", "-0.007", "0.032"),
    "llama_ic_L16_i_lc": ("0.061", "-0.061", "0.032"),
    "llama_mw_L16_i_lc": ("0.314", "-0.006", "0.032"),
}


def table_selectivity_l2() -> None:
    d = load_json("sae_v3_analysis/results/robustness/rq1_l2_selectivity.json")
    body, comments = [], []
    for key in ("gemma_sm_L24_i_lc", "gemma_ic_L24_i_lc", "gemma_mw_L24_i_lc",
                "llama_sm_L16_i_lc", "llama_ic_L16_i_lc", "llama_mw_L16_i_lc"):
        rec = d[key]
        cfg = rec["config"]
        model, task, layer = cfg["model"], cfg["paradigm"], cfg["layer"]
        r2 = rec["real_r2"]
        null95 = rec["null_r2_95pct"]
        p = rec["perm_p"]
        decision = "Pass" if p < 0.05 and r2 > null95 else "Fail"
        body.append(
            f"{MODEL_TEX[model]} & {TASK_TEX[task]}, L{layer}, $I_\\text{{LC}}$ & "
            f"{r2:.3f} & {signed3(null95)} & {p:.3f} & {decision} \\\\"
        )
        folds = np.asarray(rec["real_fold_r2s"], dtype=float)
        lo, hi = boot_ci(folds)
        comments.append(
            f"{key}: held-out R2={r2:.4f} (folds {np.round(folds, 4).tolist()}; "
            f"bootstrap-over-folds 95% CI [{lo:.4f},{hi:.4f}]), "
            f"null mean={rec['null_r2_mean']:.4f} sd={rec['null_r2_std']:.4f} "
            f"n_perm={rec['n_perms']}, n={rec['n_samples']}, "
            f"n_prompt_groups={rec['n_prompt_groups']}"
        )
        pr2, pnull, pp = SELECT_L2_PAPER[key]
        if pr2 != f"{r2:.3f}":
            flag(f"tab:neurips-selectivity-l2 {key} R2", pr2, f"{r2:.3f}",
                 "robustness/rq1_l2_selectivity.json", "held-out R2")
        got_null = f"{null95:.3f}" if null95 >= 0 else f"-{abs(null95):.3f}"
        if pnull != got_null:
            flag(f"tab:neurips-selectivity-l2 {key} null 95%ile", pnull, got_null,
                 "robustness/rq1_l2_selectivity.json", "null_r2_95pct")
        if pp != f"{p:.3f}":
            flag(f"tab:neurips-selectivity-l2 {key} perm p", pp, f"{p:.3f}",
                 "robustness/rq1_l2_selectivity.json", "perm_p")
    write_tex(
        "A04_selectivity_l2.tex",
        "tab:neurips-selectivity-l2 -- prompt-condition exclusion cross-validation.\n"
        "Source: HF sae_v3_analysis/results/robustness/rq1_l2_selectivity.json\n"
        "Columns: Model & Task/Layer/Metric & Held-out R2 & Null 95%ile & perm p & Decision\n"
        "Intervals (percentile bootstrap over the 5 held-out folds):\n" + "\n".join(comments),
        body,
    )


# ==========================================================================
# T5  tab:appendix-selectivity-controls
# ==========================================================================

SELECT_CTRL_PAPER = {
    "gemma_sm_L24_i_lc": ("12,246", "0.246", "-0.065", "0.311", "0.048"),
    "llama_sm_L16_i_lc": ("45,551", "0.345", "-0.107", "0.451", "0.048"),
    "gemma_mw_L24_i_ba": ("8,948", "0.058", "-0.099", "0.157", "0.048"),
    "llama_mw_L16_i_ba": ("57,220", "0.069", "-0.123", "0.192", "0.048"),
}


def table_selectivity_controls() -> None:
    d = load_json("sae_v3_analysis/results/robustness/probe_selectivity_controls.json")
    body, comments = [], []
    for key in ("gemma_sm_L24_i_lc", "llama_sm_L16_i_lc",
                "gemma_mw_L24_i_ba", "llama_mw_L16_i_ba"):
        rec = d[key]
        cfg = rec["config"]
        model, task, layer, metric = cfg["model"], cfg["paradigm"], cfg["layer"], cfg["metric"]
        n = rec["n_samples"]
        real = rec["real_group_r2"]
        ctrl = float(np.mean(rec["control_r2s"]))
        gap = rec["selectivity_gap"]
        p = rec["p_selectivity"]
        body.append(
            f"{MODEL_TEX[model]} & {TASK_TEX[task]}, L{layer}, {IND_TEX[metric]} & "
            f"{n:,} & {real:.3f} & {signed3(ctrl)} & {gap:.3f} & {p:.3f} \\\\"
        )
        folds = np.asarray(rec["real_fold_r2s"], dtype=float)
        lo, hi = boot_ci(folds)
        clo, chi = np.percentile(np.asarray(rec["control_r2s"], dtype=float), [2.5, 97.5])
        comments.append(
            f"{key}: real R2={real:.4f} (bootstrap-over-folds 95% CI [{lo:.4f},{hi:.4f}]); "
            f"control mean={ctrl:.4f} (2.5-97.5 pct of {rec['n_controls']} controls "
            f"[{clo:.4f},{chi:.4f}]); gap={gap:.4f}; p={p:.6f}; "
            f"n={n}, n_games={rec['n_games']}, balance_bins={rec['balance_bins']}"
        )
        pn, preal, pctrl, pgap, pp = SELECT_CTRL_PAPER[key]
        for label, want, got in (
            ("n", pn, f"{n:,}"),
            ("Real R2", preal, f"{real:.3f}"),
            ("Control mean", pctrl, f"{ctrl:.3f}" if ctrl >= 0 else f"-{abs(ctrl):.3f}"),
            ("Gap", pgap, f"{gap:.3f}"),
            ("p", pp, f"{p:.3f}"),
        ):
            if want != got:
                flag(f"tab:appendix-selectivity-controls {key} {label}", want, got,
                     "robustness/probe_selectivity_controls.json", "")
    write_tex(
        "A05_selectivity_controls.tex",
        "tab:appendix-selectivity-controls -- readout vs 20 noise-only controls.\n"
        "Source: HF sae_v3_analysis/results/robustness/probe_selectivity_controls.json\n"
        "Columns: Model & Task/Layer/Metric & n & Real R2 & Control mean & Gap & p\n"
        "Intervals:\n" + "\n".join(comments),
        body,
    )


# ==========================================================================
# T6  tab:rq2-sharing
# ==========================================================================

RQ2_TASKS = ["ic", "mw", "sm"]
RQ2_PAPER = {
    "cos": {"ic": "0.04", "mw": "0.03", "sm": "0.04"},
    "rank1": {"ic": "0.74", "mw": "0.52", "sm": "0.80"},
    "rank2": {"ic": "0.60", "mw": "0.91", "sm": "0.84"},
    "residual": {"ic": "0.93", "mw": "0.95", "sm": "0.64"},
    "combined": {"ic": "0.96", "mw": "0.95", "sm": "0.97"},
}


def _load_hidden_bk(task: str, layer: int = 22):
    p = stage(
        f"sae_features_v3/{TASK_DIR[task]}/gemma/hidden_states_dp.npz",
        f"sae_features_v3/{TASK_DIR[task]}/gemma/hidden_states_dp.npz",
    )
    d = np.load(p, allow_pickle=False)
    layers = list(d["layers"])
    li = layers.index(layer)
    H = d["hidden_states"][:, li, :].astype(np.float32)
    out = d["game_outcomes"]
    bal = d["balances"].astype(np.float32)
    valid = ((out == "bankruptcy") | (out == "voluntary_stop")) & ~np.isnan(bal)
    return H[valid], (out[valid] == "bankruptcy").astype(int)


def _bk_direction(H, bk):
    v = H[bk == 1].mean(0) - H[bk == 0].mean(0)
    return v / max(np.linalg.norm(v), 1e-12)


def _readout_cosines(layer: int = 22) -> dict:
    """Cosines between the per-task PCA(64)+logistic readout directions.

    This is the object the LOTO rows (iii)/(iv) are built from, so it is the
    cross-check that matters most for internal consistency of the table.
    """
    sys.path.insert(0, str(SAE_REPO / "src"))
    import run_rq2_aligned_hidden_transfer as rq2

    for task in RQ2_TASKS:
        stage(
            f"sae_features_v3/{TASK_DIR[task]}/gemma/hidden_states_dp.npz",
            f"sae_features_v3/{TASK_DIR[task]}/gemma/hidden_states_dp.npz",
        )
    rq2.DATA_ROOT = STAGE / "sae_features_v3"
    readouts = {}
    for task in RQ2_TASKS:
        ds = rq2.build_task_dataset("gemma", task, layer)
        readouts[task] = rq2.fit_task_readout(ds["X"], ds["y"], pca_dim=64, seed=42)["readout"]
    out = {}
    for i, a in enumerate(RQ2_TASKS):
        for b in RQ2_TASKS[i + 1:]:
            out[f"{a}-{b}"] = float(
                readouts[a] @ readouts[b]
                / (np.linalg.norm(readouts[a]) * np.linalg.norm(readouts[b]))
            )
    return out


def _feature_transfer_r2(layer: int = 22) -> dict:
    """Off-diagonal sparse-feature transfer R^2 at Gemma L22.

    Same estimator as run_iba_cross_task_probe: RF-deconfound I_BA against
    balance + round, pick the top-200 features by |Spearman| on the source,
    fit Ridge(alpha=100) on the source, score on the target.  Only features
    active in *both* tasks enter, since the readout must be re-applied.
    """
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler

    rpn, gk = _gk_pipeline()
    from run_comprehensive_robustness import compute_iba

    per = {}
    for task in RQ2_TASKS:
        stage(
            f"sae_features_v3/{TASK_DIR[task]}/gemma/sae_features_L{layer}.npz",
            f"sae_features_v3/{TASK_DIR[task]}/gemma/sae_features_L{layer}.npz",
        )
        sp, meta = rpn.load_sae_and_meta("gemma", task, layer)
        if sp is None:
            continue
        br, bal = compute_iba(meta, "gemma", task)
        bt = meta["bet_types"]
        valid = (bt == "variable") & ~np.isnan(br) & ~np.isnan(bal) & (bal > 0) & (br > 0)
        X = sp[valid]
        rn = meta["round_nums"][valid].astype(float)
        res, _ = rpn.nl_deconfound_split(
            br[valid], bal[valid], rn, br[valid], bal[valid], rn
        )
        nnz = np.diff(X.tocsc().indptr)
        per[task] = {"X": X, "y": res, "active": nnz > 10}

    out = {}
    for tgt in RQ2_TASKS:
        best = None
        detail = {}
        for src in RQ2_TASKS:
            if src == tgt or src not in per or tgt not in per:
                continue
            shared = per[src]["active"] & per[tgt]["active"]
            Xs = per[src]["X"][:, shared].toarray()
            Xt = per[tgt]["X"][:, shared].toarray()
            ys, yt = per[src]["y"], per[tgt]["y"]
            corr = np.array([
                abs(spearmanr(Xs[:, j], ys)[0]) if Xs[:, j].std() > 0 else 0.0
                for j in range(Xs.shape[1])
            ])
            idx = np.argsort(corr)[-min(200, Xs.shape[1]):]
            sc = StandardScaler()
            pred = Ridge(alpha=100.0).fit(sc.fit_transform(Xs[:, idx]), ys).predict(
                sc.transform(Xt[:, idx])
            )
            r2 = float(r2_score(yt, pred))
            detail[f"{src}_to_{tgt}"] = r2
            best = r2 if best is None else max(best, r2)
        out[tgt] = {"best": best, "detail": detail}
    return out


def table_rq2_sharing() -> None:
    r1 = load_json(
        "sae_v3_analysis/results/robustness/"
        "rq2_aligned_hidden_transfer_gemma_centroid_pca_L22_r1_e8g1_L22_r1.json"
    )
    r2j = load_json(
        "sae_v3_analysis/results/robustness/"
        "rq2_aligned_hidden_transfer_gemma_centroid_pca_L22_r2_e8g1_L22_r2.json"
    )

    # ---- (i) cosine alignment of the per-task BK directions ---------------
    # Emitted from the released hidden-subspace audit, which is the file whose
    # weight cosines the printed row reproduces.
    audit = load_json("sae_v3_analysis/results/shared_subspace_hidden_audit_20260410.json")
    wc = audit["gemma_ic_sm_mw_hidden"]["shared_subspace"]["weight_cosines"]
    cos_pair = {k.replace("_", "-"): float(v) for k, v in wc.items()}
    cos_max = {
        t: max(abs(v) for k, v in cos_pair.items() if t in k.split("-"))
        for t in RQ2_TASKS
    }
    if abs(cos_pair.get("ic-mw", 0.0) - cos_pair.get("sm-mw", 1.0)) < 1e-12:
        flag(
            "tab:rq2-sharing (i) source file",
            "one cosine per task pair",
            f"ic_mw == sm_mw == {cos_pair['ic-mw']:+.4f} (identical to 4 dp)",
            "shared_subspace_hidden_audit_20260410.json",
            "two distinct task pairs carry a bit-identical cosine in the released "
            "audit file, which a genuine measurement would not produce",
        )

    # Two independent recomputes of the same object at L22, for the record.
    data = {t: _load_hidden_bk(t) for t in RQ2_TASKS}
    centroid_dirs = {t: _bk_direction(*data[t]) for t in RQ2_TASKS}
    recompute_centroid = {}
    for i, a in enumerate(RQ2_TASKS):
        for b in RQ2_TASKS[i + 1:]:
            recompute_centroid[f"{a}-{b}"] = float(centroid_dirs[a] @ centroid_dirs[b])
    recompute_readout = _readout_cosines()

    # (ii) sparse-feature transfer, off-diagonal
    transfer = _feature_transfer_r2()

    def dec(src, task, key):
        return src["summary"]["readout_decomposition"][task][key]

    rank1 = {t: dec(r1, t, "shared_only_auc_mean") for t in RQ2_TASKS}
    rank1_sd = {t: dec(r1, t, "shared_only_auc_std") for t in RQ2_TASKS}
    rank2 = {t: dec(r2j, t, "shared_only_auc_mean") for t in RQ2_TASKS}
    rank2_sd = {t: dec(r2j, t, "shared_only_auc_std") for t in RQ2_TASKS}
    resid = {t: dec(r1, t, "residual_only_auc_mean") for t in RQ2_TASKS}
    resid_sd = {t: dec(r1, t, "residual_only_auc_std") for t in RQ2_TASKS}
    comb = {t: dec(r2j, t, "full_auc_mean") for t in RQ2_TASKS}
    comb_sd = {t: dec(r2j, t, "full_auc_std") for t in RQ2_TASKS}

    def green(v: float, s: str) -> str:
        return r"\cellcolor{green!12}\textbf{" + s + "}" if v >= 0.7 else s

    body = [
        r"\multicolumn{4}{l}{\emph{(i) Cosine alignment} (BK directions)} \\",
        r"$|\cos|$ vs others & " + " & ".join(f"{cos_max[t]:.2f}" for t in RQ2_TASKS) + r" \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\emph{(ii) Feature transfer} ($R^2$, off-diag)} \\",
        "best from other & " + " & ".join(
            (r"$<\!0$" if (transfer[t]["best"] is not None and transfer[t]["best"] < 0)
             else f3(transfer[t]["best"])) for t in RQ2_TASKS
        ) + r" \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\emph{(iii) LOTO PCA shared-only} (AUC)} \\",
        "rank-1 & " + " & ".join(green(rank1[t], f"{rank1[t]:.2f}") for t in RQ2_TASKS) + r" \\",
        "rank-2 & " + " & ".join(green(rank2[t], f"{rank2[t]:.2f}") for t in RQ2_TASKS) + r" \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\emph{(iv) LOTO PCA other slices} (AUC)} \\",
        "Residual-only & " + " & ".join(f"{resid[t]:.2f}" for t in RQ2_TASKS) + r" \\",
        "Combined (rank-2) & " + " & ".join(f"{comb[t]:.2f}" for t in RQ2_TASKS) + r" \\",
    ]

    comments = [f"pairwise cos (released audit weight_cosines): {k}={v:+.4f}"
                for k, v in cos_pair.items()]
    comments += [
        "recompute A -- cosine of the raw BK centroid-difference directions at "
        "L22 (bankruptcy vs voluntary stop, balance-valid rounds): "
        + ", ".join(f"{k}={v:+.4f}" for k, v in recompute_centroid.items()),
        "recompute B -- cosine of the per-task PCA(64)+logistic readout "
        "directions at L22 (the object the LOTO rows (iii)/(iv) are built from): "
        + ", ".join(f"{k}={v:+.4f}" for k, v in recompute_readout.items()),
        "the printed row matches the released audit file, not either recompute; "
        "the three definitions of 'the BK direction' do not agree",
    ]
    for t in RQ2_TASKS:
        comments.append(
            f"{t}: |cos|max={cos_max[t]:.4f}; "
            f"feature transfer {transfer[t]['detail']}; "
            f"rank1 shared AUC={rank1[t]:.4f}+-{rank1_sd[t]:.4f}; "
            f"rank2 shared AUC={rank2[t]:.4f}+-{rank2_sd[t]:.4f}; "
            f"residual AUC={resid[t]:.4f}+-{resid_sd[t]:.4f}; "
            f"combined AUC={comb[t]:.4f}+-{comb_sd[t]:.4f} (SD over 5 CV splits)"
        )

    got = {
        "cos": {t: f"{cos_max[t]:.2f}" for t in RQ2_TASKS},
        "rank1": {t: f"{rank1[t]:.2f}" for t in RQ2_TASKS},
        "rank2": {t: f"{rank2[t]:.2f}" for t in RQ2_TASKS},
        "residual": {t: f"{resid[t]:.2f}" for t in RQ2_TASKS},
        "combined": {t: f"{comb[t]:.2f}" for t in RQ2_TASKS},
    }
    for row, vals in got.items():
        for t in RQ2_TASKS:
            if RQ2_PAPER[row][t] != vals[t]:
                flag(
                    f"tab:rq2-sharing ({row}) {TASK_TEX[t]}",
                    RQ2_PAPER[row][t], vals[t],
                    "hidden_states_dp.npz L22 / rq2_aligned_hidden_transfer_*_L22_r{1,2}",
                    "",
                )

    write_tex(
        "A06_rq2_sharing.tex",
        "tab:rq2-sharing -- cross-task representation-sharing audit, Gemma L22.\n"
        "(i)   cosine of the per-task BK directions, from HF\n"
        "      sae_v3_analysis/results/shared_subspace_hidden_audit_20260410.json\n"
        "      (weight_cosines), reported as max |cos| against the other two tasks.\n"
        "      Two independent recomputes from hidden_states_dp.npz at L22 are\n"
        "      listed below and do NOT reproduce it -- see the note.\n"
        "(ii)  sparse-feature transfer R^2 (top-200 |Spearman| on the source,\n"
        "      Ridge alpha=100, applied to the target) on the jointly-active\n"
        "      dictionary; best off-diagonal source per target.\n"
        "(iii) LOTO PCA shared-only AUC, 5-split centroid-PCA CV, from\n"
        "      HF sae_v3_analysis/results/robustness/rq2_aligned_hidden_transfer_gemma_centroid_pca_L22_r{1,2}_e8g1_L22_r{1,2}.json.\n"
        "(iv)  residual-only = rank-1 file; combined = rank-2 full readout AUC.\n"
        "Intervals (SD over the 5 CV splits):\n" + "\n".join(comments),
        body,
    )


# ==========================================================================
# T7  tab:appendix-condition-multi
# ==========================================================================

COND_MULTI_ROWS = [
    ("gemma", "mw", "i_lc"), ("gemma", "mw", "i_ba"),
    ("gemma", "ic", "i_lc"), ("gemma", "ic", "i_ba"),
    ("llama", "mw", "i_lc"), ("llama", "mw", "i_ba"),
    ("llama", "ic", "i_lc"), ("llama", "ic", "i_ba"), ("llama", "ic", "i_ec"),
]
COND_MULTI_PAPER = {
    ("gemma", "mw", "i_lc"): ("0.081", "0.138", "+71", "0.079", "0.099"),
    ("gemma", "mw", "i_ba"): ("0.054", "0.075", "+40", "0.055", "0.055"),
    ("gemma", "ic", "i_lc"): ("-0.031", "-0.035", "---", "-0.013", "0.006"),
    ("gemma", "ic", "i_ba"): ("0.203", "0.187", "-8", "0.170", "0.204"),
    ("llama", "mw", "i_lc"): ("0.142", "0.191", "+35", "0.146", "0.146"),
    ("llama", "mw", "i_ba"): ("0.036", "0.047", "+29", "0.071", "0.063"),
    ("llama", "ic", "i_lc"): ("0.133", "-0.077", "---", "-0.125", "-0.027"),
    ("llama", "ic", "i_ba"): ("0.278", "0.255", "-8", "0.206", "n/a"),
    ("llama", "ic", "i_ec"): ("0.352", "0.343", "-2", "0.236", "n/a"),
}
# The paper marks a cell n/a when the subset R^2 collapses; the released values
# there are large negative cross-validated R^2 (unstable Ridge on a tiny subset).
NA_THRESHOLD = -1.0


def _pct(plus, minus):
    """Percent change of + over -, blanked when either endpoint is non-positive.

    A ratio through zero is not interpretable, so the paper prints ``---`` for
    those cells; this reproduces that rule instead of printing a number.
    """
    if plus is None or minus is None or minus <= 0 or plus <= 0:
        return "---"
    v = (plus - minus) / minus * 100
    return f"$-{abs(v):.0f}$" if v < 0 else f"$+{v:.0f}$"


def _texnum(x, na=False):
    if x is None:
        return "---"
    if na:
        return r"n/a$^\ddagger$"
    return f"$-${abs(x):.3f}" if x < 0 else f"{x:.3f}"


def table_condition_multi() -> None:
    cm = load_json("sae_v3_analysis/results/condition_modulation_groupkfold_L22.json")
    body, comments = [], []
    for model, task, ind in COND_MULTI_ROWS:
        subs = cm[f"{model}_{task}_{ind}_L22"]["subsets"]

        def r(k):
            return subs.get(k, {}).get("r2_mean")

        mg, pg, mm, pm = r("minus_G"), r("plus_G"), r("minus_M"), r("plus_M")
        na = {k: (v is not None and v < NA_THRESHOLD) for k, v in
              (("minus_G", mg), ("plus_G", pg), ("minus_M", mm), ("plus_M", pm))}
        dg = _pct(pg, mg)
        cells = [
            _texnum(mg, na["minus_G"]), _texnum(pg, na["plus_G"]), dg,
            _texnum(mm, na["minus_M"]), _texnum(pm, na["plus_M"]),
        ]
        # green + bold marks rows whose baseline readout is strong (R^2 >= 0.2)
        strong = mg is not None and mg >= 0.2
        if strong:
            body.append(r"\rowcolor{green!12}")
            cells[0] = r"\textbf{" + cells[0] + "}"
            cells[1] = r"\textbf{" + cells[1] + "}"
        body.append(
            f"{MODEL_TEX[model]} {TASK_TEX[task]} & {IND_TEX[ind]} (L22) & "
            + " & ".join(cells) + r" \\"
        )
        comments.append(
            f"{model} {task} {ind}: " + ", ".join(
                f"{k}={subs[k].get('r2_mean')!r} (n={subs[k].get('n')}, "
                f"sd={subs[k].get('r2_std')!r})"
                for k in ("all_variable", "minus_G", "plus_G", "minus_M", "plus_M")
                if k in subs
            )
        )
        want = COND_MULTI_PAPER[(model, task, ind)]
        got = []
        for v, isna in ((mg, na["minus_G"]), (pg, na["plus_G"])):
            got.append("n/a" if isna else ("---" if v is None else f"{v:.3f}"))
        got.append(dg.replace("$", ""))
        for v, isna in ((mm, na["minus_M"]), (pm, na["plus_M"])):
            got.append("n/a" if isna else ("---" if v is None else f"{v:.3f}"))
        for label, w, g in zip(("-G", "+G", "dG%", "-M", "+M"), want, got):
            if w != g:
                flag(f"tab:appendix-condition-multi {model} {task} {ind} {label}",
                     w, g, "condition_modulation_groupkfold_L22.json", "")
    write_tex(
        "A07_condition_multi.tex",
        "tab:appendix-condition-multi -- MW/IC +-G/+-M readout R^2 at L22.\n"
        "Source: HF sae_v3_analysis/results/condition_modulation_groupkfold_L22.json\n"
        "Columns: Cell & Indicator & -G & +G & Delta_G% & -M & +M\n"
        f"n/a is emitted when a subset R^2 falls below {NA_THRESHOLD} (unstable Ridge\n"
        "on a small subset), reproducing the paper's ddagger footnote rule.\n"
        "Per-subset n and fold SD:\n" + "\n".join(comments),
        body,
    )


# ==========================================================================
# T8  tab:appendix-condition-full  -- DEFECT 1
# ==========================================================================

COND_FULL_PAPER = {
    "all": ("0.059", "0.167", "0.051", "0.106", "0.109", "0.036"),
    "minus_G": ("0.056", "0.063", "-0.018", "0.134", "0.082", "0.020"),
    "plus_G": ("0.050", "0.153", "0.043", "0.104", "0.113", "0.039"),
    "dG": ("-12", "+143", "---", "-22", "+38", "+88"),
    "minus_M": ("0.027", "0.127", "0.008", "0.114", "0.069", "0.010"),
    "plus_M": ("0.075", "0.154", "0.073", "0.096", "0.093", "0.031"),
    "dM": ("+173", "+21", "+792", "-16", "+35", "+228"),
    "fixed": ("-0.014", "-0.010", "-1.511", "0.007", "~0", "-0.013"),
}
COND_FULL_ORDER = [("gemma", i) for i in ("i_lc", "i_ba", "i_ec")] + \
                  [("llama", i) for i in ("i_lc", "i_ba", "i_ec")]


def table_condition_full() -> None:
    gk = load_json("sae_v3_analysis/results/condition_modulation_groupkfold_L22.json")
    ci = load_json("sae_v3_analysis/results/condition_modulation_continuous_ilc_L22.json")

    def gkv(model, ind, subset):
        return gk[f"{model}_sm_{ind}_L22"]["subsets"].get(subset, {}).get("r2_mean")

    def civ(model, ind, subset):
        rec = ci.get(f"{model}_sm_{ind}_L22")
        if rec is None:
            return None
        return rec["subsets"].get(subset, {}).get("r2")

    rows = []
    for label, subset in (("All variable", "all_variable"),
                          ("$-G$ (no goal)", "minus_G"),
                          ("$+G$ (goal-setting)", "plus_G")):
        rows.append((label, [gkv(m, i, subset) for m, i in COND_FULL_ORDER], None))
    rows.append((r"$\Delta_G\%$",
                 [_pct(gkv(m, i, "plus_G"), gkv(m, i, "minus_G")) for m, i in COND_FULL_ORDER],
                 "pct"))
    for label, subset in (("$-M$ (no reward-max)", "minus_M"),
                          ("$+M$ (reward-max)", "plus_M")):
        rows.append((label, [gkv(m, i, subset) for m, i in COND_FULL_ORDER], None))
    rows.append((r"$\Delta_M\%$",
                 [_pct(gkv(m, i, "plus_M"), gkv(m, i, "minus_M")) for m, i in COND_FULL_ORDER],
                 "pct"))

    minus_g = [gkv(m, i, "minus_G") for m, i in COND_FULL_ORDER]
    body: list[str] = []
    for label, vals, kind in rows:
        if kind == "pct":
            cells = list(vals)
        else:
            cells = [_texnum(v) for v in vals]
            if label.startswith("$+G$"):
                # bold where the goal frame raises the readout above -G
                cells = [
                    (r"\textbf{" + c + "}") if (v is not None and mg is not None and v > mg) else c
                    for c, v, mg in zip(cells, vals, minus_g)
                ]
        if label.startswith("$+G$"):
            body.append(r"\rowcolor{green!12}")
        body.append(f"{label} & " + " & ".join(cells) + r" \\")
    body.append(r"\midrule")
    # ---- DEFECT 1: emit BOTH pipelines for the fixed-bet row ---------------
    fixed_gk = [gkv(m, i, "fixed_all") for m, i in COND_FULL_ORDER]
    fixed_ci = [civ(m, i, "fixed_all") for m, i in COND_FULL_ORDER]
    body.append(r"\rowcolor{red!8}")
    body.append(
        r"Fixed$^\dagger$ (GroupKFold, as rows above) & "
        + " & ".join(_texnum(v) for v in fixed_gk) + r" \\"
    )
    body.append(r"\rowcolor{red!8}")
    body.append(
        r"Fixed$^\dagger$ (continuous-$I_\text{LC}$ pipeline, as printed) & "
        + " & ".join(_texnum(v) for v in fixed_ci) + r" \\"
    )

    comments = []
    for (m, i), a, b in zip(COND_FULL_ORDER, fixed_gk, fixed_ci):
        comments.append(
            f"fixed-bet {m} sm {i}: GroupKFold R2={a!r} "
            f"(n={gk[f'{m}_sm_{i}_L22']['subsets'].get('fixed_all', {}).get('n')}), "
            f"continuous-I_LC R2={b!r} "
            f"(n={(ci.get(f'{m}_sm_{i}_L22') or {}).get('subsets', {}).get('fixed_all', {}).get('n')})"
        )
    for m, i in COND_FULL_ORDER:
        subs = gk[f"{m}_sm_{i}_L22"]["subsets"]
        for k, v in subs.items():
            if v.get("r2_mean") is not None:
                comments.append(
                    f"{m} sm {i} {k}: R2={v['r2_mean']:+.4f} +- {v.get('r2_std', float('nan')):.4f} "
                    f"(5-fold SD), n={v.get('n')}, groups={v.get('n_groups')}"
                )

    # ---- compare with the typed table ------------------------------------
    def fmt_list(vals):
        return [("---" if v is None else f"{v:.3f}") for v in vals]

    checks = {
        "all": fmt_list([gkv(m, i, "all_variable") for m, i in COND_FULL_ORDER]),
        "minus_G": fmt_list([gkv(m, i, "minus_G") for m, i in COND_FULL_ORDER]),
        "plus_G": fmt_list([gkv(m, i, "plus_G") for m, i in COND_FULL_ORDER]),
        "dG": [_pct(gkv(m, i, "plus_G"), gkv(m, i, "minus_G")).replace("$", "")
               for m, i in COND_FULL_ORDER],
        "minus_M": fmt_list([gkv(m, i, "minus_M") for m, i in COND_FULL_ORDER]),
        "plus_M": fmt_list([gkv(m, i, "plus_M") for m, i in COND_FULL_ORDER]),
        "dM": [_pct(gkv(m, i, "plus_M"), gkv(m, i, "minus_M")).replace("$", "")
               for m, i in COND_FULL_ORDER],
    }
    for key, got in checks.items():
        for (m, i), w, g in zip(COND_FULL_ORDER, COND_FULL_PAPER[key], got):
            if w != g:
                flag(f"tab:appendix-condition-full {key} {m}/{i}", w, g,
                     "condition_modulation_groupkfold_L22.json", "")
    for (m, i), w, gkfv, civ_ in zip(COND_FULL_ORDER, COND_FULL_PAPER["fixed"], fixed_gk, fixed_ci):
        got_gk = "---" if gkfv is None else f"{gkfv:.3f}"
        got_ci = "---" if civ_ is None else f"{civ_:.3f}"
        flag(
            f"tab:appendix-condition-full Fixed row {m}/{i}",
            w, f"GroupKFold {got_gk} | continuous-I_LC {got_ci}",
            "condition_modulation_groupkfold_L22.json vs "
            "condition_modulation_continuous_ilc_L22.json",
            "D1: the printed fixed row comes from the continuous-I_LC pipeline while "
            "the six rows above it come from GroupKFold; no note says so",
        )

    write_tex(
        "A08_condition_full.tex",
        "tab:appendix-condition-full -- Gemma/LLaMA SM condition modulation at L22.\n"
        "DEFECT 1: the printed Fixed row is NOT from the same pipeline as the six\n"
        "rows above it.  Rows 'All variable' / +-G / +-M come from\n"
        "  condition_modulation_groupkfold_L22.json  (GroupKFold by game id),\n"
        "while the printed Fixed values match\n"
        "  condition_modulation_continuous_ilc_L22.json  (continuous-I_LC, plain CV).\n"
        "Both pipelines' fixed-bet rows are emitted below and labelled.\n"
        "Columns: Condition & Gemma I_LC/I_BA/I_EC & LLaMA I_LC/I_BA/I_EC\n"
        "Per-cell n and fold SD:\n" + "\n".join(comments),
        body,
    )


# ==========================================================================
# T9  tab:causal-transfer-matrix  -- DEFECT 3
# ==========================================================================

W7_AXES = ["smiba", "icrc", "mwrc", "sh3c"]
W7_TARGETS = ["sm", "ic", "mw"]
W7_AXIS_TEX = {"smiba": r"\texttt{sm\_iba}", "icrc": r"\texttt{ic\_rc}",
               "mwrc": r"\texttt{mw\_rc}", "sh3c": r"\texttt{sh3c}"}
W7_PARSE_GATE = {"sm": 0.8, "ic": 0.45, "mw": 0.45}
# Pre-registered sign + low-confidence flag (INDEX.md W7 sign table, |cos|>=0.15
# is "confident"; the two |cos|=0.10 cells are low-confidence).
W7_PRED = {
    ("smiba", "sm"): ("+", False), ("smiba", "ic"): ("-", False), ("smiba", "mw"): ("+", True),
    ("icrc", "sm"): ("-", False), ("icrc", "ic"): ("+", False), ("icrc", "mw"): ("+", False),
    ("mwrc", "sm"): ("+", True), ("mwrc", "ic"): ("+", False), ("mwrc", "mw"): ("+", False),
    ("sh3c", "sm"): ("-", False), ("sh3c", "ic"): ("+", False), ("sh3c", "mw"): ("+", False),
}
# What the paper prints (tab:causal-transfer-matrix) ...
W7_PAPER_Z = {
    ("smiba", "sm"): 6.0, ("smiba", "ic"): -2.2, ("smiba", "mw"): 3.4,
    ("icrc", "sm"): -2.9, ("icrc", "ic"): 3.3, ("icrc", "mw"): 0.26,
    ("mwrc", "sm"): 4.2, ("mwrc", "ic"): 3.0, ("mwrc", "mw"): 0.32,
    ("sh3c", "sm"): -3.7, ("sh3c", "ic"): 3.5, ("sh3c", "mw"): -0.34,
}
# ... and what the project's own INDEX.md W7 adjudication line records.
W7_INDEX_Z = {
    ("smiba", "sm"): 6.0, ("smiba", "ic"): -2.2, ("smiba", "mw"): 3.4,
    ("icrc", "sm"): -3.0, ("icrc", "ic"): 3.2, ("icrc", "mw"): 0.3,
    ("mwrc", "sm"): 4.2, ("mwrc", "ic"): 3.0, ("mwrc", "mw"): 0.3,
    ("sh3c", "sm"): -3.7, ("sh3c", "ic"): 3.5, ("sh3c", "mw"): -0.3,
}


def _w7_rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in open(path)]


def _w7_outcome(rec: dict, target: str) -> float:
    if target == "sm":
        return float(rec.get("bet_ratio") or 0.0)
    if target == "ic":
        return 1.0 if rec.get("risky") else 0.0
    return 1.0 if rec.get("action") == "spin" else 0.0


def _w7_ladder(pattern: str, target: str, gate: bool):
    xs, ys, meta = [], [], []
    for path in sorted(glob.glob(pattern)):
        rs = _w7_rows(Path(path))
        if not rs:
            continue
        ok = [r for r in rs if r.get("parse_ok")]
        rate = len(ok) / len(rs)
        alpha = float(rs[0]["alpha"])
        keep = (not gate) or rate >= W7_PARSE_GATE[target]
        meta.append({"alpha": alpha, "parse_rate": rate, "n_ok": len(ok), "kept": keep,
                     "mean": float(np.mean([_w7_outcome(r, target) for r in ok])) if ok else None})
        if keep:
            xs += [alpha] * len(ok)
            ys += [_w7_outcome(r, target) for r in ok]
    if len(set(xs)) < 2:
        return None, meta
    A = np.column_stack([np.ones(len(xs)), np.asarray(xs, dtype=float)])
    slope = float(np.linalg.lstsq(A, np.asarray(ys, dtype=float), rcond=None)[0][1])
    return slope, meta


def table_steering_matrix() -> None:
    nulls = {}
    for target in W7_TARGETS:
        vals = []
        for i in (1, 2, 3):
            s, _ = _w7_ladder(str(W7_DIR / f"sec4_w7_null_{target}{i}_a*.jsonl"), target, gate=False)
            if s is not None:
                vals.append(s)
        arr = np.asarray(vals, dtype=float)
        # population SD over the three random directions -- this is the
        # convention that reproduces the published z to two decimals.
        nulls[target] = {"slopes": vals, "mean": float(arr.mean()),
                         "sd_pop": float(arr.std(ddof=0)),
                         "sd_sample": float(arr.std(ddof=1))}
        print(f"  [w7 null] {target}: slopes={np.round(vals, 5).tolist()} "
              f"mean={arr.mean():+.5f} sd_pop={arr.std(ddof=0):.5f} "
              f"sd_samp={arr.std(ddof=1):.5f}")

    z_pop, z_samp, slopes, parse = {}, {}, {}, {}
    for axis in W7_AXES:
        for target in W7_TARGETS:
            s, meta = _w7_ladder(
                str(W7_DIR / f"sec4_w7_{axis}_{target}_a*.jsonl"), target, gate=True
            )
            slopes[(axis, target)] = s
            parse[(axis, target)] = meta
            nb = nulls[target]
            z_pop[(axis, target)] = (s - nb["mean"]) / nb["sd_pop"]
            z_samp[(axis, target)] = (s - nb["mean"]) / nb["sd_sample"]

    def zfmt(z):
        return f"{z:+.1f}" if abs(z) >= 1 else f"{z:+.2f}"

    def verdict(axis, target, z):
        """Pre-registered adjudication: |z|>=2 with the predicted sign is a hit;
        |z|<2 with the predicted sign is null (inside the band); a wrong sign is
        a sign miss at any magnitude.  Low-confidence cells carry a dagger."""
        pred, low = W7_PRED[(axis, target)]
        sign_ok = sign_matches(pred, z)
        if not sign_ok:
            return "sign miss$^\\dagger$" if low else "sign miss"
        if abs(z) < 2:
            return "null"
        return "sign hit$^\\dagger$" if low else "hit"

    body: list[str] = []
    for axis in W7_AXES:
        cells = []
        for target in W7_TARGETS:
            z = z_pop[(axis, target)]
            pred, low = W7_PRED[(axis, target)]
            v = verdict(axis, target, z)
            txt = f"${pred}$ / ${zfmt(z)}$ / {v}"
            if target == "mw":
                txt = r"\cellcolor{gray!15}" + txt
            cells.append(txt)
        body.append(W7_AXIS_TEX[axis] + " & " + " & ".join(cells) + r" \\")

    # tallies, computed rather than asserted
    confident = [(a, t) for a in W7_AXES for t in W7_TARGETS if not W7_PRED[(a, t)][1]]
    primary_hits = sum(
        1 for a, t in confident
        if abs(z_pop[(a, t)]) >= 2 and sign_matches(W7_PRED[(a, t)][0], z_pop[(a, t)])
    )
    sign_hits = sum(
        1 for a in W7_AXES for t in W7_TARGETS
        if sign_matches(W7_PRED[(a, t)][0], z_pop[(a, t)])
    )

    comments = [
        f"primary tally (10 confident cells, |z|>=2 AND predicted sign): "
        f"{primary_hits}/{len(confident)}  [paper text: 7/10]",
        f"sign agreement over all 12 cells: {sign_hits}/12  [paper text: 11/12]",
    ]
    for t in W7_TARGETS:
        nb = nulls[t]
        comments.append(
            f"null band {t}: 3 random directions, slopes "
            f"{np.round(nb['slopes'], 6).tolist()}, mean={nb['mean']:+.6f}, "
            f"population SD={nb['sd_pop']:.6f} (used), sample SD={nb['sd_sample']:.6f}"
        )
    for axis in W7_AXES:
        for target in W7_TARGETS:
            kept = [m for m in parse[(axis, target)] if m["kept"]]
            dropped = [m for m in parse[(axis, target)] if not m["kept"]]
            comments.append(
                f"{axis}->{target}: slope={slopes[(axis, target)]:+.6f}; "
                f"z(pop SD)={z_pop[(axis, target)]:+.3f}; "
                f"z(sample SD)={z_samp[(axis, target)]:+.3f}; "
                f"doses kept={[m['alpha'] for m in kept]}; "
                f"dropped by parse gate={[(m['alpha'], round(m['parse_rate'], 2)) for m in dropped]}"
            )
            got = zfmt(z_pop[(axis, target)])
            paper = W7_PAPER_Z[(axis, target)]
            paper_s = f"{paper:+.1f}" if abs(paper) >= 1 else f"{paper:+.2f}"
            index = W7_INDEX_Z[(axis, target)]
            index_s = f"{index:+.1f}" if abs(index) >= 1 else f"{index:+.2f}"
            if got != paper_s or paper_s != index_s:
                flag(
                    f"tab:causal-transfer-matrix {axis}->{target} z",
                    f"paper {paper_s} / INDEX.md {index_s}", got,
                    "experiments/sec4_causal/checkpoints/sec4_w7/*.jsonl",
                    "D3: recomputed = trial-level OLS slope vs 3-direction null band "
                    "(population SD)",
                )

    write_tex(
        "A09_steering_matrix.tex",
        "tab:causal-transfer-matrix -- pre-registered 12-cell cross-task steering matrix.\n"
        "DEFECT 3: the published cells existed only as literals in\n"
        "scripts/gen_fig_cross_context_write.py.  Recomputed here from\n"
        "experiments/sec4_causal/checkpoints/sec4_w7/*.jsonl.\n"
        "Statistic: trial-level OLS slope of the target outcome (SM bet ratio,\n"
        "IC risky rate, MW spin rate) on dose alpha, over parse-gated dose cells\n"
        "(gates SM 0.8, IC 0.45, MW 0.45), z-scored against the three random-direction\n"
        "null slopes for that target.\n"
        "Columns: Axis & SM & IC & MW  (each cell: pred sign / observed z / verdict)\n"
        + "\n".join(comments),
        body,
    )


def sign_matches(pred: str, z: float) -> bool:
    return (z > 0) if pred == "+" else (z < 0)


# ==========================================================================
# T10  tab:causal-condition-writability
# ==========================================================================

WRITE_PAPER = {
    "slope_minusG": ("0.0358", "---"),
    "slope_plusG": ("0.0469", "---"),
    "slope_plusM": ("0.0218", "---"),
    "plusG_vs_minusG": ("+0.0111 [+0.0069,+0.0156]", "---"),
    "plusM_vs_minusG": ("-0.0140 [-0.0218,-0.0066]", "---"),
    "twin": ("+0.0237 [+0.0179,+0.0297]", "-0.0156 [-0.0261,-0.0046]"),
    "collider": ("+0.0242 [+0.0184,+0.0295]", "-0.0124 [-0.0211,-0.0036]"),
}


def _ci(rec) -> str:
    d = rec["diff"]
    lo, hi = rec["ci95"]
    return f"${d:+.4f}$~~$[{lo:+.4f},{hi:+.4f}]$"


def _ci_plain(rec) -> str:
    d = rec["diff"]
    lo, hi = rec["ci95"]
    return f"{d:+.4f} [{lo:+.4f},{hi:+.4f}]"


def table_condition_writability() -> None:
    sys.path.insert(0, str(MLC_REPO))
    from src import sec4_stats

    res = sec4_stats.analyze_w14(results_dir=str(W14_DIR))
    g, l = res["gemma"], res["llama"]

    body = [
        f"slope, matched $-G$ pool & ${g['slopes_common_grid']['minusG']:.4f}$ & ---$^\\dagger$ \\\\",
        f"slope, $+G^{{\\text{{twin}}}}$ & ${g['slopes_common_grid']['plusG']:.4f}$ & ---$^\\dagger$ \\\\",
        f"slope, $+M^{{\\text{{twin}}}}$ & ${g['slopes_common_grid']['plusM']:.4f}$ & ---$^\\dagger$ \\\\",
        r"\midrule",
        r"$+G^{\text{twin}} - (-G)$ & " + _ci(g["vs_minusG_common_grid"]["plusG"]) + r" & --- \\",
        r"$+M^{\text{twin}} - (-G)$ & " + _ci(g["vs_minusG_common_grid"]["plusM"]) + r" & --- \\",
        r"$+G^{\text{twin}} - (+M^{\text{twin}})$ & "
        + _ci(g["primary_plusG_minus_plusM"]) + " & "
        + _ci(l["primary_plusG_minus_plusM"]) + r" \\",
        r"collider-corrected $+G^{\text{twin}} - (+M^{\text{twin}})$ & "
        + _ci(g["robust_impute_stop_plusG_minus_plusM"]) + " & "
        + _ci(l["robust_impute_stop_plusG_minus_plusM"]) + r" \\",
    ]

    comments = []
    for name, m in (("gemma", g), ("llama", l)):
        comments.append(f"{name}: verdict={m['verdict']}, n_boot={m['n_boot']}, "
                        f"shared doses={m['shared_doses']}, common grid={m['common_doses']}")
        comments.append(f"{name}: slopes on common grid={m['slopes_common_grid']}")
        for k in ("primary_plusG_minus_plusM", "robust_impute_stop_plusG_minus_plusM",
                  "robust_grid_restricted_plusG_minus_plusM",
                  "robust_drop_top_dose_plusG_minus_plusM",
                  "robust_extreme_removed_plusG_minus_plusM"):
            comments.append(f"{name} {k}: {_ci_plain(m[k])} "
                            f"excludes_zero={m[k]['excludes_zero']}")
        comments.append(f"{name} parse by dose: {m['parse_by_dose']}")

    got = {
        "slope_minusG": (f"{g['slopes_common_grid']['minusG']:.4f}", "---"),
        "slope_plusG": (f"{g['slopes_common_grid']['plusG']:.4f}", "---"),
        "slope_plusM": (f"{g['slopes_common_grid']['plusM']:.4f}", "---"),
        "plusG_vs_minusG": (_ci_plain(g["vs_minusG_common_grid"]["plusG"]).replace(" ", " "), "---"),
        "plusM_vs_minusG": (_ci_plain(g["vs_minusG_common_grid"]["plusM"]), "---"),
        "twin": (_ci_plain(g["primary_plusG_minus_plusM"]),
                 _ci_plain(l["primary_plusG_minus_plusM"])),
        "collider": (_ci_plain(g["robust_impute_stop_plusG_minus_plusM"]),
                     _ci_plain(l["robust_impute_stop_plusG_minus_plusM"])),
    }
    for key, (wg, wl) in WRITE_PAPER.items():
        gg, gl = got[key]
        for col, want, have in (("Gemma", wg, gg), ("LLaMA", wl, gl)):
            want_n = want.replace(" ", "")
            have_n = have.replace(" ", "")
            if want_n != have_n:
                flag(f"tab:causal-condition-writability {key} ({col})", want, have,
                     "experiments/sec4_causal/checkpoints/sec4_w14/*.jsonl via sec4_stats.analyze_w14", "")

    write_tex(
        "A10_condition_writability.tex",
        "tab:causal-condition-writability -- condition writability of the behavioural axis.\n"
        "Recomputed from experiments/sec4_causal/checkpoints/sec4_w14/*.jsonl by the canonical\n"
        "analyser sec4_stats.analyze_w14 (seeded 1000x bootstrap, so the CIs are\n"
        "reproducible).  Per-condition slopes and the vs-control contrasts use the\n"
        "common dose grid {-3,0,+3}; the twin head-to-head uses all shared doses.\n"
        "Columns: & Gemma & LLaMA\n" + "\n".join(comments),
        body,
    )


# ==========================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(parents=True, exist_ok=True)

    print("=== T1  tab:appendix-sae-full ===")
    table_sae_full()

    print("\n=== T2/T3  wired-in generated tables ===")
    gk_dir = stage_groupkfold_results()
    table_layer_sweep(gk_dir)
    table_band_readout(gk_dir)

    print("\n=== T3b  DEFECT 2: band readout at protocol layers ===")
    table_band_protocol_layers()

    print("\n=== T4  tab:neurips-selectivity-l2 ===")
    table_selectivity_l2()

    print("\n=== T5  tab:appendix-selectivity-controls ===")
    table_selectivity_controls()

    print("\n=== T6  tab:rq2-sharing ===")
    table_rq2_sharing()

    print("\n=== T7  tab:appendix-condition-multi ===")
    table_condition_multi()

    print("\n=== T8  DEFECT 1: tab:appendix-condition-full ===")
    table_condition_full()

    print("\n=== T9  DEFECT 3: tab:causal-transfer-matrix ===")
    table_steering_matrix()

    print("\n=== T10  tab:causal-condition-writability ===")
    table_condition_writability()

    print("\n" + "=" * 74)
    print(f"DISCREPANCIES vs the typed appendix: {len(DISCREPANCIES)}")
    for d in DISCREPANCIES:
        print(f"  - {d['element']}\n      paper: {d['paper_value']}\n"
              f"      code : {d['computed_value']}\n      src  : {d['source']}"
              + (f"\n      note : {d['note']}" if d["note"] else ""))
    print("=" * 74)


if __name__ == "__main__":
    main()
