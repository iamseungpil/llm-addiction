"""C4 read/write spine — integration + overlay plot (CPU, thin analysis).

Reads the §4 all-layer READ profiles (produced by the C1 runner
run_spine_layer.py) and the existing causal WRITE profiles (W1-W3 gap
recovery, keyed by layer-window arm name) and emits, per (model, section),
a single read(layer) vs write(layer) overlay on a shared layer axis.

  read  = the section's own readout metric per layer
            §4.1 r2_mean   §4.2 auc_shared   §4.3 delta_r2
  write = recovery vs the -G gap per layer-window (W1-W3 `gap.recovery`),
          plotted at the window-center layer.

Both curves are min-max normalized to [0,1] independently so "where it is
read" and "where it is written" align on one panel. Read profiles are read
from results/spine/read_profile_{model}_{task}_{section}.json; an absent file
is SKIPPED (noted), never fabricated. Write recovery is read from the W1-W3
stats JSON already on disk (multilayer_causal/out/{w1,w3}_stats.json).

Usage:
  python -m multilayer_causal.src.spine_stats \
      --read-dir results/spine --w1 multilayer_causal/out/w1_stats.json \
      --w3 multilayer_causal/out/w3_stats.json --out results/spine
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# Section -> (read profile metric key, write arm-window source).
# Write windows are named by their layer span (e.g. w1e_1617 = layers 16-17,
# w3m_1621 = 16-21); we plot recovery at the window-center layer.
READ_KEY = {"section41": "r2_mean", "section42": "auc_shared",
            "section43": "delta_r2"}
TASKS = ("sm", "ic", "mw")

# ---- C1 runner <-> spine contract bridge -----------------------------------
# run_spine_layer.py writes ONE file per layer, named with the long task form
# and its own section names, with the metric nested under result/{indicator}:
#   read_profile_{model}_{slot_machine}_{indicators}_L{layer}.json
#   { "result": { "i_ba": {"r2_mean": ...}, ... } }              (§4.1)
#   { "result": { "i_ba": {"delta_r2_G":.., "delta_r2_M":..}} }  (§4.3)
#   { "result": { "loto_pca": {"auc_shared": ...}, ...} }        (§4.2)
# spine_stats expects ONE collapsed file per (model, short-task, sectionNN) with
# top-level layers-array columns. aggregate_read_profiles() does that collapse.
SECTION_LONG2SHORT = {"indicators": "section41", "crosstask": "section42",
                      "condition": "section43"}
TASK_SHORT = {"slot_machine": "sm", "investment_choice": "ic",
              "mystery_wheel": "mw"}
# Which indicator owns the §4.1/§4.3 read curve. Body anchor for §4.1 is I_BA
# (LLaMA-IC I_BA L22 ~= 0.109; Gemma-SM I_BA L22 ~= 0.167); §4.3 sharpening is
# reported on the same indicator so read and ΔR2 curves are on one indicator.
SPINE_INDICATOR = "i_ba"


def _per_layer_value(section_long, result, indicator):
    """Pull the single read scalar for one layer's runner `result` dict.
    Returns None when the cell was skipped (e.g. {'reason': ...})."""
    if not isinstance(result, dict):
        return None
    if section_long == "indicators":          # §4.1 r2_mean
        cell = result.get(indicator)
        return cell.get("r2_mean") if isinstance(cell, dict) else None
    if section_long == "condition":           # §4.3 delta_r2 (use +M sharpening)
        cell = result.get(indicator)
        if not isinstance(cell, dict):
            return None
        return cell.get("delta_r2_M")
    if section_long == "crosstask":           # §4.2 auc_shared (held-out task)
        loto = result.get("loto_pca")
        return loto.get("auc_shared") if isinstance(loto, dict) else None
    return None


def aggregate_read_profiles(read_dir, models=("gemma", "llama"),
                            indicator=SPINE_INDICATOR):
    """Collapse run_spine_layer.py per-layer JSONs into the layers-array
    profiles load_read_profile() consumes. For each (model, long-task, long-
    section) it globs read_profile_{model}_{task}_{section}_L*.json, sorts by
    layer, and writes read_profile_{model}_{short}_{sectionNN}.json with
    top-level {layers:[...], <metric_key>:[...]} so the contract aligns on
    task-form (sm/ic/mw) and section name (section41/42/43)."""
    read_dir = Path(read_dir)
    written = []
    for model in models:
        for task_long, task_short in TASK_SHORT.items():
            for sec_long, sec_short in SECTION_LONG2SHORT.items():
                pat = f"read_profile_{model}_{task_long}_{sec_long}_L*.json"
                files = sorted(read_dir.glob(pat))
                rows = []
                for fp in files:
                    d = json.loads(fp.read_text())
                    lay = d.get("layer")
                    val = _per_layer_value(sec_long, d.get("result", {}),
                                           indicator)
                    if lay is not None:
                        rows.append((int(lay), val))
                if not rows:
                    continue
                rows.sort()
                key = READ_KEY[sec_short]
                out = {"model": model, "task": task_short, "section": sec_short,
                       "indicator": indicator,
                       "layers": [r[0] for r in rows],
                       key: [r[1] for r in rows]}
                op = read_dir / (f"read_profile_{model}_{task_short}"
                                 f"_{sec_short}.json")
                op.write_text(json.dumps(out, indent=1))
                written.append(str(op))
    return written


def load_read_profile(read_dir, model, task, section):
    """Read one C1 read profile; return (layers, values) or None if absent."""
    p = Path(read_dir) / f"read_profile_{model}_{task}_{section}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    key = READ_KEY[section]
    if "layers" not in d or key not in d:
        return None
    layers = np.asarray(d["layers"], float)
    vals = np.asarray([np.nan if v is None else v for v in d[key]], float)
    return layers, vals


def window_center(arm):
    """Layer-window arm name -> center layer. Two naming forms:
      - 4-digit span:  'w1e_1617'->16.5, 'w3m_1621'->18.5  (lo,hi inclusive)
      - single window: 'spinew_bk_w00'->2.5, 'spinew_m_w06'->8.5  (the width-6
        e1 tiling: window starts at NN, spans 6 layers [NN, NN+5], center NN+2.5)
    Returns None if neither suffix form is present."""
    m = re.search(r"_(\d{2})(\d{2})$", arm)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (lo + hi) / 2.0
    m = re.search(r"_w(\d{2})$", arm)        # spinew_* width-6 single window
    if m:
        lo = int(m.group(1))
        return lo + 2.5
    return None


def write_profile(arm_dict, name_filter=None):
    """Pull (layer, recovery) points from a W1-W3 arm dict — the arms whose
    gap.recovery is the section's write metric, plotted at window-center
    layer. Arms without a layer-window suffix (e.g. w3bk_*) are ignored. The
    'spinew' block holds BOTH §4.2 (spinew_bk_*) and §4.3 (spinew_m_*) arms, so
    name_filter (a substring, e.g. 'spinew_bk' / 'spinew_m') restricts which
    arms a given section consumes."""
    pts = []
    for arm, e in arm_dict.items():
        if name_filter is not None and name_filter not in arm:
            continue
        c = window_center(arm)
        if c is None or not isinstance(e, dict):
            continue
        gap = e.get("gap")
        if isinstance(gap, dict) and gap.get("recovery") is not None:
            pts.append((c, float(gap["recovery"])))
    pts.sort()
    if not pts:
        return None
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


# Per section: (stats tag, key holding the layer-window arms).
# §4.1 write = W1 e1/w1e tiling (w1be block). §4.2/§4.3 write = the spinew
# width-6 blank-fill (spinew_bk_* / spinew_m_* under the w3 'spinew' block,
# emitted by w3_stats.py): these complete the 42-layer write cover and are
# plotted at each window center via window_center's _wNN parse. The pre-existing
# 7 w1e_* (§4.1) and 3 w3m_* (§4.3, under 'arms') remain valid in-grid windows;
# §4.3 reads BOTH 'arms' (w3m_*) and 'spinew' (spinew_m_*). An absent spinew
# block (arms not yet run) is SKIPPED+noted, never fabricated.
# Each source = (stats tag, block key, arm-name filter or None).
WRITE_SOURCE = {"section41": [("w1", "w1be", None)],
                "section42": [("w3", "spinew", "spinew_bk")],
                "section43": [("w3", "arms", "w3m"),
                              ("w3", "spinew", "spinew_m")]}


def normalize(v):
    """Min-max to [0,1] over finite entries; flat/empty -> all NaN."""
    v = np.asarray(v, float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return v
    lo, hi = finite.min(), finite.max()
    if hi - lo < 1e-12:
        return np.where(np.isfinite(v), 0.5, np.nan)
    return (v - lo) / (hi - lo)


def build_spine(read_dir, stats, model, section):
    """One (model, section) spine: read curves per task + one write curve,
    each normalized. Returns a dict + a `notes` list of skipped curves."""
    notes = []
    reads = {}
    for task in TASKS:
        prof = load_read_profile(read_dir, model, task, section)
        if prof is None:
            notes.append(f"read_profile_{model}_{task}_{section} absent")
            continue
        layers, vals = prof
        reads[task] = {"layers": layers.tolist(),
                       "read_raw": vals.tolist(),
                       "read_norm": normalize(vals).tolist()}

    write = None
    srcs = WRITE_SOURCE.get(section)
    if not srcs:
        notes.append(f"no write source defined for {section}")
    else:
        # A section can draw from several blocks (e.g. §4.3 = w3m_* + spinew_m_*);
        # merge their (layer, recovery) points onto one curve.
        pts = []
        for tag, key, nf in srcs:
            if stats.get(tag) is None:
                notes.append(f"write stats '{tag}' absent for {section}")
                continue
            wp = write_profile(stats[tag].get(key, {}), name_filter=nf)
            if wp is None:
                notes.append(f"no layer-window recovery arms in {tag}/{key}"
                             + (f" [{nf}]" if nf else ""))
                continue
            wl, wr = wp
            pts.extend(zip(wl.tolist(), wr.tolist()))
        if pts:
            pts.sort()
            wl = np.array([p[0] for p in pts], float)
            wr = np.array([p[1] for p in pts], float)
            write = {"layers": wl.tolist(), "write_raw": wr.tolist(),
                     "write_norm": normalize(wr).tolist()}

    return {"model": model, "section": section,
            "read_key": READ_KEY[section], "reads": reads,
            "write": write, "notes": notes}


def plot_spine(spine, pdf_path):
    """read(layer) (per task) vs write(layer) overlay on normalized y."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for task, r in spine["reads"].items():
        ax.plot(r["layers"], r["read_norm"], "-o", ms=3,
                label=f"read {task} ({spine['read_key']})")
    if spine["write"]:
        w = spine["write"]
        ax.plot(w["layers"], w["write_norm"], "-s", color="k", lw=2,
                label="write (gap recovery)")
    ax.set_xlabel("layer")
    ax.set_ylabel("normalized read / write")
    ax.set_title(f"{spine['model']} {spine['section']} read vs write spine")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--read-dir",
                    default=str(root / "results" / "spine"))
    ap.add_argument("--w1", default=str(Path(__file__).resolve().parents[1]
                                        / "out" / "w1_stats.json"))
    ap.add_argument("--w3", default=str(Path(__file__).resolve().parents[1]
                                        / "out" / "w3_stats.json"))
    ap.add_argument("--out", default=str(root / "results" / "spine"))
    ap.add_argument("--models", nargs="+", default=["gemma", "llama"])
    ap.add_argument("--no-aggregate", action="store_true",
                    help="Skip collapsing run_spine_layer per-layer JSONs "
                         "(assume read_profile_{model}_{task}_{sectionNN}.json "
                         "already exist).")
    args = ap.parse_args()

    if not args.no_aggregate:
        # Collapse the C1 runner's per-layer _L{n} JSONs (long task/section
        # names, nested result) into the layers-array profiles build_spine
        # reads. Without this the contract never lines up and every cell is None.
        agg = aggregate_read_profiles(args.read_dir, models=args.models)
        for a in agg:
            print(f"aggregated: {a}")

    stats = {}
    for tag, path in (("w1", args.w1), ("w3", args.w3)):
        p = Path(path)
        stats[tag] = json.loads(p.read_text()) if p.exists() else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for model in args.models:
        for section in READ_KEY:
            spine = build_spine(args.read_dir, stats, model, section)
            base = out / f"spine_{model}_{section}"
            base.with_suffix(".json").write_text(json.dumps(spine, indent=1))
            if spine["reads"] or spine["write"]:
                plot_spine(spine, str(base.with_suffix(".pdf")))
            summary[f"{model}_{section}"] = {
                "read_tasks": list(spine["reads"]),
                "has_write": spine["write"] is not None,
                "notes": spine["notes"]}
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
