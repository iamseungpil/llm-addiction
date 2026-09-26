"""Final read-vs-write spine overlay (per model, all §4 sections).

Per §4 section, one panel: the read(layer) curve (left axis) overlaid with the
causal write-recovery(window-center) curve (right axis), with the L16-21 write
window shaded. Shows "where it is read (broad) vs where writing it moves
behavior (local)".

  read  §4.1 = I_BA readout R2(layer) per task
        §4.2 = cross-task mean transfer AUC(layer) per model
        §4.3 = +M sharpening ΔR2(layer) per task
  write recovery vs -G anchor (gap = +G - -G), per width-6 window center:
        §4.1 = W1 e1 tiling (from w1_stats.json w1be)
        §4.2 = spinew_bk_* (BK control axis)
        §4.3 = spinew_m_* + w3m_* (+M twin)

Write recovery for §4.2/§4.3 is computed from the arm jsonls on HF (cached to
results/spine/write_recovery.json); §4.1 is read from the W1 stats block.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/v-seungplee/llm-addiction")
SPINE = REPO / "multilayer_causal/results/spine"
W1_STATS = REPO / "multilayer_causal/out/w1_stats.json"
WRITE_CACHE = SPINE / "write_recovery.json"
HF = "llm-addiction-research/llm-addiction"
WRITE_LO, WRITE_HI = 16, 21

# §4.2 / §4.3 write arms (vs the W2 ±G anchors they share a population with).
BK_ARMS = [f"spinew/spinew_bk_{w}" for w in
           ("w00", "w06", "w12", "w18", "w24", "w30", "w36")]
M_ARMS = ([f"spinew/spinew_m_{w}" for w in ("w00", "w06", "w12", "w18", "w30", "w36")]
          + [f"w3m/w3m_{w}" for w in ("0813", "1621", "2429")])


def _arm_rows(arm):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(HF, f"experiments/multilayer_causal/checkpoints/{arm}.jsonl",
                        repo_type="dataset")
    return [json.loads(l) for l in open(p) if l.strip()]


def _mean_br(arm):
    r = [t for t in _arm_rows(arm) if t.get("parse_ok")]
    br = [float(t["bet_ratio"]) for t in r]
    ctr = [(t["layers"][0] + t["layers"][-1]) / 2 for t in r if isinstance(t.get("layers"), list)]
    return (float(np.mean(br)) if br else None,
            float(np.mean(ctr)) if ctr else None)


def compute_write_recovery():
    """Recovery = (mean_br(arm) - -G) / (+G - -G) per arm, x = window center."""
    if WRITE_CACHE.exists():
        return json.loads(WRITE_CACHE.read_text())
    mminus = _mean_br("w2/w2_anchor_minus")[0]
    mplus = _mean_br("w2/w2_anchor_plus")[0]
    gap = mplus - mminus
    out = {"section42": [], "section43": []}
    for sec, arms in (("section42", BK_ARMS), ("section43", M_ARMS)):
        for arm in arms:
            m, ctr = _mean_br(arm)
            if m is not None and ctr is not None:
                out[sec].append([ctr, (m - mminus) / gap])
        out[sec].sort()
    SPINE.mkdir(parents=True, exist_ok=True)
    WRITE_CACHE.write_text(json.dumps(out, indent=1))
    return out


def section41_write():
    """§4.1 write recovery from the W1 stats (w1be block). Layers are encoded in
    the arm name (w1e_1617 = layers 16-17 -> center 16.5); the 5-layer w1b_*
    arms are not contiguous windows and are skipped."""
    import re
    if not W1_STATS.exists():
        return []
    blk = json.loads(W1_STATS.read_text()).get("w1be", {})
    pts = []
    for arm, e in blk.items() if isinstance(blk, dict) else []:
        m = re.search(r"w1e_(\d{2})(\d{2})$", arm)
        g = e.get("gap") if isinstance(e, dict) else None
        if m and isinstance(g, dict) and g.get("recovery") is not None:
            lo, hi = int(m.group(1)), int(m.group(2))
            pts.append([(lo + hi) / 2, g["recovery"]])
    pts.sort()
    return pts


def read_curve(model, task, sec):
    p = SPINE / f"read_profile_{model}_{task}_{sec}.json"
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    key = "r2_mean" if sec == "section41" else "delta_r2"
    pts = [(x, y) for x, y in zip(d.get("layers", []), d.get(key, [])) if y is not None]
    return (np.array([p[0] for p in pts]), np.array([p[1] for p in pts])) if pts else (None, None)


def crosstask_curve(model):
    rows = []
    for fp in sorted(SPINE.glob(f"read_profile_{model}_crosstask42_L*.json")):
        d = json.loads(fp.read_text()).get("result", {})
        ta = d.get("transfer_auc")
        if isinstance(ta, dict) and ta:
            rows.append((int(d["layer"]), float(np.mean(list(ta.values())))))
    rows.sort()
    return (np.array([r[0] for r in rows]), np.array([r[1] for r in rows])) if rows else (None, None)


def panel(ax, model, sec, title, ylabel, write_pts):
    ax.axvspan(WRITE_LO, WRITE_HI, color="0.88", zorder=0, label="write window L16-21")
    if sec == "section42":
        x, y = crosstask_curve(model)
        if x is not None:
            ax.plot(x, y, "-o", ms=3, color="C0", label="read: transfer AUC")
        ax.axhline(0.5, color="0.6", lw=0.6, ls=":")
    else:
        for i, ts in enumerate(("sm", "ic", "mw")):
            x, y = read_curve(model, ts, sec)
            if x is not None:
                ax.plot(x, y, "-o", ms=2.3, color=f"C{i}", label=f"read: {ts}")
        ax.axhline(0.0, color="0.6", lw=0.6, ls=":")
    ax.set_title(title); ax.set_xlabel("layer"); ax.set_ylabel(ylabel)
    if write_pts:
        axw = ax.twinx()
        wx = [p[0] for p in write_pts]; wy = [p[1] for p in write_pts]
        axw.plot(wx, wy, "-s", ms=5, color="k", lw=1.6, label="write: recovery vs -G")
        axw.axhline(0.0, color="k", lw=0.4, ls=":")
        axw.set_ylabel("write recovery (frac of +G gap)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = axw.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="best")
    else:
        ax.legend(fontsize=6.5, loc="best")


def main(model="gemma"):
    wr = compute_write_recovery()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    panel(axes[0], model, "section41", "§4.1 I_BA readout", "read R²", section41_write())
    panel(axes[1], model, "section42", "§4.2 cross-task shareability", "read mean transfer AUC", wr["section42"])
    panel(axes[2], model, "section43", "§4.3 +M sharpening", "read ΔR² (+M)", wr["section43"])
    fig.suptitle(f"{model}: §4 read (all layers) vs causal write recovery (window center)")
    fig.tight_layout()
    out = SPINE / f"spine_overlay_{model}"
    fig.savefig(str(out) + ".pdf"); fig.savefig(str(out) + ".png", dpi=130)
    plt.close(fig)
    print("wrote", out.with_suffix(".png"))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "gemma")
