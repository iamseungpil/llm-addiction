"""Preliminary read-vs-write spine overlay (Gemma, complete cells).

Isolated previewer: reads the per-layer read profiles already on disk and draws,
per §4 section, the read(layer) curve with the causal write window L16-21 shaded.
This is a PREVIEW (Gemma read + the L16-21 causal window from W1); the final
spine_stats overlay adds LLaMA + the spinew §4.2/§4.3 write-recovery curves.

  §4.1 read = SAE-feature I_BA readout R2(layer), per task
  §4.2 read = cross-task shareability = mean transfer AUC(layer), per model
  §4.3 read = +M sharpening ΔR2(layer), per task
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPINE = Path("multilayer_causal/results/spine")
WRITE_LO, WRITE_HI = 16, 21          # causal write window (W1/W2/W3)
TASKS = {"sm": "slot_machine", "ic": "investment_choice", "mw": "mystery_wheel"}


def read_curve(model, task, sec):
    p = SPINE / f"read_profile_{model}_{task}_{sec}.json"
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    key = "r2_mean" if sec == "section41" else "delta_r2"
    xs, ys = d.get("layers", []), d.get(key, [])
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pts:
        return None, None
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def crosstask_curve(model):
    """Mean transfer AUC per layer from the per-model crosstask42 files."""
    rows = []
    for fp in sorted(SPINE.glob(f"read_profile_{model}_crosstask42_L*.json")):
        d = json.loads(fp.read_text()).get("result", {})
        ta = d.get("transfer_auc")
        if isinstance(ta, dict) and ta:
            rows.append((int(d["layer"]), float(np.mean(list(ta.values())))))
    if not rows:
        return None, None
    rows.sort()
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


def panel(ax, model, sec, title, ylabel):
    ax.axvspan(WRITE_LO, WRITE_HI, color="0.85", label="causal write window L16-21")
    if sec == "section42":
        x, y = crosstask_curve(model)
        if x is not None:
            ax.plot(x, y, "-o", ms=3, label="mean transfer AUC")
        ax.axhline(0.5, color="r", lw=0.6, ls=":")
    else:
        for ts in ("sm", "ic", "mw"):
            x, y = read_curve(model, ts, sec)
            if x is not None:
                ax.plot(x, y, "-o", ms=2.5, label=ts)
        ax.axhline(0.0, color="r", lw=0.6, ls=":")
    pk = None
    if sec == "section41":
        x, y = read_curve(model, "sm", sec)
        if x is not None:
            pk = int(x[int(np.argmax(y))])
            ax.axvline(pk, color="g", lw=0.8, ls="--", label=f"sm read peak L{pk}")
    ax.set_title(title)
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, loc="best")


def main():
    model = "gemma"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panel(axes[0], model, "section41", "§4.1 I_BA readout (read)", "R²")
    panel(axes[1], model, "section42", "§4.2 cross-task shareability (read)", "mean transfer AUC")
    panel(axes[2], model, "section43", "§4.3 +M sharpening (read)", "ΔR² (+M)")
    fig.suptitle(f"{model}: §4 read profiles across ALL layers vs causal write window (preview)")
    fig.tight_layout()
    out = SPINE / f"spine_overlay_preview_{model}"
    fig.savefig(str(out) + ".pdf")
    fig.savefig(str(out) + ".png", dpi=130)
    print("wrote", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
