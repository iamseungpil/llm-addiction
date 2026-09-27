"""§4.2 figure: 2D LOTO PCA scatter (Gemma L22).

For each held-out task (IC, MW, SM):
  X-axis = projection onto LOTO rank-1 SHARED direction
           (built from the other two tasks' BK contrasts).
  Y-axis = projection onto the held-out task's OWN BK direction
           after orthogonalising vs the shared axis (residual).

Visual story:
  * IC/SM panel: BK and voluntary-stop separate along BOTH axes (shared
    geometry transfers).
  * MW panel: separates only along Y (own direction); X shows BK and
    voluntary-stop overlapping (no shared geometry).

Voluntary-stop is downsampled to 3× the bankruptcy count for visual balance
(class ratio is otherwise 30-50:1, washing out the BK cluster).
"""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

REPO_ID = "llm-addiction-research/llm-addiction"
# Optional on-disk mirror of the released feature dumps; see
# ``scripts/build_figure_data.py``.  When it is absent the same npz files come
# from the public dataset, the way scripts/figures/fig5b_pca_appendix.py reads
# them, so this generator runs anywhere.
DATA_ROOT = Path(os.environ.get("LLM_ADDICTION_DATA", Path.home() / "llm-addiction-data"))
DATA = DATA_ROOT / "sae_features_v3"
OUT = Path(__file__).resolve().parents[1] / "images"
RNG = np.random.default_rng(42)

TASKS = ["ic", "mw", "sm"]
TASK_LABEL = {"ic": "IC (Investment Choice)",
              "mw": "MW (Mystery Wheel)",
              "sm": "SM (Slot Machine)"}
TASK_DIRS = {"sm": "slot_machine", "ic": "investment_choice",
             "mw": "mystery_wheel"}
LAYER = 22


def _npz_path(task: str) -> Path:
    rel = f"sae_features_v3/{TASK_DIRS[task]}/gemma/hidden_states_dp.npz"
    local = DATA / TASK_DIRS[task] / "gemma" / "hidden_states_dp.npz"
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO_ID, rel, repo_type="dataset",
                                token=os.environ.get("HF_TOKEN")))


def load_hs_bk(task: str):
    d = np.load(_npz_path(task), allow_pickle=False)
    layers = list(d["layers"])
    li = layers.index(LAYER)
    H = d["hidden_states"][:, li, :]
    out = d["game_outcomes"]
    valid = (out == "bankruptcy") | (out == "voluntary_stop")
    return H[valid], (out[valid] == "bankruptcy").astype(int)


def bk_contrast(H, bk):
    v = H[bk == 1].mean(0) - H[bk == 0].mean(0)
    return v / max(np.linalg.norm(v), 1e-12)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.0,
        "axes.labelsize": 10.0, "axes.titlesize": 10.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 8.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.8,
    })

    data = {t: load_hs_bk(t) for t in TASKS}

    # ---- Body figure: SM-only single panel (cleanest separation) ----
    fig, ax_only = plt.subplots(1, 1, figsize=(4.0, 3.2))
    axes_body = [ax_only]
    held_body = ["sm"]

    # ---- Appendix figure: full 3-panel (IC + MW + SM) ----
    fig_app, axes_app = plt.subplots(1, 3, figsize=(8.7, 2.8), sharey=False)
    axes_app_list = list(axes_app)
    held_app = TASKS[:]

    body_targets = list(zip(axes_body, held_body))
    app_targets = list(zip(axes_app_list, held_app))

    for ax, held in body_targets + app_targets:
        others = [t for t in TASKS if t != held]
        # Shared rank-1 axis from other 2 tasks' BK contrasts
        contrasts = np.stack([bk_contrast(*data[t]) for t in others], axis=0)
        U, S, Vt = np.linalg.svd(contrasts, full_matrices=False)
        shared = Vt[0]                          # (D,)

        H_h, bk_h = data[held]
        # Held-out task's own BK direction, orthogonalised against `shared`
        own = bk_contrast(H_h, bk_h)
        own_orth = own - shared * (shared @ own)
        own_orth = own_orth / max(np.linalg.norm(own_orth), 1e-12)

        x = H_h @ shared
        y = H_h @ own_orth

        # Orient so bankruptcy mean > stop mean on each axis
        if x[bk_h == 1].mean() < x[bk_h == 0].mean():
            x = -x
        if y[bk_h == 1].mean() < y[bk_h == 0].mean():
            y = -y

        # AUC of held-out using shared axis only
        auc_shared = roc_auc_score(bk_h, x)
        auc_orth = roc_auc_score(bk_h, y)

        # Downsample voluntary stop to 3× BK count for visual balance
        n_bk = int((bk_h == 1).sum())
        idx_st = np.where(bk_h == 0)[0]
        if len(idx_st) > 3 * n_bk:
            idx_st = RNG.choice(idx_st, size=3 * n_bk, replace=False)

        ax.scatter(x[idx_st], y[idx_st],
                   s=10, c="#3b6db5", alpha=0.45, edgecolors="none",
                   label=f"voluntary stop (n={3*n_bk if len(idx_st) == 3*n_bk else len(idx_st)})")
        ax.scatter(x[bk_h == 1], y[bk_h == 1],
                   s=18, c="#c44e52", alpha=0.85, edgecolors="white",
                   linewidths=0.4,
                   label=f"bankruptcy (n={n_bk})")
        ax.axhline(0, color="0.7", linewidth=0.5, zorder=0)
        ax.axvline(0, color="0.7", linewidth=0.5, zorder=0)
        # No internal panel title — task identity carried in caption + legend AUC value
        ax.text(0.04, 0.96, f"AUC$_{{\\rm shared}}$={auc_shared:.2f}",
                transform=ax.transAxes, fontsize=8.5,
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.7', lw=0.4))
        ax.set_xlabel(f"LOTO shared axis ({TASK_LABEL[held]})")
        if held == TASKS[0]:
            ax.set_ylabel("own BK axis (residual)")
        ax.legend(frameon=False, fontsize=8.0, loc="best",
                  scatterpoints=1, markerscale=0.8)

    fig.tight_layout(pad=0.4)
    out = OUT / "fig5b_pca.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    fig_app.tight_layout(pad=0.5)
    out_app = OUT / "fig5b_pca_appendix.pdf"
    fig_app.savefig(out_app, bbox_inches="tight")
    plt.close(fig_app)
    print(f"saved {out_app}")


if __name__ == "__main__":
    main()
