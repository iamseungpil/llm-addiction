"""
Round-by-Round Bet Escalation Analysis
=======================================
의도: Variable betting에서 베팅이 라운드 진행에 따라 체계적으로
      증가하는지(within-game escalation) 정량적으로 검증한다.

가설:
  H1: Variable에서 bet_ratio가 라운드에 따라 증가 (Spearman rho > 0)
  H2: Fixed에서는 이 경향 없음 → rho 차이가 유의 (permutation test)
  H3: 손실 직후 bet_ratio 변화가 승리 직후보다 큼 (Wilcoxon, 게임 내 paired)

검증:
  - LLaMA SM 3,200 게임의 라운드별 bet/balance ratio
  - 라운드 정규화 (0-1) → survivorship bias 완화
  - Permutation test for rho difference (H2)
  - Holm-Bonferroni correction (k=3)
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

DATA_PATH = Path("/home/v-seungplee/data/llm-addiction/behavioral/slot_machine/llama_v4_role/final_llama_20260315_062428.json")
OUT_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/escalation")
FIG_PATH = Path("/home/v-seungplee/LLM_Addiction_NMT_KOR/images/escalation_trajectory.pdf")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_round_data():
    """Extract per-round bet/balance ratio from game logs."""
    d = json.load(open(DATA_PATH))
    games = []
    for r in d["results"]:
        rounds_data = []
        for dec_idx, dec in enumerate(r["decisions"]):
            bal_before = float(dec.get("balance_before", 0))
            action = str(dec.get("action", ""))
            if bal_before > 0 and action == "bet":
                bet = float(dec.get("parsed_bet", dec.get("bet", 10)))
                ratio = min(bet / bal_before, 1.0)
                # Previous outcome from history
                prev_win = None
                if dec_idx > 0 and r.get("history"):
                    hist_idx = dec_idx - 1
                    if hist_idx < len(r["history"]):
                        h = r["history"][hist_idx]
                        if "result" in h:
                            prev_win = str(h["result"]) == "W"
                        elif "win" in h:
                            prev_win = bool(h["win"])

                rounds_data.append({
                    "round": int(dec["round"]),
                    "bet_ratio": ratio,
                    "prev_win": prev_win,
                })

        if len(rounds_data) >= 2:  # need at least 2 rounds for correlation
            total_rounds = len(rounds_data)
            for rd in rounds_data:
                rd["round_norm"] = (rd["round"] - 1) / max(total_rounds - 1, 1)

            games.append({
                "bet_type": r["bet_type"],
                "prompt_combo": r["prompt_combo"],
                "outcome": r["outcome"],
                "total_rounds": total_rounds,
                "rounds": rounds_data,
            })
    return games


def test_h1_h2(games):
    """H1: Variable escalation. H2: Variable rho > Fixed rho (permutation)."""
    # Per-game Spearman rho (round_norm vs bet_ratio)
    def game_rhos(game_list):
        rhos = []
        for g in game_list:
            rounds = [r["round_norm"] for r in g["rounds"]]
            ratios = [r["bet_ratio"] for r in g["rounds"]]
            if len(rounds) >= 3:
                rho, _ = spearmanr(rounds, ratios)
                if not np.isnan(rho):
                    rhos.append(rho)
        return np.array(rhos)

    var_games = [g for g in games if g["bet_type"] == "variable"]
    fix_games = [g for g in games if g["bet_type"] == "fixed"]

    var_rhos = game_rhos(var_games)
    fix_rhos = game_rhos(fix_games)

    print(f"\n=== H1 & H2: Within-Game Escalation ===")
    print(f"Variable: n={len(var_rhos)}, mean rho={np.mean(var_rhos):.4f}, median={np.median(var_rhos):.4f}")
    print(f"Fixed:    n={len(fix_rhos)}, mean rho={np.mean(fix_rhos):.4f}, median={np.median(fix_rhos):.4f}")

    # H1: Is Variable rho > 0?
    from scipy.stats import ttest_1samp
    t_var, p_var = ttest_1samp(var_rhos, 0, alternative="greater")
    print(f"\nH1 (Variable rho > 0): t={t_var:.3f}, p={p_var:.6f}")

    # H2: Permutation test for rho difference
    observed_diff = np.mean(var_rhos) - np.mean(fix_rhos)
    combined = np.concatenate([var_rhos, fix_rhos])
    n_var = len(var_rhos)
    n_perm = 10000
    perm_diffs = []
    for _ in range(n_perm):
        perm = np.random.permutation(combined)
        perm_diff = np.mean(perm[:n_var]) - np.mean(perm[n_var:])
        perm_diffs.append(perm_diff)
    p_perm = np.mean(np.array(perm_diffs) >= observed_diff)
    print(f"\nH2 (Variable rho > Fixed rho):")
    print(f"  Observed diff: {observed_diff:.4f}")
    print(f"  Permutation p: {p_perm:.6f} (n_perm={n_perm})")

    return {
        "h1_var_mean_rho": float(np.mean(var_rhos)),
        "h1_var_median_rho": float(np.median(var_rhos)),
        "h1_p": float(p_var),
        "h2_fix_mean_rho": float(np.mean(fix_rhos)),
        "h2_observed_diff": float(observed_diff),
        "h2_perm_p": float(p_perm),
        "n_var": len(var_rhos),
        "n_fix": len(fix_rhos),
    }


def test_h3(games):
    """H3: Loss-chasing > win-chasing (within-game paired comparison)."""
    var_games = [g for g in games if g["bet_type"] == "variable"]

    deltas_after_loss = []
    deltas_after_win = []

    for g in var_games:
        game_loss_deltas = []
        game_win_deltas = []
        for i in range(1, len(g["rounds"])):
            curr = g["rounds"][i]["bet_ratio"]
            prev = g["rounds"][i - 1]["bet_ratio"]
            delta = curr - prev
            if g["rounds"][i]["prev_win"] is True:
                game_win_deltas.append(delta)
            elif g["rounds"][i]["prev_win"] is False:
                game_loss_deltas.append(delta)

        if game_loss_deltas and game_win_deltas:
            deltas_after_loss.append(np.mean(game_loss_deltas))
            deltas_after_win.append(np.mean(game_win_deltas))

    deltas_after_loss = np.array(deltas_after_loss)
    deltas_after_win = np.array(deltas_after_win)

    print(f"\n=== H3: Loss-Chasing vs Win-Chasing ===")
    print(f"Games with both loss & win transitions: {len(deltas_after_loss)}")
    print(f"Mean delta after loss: {np.mean(deltas_after_loss):.4f}")
    print(f"Mean delta after win:  {np.mean(deltas_after_win):.4f}")

    stat, p = wilcoxon(deltas_after_loss, deltas_after_win, alternative="greater")
    print(f"Wilcoxon signed-rank: stat={stat:.1f}, p={p:.6f}")

    return {
        "n_paired": len(deltas_after_loss),
        "mean_delta_loss": float(np.mean(deltas_after_loss)),
        "mean_delta_win": float(np.mean(deltas_after_win)),
        "wilcoxon_p": float(p),
    }


def plot_trajectory(games):
    """Plot bet_ratio trajectory by round (Fixed vs Variable)."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for bt, color, label in [("fixed", "#27ae60", "Fixed"), ("variable", "#e74c3c", "Variable")]:
        bt_games = [g for g in games if g["bet_type"] == bt]

        # Collect ratios by normalized round bin (0-1 in 10 bins)
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_means = []
        bin_ses = []

        for i in range(len(bins) - 1):
            vals = []
            for g in bt_games:
                for r in g["rounds"]:
                    if bins[i] <= r["round_norm"] < bins[i + 1]:
                        vals.append(r["bet_ratio"])
            if vals:
                bin_means.append(np.mean(vals))
                bin_ses.append(np.std(vals) / np.sqrt(len(vals)))
            else:
                bin_means.append(np.nan)
                bin_ses.append(0)

        ax.errorbar(bin_centers, bin_means, yerr=bin_ses, color=color,
                     marker="o", markersize=5, linewidth=2, capsize=3, label=label)

    ax.set_xlabel("Normalized Round (0=start, 1=end)", fontsize=11)
    ax.set_ylabel("Bet / Balance Ratio", fontsize=11)
    ax.set_title("Within-Game Bet Escalation Trajectory", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved to {FIG_PATH}")


def main():
    print("Loading round-level data...")
    games = load_round_data()
    print(f"Games with ≥2 rounds: {len(games)}")
    print(f"  Fixed: {sum(1 for g in games if g['bet_type']=='fixed')}")
    print(f"  Variable: {sum(1 for g in games if g['bet_type']=='variable')}")

    h1h2 = test_h1_h2(games)
    h3 = test_h3(games)

    # Holm-Bonferroni correction (k=3)
    p_values = [h1h2["h1_p"], h1h2["h2_perm_p"], h3["wilcoxon_p"]]
    labels = ["H1 (Variable escalation)", "H2 (Var > Fix rho)", "H3 (Loss > Win chasing)"]

    sorted_idx = np.argsort(p_values)
    k = len(p_values)
    print(f"\n=== Holm-Bonferroni Correction (k={k}) ===")
    for rank, idx in enumerate(sorted_idx):
        threshold = 0.05 / (k - rank)
        sig = p_values[idx] < threshold
        print(f"  {labels[idx]}: p={p_values[idx]:.6f}, threshold={threshold:.4f}, sig={sig}")

    # Plot
    plot_trajectory(games)

    # Save results
    results = {"h1_h2": h1h2, "h3": h3}
    out_path = OUT_DIR / "escalation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
