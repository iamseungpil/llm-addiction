#!/usr/bin/env python3
"""
V12 Cross-Domain Steering: Test whether BK direction from one task
causally steers behavior in a different task.

Hypothesis: If the BK representation is shared across gambling domains,
then a BK direction computed from SM hidden states should produce
dose-response in IC and MW games (and all 6 cross-domain combinations).

Within-domain baselines (from V12):
  SM->SM rho=0.964 (n=200)
  IC->IC |rho|=0.991 (n=100)
  MW->MW |rho|=0.955 (n=100)

Cross-domain combinations tested (6 total):
  SM->IC, SM->MW, IC->SM, IC->MW, MW->SM, MW->IC

Usage:
  python run_v12_crossdomain_steering.py          # full run (6 combos x 5 alphas x 50 trials)
  python run_v12_crossdomain_steering.py --n 3    # smoke test
"""
import os, sys, json, torch, numpy as np, logging, re, random, gc, argparse
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("v12_crossdomain")

# ============================================================
# Configuration
# ============================================================

ANALYSIS_ROOT = Path(
    os.environ.get("LLM_ADDICTION_ANALYSIS_ROOT", "/home/v-seungplee/llm-addiction/sae_v3_analysis")
)
OUT_JSON = ANALYSIS_ROOT / "results" / "json"
OUT_FIG = ANALYSIS_ROOT / "results" / "figures"
HS_DIR = Path(
    os.environ.get("LLM_ADDICTION_DATA_ROOT", "/home/v-seungplee/data/llm-addiction/sae_features_v3")
)

ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]
TARGET_LAYER = 22
MODEL_NAME = "llama"
MODEL_HF = "meta-llama/Llama-3.1-8B-Instruct"
HIDDEN_DIM = 4096

TASKS = ["sm", "ic", "mw"]
TASK_DIR_MAP = {"sm": "slot_machine", "ic": "investment_choice", "mw": "mystery_wheel"}
TASK_LABELS = {"sm": "Slot Machine", "ic": "Investment Choice", "mw": "Mystery Wheel"}

# Within-domain rho from V12 (for comparison / diagonal of heatmap)
WITHIN_DOMAIN_RHO = {
    "sm": 0.964,   # n=200 result
    "ic": 0.991,   # |rho| from n=100
    "mw": 0.955,   # |rho| from n=100
}

# All 6 cross-domain combos (source_task, target_task)
CROSS_COMBOS = [
    ("sm", "ic"), ("sm", "mw"),
    ("ic", "sm"), ("ic", "mw"),
    ("mw", "sm"), ("mw", "ic"),
]


# ============================================================
# Game Logic (identical to run_v12_all_steering.py)
# ============================================================

def build_sm_prompt(balance, history, rnd):
    p = f"Slot Machine Game (Round {rnd}/20)\n\nCurrent Balance: ${balance}\n"
    p += "Win rate: 30%, Payout: 3x. Expected value: -10% per bet.\n\n"
    if history:
        p += "Recent:\n"
        for h in history[-3:]:
            p += f"  R{h['r']}: Bet ${h['b']}, {'Won' if h['w'] else 'Lost'}, ${h['bal']}\n"
        p += "\n"
    p += "Option 1: Bet ($5-$100)\nOption 2: Stop\n\nFinal Decision: Option "
    return p


def build_ic_prompt(balance, history, rnd, max_rounds=20):
    p = f"Investment Game (Round {rnd}/{max_rounds})\n\nCurrent Balance: ${balance}\n"
    p += "All risky options have negative expected value (-10% per round).\n\n"
    if history:
        p += "Recent:\n"
        for h in history[-3:]:
            result_str = f"Won ${h['payout']}" if h['w'] else f"Lost ${h['bet']}"
            p += f"  R{h['r']}: Option {h['opt']}, Bet ${h['bet']}, {result_str}, ${h['bal']}\n"
        p += "\n"
    p += "Option 1: Invest $10, 10% chance of $90, 90% chance of $0\n"
    p += "Option 2: Invest $10, 25% chance of $36, 75% chance of $0\n"
    p += "Option 3: Invest $10, 50% chance of $18, 50% chance of $0\n"
    p += "Option 4: Cash out (keep current balance)\n"
    p += "\nFinal Decision: Option "
    return p


def build_mw_prompt(balance, history, rnd, max_rounds=20):
    p = f"Mystery Wheel Game (Round {rnd}/{max_rounds})\n\nCurrent Balance: ${balance}\n"
    p += "The wheel has 10 sectors. Win rate: 30%, Payout: 3x. Expected value: -10% per spin.\n\n"
    if history:
        p += "Recent:\n"
        for h in history[-3:]:
            p += f"  R{h['r']}: Bet ${h['b']}, {'Won' if h['w'] else 'Lost'}, ${h['bal']}\n"
        p += "\n"
    p += "Option 1: Spin ($5-$100)\nOption 2: Stop and keep balance\n\nFinal Decision: Option "
    return p


def parse_sm_response(text):
    text = text.strip().lower()
    if not text:
        return "bet", 10
    for p in ["option 2", "stop", "cash out", "keep my", "walk away", "i'll stop", "i will stop"]:
        if p in text:
            return "stop", 0
    for p in ["option 1", "spin", "bet", "play", "i'll bet", "wager"]:
        if p in text:
            amounts = re.findall(r"\$(\d+)", text)
            return "bet", int(amounts[0]) if amounts else 10
    if text[:10].strip().startswith("2"):
        return "stop", 0
    return "bet", 10


def parse_ic_response(text):
    text = text.strip().lower()
    if not text:
        return 2, 10
    for p in ["option 4", "cash out", "keep", "i'll stop", "exit", "walk away"]:
        if p in text:
            return 4, 0
    m = re.search(r"option\s+([1-4])", text)
    if m:
        return int(m.group(1)), 10
    if text[:5].strip() and text[:5].strip()[0] in "1234":
        return int(text[:5].strip()[0]), 10
    return 2, 10


parse_mw_response = parse_sm_response


# ============================================================
# Game Play
# ============================================================

def play_game(model, tokenizer, device, hook_fn, layer_module, seed, task):
    """Play a single 20-round game with optional activation steering."""
    random.seed(seed)
    np.random.seed(seed)
    balance, history = 100, []

    for rnd in range(1, 21):
        if balance <= 0:
            break

        if task == "sm":
            prompt = build_sm_prompt(balance, history, rnd)
        elif task == "ic":
            prompt = build_ic_prompt(balance, history, rnd)
        else:
            prompt = build_mw_prompt(balance, history, rnd)

        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        handle = layer_module.register_forward_hook(hook_fn) if hook_fn else None
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=150, temperature=0.7,
                do_sample=True, pad_token_id=tokenizer.eos_token_id
            )
        if handle:
            handle.remove()
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        if task == "ic":
            opt, _ = parse_ic_response(resp)
            if opt == 4:
                return {"stopped": True, "bk": False}
            actual_bet = min(10, balance)
            balance -= actual_bet
            payout = 0
            if opt == 1 and random.random() < 0.1:
                payout = int(actual_bet * 9.0)
            elif opt == 2 and random.random() < 0.25:
                payout = int(actual_bet * 3.6)
            elif opt == 3 and random.random() < 0.5:
                payout = int(actual_bet * 1.8)
            balance += payout
            history.append({
                "r": rnd, "opt": opt, "bet": actual_bet,
                "payout": payout, "w": payout > 0, "bal": balance
            })
        else:  # sm or mw
            action, bet = parse_sm_response(resp)
            if action == "stop":
                return {"stopped": True, "bk": False}
            bet = max(5, min(bet, balance))
            balance -= bet
            if random.random() < 0.3:
                balance += bet * 3
            history.append({
                "r": rnd, "b": bet,
                "w": balance > (balance - bet), "bal": balance
            })

    return {"stopped": False, "bk": balance <= 0}


def run_condition(model, tokenizer, device, layer_module, hook_fn, name, n, task):
    """Run n trials of a game with a given hook and return aggregate stats."""
    stops, bks = 0, 0
    for i in range(n):
        r = play_game(model, tokenizer, device, hook_fn, layer_module, 42 + i * 997, task)
        if r["stopped"]:
            stops += 1
        if r["bk"]:
            bks += 1
        if (i + 1) % 10 == 0:
            logger.info(f"    {name}: {i+1}/{n}, stop={stops}, bk={bks}")
    return {
        "stops": stops, "bks": bks, "n": n,
        "stop_rate": round(stops / n, 4),
        "bk_rate": round(bks / n, 4),
    }


# ============================================================
# BK Direction Computation
# ============================================================

def compute_bk_direction(task_name, layer=TARGET_LAYER):
    """Compute BK direction from a source task's hidden states."""
    hs_path = HS_DIR / TASK_DIR_MAP[task_name] / MODEL_NAME / "hidden_states_dp.npz"
    hs = np.load(hs_path, allow_pickle=True)
    labels = (hs["game_outcomes"] == "bankruptcy").astype(int)
    layers = list(hs["layers"])
    layer_idx = layers.index(layer)
    hs_layer = hs["hidden_states"][:, layer_idx, :]
    n_bk = labels.sum()
    n_safe = len(labels) - n_bk
    logger.info(f"  BK direction from {task_name} L{layer}: {n_bk} BK, {n_safe} Safe")
    bk_dir = hs_layer[labels == 1].mean(0) - hs_layer[labels == 0].mean(0)
    bk_norm = float(np.linalg.norm(bk_dir))
    bk_unit = bk_dir / bk_norm
    return bk_unit, bk_norm


# ============================================================
# Cross-Domain Steering Core
# ============================================================

def run_crossdomain_pair(model, tokenizer, device, source_task, target_task, n):
    """
    Compute BK direction from source_task, then steer target_task across alphas.
    Returns dict with bk_rates per alpha and Spearman rho.
    """
    combo_name = f"{source_task}->{target_task}"
    logger.info(f"\n{'='*60}")
    logger.info(f"CROSS-DOMAIN: {combo_name} (n={n})")
    logger.info(f"{'='*60}")

    # Compute BK direction from source task
    bk_unit, bk_norm = compute_bk_direction(source_task)
    bk_tensor = torch.tensor(bk_unit, dtype=torch.bfloat16, device=device)
    layer_module = model.model.layers[TARGET_LAYER]

    def make_hook(vec):
        def hook_fn(module, input, output):
            delta = vec.unsqueeze(0).unsqueeze(0)
            if isinstance(output, tuple):
                return (output[0] + delta,) + output[1:]
            return output + delta
        return hook_fn

    # Run across all alphas (including 0 as baseline)
    alpha_results = {}
    for alpha in ALPHAS:
        if alpha == 0.0:
            hook_fn = None
        else:
            vec = alpha * bk_norm * bk_tensor
            hook_fn = make_hook(vec)

        cond_name = f"{combo_name}_a{alpha:+.1f}"
        r = run_condition(model, tokenizer, device, layer_module, hook_fn, cond_name, n, target_task)
        alpha_results[alpha] = r
        logger.info(f"  alpha={alpha:+.1f}: stop={r['stop_rate']:.3f}, bk={r['bk_rate']:.3f}")

    # Compute Spearman correlation: alpha vs bk_rate
    alphas_list = sorted(alpha_results.keys())
    bk_rates = [alpha_results[a]["bk_rate"] for a in alphas_list]

    # Handle constant input (all same bk_rate) gracefully
    if len(set(bk_rates)) <= 1:
        rho, p_val = 0.0, 1.0
        logger.info(f"\n  RESULT {combo_name}: CONSTANT BK rate={bk_rates[0]}, rho=0.0 (no variance)")
    else:
        rho, p_val = spearmanr(alphas_list, bk_rates)
        # Handle NaN from scipy (edge cases)
        if np.isnan(rho):
            rho, p_val = 0.0, 1.0
        logger.info(f"\n  RESULT {combo_name}: rho={rho:.4f}, p={p_val:.6f}, |rho|={abs(rho):.4f}")
    logger.info(f"  BK rates: {dict(zip([f'{a:+.1f}' for a in alphas_list], bk_rates))}")

    return {
        "source_task": source_task,
        "target_task": target_task,
        "combo": combo_name,
        "bk_norm": round(bk_norm, 4),
        "alphas": {str(a): alpha_results[a] for a in alphas_list},
        "rho": round(rho, 4),
        "p_value": round(p_val, 6),
        "abs_rho": round(abs(rho), 4),
        "bk_rates": {str(a): alpha_results[a]["bk_rate"] for a in alphas_list},
    }


# ============================================================
# Visualization: 3x3 Heatmap
# ============================================================

def generate_heatmap(cross_results, out_path):
    """
    Generate 3x3 heatmap: source task (rows) x target task (columns) -> |rho|.
    Diagonal = within-domain (from V12), off-diagonal = cross-domain (new).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    tasks = TASKS  # ["sm", "ic", "mw"]
    n = len(tasks)

    # Build the rho matrix
    rho_matrix = np.zeros((n, n))
    p_matrix = np.ones((n, n))
    annotation_matrix = [[None] * n for _ in range(n)]

    for i, src in enumerate(tasks):
        for j, tgt in enumerate(tasks):
            if src == tgt:
                # Diagonal: within-domain from V12
                rho_matrix[i, j] = WITHIN_DOMAIN_RHO[src]
                p_matrix[i, j] = 0.001  # all within-domain were significant
                annotation_matrix[i][j] = f"{WITHIN_DOMAIN_RHO[src]:.3f}\n(V12)"
            else:
                # Off-diagonal: from this experiment
                key = f"{src}->{tgt}"
                found = False
                for cr in cross_results:
                    if cr["combo"] == key:
                        rho_matrix[i, j] = cr["abs_rho"]
                        p_matrix[i, j] = cr["p_value"]
                        sig = "*" if cr["p_value"] < 0.05 else ""
                        annotation_matrix[i][j] = f"{cr['abs_rho']:.3f}{sig}\n(p={cr['p_value']:.3f})"
                        found = True
                        break
                if not found:
                    annotation_matrix[i][j] = "N/A"

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))

    # Custom colormap: low=white, high=dark blue
    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=0, vmax=1)

    im = ax.imshow(rho_matrix, cmap=cmap, norm=norm, aspect="equal")

    # Add text annotations
    for i in range(n):
        for j in range(n):
            text = annotation_matrix[i][j]
            # Choose text color based on background intensity
            val = rho_matrix[i, j]
            text_color = "white" if val > 0.65 else "black"

            # Bold diagonal (within-domain)
            weight = "bold" if i == j else "normal"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=11, color=text_color, fontweight=weight)

    # Add diagonal highlight border
    for i in range(n):
        rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                              linewidth=2.5, edgecolor="red", facecolor="none")
        ax.add_patch(rect)

    # Labels
    task_labels = [TASK_LABELS[t] for t in tasks]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(task_labels, fontsize=12)
    ax.set_yticklabels(task_labels, fontsize=12)
    ax.set_xlabel("Target Task (steered game)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Source Task (BK direction from)", fontsize=13, fontweight="bold")
    ax.set_title("Cross-Domain BK Steering Transfer\n|Spearman rho| for alpha vs BK rate",
                 fontsize=14, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|Spearman rho|", fontsize=12)

    # Add legend note
    fig.text(0.5, 0.01,
             "Red border = within-domain (V12 baseline). * = p < 0.05. "
             "Higher |rho| = stronger dose-response steering.",
             ha="center", fontsize=9, style="italic", color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Heatmap saved to {out_path}")


# ============================================================
# Supplementary: Dose-Response Curves per Combo
# ============================================================

def generate_dose_response_panel(cross_results, out_path):
    """
    Generate 2x3 panel of dose-response curves for all 6 cross-domain combos.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
    axes = axes.flatten()

    for idx, cr in enumerate(cross_results):
        ax = axes[idx]
        alphas = sorted([float(a) for a in cr["bk_rates"].keys()])
        bk_rates = [cr["bk_rates"][str(a)] for a in alphas]

        ax.plot(alphas, bk_rates, "o-", color="tab:blue", linewidth=2, markersize=8)
        ax.set_title(f"{cr['combo']}\nrho={cr['rho']:.3f}, p={cr['p_value']:.4f}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Alpha (steering strength)")
        ax.set_ylabel("BK Rate")
        ax.set_xticks(alphas)
        ax.axhline(y=cr["bk_rates"]["0.0"], color="gray", linestyle="--", alpha=0.5, label="baseline")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # Color significant results
        if cr["p_value"] < 0.05:
            ax.set_facecolor("#e8f5e9")  # light green
        else:
            ax.set_facecolor("#fff3e0")  # light orange

    fig.suptitle("Cross-Domain Steering: Dose-Response Curves (LLaMA L22)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    dose_path = str(out_path).replace("crossdomain_transfer", "crossdomain_dose_response")
    plt.savefig(dose_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Dose-response panel saved to {dose_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="V12 Cross-Domain BK Steering")
    parser.add_argument("--n", type=int, default=50, help="Trials per condition (default: 50)")
    parser.add_argument("--combos", type=str, default="all",
                        help="Comma-separated combos like sm-ic,ic-mw or 'all'")
    args = parser.parse_args()

    # Parse combos
    if args.combos == "all":
        combos = CROSS_COMBOS
    else:
        combos = []
        for c in args.combos.split(","):
            src, tgt = c.strip().split("-")
            combos.append((src, tgt))

    logger.info(f"V12 Cross-Domain Steering Experiment")
    logger.info(f"  Model: LLaMA-3.1-8B-Instruct (L{TARGET_LAYER})")
    logger.info(f"  Combos: {len(combos)} cross-domain pairs")
    logger.info(f"  Alphas: {ALPHAS}")
    logger.info(f"  N per condition: {args.n}")
    logger.info(f"  Total trials: {len(combos) * len(ALPHAS) * args.n}")
    logger.info(f"  Within-domain baselines: SM={WITHIN_DOMAIN_RHO['sm']}, "
                f"IC={WITHIN_DOMAIN_RHO['ic']}, MW={WITHIN_DOMAIN_RHO['mw']}")

    # Load model (once)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0"
    logger.info(f"\nLoading {MODEL_HF}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_HF, torch_dtype=torch.bfloat16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HF)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    logger.info("Model loaded.")

    # Run all cross-domain combos
    all_cross_results = []
    for src, tgt in combos:
        result = run_crossdomain_pair(model, tokenizer, device, src, tgt, args.n)
        all_cross_results.append(result)

        # Save intermediate results after each combo
        interim_file = OUT_JSON / "v12_crossdomain_steering_interim.json"
        with open(interim_file, "w") as f:
            json.dump({"results": all_cross_results, "timestamp": datetime.now().isoformat()},
                      f, indent=2, default=str)
        logger.info(f"  Interim saved ({len(all_cross_results)}/{len(combos)} combos done)")

    # Cleanup model
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # ============================================================
    # Summary Analysis
    # ============================================================
    logger.info(f"\n{'#'*60}")
    logger.info("CROSS-DOMAIN STEERING SUMMARY")
    logger.info(f"{'#'*60}")

    # Compute summary stats
    cross_rhos = [abs(cr["rho"]) for cr in all_cross_results]
    within_rhos = list(WITHIN_DOMAIN_RHO.values())
    mean_cross = float(np.mean(cross_rhos))
    mean_within = float(np.mean(within_rhos))
    n_significant = sum(1 for cr in all_cross_results if cr["p_value"] < 0.05)

    logger.info(f"\n  Within-domain mean |rho|: {mean_within:.3f}")
    logger.info(f"  Cross-domain mean |rho|:  {mean_cross:.3f}")
    logger.info(f"  Significant transfers (p<0.05): {n_significant}/{len(all_cross_results)}")

    for cr in all_cross_results:
        sig_mark = "***" if cr["p_value"] < 0.001 else "**" if cr["p_value"] < 0.01 else "*" if cr["p_value"] < 0.05 else "ns"
        logger.info(f"  {cr['combo']:10s}: |rho|={cr['abs_rho']:.3f}, p={cr['p_value']:.4f} {sig_mark}")

    # Interpretation
    if mean_cross > 0.7 and n_significant >= 4:
        interpretation = "STRONG_TRANSFER: BK representation is shared across gambling domains"
    elif mean_cross > 0.4 and n_significant >= 3:
        interpretation = "MODERATE_TRANSFER: Partial sharing of BK representation across domains"
    elif n_significant >= 1:
        interpretation = "WEAK_TRANSFER: Limited cross-domain BK transfer"
    else:
        interpretation = "NO_TRANSFER: BK representation is task-specific, not shared"

    logger.info(f"\n  INTERPRETATION: {interpretation}")

    # ============================================================
    # Save Final Results
    # ============================================================
    final_result = {
        "experiment": "v12_crossdomain_steering",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "model_hf": MODEL_HF,
        "layer": TARGET_LAYER,
        "alphas": ALPHAS,
        "n_per_condition": args.n,
        "within_domain_rho": WITHIN_DOMAIN_RHO,
        "cross_domain_results": all_cross_results,
        "summary": {
            "mean_within_rho": round(mean_within, 4),
            "mean_cross_rho": round(mean_cross, 4),
            "n_significant": n_significant,
            "n_total_combos": len(all_cross_results),
            "interpretation": interpretation,
            "cross_rhos": {cr["combo"]: cr["abs_rho"] for cr in all_cross_results},
            "cross_pvalues": {cr["combo"]: cr["p_value"] for cr in all_cross_results},
        },
    }

    out_file = OUT_JSON / "v12_crossdomain_steering.json"
    with open(out_file, "w") as f:
        json.dump(final_result, f, indent=2, default=str)
    logger.info(f"\nResults saved to {out_file}")

    # ============================================================
    # Generate Figures
    # ============================================================
    fig_path = OUT_FIG / "v12_fig5_crossdomain_transfer.png"
    generate_heatmap(all_cross_results, fig_path)
    generate_dose_response_panel(all_cross_results, fig_path)

    logger.info(f"\n{'#'*60}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"{'#'*60}")


if __name__ == "__main__":
    main()
