#!/usr/bin/env python3
"""
V14 parallel worker. Run a single experiment subset.
Usage:
  python run_v14_parallel.py --exp exp2a   # LLaMA IC
  python run_v14_parallel.py --exp exp2b   # LLaMA MW
  python run_v14_parallel.py --exp exp3    # LLaMA cross-domain
  python run_v14_parallel.py --exp exp4    # Gemma MW
  python run_v14_parallel.py --exp exp5    # Gemma SM
  python run_v14_parallel.py --exp exp6    # Gemma IC
"""
import os, sys, json, torch, numpy as np, logging, gc, random, argparse
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from run_v12_all_steering import (
    play_game, run_condition, compute_bk_direction,
    build_sm_prompt, build_ic_prompt, build_mw_prompt,
    parse_sm_response, parse_ic_response, parse_mw_response,
    HS_DIR, OUT, TARGET_LAYER, MODELS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v14p")

ALPHAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def make_hook(alpha, direction_tensor):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            return (output[0] + alpha * direction_tensor.unsqueeze(0).unsqueeze(0),) + output[1:]
        return output + alpha * direction_tensor.unsqueeze(0).unsqueeze(0)
    return hook_fn


def steering_sweep(model, tokenizer, device, layer_module, direction, task, n_games, alphas=ALPHAS):
    d_tensor = torch.tensor(direction, dtype=torch.float16).to(device)
    results = {}
    for a in alphas:
        hook_fn = None if a == 0.0 else make_hook(a, d_tensor)
        res = run_condition(model, tokenizer, device, layer_module, hook_fn, f"alpha={a:.1f}", n_games, task)
        results[a] = res["bk_rate"]
        log.info(f"    alpha={a:+.1f}: BK={res['bks']}/{res['n']}={res['bk_rate']:.3f}")
    return results


def compute_rho(results, alphas=ALPHAS):
    rates = [results[a] for a in alphas]
    if len(set(rates)) <= 1:
        return float('nan'), float('nan')
    rho, p = spearmanr(alphas, rates)
    return float(rho), float(p)


def gen_random_dirs(dim, n, bk_norm, seed=42):
    rng = np.random.RandomState(seed)
    dirs = rng.randn(n, dim).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs *= bk_norm
    return dirs


def run_experiment(model, tokenizer, device, model_name, task, bk_dir, bk_norm,
                   n_games, n_random, exp_name, source_task=None):
    log.info(f"\n{'='*60}")
    log.info(f"  {exp_name}: {model_name} {task} (n={n_games}, {n_random} randoms)")
    log.info(f"{'='*60}")

    hidden_dim = len(bk_dir)
    layer_module = model.model.layers[TARGET_LAYER]

    log.info("  [BK Direction]")
    bk_results = steering_sweep(model, tokenizer, device, layer_module, bk_dir, task, n_games)
    bk_rho, bk_p = compute_rho(bk_results)
    log.info(f"  >> BK Direction: rho={bk_rho:.4f}, p={bk_p:.6f}")

    rand_dirs = gen_random_dirs(hidden_dim, n_random, bk_norm, seed=hash(exp_name) % (2**31))
    rand_results = []
    rand_abs_rhos = []

    for i in range(n_random):
        log.info(f"  [Random {i+1}/{n_random}]")
        r_res = steering_sweep(model, tokenizer, device, layer_module, rand_dirs[i], task, n_games)
        r_rho, r_p = compute_rho(r_res)
        log.info(f"  >> Random {i}: rho={r_rho:.4f}, p={r_p:.6f}")
        rand_results.append({
            "rho": None if np.isnan(r_rho) else round(r_rho, 4),
            "p": None if np.isnan(r_p) else round(r_p, 6),
            "bk_rates": {str(a): round(r_res[a], 4) for a in ALPHAS}
        })
        if not np.isnan(r_rho):
            rand_abs_rhos.append(abs(r_rho))

    if rand_abs_rhos and not np.isnan(bk_rho):
        perm_p = (sum(1 for r in rand_abs_rhos if r >= abs(bk_rho)) + 1) / (len(rand_abs_rhos) + 1)
    else:
        perm_p = float('nan')

    n_rand_sig = sum(1 for r in rand_results if r["p"] is not None and r["p"] < 0.05)
    bk_sig = not np.isnan(bk_p) and bk_p < 0.05

    if bk_sig and not np.isnan(perm_p) and perm_p < 0.05:
        verdict = "BK_SPECIFIC_CONFIRMED"
    elif bk_sig:
        verdict = "BK_SIGNIFICANT_NOT_SPECIFIC"
    else:
        verdict = "NOT_SIGNIFICANT"

    result = {
        "experiment": exp_name, "timestamp": TS,
        "model": model_name, "task": task, "source_task": source_task,
        "layer": TARGET_LAYER, "n_games": n_games, "n_random": n_random,
        "baseline_bk": bk_results.get(0.0),
        "bk_direction": {
            "rho": None if np.isnan(bk_rho) else round(bk_rho, 4),
            "p": None if np.isnan(bk_p) else round(bk_p, 6),
            "bk_rates": {str(a): round(bk_results[a], 4) for a in ALPHAS},
            "norm": round(float(bk_norm), 4)
        },
        "random_controls": rand_results,
        "random_abs_rhos_sorted": [round(r, 4) for r in sorted(rand_abs_rhos, reverse=True)[:20]],
        "permutation_p": None if np.isnan(perm_p) else round(perm_p, 4),
        "n_random_significant": n_rand_sig,
        "verdict": verdict
    }

    fname = f"v14_{exp_name}_{TS}.json"
    with open(OUT / fname, "w") as f:
        json.dump(result, f, indent=2)

    log.info(f"  VERDICT: {verdict}")
    log.info(f"  BK rho={bk_rho:.4f}, perm_p={perm_p}, rand_sig={n_rand_sig}/{n_random}")
    log.info(f"  Saved: {fname}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, choices=["exp2a", "exp2b", "exp3", "exp4", "exp5", "exp6"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.exp in ["exp2a", "exp2b", "exp3"]:
        log.info(f"Loading LLaMA for {args.exp}...")
        model_id = MODELS["llama"]["name"]
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        mdl.eval()

        ic_unit, ic_norm = compute_bk_direction("llama", "ic")
        mw_unit, mw_norm = compute_bk_direction("llama", "mw")
        ic_dir = ic_unit * ic_norm
        mw_dir = mw_unit * mw_norm

        if args.exp == "exp2a":
            run_experiment(mdl, tok, device, "llama", "ic", ic_dir, ic_norm,
                           n_games=100, n_random=10, exp_name="exp2a_llama_ic_n100")

        elif args.exp == "exp2b":
            run_experiment(mdl, tok, device, "llama", "mw", mw_dir, mw_norm,
                           n_games=100, n_random=10, exp_name="exp2b_llama_mw_n100")

        elif args.exp == "exp3":
            sm_unit, sm_norm = compute_bk_direction("llama", "sm")
            sm_dir = sm_unit * sm_norm
            for src, tgt, d, norm, label in [
                ("mw", "ic", mw_dir, mw_norm, "mw2ic"),
                ("mw", "sm", mw_dir, mw_norm, "mw2sm"),
                ("ic", "sm", ic_dir, ic_norm, "ic2sm"),
            ]:
                run_experiment(mdl, tok, device, "llama", tgt, d, norm,
                               n_games=50, n_random=5, exp_name=f"exp3_{label}",
                               source_task=src)

    elif args.exp in ["exp4", "exp5", "exp6"]:
        log.info(f"Loading Gemma for {args.exp}...")
        model_id = MODELS["gemma"]["name"]
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        mdl.eval()

        gemma_sm_unit, gemma_sm_norm = compute_bk_direction("gemma", "sm")
        gemma_sm_dir = gemma_sm_unit * gemma_sm_norm
        gemma_ic_unit, gemma_ic_norm = compute_bk_direction("gemma", "ic")
        gemma_ic_dir = gemma_ic_unit * gemma_ic_norm
        gemma_mw_unit, gemma_mw_norm = compute_bk_direction("gemma", "mw")
        gemma_mw_dir = gemma_mw_unit * gemma_mw_norm

        if args.exp == "exp4":
            run_experiment(mdl, tok, device, "gemma", "mw", gemma_mw_dir, gemma_mw_norm,
                           n_games=100, n_random=10, exp_name="exp4_gemma_mw_n100")
        elif args.exp == "exp5":
            run_experiment(mdl, tok, device, "gemma", "sm", gemma_sm_dir, gemma_sm_norm,
                           n_games=100, n_random=10, exp_name="exp5_gemma_sm_n100")
        elif args.exp == "exp6":
            run_experiment(mdl, tok, device, "gemma", "ic", gemma_ic_dir, gemma_ic_norm,
                           n_games=100, n_random=10, exp_name="exp6_gemma_ic_n100")

    log.info("Done.")


if __name__ == "__main__":
    main()
