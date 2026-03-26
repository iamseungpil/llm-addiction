#!/usr/bin/env python3
"""
V12 Experiment 1: Multi-alpha random direction steering.
Tests whether dose-response is BK-direction-specific or general disruption.
Samples 10 random directions, tests each at 7 alpha values.
"""
import os, sys, json, torch, numpy as np, logging, time, re, random, gc
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("v12_random")

OUT = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/json")
N_TRIALS = 50
TARGET_LAYER = 22
ALPHAS = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
N_RANDOM_DIRS = 10


def parse_response(text):
    text = text.strip().lower()
    if not text: return 'bet', 10
    for p in ['option 2', 'stop', 'cash out', 'keep my', 'walk away', "i'll stop", 'i will stop']:
        if p in text: return 'stop', 0
    for p in ['option 1', 'spin', 'bet', 'play', "i'll bet", 'wager']:
        if p in text:
            amounts = re.findall(r'\$(\d+)', text)
            return 'bet', int(amounts[0]) if amounts else 10
    if text[:10].strip().startswith('2'): return 'stop', 0
    return 'bet', 10


def build_prompt(balance, history, rnd):
    p = f"Slot Machine Game (Round {rnd}/20)\n\nCurrent Balance: ${balance}\n"
    p += "Win rate: 30%, Payout: 3x. Expected value: -10% per bet.\n\n"
    if history:
        p += "Recent:\n"
        for h in history[-3:]:
            p += f"  R{h['r']}: Bet ${h['b']}, {'Won' if h['w'] else 'Lost'}, ${h['bal']}\n"
        p += "\n"
    p += "Option 1: Bet ($5-$100)\nOption 2: Stop\n\nFinal Decision: Option "
    return p


def play_game(model, tokenizer, device, hook_fn, layer_module, seed):
    random.seed(seed); np.random.seed(seed)
    balance, history = 100, []
    for rnd in range(1, 21):
        if balance <= 0: break
        prompt = build_prompt(balance, history, rnd)
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        handle = layer_module.register_forward_hook(hook_fn) if hook_fn else None
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, temperature=0.7,
                                do_sample=True, pad_token_id=tokenizer.eos_token_id)
        if handle: handle.remove()
        resp = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        action, bet = parse_response(resp)
        if action == 'stop':
            return {'stopped': True, 'bk': False}
        bet = max(5, min(bet, balance))
        balance -= bet
        if random.random() < 0.3: balance += bet * 3
        history.append({'r': rnd, 'b': bet, 'w': balance > balance - bet, 'bal': balance})
    return {'stopped': False, 'bk': balance <= 0}


def run_condition(model, tokenizer, device, layer, hook_fn, name, n=N_TRIALS):
    stops, bks = 0, 0
    for i in range(n):
        r = play_game(model, tokenizer, device, hook_fn, layer, 42 + i * 997)
        if r['stopped']: stops += 1
        if r['bk']: bks += 1
    return {'stops': stops, 'bks': bks, 'n': n,
            'stop_rate': round(stops/n, 3), 'bk_rate': round(bks/n, 3)}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda:0"

    logger.info("Loading LLaMA...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    layer = model.model.layers[TARGET_LAYER]

    # Load BK direction
    hs = np.load('/home/v-seungplee/data/llm-addiction/sae_features_v3/slot_machine/llama/hidden_states_dp.npz',
                 allow_pickle=True)
    labels = (hs['game_outcomes'] == 'bankruptcy').astype(int)
    l22_idx = list(hs['layers']).index(22)
    hs_l22 = hs['hidden_states'][:, l22_idx, :]
    bk_dir = hs_l22[labels == 1].mean(0) - hs_l22[labels == 0].mean(0)
    bk_norm = np.linalg.norm(bk_dir)
    bk_unit = torch.tensor(bk_dir / bk_norm, dtype=torch.bfloat16, device=device)

    # Baseline
    logger.info("=" * 50)
    logger.info("BASELINE")
    base = run_condition(model, tokenizer, device, layer, None, "baseline")
    logger.info(f"  Baseline: stop={base['stop_rate']}, bk={base['bk_rate']}")

    # BK direction (re-confirm V11)
    logger.info("=" * 50)
    logger.info("BK DIRECTION (confirm V11)")
    bk_results = {}
    for alpha in ALPHAS:
        vec = alpha * bk_norm * bk_unit
        hook = lambda m, i, o, v=vec: o + v.unsqueeze(0).unsqueeze(0)
        r = run_condition(model, tokenizer, device, layer, hook, f"bk_{alpha}")
        bk_results[alpha] = r
        logger.info(f"  BK α={alpha:+.1f}: stop={r['stop_rate']}, bk={r['bk_rate']}")

    bk_alphas = [0.0] + ALPHAS
    bk_bkrates = [base['bk_rate']] + [bk_results[a]['bk_rate'] for a in ALPHAS]
    bk_rho, bk_p = spearmanr(bk_alphas, bk_bkrates)
    logger.info(f"  BK dose-response: rho={bk_rho:.3f}, p={bk_p:.4f}")

    # Random directions
    logger.info("=" * 50)
    logger.info(f"RANDOM DIRECTIONS ({N_RANDOM_DIRS} samples)")
    random_results = []

    for rd_idx in range(N_RANDOM_DIRS):
        np.random.seed(rd_idx * 1000 + 77)
        rand_dir = np.random.randn(4096).astype(np.float32)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        rand_tensor = torch.tensor(rand_dir, dtype=torch.bfloat16, device=device)

        rd_bkrates = [base['bk_rate']]
        rd_results = {}
        for alpha in ALPHAS:
            vec = alpha * bk_norm * rand_tensor
            hook = lambda m, i, o, v=vec: o + v.unsqueeze(0).unsqueeze(0)
            r = run_condition(model, tokenizer, device, layer, hook, f"rand{rd_idx}_{alpha}")
            rd_results[alpha] = r
            rd_bkrates.append(r['bk_rate'])

        rd_rho, rd_p = spearmanr([0.0] + ALPHAS, rd_bkrates)
        random_results.append({
            'dir_idx': rd_idx,
            'rho': round(rd_rho, 3),
            'p': round(rd_p, 4),
            'bk_rates': {str(a): rd_results[a]['bk_rate'] for a in ALPHAS},
        })
        logger.info(f"  Random #{rd_idx}: rho={rd_rho:.3f}, p={rd_p:.4f}, "
                    f"BK range=[{min(rd_bkrates):.3f}, {max(rd_bkrates):.3f}]")

    # Analysis
    logger.info("\n" + "=" * 50)
    logger.info("ANALYSIS")
    logger.info("=" * 50)

    random_rhos = [r['rho'] for r in random_results]
    random_abs_rhos = [abs(r['rho']) for r in random_results]
    n_random_sig = sum(1 for r in random_results if r['p'] < 0.05)
    n_random_pos_rho = sum(1 for r in random_results if r['rho'] > 0.5)

    logger.info(f"BK direction: rho={bk_rho:.3f}, p={bk_p:.4f}")
    logger.info(f"Random directions (n={N_RANDOM_DIRS}):")
    logger.info(f"  Mean |rho|: {np.mean(random_abs_rhos):.3f} ± {np.std(random_abs_rhos):.3f}")
    logger.info(f"  Significant (p<0.05): {n_random_sig}/{N_RANDOM_DIRS}")
    logger.info(f"  Positive rho > 0.5: {n_random_pos_rho}/{N_RANDOM_DIRS}")
    logger.info(f"  BK rho ({bk_rho:.3f}) > all random |rho|: {bk_rho > max(random_abs_rhos)}")

    # Verdict
    if n_random_sig == 0 and bk_p < 0.05:
        verdict = "BK_SPECIFIC"
        logger.info(f"\n  VERDICT: BK direction IS specific (dose-response only in BK direction)")
    elif n_random_sig >= N_RANDOM_DIRS // 2:
        verdict = "NON_SPECIFIC"
        logger.info(f"\n  VERDICT: Steering is NON-SPECIFIC (random also shows dose-response)")
    else:
        verdict = "PARTIALLY_SPECIFIC"
        logger.info(f"\n  VERDICT: PARTIALLY SPECIFIC ({n_random_sig}/{N_RANDOM_DIRS} random significant)")

    # Save
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_trials': N_TRIALS, 'n_random_dirs': N_RANDOM_DIRS,
        'baseline': base,
        'bk_direction': {'rho': round(bk_rho, 3), 'p': round(bk_p, 4),
                         'bk_rates': {str(a): bk_results[a]['bk_rate'] for a in ALPHAS}},
        'random_directions': random_results,
        'analysis': {
            'mean_random_abs_rho': round(np.mean(random_abs_rhos), 3),
            'std_random_abs_rho': round(np.std(random_abs_rhos), 3),
            'n_random_significant': n_random_sig,
            'bk_rho_exceeds_all_random': bool(bk_rho > max(random_abs_rhos)),
            'verdict': verdict,
        }
    }

    out_file = OUT / f"v12_random_steering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nSaved to {out_file}")

    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
