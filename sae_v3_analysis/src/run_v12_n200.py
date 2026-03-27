#!/usr/bin/env python3
"""
V12 Iteration 2: n=200 BK steering + 3 random controls in same run.
Higher power to resolve borderline p=0.068 from V12 iteration 1.
"""
import sys, json, torch, numpy as np, logging, time, re, random, gc
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, mannwhitneyu

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("v12_n200")

OUT = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/json")
N_TRIALS = 200
TARGET_LAYER = 22
ALPHAS = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]


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
        history.append({'r': rnd, 'b': bet, 'w': balance >= balance, 'bal': balance})
    return {'stopped': False, 'bk': balance <= 0}


def run_condition(model, tokenizer, device, layer, hook_fn, name, n=N_TRIALS):
    stops, bks = 0, 0
    for i in range(n):
        r = play_game(model, tokenizer, device, hook_fn, layer, 42 + i * 997)
        if r['stopped']: stops += 1
        if r['bk']: bks += 1
        if (i+1) % 50 == 0:
            logger.info(f"  {name}: {i+1}/{n}, stop={stops}, bk={bks}")
    return {'stops': stops, 'bks': bks, 'n': n,
            'stop_rate': round(stops/n, 4), 'bk_rate': round(bks/n, 4)}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda:0"

    logger.info(f"Loading LLaMA... (n={N_TRIALS} per condition)")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    layer = model.model.layers[TARGET_LAYER]

    # BK direction
    hs = np.load('/home/v-seungplee/data/llm-addiction/sae_features_v3/slot_machine/llama/hidden_states_dp.npz',
                 allow_pickle=True)
    labels = (hs['game_outcomes'] == 'bankruptcy').astype(int)
    l22_idx = list(hs['layers']).index(22)
    hs_l22 = hs['hidden_states'][:, l22_idx, :]
    bk_dir = hs_l22[labels == 1].mean(0) - hs_l22[labels == 0].mean(0)
    bk_norm = np.linalg.norm(bk_dir)
    bk_unit = torch.tensor(bk_dir / bk_norm, dtype=torch.bfloat16, device=device)

    results = {'n_trials': N_TRIALS, 'timestamp': datetime.now().isoformat()}

    # Baseline
    logger.info("=" * 50)
    logger.info("BASELINE (n=200)")
    base = run_condition(model, tokenizer, device, layer, None, "baseline")
    results['baseline'] = base
    logger.info(f"  → stop={base['stop_rate']}, bk={base['bk_rate']}")

    # BK direction
    logger.info("=" * 50)
    logger.info("BK DIRECTION (n=200 per alpha)")
    bk_rates = [base['bk_rate']]
    bk_all = {}
    for alpha in ALPHAS:
        vec = alpha * bk_norm * bk_unit
        hook = lambda m, i, o, v=vec: o + v.unsqueeze(0).unsqueeze(0)
        r = run_condition(model, tokenizer, device, layer, hook, f"bk_{alpha:+.1f}")
        bk_all[str(alpha)] = r
        bk_rates.append(r['bk_rate'])
        logger.info(f"  BK α={alpha:+.1f}: stop={r['stop_rate']}, bk={r['bk_rate']}")

    bk_rho, bk_p = spearmanr([0.0] + ALPHAS, bk_rates)
    results['bk_direction'] = {'rho': round(bk_rho, 4), 'p': round(bk_p, 6), 'results': bk_all}
    logger.info(f"  BK dose-response: rho={bk_rho:.4f}, p={bk_p:.6f}")

    # 3 Random directions
    logger.info("=" * 50)
    logger.info("RANDOM DIRECTIONS (3 dirs, n=200 each)")
    rand_results = []
    for rd in range(3):
        np.random.seed(rd * 7777 + 13)
        rand_dir = np.random.randn(4096).astype(np.float32)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        rand_t = torch.tensor(rand_dir, dtype=torch.bfloat16, device=device)

        rd_rates = [base['bk_rate']]
        rd_all = {}
        for alpha in ALPHAS:
            vec = alpha * bk_norm * rand_t
            hook = lambda m, i, o, v=vec: o + v.unsqueeze(0).unsqueeze(0)
            r = run_condition(model, tokenizer, device, layer, hook, f"rand{rd}_{alpha:+.1f}")
            rd_all[str(alpha)] = r
            rd_rates.append(r['bk_rate'])

        rd_rho, rd_p = spearmanr([0.0] + ALPHAS, rd_rates)
        rand_results.append({'dir': rd, 'rho': round(rd_rho, 4), 'p': round(rd_p, 6), 'results': rd_all})
        logger.info(f"  Random #{rd}: rho={rd_rho:.4f}, p={rd_p:.6f}")

    results['random_directions'] = rand_results

    # Final analysis
    logger.info("\n" + "=" * 50)
    logger.info("FINAL ANALYSIS (n=200)")
    logger.info("=" * 50)
    logger.info(f"BK direction: rho={bk_rho:.4f}, p={bk_p:.6f}")
    for rd in rand_results:
        logger.info(f"Random #{rd['dir']}: rho={rd['rho']:.4f}, p={rd['p']:.6f}")

    rand_rhos = [abs(r['rho']) for r in rand_results]
    bk_exceeds = bk_rho > max(rand_rhos) if rand_rhos else True
    rand_any_sig = any(r['p'] < 0.05 for r in rand_results)

    if bk_p < 0.05 and not rand_any_sig:
        verdict = "BK_SPECIFIC_CONFIRMED"
        logger.info("VERDICT: BK DIRECTION IS SPECIFIC (p<0.05, 0 random significant)")
    elif bk_p < 0.05 and rand_any_sig:
        verdict = "PARTIALLY_SPECIFIC"
        logger.info("VERDICT: PARTIALLY SPECIFIC")
    else:
        verdict = "NOT_SIGNIFICANT"
        logger.info(f"VERDICT: BK direction not significant at n=200 (p={bk_p:.4f})")

    results['verdict'] = verdict
    results['bk_exceeds_all_random'] = bk_exceeds

    out_file = OUT / f"v12_n200_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved to {out_file}")

    del model, tokenizer; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
