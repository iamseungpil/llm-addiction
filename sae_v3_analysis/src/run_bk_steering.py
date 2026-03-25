#!/usr/bin/env python3
"""
Phase 1: BK Direction Steering at L22.
Adds/subtracts BK direction vector from residual stream.
Tests dose-response relationship with BK rate.
"""
import os, sys, json, torch, numpy as np, logging, time, re, random, gc
from pathlib import Path
from datetime import datetime
from scipy.stats import fisher_exact, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("steering")

OUT = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/json")
N_TRIALS = 50
TARGET_LAYER = 22
ALPHAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


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
            return {'stopped': True, 'bk': False, 'rounds': rnd}

        bet = max(5, min(bet, balance))
        balance -= bet
        won = random.random() < 0.3
        if won: balance += bet * 3
        history.append({'r': rnd, 'b': bet, 'w': won, 'bal': balance})

    return {'stopped': False, 'bk': balance <= 0, 'rounds': len(history)}


def run_condition(model, tokenizer, device, layer, hook_fn, name, n=N_TRIALS):
    stops, bks = 0, 0
    for i in range(n):
        r = play_game(model, tokenizer, device, hook_fn, layer, 42 + i * 997)
        if r['stopped']: stops += 1
        if r['bk']: bks += 1
        if (i+1) % 10 == 0:
            logger.info(f"  {name}: {i+1}/{n}, stop={stops}, bk={bks}")
    return {'name': name, 'n': n, 'stops': stops, 'bks': bks,
            'stop_rate': round(stops/n, 3), 'bk_rate': round(bks/n, 3)}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda:0"

    logger.info("Loading LLaMA-3.1-8B-Instruct...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    layer = model.model.layers[TARGET_LAYER]

    # Compute BK direction vector from hidden states
    logger.info("Computing BK direction vector from SM hidden states...")
    hs = np.load('/home/v-seungplee/data/llm-addiction/sae_features_v3/slot_machine/llama/hidden_states_dp.npz',
                 allow_pickle=True)
    labels = (hs['game_outcomes'] == 'bankruptcy').astype(int)
    l22_idx = list(hs['layers']).index(22)
    hs_l22 = hs['hidden_states'][:, l22_idx, :]

    bk_mean = hs_l22[labels == 1].mean(0)
    safe_mean = hs_l22[labels == 0].mean(0)
    bk_direction = bk_mean - safe_mean  # (4096,)
    bk_norm = np.linalg.norm(bk_direction)
    bk_unit = bk_direction / bk_norm  # unit vector

    # Random direction (same norm) for control
    np.random.seed(99)
    random_dir = np.random.randn(4096).astype(np.float32)
    random_dir = random_dir / np.linalg.norm(random_dir)

    bk_tensor = torch.tensor(bk_unit, dtype=torch.bfloat16, device=device)
    random_tensor = torch.tensor(random_dir, dtype=torch.bfloat16, device=device)

    logger.info(f"BK direction norm: {bk_norm:.3f}")
    logger.info(f"BK direction top-5 dims: {np.argsort(-np.abs(bk_direction))[:5]}")

    results = {'model': 'llama', 'layer': TARGET_LAYER, 'n_trials': N_TRIALS,
               'bk_direction_norm': float(bk_norm), 'timestamp': datetime.now().isoformat()}

    # === Run all conditions ===
    conditions = []

    # Baseline
    logger.info("=" * 50)
    logger.info("BASELINE (alpha=0)")
    base = run_condition(model, tokenizer, device, layer, None, "baseline")
    results['baseline'] = base
    conditions.append((0.0, base))
    logger.info(f"  → stop={base['stop_rate']}, bk={base['bk_rate']}")

    # BK direction steering
    for alpha in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
        logger.info("=" * 50)
        logger.info(f"BK STEERING alpha={alpha}")
        scaled_vec = alpha * bk_norm * bk_tensor

        def make_hook(vec):
            def hook(module, input, output):
                return output + vec.unsqueeze(0).unsqueeze(0)
            return hook

        cond = run_condition(model, tokenizer, device, layer,
                           make_hook(scaled_vec), f"bk_alpha_{alpha}")
        results[f'bk_alpha_{alpha}'] = cond
        conditions.append((alpha, cond))
        logger.info(f"  → stop={cond['stop_rate']}, bk={cond['bk_rate']}")

    # Random direction control
    logger.info("=" * 50)
    logger.info("RANDOM DIRECTION (alpha=1.0)")
    rand_vec = 1.0 * bk_norm * random_tensor
    rand = run_condition(model, tokenizer, device, layer,
                        lambda m, i, o: o + rand_vec.unsqueeze(0).unsqueeze(0),
                        "random_alpha_1.0")
    results['random_control'] = rand
    logger.info(f"  → stop={rand['stop_rate']}, bk={rand['bk_rate']}")

    # === Analysis: Dose-response ===
    logger.info("\n" + "=" * 50)
    logger.info("DOSE-RESPONSE ANALYSIS")
    logger.info("=" * 50)

    alphas_list = [a for a, _ in conditions]
    bk_rates = [c['bk_rate'] for _, c in conditions]
    stop_rates = [c['stop_rate'] for _, c in conditions]

    rho_bk, p_bk = spearmanr(alphas_list, bk_rates)
    rho_stop, p_stop = spearmanr(alphas_list, stop_rates)

    logger.info(f"Alpha vs BK rate: Spearman rho={rho_bk:.3f}, p={p_bk:.4f}")
    logger.info(f"Alpha vs Stop rate: Spearman rho={rho_stop:.3f}, p={p_stop:.4f}")

    # Fisher test: most extreme conditions vs baseline
    for alpha in [-2.0, 2.0]:
        cond = results[f'bk_alpha_{alpha}']
        table = [[base['stops'], base['n']-base['stops']],
                 [cond['stops'], cond['n']-cond['stops']]]
        _, p = fisher_exact(table)
        effect = cond['stop_rate'] - base['stop_rate']
        logger.info(f"Alpha={alpha} vs baseline: stop {effect:+.3f}, Fisher p={p:.4f}")

    results['dose_response'] = {
        'spearman_bk': {'rho': round(rho_bk, 3), 'p': round(p_bk, 4)},
        'spearman_stop': {'rho': round(rho_stop, 3), 'p': round(p_stop, 4)},
        'random_control_stop': rand['stop_rate'],
        'random_control_bk': rand['bk_rate'],
    }

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    for alpha, cond in sorted(conditions, key=lambda x: x[0]):
        logger.info(f"  alpha={alpha:+.1f}: stop={cond['stop_rate']:.3f}, bk={cond['bk_rate']:.3f}")
    logger.info(f"  random:   stop={rand['stop_rate']:.3f}, bk={rand['bk_rate']:.3f}")
    logger.info(f"  Dose-response BK: rho={rho_bk:.3f}, p={p_bk:.4f}")
    logger.info(f"  Dose-response Stop: rho={rho_stop:.3f}, p={p_stop:.4f}")

    success = p_bk < 0.05 and rho_bk > 0
    logger.info(f"\n  CAUSAL EVIDENCE: {'YES — dose-response confirmed' if success else 'NO — not significant'}")

    out_file = OUT / f"bk_steering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved to {out_file}")

    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
