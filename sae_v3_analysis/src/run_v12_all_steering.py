#!/usr/bin/env python3
"""
V12 All-Steering: Run L22 steering for all 5 remaining model×task combinations.
(LLaMA SM L22 n=200 already done in v12_n200)

Usage:
  python run_v12_all_steering.py --model llama --task ic   # single combo
  python run_v12_all_steering.py --model gemma --task all  # all Gemma tasks
  python run_v12_all_steering.py --all                     # all 5 remaining
  python run_v12_all_steering.py --all --n 3               # smoke test
"""
import os, sys, json, torch, numpy as np, logging, re, random, gc, argparse
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("v12_all")

ANALYSIS_ROOT = Path(
    os.environ.get("LLM_ADDICTION_ANALYSIS_ROOT", "/home/v-seungplee/llm-addiction/sae_v3_analysis")
)
OUT = ANALYSIS_ROOT / "results" / "json"
HS_DIR = Path(
    os.environ.get("LLM_ADDICTION_DATA_ROOT", "/home/v-seungplee/data/llm-addiction/sae_features_v3")
)
ALPHAS = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
N_RANDOM = 3
TARGET_LAYER = 22

MODELS = {
    "llama": {"name": "meta-llama/Llama-3.1-8B-Instruct", "hidden_dim": 4096},
    "gemma": {"name": "google/gemma-2-9b-it", "hidden_dim": 3584},
}

# ============================================================
# Game Logic (from run_v12_full.py)
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
    """Mystery Wheel prompt."""
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
    if not text: return 'bet', 10
    for p in ['option 2', 'stop', 'cash out', 'keep my', 'walk away', "i'll stop", 'i will stop']:
        if p in text: return 'stop', 0
    for p in ['option 1', 'spin', 'bet', 'play', "i'll bet", 'wager']:
        if p in text:
            amounts = re.findall(r'\$(\d+)', text)
            return 'bet', int(amounts[0]) if amounts else 10
    if text[:10].strip().startswith('2'): return 'stop', 0
    return 'bet', 10

def parse_ic_response(text):
    text = text.strip().lower()
    if not text: return 2, 10
    for p in ['option 4', 'cash out', 'keep', "i'll stop", 'exit', 'walk away']:
        if p in text: return 4, 0
    m = re.search(r'option\s+([1-4])', text)
    if m: return int(m.group(1)), 10
    if text[:5].strip() and text[:5].strip()[0] in '1234':
        return int(text[:5].strip()[0]), 10
    return 2, 10

# MW uses same parse as SM (option 1=spin, option 2=stop)
parse_mw_response = parse_sm_response

# ============================================================
# Game Play Functions
# ============================================================

def play_game(model, tokenizer, device, hook_fn, layer_module, seed, task):
    random.seed(seed); np.random.seed(seed)
    balance, history = 100, []
    for rnd in range(1, 21):
        if balance <= 0: break
        if task == 'sm':
            prompt = build_sm_prompt(balance, history, rnd)
        elif task == 'ic':
            prompt = build_ic_prompt(balance, history, rnd)
        else:
            prompt = build_mw_prompt(balance, history, rnd)

        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        handle = layer_module.register_forward_hook(hook_fn) if hook_fn else None
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, temperature=0.7,
                                do_sample=True, pad_token_id=tokenizer.eos_token_id)
        if handle: handle.remove()
        resp = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        if task == 'ic':
            opt, _ = parse_ic_response(resp)
            if opt == 4: return {'stopped': True, 'bk': False}
            actual_bet = min(10, balance); balance -= actual_bet
            payout = 0
            if opt == 1 and random.random() < 0.1: payout = int(actual_bet * 9.0)
            elif opt == 2 and random.random() < 0.25: payout = int(actual_bet * 3.6)
            elif opt == 3 and random.random() < 0.5: payout = int(actual_bet * 1.8)
            balance += payout
            history.append({'r': rnd, 'opt': opt, 'bet': actual_bet, 'payout': payout,
                            'w': payout > 0, 'bal': balance})
        else:  # sm or mw
            action, bet = parse_sm_response(resp)
            if action == 'stop': return {'stopped': True, 'bk': False}
            bet = max(5, min(bet, balance)); balance -= bet
            if random.random() < 0.3: balance += bet * 3
            history.append({'r': rnd, 'b': bet, 'w': balance > (balance - bet), 'bal': balance})

    return {'stopped': False, 'bk': balance <= 0}


def run_condition(model, tokenizer, device, layer_module, hook_fn, name, n, task):
    stops, bks = 0, 0
    for i in range(n):
        r = play_game(model, tokenizer, device, hook_fn, layer_module, 42 + i * 997, task)
        if r['stopped']: stops += 1
        if r['bk']: bks += 1
        if (i + 1) % 50 == 0:
            logger.info(f"  {name}: {i+1}/{n}, stop={stops}, bk={bks}")
    return {'stops': stops, 'bks': bks, 'n': n,
            'stop_rate': round(stops/n, 4), 'bk_rate': round(bks/n, 4)}


# ============================================================
# Steering + T1-T6 Verification
# ============================================================

def compute_bk_direction(model_name, task_name, layer=TARGET_LAYER):
    task_dir_map = {'sm': 'slot_machine', 'ic': 'investment_choice', 'mw': 'mystery_wheel'}
    hs_path = HS_DIR / task_dir_map[task_name] / model_name / "hidden_states_dp.npz"
    hs = np.load(hs_path, allow_pickle=True)
    labels = (hs['game_outcomes'] == 'bankruptcy').astype(int)
    layers = list(hs['layers'])
    layer_idx = layers.index(layer)
    hs_layer = hs['hidden_states'][:, layer_idx, :]
    n_bk = labels.sum()
    n_safe = len(labels) - n_bk
    logger.info(f"  BK direction ({model_name} {task_name} L{layer}): {n_bk} BK, {n_safe} Safe")
    bk_dir = hs_layer[labels == 1].mean(0) - hs_layer[labels == 0].mean(0)
    bk_norm = float(np.linalg.norm(bk_dir))
    bk_unit = bk_dir / bk_norm
    return bk_unit, bk_norm


def run_steering(model, tokenizer, device, model_name, task_name, n):
    """Run full steering experiment for one model×task combo."""
    hidden_dim = MODELS[model_name]["hidden_dim"]
    bk_unit, bk_norm = compute_bk_direction(model_name, task_name)
    bk_tensor = torch.tensor(bk_unit, dtype=torch.bfloat16, device=device)
    layer_module = model.model.layers[TARGET_LAYER]

    exp_name = f"{model_name}_{task_name}_L{TARGET_LAYER}"
    logger.info(f"\n{'#'*60}")
    logger.info(f"STEERING: {exp_name} (n={n})")
    logger.info(f"{'#'*60}")

    # Baseline
    base = run_condition(model, tokenizer, device, layer_module, None, "baseline", n, task_name)
    logger.info(f"  Baseline: stop={base['stop_rate']}, bk={base['bk_rate']}")

    # BK direction
    bk_results = {}
    for alpha in ALPHAS:
        vec = alpha * bk_norm * bk_tensor
        hook = lambda m, inp, o, v=vec: o + v.unsqueeze(0).unsqueeze(0)
        r = run_condition(model, tokenizer, device, layer_module, hook, f"bk_{alpha:+.1f}", n, task_name)
        bk_results[alpha] = r
        logger.info(f"  BK α={alpha:+.1f}: stop={r['stop_rate']}, bk={r['bk_rate']}")

    # Random controls
    rand_results = []
    for rd in range(N_RANDOM):
        np.random.seed(rd * 7777 + 13)
        rand_dir = np.random.randn(hidden_dim).astype(np.float32)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        rand_tensor = torch.tensor(rand_dir, dtype=torch.bfloat16, device=device)

        rd_bk_rates = {}
        for alpha in ALPHAS:
            vec = alpha * bk_norm * rand_tensor
            hook = lambda m, inp, o, v=vec: o + v.unsqueeze(0).unsqueeze(0)
            r = run_condition(model, tokenizer, device, layer_module, hook,
                             f"rand{rd}_{alpha:+.1f}", n, task_name)
            rd_bk_rates[alpha] = r['bk_rate']
        rand_results.append({'dir': rd, 'bk_rates': rd_bk_rates})
        logger.info(f"  Random #{rd}: BK range [{min(rd_bk_rates.values()):.3f}, {max(rd_bk_rates.values()):.3f}]")

    # T1-T6 Verification
    bk_rates_all = [base['bk_rate']] + [bk_results[a]['bk_rate'] for a in ALPHAS]
    all_alphas = [0.0] + ALPHAS
    bk_rho, bk_p = spearmanr(all_alphas, bk_rates_all)

    # Relaxed thresholds for IC (low BK rate) and MW (high BK rate)
    is_ic = task_name == 'ic'
    is_mw = task_name == 'mw'
    t2_thresh = 0.03 if is_ic else 0.05
    t3_thresh = 0.03 if is_ic else 0.05
    t4_rho_min = 0.3 if is_ic else 0.5

    verification = {}
    verification['T1'] = base['bk_rate'] >= 0.05 and base['bk_rate'] <= 0.95
    verification['T2'] = bk_results[-2.0]['stop_rate'] - base['stop_rate'] > t2_thresh
    verification['T3'] = bk_results[2.0]['bk_rate'] - base['bk_rate'] > t3_thresh
    verification['T4'] = bk_rho > t4_rho_min and bk_p < 0.05

    rand_stats = []
    all_rand_nonsig = True
    for rd in rand_results:
        rd_rates = [base['bk_rate']] + [rd['bk_rates'][a] for a in ALPHAS]
        rd_rho, rd_p = spearmanr(all_alphas, rd_rates)
        rand_stats.append({'rho': round(rd_rho, 4), 'p': round(rd_p, 6)})
        if rd_p < 0.05: all_rand_nonsig = False

    verification['T5'] = all_rand_nonsig
    verification['T6'] = verification['T4'] and verification['T5']

    # Print
    logger.info(f"\n{'='*60}")
    logger.info(f"VERIFICATION: {exp_name}")
    logger.info(f"  T1 Baseline: {'PASS' if verification['T1'] else 'FAIL'} (BK={base['bk_rate']})")
    logger.info(f"  T2 α=-2 stop: {'PASS' if verification['T2'] else 'FAIL'} (+{bk_results[-2.0]['stop_rate']-base['stop_rate']:.3f})")
    logger.info(f"  T3 α=+2 BK: {'PASS' if verification['T3'] else 'FAIL'} (+{bk_results[2.0]['bk_rate']-base['bk_rate']:.3f})")
    logger.info(f"  T4 Dose-resp: {'PASS' if verification['T4'] else 'FAIL'} (rho={bk_rho:.3f}, p={bk_p:.5f})")
    logger.info(f"  T5 Random: {'PASS' if verification['T5'] else 'FAIL'} ({[r['p'] for r in rand_stats]})")
    logger.info(f"  T6 Final: {'PASS' if verification['T6'] else 'FAIL'}")
    verdict = "BK_SPECIFIC_CONFIRMED" if verification['T6'] else \
              "DOSE_RESPONSE_ONLY" if verification['T4'] else "NOT_SIGNIFICANT"
    logger.info(f"  VERDICT: {verdict}")
    logger.info(f"{'='*60}")

    # Save
    result = {
        'experiment': exp_name, 'timestamp': datetime.now().isoformat(),
        'model': model_name, 'task': task_name, 'layer': TARGET_LAYER, 'n_trials': n,
        'baseline': base,
        'bk_direction': {'rho': round(bk_rho, 4), 'p': round(bk_p, 6),
                         'results': {str(a): bk_results[a] for a in ALPHAS}},
        'random_directions': [
            {'dir': rd['dir'], 'rho': rand_stats[i]['rho'], 'p': rand_stats[i]['p'],
             'bk_rates': rd['bk_rates']}
            for i, rd in enumerate(rand_results)],
        'verification': verification,
        'verdict': verdict,
    }
    out_file = OUT / f"v12_{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Saved to {out_file}")
    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['llama', 'gemma'])
    parser.add_argument('--task', choices=['sm', 'ic', 'mw', 'all'])
    parser.add_argument('--all', action='store_true', help='Run all 5 remaining combos')
    parser.add_argument('--n', type=int, default=100)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.all:
        # Run all 5 remaining (LLaMA SM already done)
        combos = [
            ("llama", "ic"), ("llama", "mw"),
            ("gemma", "sm"), ("gemma", "ic"), ("gemma", "mw"),
        ]
    else:
        tasks = ['sm', 'ic', 'mw'] if args.task == 'all' else [args.task]
        combos = [(args.model, t) for t in tasks]

    # Group by model to minimize model loads
    from collections import defaultdict
    by_model = defaultdict(list)
    for m, t in combos:
        by_model[m].append(t)

    all_results = {}
    for model_name, tasks in by_model.items():
        cfg = MODELS[model_name]
        device = "cuda:0"
        logger.info(f"\nLoading {cfg['name']}...")
        model = AutoModelForCausalLM.from_pretrained(
            cfg['name'], torch_dtype=torch.bfloat16, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(cfg['name'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        for task_name in tasks:
            result = run_steering(model, tokenizer, device, model_name, task_name, args.n)
            all_results[f"{model_name}_{task_name}"] = result

        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    # Final summary
    logger.info(f"\n{'#'*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'#'*60}")
    for key, r in all_results.items():
        v = r['verification']
        logger.info(f"  {key}: T4={'PASS' if v['T4'] else 'FAIL'}, T6={'PASS' if v['T6'] else 'FAIL'}, verdict={r['verdict']}")


if __name__ == '__main__':
    main()
