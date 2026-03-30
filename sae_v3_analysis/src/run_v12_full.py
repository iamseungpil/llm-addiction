#!/usr/bin/env python3
"""
V12 Full Experiment Suite: Multi-layer, Cross-task, Cross-model steering.
Autoresearch-style: implement → run → verify T1-T6 → next step.

Usage:
  python run_v12_full.py --experiment s1   # LLaMA SM multi-layer
  python run_v12_full.py --experiment s2   # LLaMA IC cross-task
  python run_v12_full.py --experiment s3   # Gemma SM hidden state extraction
  python run_v12_full.py --experiment s4   # Gemma SM steering
  python run_v12_full.py --experiment smoke_s1  # Smoke test (n=3)
"""
import os, sys, json, torch, numpy as np, logging, re, random, gc, argparse
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("v12_full")

OUT = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/json")
HS_DIR = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3")
ALPHAS = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
N_RANDOM = 3

# ============================================================
# Game Logic
# ============================================================

def build_sm_prompt(balance, history, rnd):
    """Slot Machine prompt (same as V12 iter2)."""
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
    """Investment Choice prompt (simplified for steering)."""
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


def parse_sm_response(text):
    """Parse SM response: bet or stop."""
    text = text.strip().lower()
    if not text:
        return 'bet', 10
    for p in ['option 2', 'stop', 'cash out', 'keep my', 'walk away', "i'll stop", 'i will stop']:
        if p in text:
            return 'stop', 0
    for p in ['option 1', 'spin', 'bet', 'play', "i'll bet", 'wager']:
        if p in text:
            amounts = re.findall(r'\$(\d+)', text)
            return 'bet', int(amounts[0]) if amounts else 10
    if text[:10].strip().startswith('2'):
        return 'stop', 0
    return 'bet', 10


def parse_ic_response(text):
    """Parse IC response: option 1-3 (risky) or option 4 (stop)."""
    text = text.strip().lower()
    if not text:
        return 2, 10  # default: moderate risk
    # Check for option 4 / stop / cash out
    for p in ['option 4', 'cash out', 'keep', "i'll stop", 'exit', 'walk away']:
        if p in text:
            return 4, 0
    # Check for specific options
    m = re.search(r'option\s+([1-4])', text)
    if m:
        return int(m.group(1)), 10
    # First digit
    if text[:5].strip() and text[:5].strip()[0] in '1234':
        return int(text[:5].strip()[0]), 10
    return 2, 10  # default


def play_sm_game(model, tokenizer, device, hook_fn, layer_module, seed):
    """Play one SM game with optional steering hook."""
    random.seed(seed); np.random.seed(seed)
    balance, history = 100, []
    for rnd in range(1, 21):
        if balance <= 0:
            break
        prompt = build_sm_prompt(balance, history, rnd)
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        handle = layer_module.register_forward_hook(hook_fn) if hook_fn else None
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, temperature=0.7,
                                do_sample=True, pad_token_id=tokenizer.eos_token_id)
        if handle:
            handle.remove()
        resp = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        action, bet = parse_sm_response(resp)
        if action == 'stop':
            return {'stopped': True, 'bk': False}
        bet = max(5, min(bet, balance))
        balance -= bet
        if random.random() < 0.3:
            balance += bet * 3
        history.append({'r': rnd, 'b': bet, 'w': balance > (balance - bet), 'bal': balance})
    return {'stopped': False, 'bk': balance <= 0}


def play_ic_game(model, tokenizer, device, hook_fn, layer_module, seed):
    """Play one IC game with optional steering hook."""
    random.seed(seed); np.random.seed(seed)
    balance, history = 100, []
    bet_amount = 10  # fixed $10 bets for simplicity
    for rnd in range(1, 21):
        if balance <= 0:
            break
        prompt = build_ic_prompt(balance, history, rnd)
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        handle = layer_module.register_forward_hook(hook_fn) if hook_fn else None
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, temperature=0.7,
                                do_sample=True, pad_token_id=tokenizer.eos_token_id)
        if handle:
            handle.remove()
        resp = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        opt, _ = parse_ic_response(resp)
        if opt == 4:  # cash out
            return {'stopped': True, 'bk': False}
        actual_bet = min(bet_amount, balance)
        balance -= actual_bet
        # Apply outcome based on option
        payout = 0
        if opt == 1:    # 10% chance of 9x
            if random.random() < 0.1:
                payout = int(actual_bet * 9.0)
        elif opt == 2:  # 25% chance of 3.6x
            if random.random() < 0.25:
                payout = int(actual_bet * 3.6)
        else:           # opt == 3: 50% chance of 1.8x
            if random.random() < 0.5:
                payout = int(actual_bet * 1.8)
        balance += payout
        won = payout > 0
        history.append({'r': rnd, 'opt': opt, 'bet': actual_bet, 'payout': payout,
                        'w': won, 'bal': balance})
    return {'stopped': False, 'bk': balance <= 0}


# ============================================================
# Steering Engine
# ============================================================

def run_condition(model, tokenizer, device, layer_modules, hook_fn, name, n, game_fn):
    """Run n games with optional hooks on one or more layers."""
    stops, bks = 0, 0
    for i in range(n):
        seed = 42 + i * 997
        # Register hooks on all layers
        handles = []
        if hook_fn is not None:
            for lm in layer_modules:
                handles.append(lm.register_forward_hook(hook_fn))
        r = game_fn(model, tokenizer, device, None, layer_modules[0], seed)
        for h in handles:
            h.remove()
        if r['stopped']:
            stops += 1
        if r['bk']:
            bks += 1
        if (i + 1) % 50 == 0:
            logger.info(f"  {name}: {i+1}/{n}, stop={stops}, bk={bks}")
    return {'stops': stops, 'bks': bks, 'n': n,
            'stop_rate': round(stops / n, 4), 'bk_rate': round(bks / n, 4)}


def play_game_with_multi_hook(model, tokenizer, device, hooks_and_layers, game_fn, seed):
    """Play one game with hooks on multiple layers simultaneously."""
    random.seed(seed); np.random.seed(seed)
    balance, history = 100, []
    max_rounds = 20

    for rnd in range(1, max_rounds + 1):
        if balance <= 0:
            break
        if game_fn == 'sm':
            prompt = build_sm_prompt(balance, history, rnd)
        else:
            prompt = build_ic_prompt(balance, history, rnd)

        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # Register all hooks
        handles = []
        for layer_mod, hook_fn in hooks_and_layers:
            handles.append(layer_mod.register_forward_hook(hook_fn))

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, temperature=0.7,
                                do_sample=True, pad_token_id=tokenizer.eos_token_id)

        for h in handles:
            h.remove()

        resp = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        if game_fn == 'sm':
            action, bet = parse_sm_response(resp)
            if action == 'stop':
                return {'stopped': True, 'bk': False}
            bet = max(5, min(bet, balance))
            balance -= bet
            if random.random() < 0.3:
                balance += bet * 3
            history.append({'r': rnd, 'b': bet, 'w': balance > (balance - bet), 'bal': balance})
        else:
            opt, _ = parse_ic_response(resp)
            if opt == 4:
                return {'stopped': True, 'bk': False}
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
            history.append({'r': rnd, 'opt': opt, 'bet': actual_bet,
                            'payout': payout, 'w': payout > 0, 'bal': balance})

    return {'stopped': False, 'bk': balance <= 0}


def run_multi_hook_condition(model, tokenizer, device, hooks_and_layers, name, n, game_fn):
    """Run n games with hooks on multiple layers."""
    stops, bks = 0, 0
    for i in range(n):
        seed = 42 + i * 997
        r = play_game_with_multi_hook(model, tokenizer, device, hooks_and_layers, game_fn, seed)
        if r['stopped']:
            stops += 1
        if r['bk']:
            bks += 1
        if (i + 1) % 50 == 0:
            logger.info(f"  {name}: {i+1}/{n}, stop={stops}, bk={bks}")
    return {'stops': stops, 'bks': bks, 'n': n,
            'stop_rate': round(stops / n, 4), 'bk_rate': round(bks / n, 4)}


# ============================================================
# BK Direction Computation
# ============================================================

def compute_bk_direction(hs_path, target_layer):
    """Compute BK direction vector from hidden states NPZ."""
    hs = np.load(hs_path, allow_pickle=True)
    labels = (hs['game_outcomes'] == 'bankruptcy').astype(int)
    layers = list(hs['layers'])
    if target_layer not in layers:
        raise ValueError(f"Layer {target_layer} not in {layers}")
    layer_idx = layers.index(target_layer)
    hs_layer = hs['hidden_states'][:, layer_idx, :]
    n_bk = labels.sum()
    n_safe = len(labels) - n_bk
    logger.info(f"  BK direction: {n_bk} BK, {n_safe} Safe samples at L{target_layer}")
    bk_dir = hs_layer[labels == 1].mean(0) - hs_layer[labels == 0].mean(0)
    bk_norm = np.linalg.norm(bk_dir)
    bk_unit = bk_dir / bk_norm
    return bk_unit, bk_norm


# ============================================================
# T1-T6 Verification
# ============================================================

def verify_t1_t6(baseline, bk_results, random_results, alphas=ALPHAS,
                 t2_threshold=0.05, t3_threshold=0.05, t4_rho_min=0.5):
    """Run T1-T6 verification. Returns dict of test results."""
    results = {}

    # T1: Baseline reasonable
    bk_base = baseline['bk_rate']
    results['T1'] = {'pass': 0.20 <= bk_base <= 0.80,
                     'value': f"BK={bk_base}", 'criterion': 'BK ∈ [0.20, 0.80]'}

    # Dose-response data
    bk_rates = [bk_base] + [bk_results[a]['bk_rate'] for a in alphas]
    stop_rates = [baseline['stop_rate']] + [bk_results[a]['stop_rate'] for a in alphas]
    all_alphas = [0.0] + alphas

    # T2: α=-2 increases stop
    alpha_neg2 = bk_results[-2.0]
    stop_diff = alpha_neg2['stop_rate'] - baseline['stop_rate']
    results['T2'] = {'pass': stop_diff > t2_threshold,
                     'value': f"+{stop_diff:.3f}", 'criterion': f'>baseline+{t2_threshold}'}

    # T3: α=+2 increases BK
    alpha_pos2 = bk_results[2.0]
    bk_diff = alpha_pos2['bk_rate'] - bk_base
    results['T3'] = {'pass': bk_diff > t3_threshold,
                     'value': f"+{bk_diff:.3f}", 'criterion': f'>baseline+{t3_threshold}'}

    # T4: Dose-response
    rho, p = spearmanr(all_alphas, bk_rates)
    results['T4'] = {'pass': rho > t4_rho_min and p < 0.05,
                     'value': f"rho={rho:.3f}, p={p:.5f}",
                     'criterion': f'rho>{t4_rho_min}, p<0.05'}

    # T5: Random directions not significant
    rand_results_list = []
    all_rand_nonsig = True
    for rd in random_results:
        rd_rates = [bk_base] + [rd['bk_rates'][a] for a in alphas]
        rd_rho, rd_p = spearmanr(all_alphas, rd_rates)
        rand_results_list.append({'rho': rd_rho, 'p': rd_p})
        if rd_p < 0.05:
            all_rand_nonsig = False
    results['T5'] = {
        'pass': all_rand_nonsig,
        'value': ', '.join([f"p={r['p']:.3f}" for r in rand_results_list]),
        'criterion': f'All {len(random_results)} random p>0.05'
    }

    # T6: Final verdict
    results['T6'] = {
        'pass': results['T4']['pass'] and results['T5']['pass'],
        'value': 'BK_SPECIFIC_CONFIRMED' if (results['T4']['pass'] and results['T5']['pass']) else 'NOT_CONFIRMED',
        'criterion': 'T4 AND T5'
    }

    return results, rho, p, rand_results_list


def print_verification(results, experiment_name):
    """Print T1-T6 verification table."""
    logger.info(f"\n{'='*60}")
    logger.info(f"VERIFICATION: {experiment_name}")
    logger.info(f"{'='*60}")
    all_pass = True
    for t, r in results.items():
        status = "PASS" if r['pass'] else "FAIL"
        if not r['pass']:
            all_pass = False
        logger.info(f"  {t}: [{status}] {r['criterion']} → {r['value']}")
    verdict = "ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED"
    logger.info(f"\n  VERDICT: {verdict}")
    logger.info(f"{'='*60}\n")
    return all_pass


# ============================================================
# Experiment Runners
# ============================================================

def run_steering_experiment(model, tokenizer, device, model_layers, bk_unit, bk_norm,
                            target_layer_indices, layer_names, n, game_fn_name,
                            experiment_name):
    """Generic steering experiment with T1-T6 verification.

    target_layer_indices: list of layer indices to steer (e.g., [22] or [22, 25, 30])
    """
    game_fn = 'sm' if game_fn_name == 'sm' else 'ic'
    play_fn = play_sm_game if game_fn_name == 'sm' else play_ic_game

    # Get layer modules
    target_modules = [model_layers[i] for i in target_layer_indices]

    logger.info(f"\n{'#'*60}")
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info(f"  Layers: {target_layer_indices}, n={n}, game={game_fn_name}")
    logger.info(f"{'#'*60}")

    # Baseline (no hooks)
    logger.info("--- BASELINE ---")
    base = run_multi_hook_condition(model, tokenizer, device, [], "baseline", n, game_fn)
    logger.info(f"  Baseline: stop={base['stop_rate']}, bk={base['bk_rate']}")

    # BK direction steering
    logger.info("--- BK DIRECTION ---")
    bk_results = {}
    for alpha in ALPHAS:
        hooks_and_layers = []
        for i, layer_idx in enumerate(target_layer_indices):
            # Each layer gets its own direction vector and hook
            bk_u = bk_unit[layer_idx] if isinstance(bk_unit, dict) else bk_unit
            bk_n = bk_norm[layer_idx] if isinstance(bk_norm, dict) else bk_norm
            vec = alpha * bk_n * torch.tensor(bk_u, dtype=torch.bfloat16, device=device)
            hook = lambda m, inp, o, v=vec: o + v.unsqueeze(0).unsqueeze(0)
            hooks_and_layers.append((target_modules[i], hook))

        r = run_multi_hook_condition(model, tokenizer, device, hooks_and_layers,
                                     f"bk_{alpha:+.1f}", n, game_fn)
        bk_results[alpha] = r
        logger.info(f"  BK α={alpha:+.1f}: stop={r['stop_rate']}, bk={r['bk_rate']}")

    # Random direction controls
    logger.info("--- RANDOM DIRECTIONS ---")
    random_results = []
    for rd in range(N_RANDOM):
        np.random.seed(rd * 7777 + 13)
        rd_bk_rates = {}
        for alpha in ALPHAS:
            hooks_and_layers = []
            for i, layer_idx in enumerate(target_layer_indices):
                bk_n = bk_norm[layer_idx] if isinstance(bk_norm, dict) else bk_norm
                hidden_dim = len(bk_unit[layer_idx]) if isinstance(bk_unit, dict) else len(bk_unit)
                np.random.seed(rd * 7777 + 13 + layer_idx * 100)
                rand_dir = np.random.randn(hidden_dim).astype(np.float32)
                rand_dir = rand_dir / np.linalg.norm(rand_dir)
                vec = alpha * bk_n * torch.tensor(rand_dir, dtype=torch.bfloat16, device=device)
                hook = lambda m, inp, o, v=vec: o + v.unsqueeze(0).unsqueeze(0)
                hooks_and_layers.append((target_modules[i], hook))

            r = run_multi_hook_condition(model, tokenizer, device, hooks_and_layers,
                                         f"rand{rd}_{alpha:+.1f}", n, game_fn)
            rd_bk_rates[alpha] = r['bk_rate']

        random_results.append({'dir': rd, 'bk_rates': rd_bk_rates})
        logger.info(f"  Random #{rd}: BK range [{min(rd_bk_rates.values()):.3f}, {max(rd_bk_rates.values()):.3f}]")

    # T1-T6 Verification
    verification, bk_rho, bk_p, rand_stats = verify_t1_t6(
        base, bk_results, random_results,
        t2_threshold=0.03 if game_fn_name == 'ic' else 0.05,
        t3_threshold=0.03 if game_fn_name == 'ic' else 0.05,
        t4_rho_min=0.3 if game_fn_name == 'ic' else 0.5,
    )
    all_pass = print_verification(verification, experiment_name)

    # Save results
    result = {
        'experiment': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'n_trials': n,
        'layers': target_layer_indices,
        'layer_names': layer_names,
        'game': game_fn_name,
        'baseline': base,
        'bk_direction': {
            'rho': round(bk_rho, 4), 'p': round(bk_p, 6),
            'results': {str(a): bk_results[a] for a in ALPHAS}
        },
        'random_directions': [
            {'dir': rd['dir'], 'rho': round(rand_stats[i]['rho'], 4),
             'p': round(rand_stats[i]['p'], 6), 'bk_rates': rd['bk_rates']}
            for i, rd in enumerate(random_results)
        ],
        'verification': {k: {'pass': v['pass'], 'value': v['value']} for k, v in verification.items()},
        'all_pass': all_pass,
    }

    tag = experiment_name.replace(' ', '_').lower()
    out_file = OUT / f"v12_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Saved to {out_file}")

    return result


# ============================================================
# S1: LLaMA SM Multi-Layer
# ============================================================

def run_s1(n=100):
    """S1: LLaMA SM multi-layer steering."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda:0"

    logger.info("Loading LLaMA-3.1-8B-Instruct...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    hs_path = HS_DIR / "slot_machine/llama/hidden_states_dp.npz"

    # Compute BK directions for each layer
    bk_units, bk_norms = {}, {}
    for layer in [22, 25, 30]:
        bk_u, bk_n = compute_bk_direction(hs_path, layer)
        bk_units[layer] = bk_u
        bk_norms[layer] = bk_n

    all_results = {}

    # S1a: L25 single
    logger.info("\n" + "=" * 60)
    logger.info("S1a: L25 single-layer steering")
    r = run_steering_experiment(
        model, tokenizer, device, model.model.layers,
        {25: bk_units[25]}, {25: bk_norms[25]},
        [25], ["L25"], n, 'sm', 'S1a LLaMA SM L25')
    all_results['s1a_l25'] = r

    # S1b: L30 single
    logger.info("\n" + "=" * 60)
    logger.info("S1b: L30 single-layer steering")
    r = run_steering_experiment(
        model, tokenizer, device, model.model.layers,
        {30: bk_units[30]}, {30: bk_norms[30]},
        [30], ["L30"], n, 'sm', 'S1b LLaMA SM L30')
    all_results['s1b_l30'] = r

    # S1c: L22+L25+L30 combined
    logger.info("\n" + "=" * 60)
    logger.info("S1c: L22+L25+L30 combined steering")
    r = run_steering_experiment(
        model, tokenizer, device, model.model.layers,
        bk_units, bk_norms,
        [22, 25, 30], ["L22", "L25", "L30"], n, 'sm', 'S1c LLaMA SM L22+L25+L30')
    all_results['s1c_combined'] = r

    # Summary
    logger.info("\n" + "#" * 60)
    logger.info("S1 SUMMARY")
    logger.info("#" * 60)
    for key, r in all_results.items():
        v = r['verification']
        logger.info(f"  {key}: T4={v['T4']['value']}, T6={'PASS' if v['T6']['pass'] else 'FAIL'}")

    del model, tokenizer
    torch.cuda.empty_cache()
    return all_results


# ============================================================
# S2: LLaMA IC Cross-Task
# ============================================================

def run_s2(n=100):
    """S2: LLaMA IC cross-task steering."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda:0"

    logger.info("Loading LLaMA-3.1-8B-Instruct...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    hs_path = HS_DIR / "investment_choice/llama/hidden_states_dp.npz"
    bk_unit, bk_norm = compute_bk_direction(hs_path, 22)

    r = run_steering_experiment(
        model, tokenizer, device, model.model.layers,
        bk_unit, bk_norm,
        [22], ["L22"], n, 'ic', 'S2 LLaMA IC L22')

    del model, tokenizer
    torch.cuda.empty_cache()
    return r


# ============================================================
# S3: Gemma SM Hidden State Extraction
# ============================================================

def run_s3(n_games=400):
    """S3: Extract Gemma SM hidden states for BK direction computation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda:0"

    logger.info("Loading Gemma-2-9B-IT...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it", torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    model.eval()

    target_layer = 22
    hidden_dim = model.config.hidden_size  # 3584 for Gemma-2-9b
    logger.info(f"Gemma hidden_dim={hidden_dim}, extracting L{target_layer}")

    all_hidden_states = []
    all_outcomes = []

    for game_idx in range(n_games):
        seed = 42 + game_idx * 997
        random.seed(seed); np.random.seed(seed)
        balance, history = 100, []
        game_hs = None

        for rnd in range(1, 21):
            if balance <= 0:
                break
            prompt = build_sm_prompt(balance, history, rnd)
            msgs = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)

            # Hook to capture hidden state at target layer
            captured = {}
            def capture_hook(module, inp, out, captured=captured):
                # out is a tuple; first element is the hidden state
                if isinstance(out, tuple):
                    captured['hs'] = out[0][:, -1, :].detach().cpu().numpy()
                else:
                    captured['hs'] = out[:, -1, :].detach().cpu().numpy()

            layer_module = model.model.layers[target_layer]
            handle = layer_module.register_forward_hook(capture_hook)

            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=150, temperature=0.7,
                                    do_sample=True, pad_token_id=tokenizer.eos_token_id)
            handle.remove()

            # Save the hidden state from the last decision point
            if 'hs' in captured:
                game_hs = captured['hs'][0]  # shape: (hidden_dim,)

            resp = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            action, bet = parse_sm_response(resp)
            if action == 'stop':
                if game_hs is not None:
                    all_hidden_states.append(game_hs)
                    all_outcomes.append('voluntary_stop')
                break
            bet = max(5, min(bet, balance))
            balance -= bet
            if random.random() < 0.3:
                balance += bet * 3
            history.append({'r': rnd, 'b': bet, 'w': balance > (balance - bet), 'bal': balance})

        else:
            # Reached end of loop without break
            if game_hs is not None:
                outcome = 'bankruptcy' if balance <= 0 else 'voluntary_stop'
                all_hidden_states.append(game_hs)
                all_outcomes.append(outcome)
            continue

        # If broke out of the loop (stopped or bankrupt)
        if action != 'stop':
            if game_hs is not None:
                outcome = 'bankruptcy' if balance <= 0 else 'voluntary_stop'
                all_hidden_states.append(game_hs)
                all_outcomes.append(outcome)

        if (game_idx + 1) % 50 == 0:
            n_bk = sum(1 for o in all_outcomes if o == 'bankruptcy')
            logger.info(f"  Game {game_idx+1}/{n_games}: {n_bk} BK, {len(all_outcomes)-n_bk} Safe")

    # Save
    out_dir = HS_DIR / "slot_machine/gemma"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hidden_states_dp.npz"

    hs_array = np.array(all_hidden_states)
    outcomes_array = np.array(all_outcomes)
    np.savez(out_path,
             hidden_states=hs_array.reshape(-1, 1, hidden_dim),
             layers=np.array([target_layer]),
             game_outcomes=outcomes_array)

    n_bk = sum(1 for o in all_outcomes if o == 'bankruptcy')
    logger.info(f"\nS3 Complete: {len(all_outcomes)} games, {n_bk} BK ({n_bk/len(all_outcomes)*100:.1f}%)")
    logger.info(f"Saved to {out_path}")
    logger.info(f"Hidden states shape: {hs_array.shape}")

    # Verification
    assert hs_array.shape[1] == hidden_dim, f"Hidden dim mismatch: {hs_array.shape[1]} vs {hidden_dim}"
    assert n_bk > 10, f"Too few BK samples: {n_bk} (need >10 for direction computation)"
    assert len(all_outcomes) - n_bk > 10, f"Too few Safe samples"
    logger.info("S3 verification: PASS (shape OK, sufficient BK/Safe samples)")

    del model, tokenizer
    torch.cuda.empty_cache()
    return {'n_games': len(all_outcomes), 'n_bk': n_bk, 'hs_shape': list(hs_array.shape),
            'out_path': str(out_path)}


# ============================================================
# S4: Gemma SM Steering
# ============================================================

def run_s4(n=100):
    """S4: Gemma SM steering with its own BK direction."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda:0"

    hs_path = HS_DIR / "slot_machine/gemma/hidden_states_dp.npz"
    if not hs_path.exists():
        logger.error(f"Gemma hidden states not found at {hs_path}. Run S3 first.")
        return None

    logger.info("Loading Gemma-2-9B-IT...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it", torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    model.eval()

    # Compute Gemma BK direction
    hs = np.load(hs_path, allow_pickle=True)
    labels = (hs['game_outcomes'] == 'bankruptcy').astype(int)
    hs_l22 = hs['hidden_states'][:, 0, :]  # only one layer stored
    bk_dir = hs_l22[labels == 1].mean(0) - hs_l22[labels == 0].mean(0)
    bk_norm = np.linalg.norm(bk_dir)
    bk_unit = bk_dir / bk_norm
    logger.info(f"Gemma BK direction: norm={bk_norm:.4f}, dim={len(bk_unit)}")

    r = run_steering_experiment(
        model, tokenizer, device, model.model.layers,
        bk_unit, bk_norm,
        [22], ["L22"], n, 'sm', 'S4 Gemma SM L22')

    del model, tokenizer
    torch.cuda.empty_cache()
    return r


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True,
                        choices=['s1', 's2', 's3', 's4', 'smoke_s1', 'smoke_s2', 'smoke_s4', 'all'])
    parser.add_argument('--n', type=int, default=None, help='Override n_trials')
    args = parser.parse_args()

    if args.experiment == 'smoke_s1':
        logger.info("SMOKE TEST S1 (n=3)")
        run_s1(n=args.n or 3)
    elif args.experiment == 'smoke_s2':
        logger.info("SMOKE TEST S2 (n=3)")
        run_s2(n=args.n or 3)
    elif args.experiment == 'smoke_s4':
        logger.info("SMOKE TEST S4 (n=3)")
        run_s4(n=args.n or 3)
    elif args.experiment == 's1':
        run_s1(n=args.n or 100)
    elif args.experiment == 's2':
        run_s2(n=args.n or 100)
    elif args.experiment == 's3':
        run_s3(n_games=args.n or 400)
    elif args.experiment == 's4':
        run_s4(n=args.n or 100)
    elif args.experiment == 'all':
        n = args.n or 100
        logger.info(f"Running ALL experiments (n={n})")
        run_s1(n=n)
        gc.collect(); torch.cuda.empty_cache()
        run_s2(n=n)
        gc.collect(); torch.cuda.empty_cache()
        run_s3(n_games=max(n * 4, 400))
        gc.collect(); torch.cuda.empty_cache()
        run_s4(n=n)


if __name__ == '__main__':
    main()
