#!/usr/bin/env python3
"""
Causal validation v2: Correct residual stream hook + n=50 + random control.

Fixes from v1:
  - Hook on decoder layer output (plain tensor), not MLP output
  - output.clone() and modify, return modified tensor
  - n=50 trials for sufficient power
  - Random neuron control to distinguish specific vs disruption effects
"""
import os, sys, json, torch, numpy as np, logging, time, re, gc, random
from pathlib import Path
from datetime import datetime
from scipy.stats import fisher_exact

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("causal_v2")

OUT_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/json")
N_TRIALS = 50
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 150
TARGET_LAYER = 22


def parse_response(text):
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


def build_prompt(balance, history, round_num):
    p = f"Slot Machine Game (Round {round_num}/20)\n\n"
    p += f"Current Balance: ${balance}\n"
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
        if balance <= 0:
            break
        prompt = build_prompt(balance, history, rnd)
        msgs = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        handle = layer_module.register_forward_hook(hook_fn) if hook_fn else None
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                temperature=TEMPERATURE, do_sample=True,
                                pad_token_id=tokenizer.eos_token_id)
        if handle:
            handle.remove()

        resp = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        action, bet = parse_response(resp)

        if action == 'stop':
            return {'stopped': True, 'bk': False, 'rounds': rnd}

        bet = max(5, min(bet, balance))
        balance -= bet
        won = random.random() < 0.3
        if won:
            balance += bet * 3
        history.append({'r': rnd, 'b': bet, 'w': won, 'bal': balance})

    return {'stopped': False, 'bk': balance <= 0, 'rounds': len(history)}


def run_condition(model, tokenizer, device, layer_module, hook_fn, name, n=N_TRIALS):
    stops, bks = 0, 0
    for i in range(n):
        r = play_game(model, tokenizer, device, hook_fn, layer_module, 42 + i * 997)
        if r['stopped']: stops += 1
        if r['bk']: bks += 1
        if (i+1) % 10 == 0:
            logger.info(f"  {name}: {i+1}/{n}, stop={stops}, bk={bks}")
    return {'name': name, 'n': n, 'stops': stops, 'bks': bks,
            'stop_rate': round(stops/n, 3), 'bk_rate': round(bks/n, 3)}


def compare(base, exp):
    table = [[base['stops'], base['n']-base['stops']],
             [exp['stops'], exp['n']-exp['stops']]]
    _, p = fisher_exact(table)
    return {
        'stop_effect': round(exp['stop_rate'] - base['stop_rate'], 3),
        'bk_effect': round(exp['bk_rate'] - base['bk_rate'], 3),
        'p': round(p, 4),
        'sig': p < 0.05,
    }


def make_zero_hook(neuron_ids):
    def hook(module, input, output):
        m = output.clone()
        m[:, :, neuron_ids] = 0.0
        return m
    return hook


def make_mean_hook(neuron_ids, values):
    def hook(module, input, output):
        m = output.clone()
        for i, nid in enumerate(neuron_ids):
            m[:, :, nid] = values[i]
        return m
    return hook


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

    # Load neuron IDs
    with open(OUT_DIR / 'llama_universal_neurons_full.json') as f:
        neurons = json.load(f)
    l22 = neurons['L22']
    pro_ids = set(l22['promoting_ids'])
    inh_ids = set(l22['inhibiting_ids'])
    ranked = l22['all_ids_ranked']
    ranked_r = l22['all_min_abs_r_ranked']

    strong_pro = [ranked[i] for i in range(len(ranked)) if ranked_r[i] >= 0.2 and ranked[i] in pro_ids]
    strong_inh = [ranked[i] for i in range(len(ranked)) if ranked_r[i] >= 0.2 and ranked[i] in inh_ids]
    top20_pro = strong_pro[:20]
    top20_inh = strong_inh[:20]

    # Random control (same size as strong_pro)
    np.random.seed(99)
    all_neurons = set(range(4096))
    universal = set(l22['promoting_ids'] + l22['inhibiting_ids'])
    non_universal = list(all_neurons - universal)
    random_ids = np.random.choice(non_universal, len(strong_pro), replace=False).tolist()

    logger.info(f"Strong promoting: {len(strong_pro)}, inhibiting: {len(strong_inh)}")
    logger.info(f"Top-20 promoting: {top20_pro[:5]}...")
    logger.info(f"Random control: {len(random_ids)} non-universal neurons")

    # Load mean activations for patching
    hs = np.load('/home/v-seungplee/data/llm-addiction/sae_features_v3/slot_machine/llama/hidden_states_dp.npz',
                 allow_pickle=True)
    labels = (hs['game_outcomes'] == 'bankruptcy').astype(int)
    l22_idx = list(hs['layers']).index(22)
    hs_l22 = hs['hidden_states'][:, l22_idx, :]
    bk_mean = hs_l22[labels == 1].mean(0)
    safe_mean = hs_l22[labels == 0].mean(0)

    results = {'model': 'llama', 'layer': TARGET_LAYER, 'n_trials': N_TRIALS,
               'timestamp': datetime.now().isoformat()}

    # === Baseline ===
    logger.info("=" * 50)
    logger.info("BASELINE")
    base = run_condition(model, tokenizer, device, layer, None, "baseline")
    results['baseline'] = base
    logger.info(f"  → stop={base['stop_rate']}, bk={base['bk_rate']}")

    # === Random Control ===
    logger.info("=" * 50)
    logger.info(f"RANDOM CONTROL ({len(random_ids)} non-universal neurons)")
    ctrl = run_condition(model, tokenizer, device, layer,
                        make_zero_hook(random_ids), f"random_{len(random_ids)}")
    ctrl_cmp = compare(base, ctrl)
    results['random_control'] = {**ctrl, **ctrl_cmp}
    logger.info(f"  → stop {ctrl_cmp['stop_effect']:+.3f}, bk {ctrl_cmp['bk_effect']:+.3f}, p={ctrl_cmp['p']}")

    # === Zero Promoting ===
    logger.info("=" * 50)
    logger.info(f"ZERO PROMOTING ({len(strong_pro)} neurons)")
    zp = run_condition(model, tokenizer, device, layer,
                      make_zero_hook(strong_pro), f"zero_pro_{len(strong_pro)}")
    zp_cmp = compare(base, zp)
    results['zero_promoting'] = {**zp, **zp_cmp, 'n_neurons': len(strong_pro)}
    logger.info(f"  → stop {zp_cmp['stop_effect']:+.3f}, bk {zp_cmp['bk_effect']:+.3f}, p={zp_cmp['p']}")

    # === Zero Inhibiting ===
    logger.info("=" * 50)
    logger.info(f"ZERO INHIBITING ({len(strong_inh)} neurons)")
    zi = run_condition(model, tokenizer, device, layer,
                      make_zero_hook(strong_inh), f"zero_inh_{len(strong_inh)}")
    zi_cmp = compare(base, zi)
    results['zero_inhibiting'] = {**zi, **zi_cmp, 'n_neurons': len(strong_inh)}
    logger.info(f"  → stop {zi_cmp['stop_effect']:+.3f}, bk {zi_cmp['bk_effect']:+.3f}, p={zi_cmp['p']}")

    # === Activation Patching: Promoting → BK mean ===
    logger.info("=" * 50)
    logger.info(f"PATCH PROMOTING → BK MEAN (top 20)")
    bk_vals = [float(bk_mean[n]) for n in top20_pro]
    pp = run_condition(model, tokenizer, device, layer,
                      make_mean_hook(top20_pro, bk_vals), "patch_pro_bk")
    pp_cmp = compare(base, pp)
    results['patch_promoting_bk'] = {**pp, **pp_cmp, 'n_neurons': 20}
    logger.info(f"  → stop {pp_cmp['stop_effect']:+.3f}, bk {pp_cmp['bk_effect']:+.3f}, p={pp_cmp['p']}")

    # === Activation Patching: Inhibiting → Safe mean ===
    logger.info("=" * 50)
    logger.info(f"PATCH INHIBITING → SAFE MEAN (top 20)")
    safe_vals = [float(safe_mean[n]) for n in top20_inh]
    pi = run_condition(model, tokenizer, device, layer,
                      make_mean_hook(top20_inh, safe_vals), "patch_inh_safe")
    pi_cmp = compare(base, pi)
    results['patch_inhibiting_safe'] = {**pi, **pi_cmp, 'n_neurons': 20}
    logger.info(f"  → stop {pi_cmp['stop_effect']:+.3f}, bk {pi_cmp['bk_effect']:+.3f}, p={pi_cmp['p']}")

    # === Summary ===
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Baseline: stop={base['stop_rate']}, bk={base['bk_rate']}")
    logger.info(f"Random Control: stop {ctrl_cmp['stop_effect']:+.3f} (p={ctrl_cmp['p']})")
    logger.info(f"Zero Promoting: stop {zp_cmp['stop_effect']:+.3f} (p={zp_cmp['p']}), EXPECT: +")
    logger.info(f"Zero Inhibiting: stop {zi_cmp['stop_effect']:+.3f} (p={zi_cmp['p']}), EXPECT: -")
    logger.info(f"Patch Pro→BK: stop {pp_cmp['stop_effect']:+.3f} (p={pp_cmp['p']}), EXPECT: -")
    logger.info(f"Patch Inh→Safe: stop {pi_cmp['stop_effect']:+.3f} (p={pi_cmp['p']}), EXPECT: +")

    # Metric: count of experiments with correct direction AND p<0.05
    correct_dir = [
        zp_cmp['stop_effect'] > 0,     # zero promoting → more stopping
        zi_cmp['stop_effect'] < 0,     # zero inhibiting → less stopping
        pp_cmp['stop_effect'] < 0,     # patch promoting BK → less stopping
        pi_cmp['stop_effect'] > 0,     # patch inhibiting safe → more stopping
    ]
    significant = [
        zp_cmp['p'] < 0.05,
        zi_cmp['p'] < 0.05,
        pp_cmp['p'] < 0.05,
        pi_cmp['p'] < 0.05,
    ]
    n_correct = sum(correct_dir)
    n_sig = sum(significant)
    n_both = sum(c and s for c, s in zip(correct_dir, significant))
    logger.info(f"\nCorrect direction: {n_correct}/4")
    logger.info(f"Significant (p<0.05): {n_sig}/4")
    logger.info(f"BOTH correct + significant: {n_both}/4")

    results['summary'] = {
        'n_correct_direction': n_correct,
        'n_significant': n_sig,
        'n_correct_and_significant': n_both,
    }

    out_file = OUT_DIR / f"causal_patching_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nSaved to {out_file}")

    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
