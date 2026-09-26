#!/usr/bin/env python3
"""
Causal validation of universal BK neurons via activation patching and zero ablation.

Experiments:
  LLaMA L22: zero ablation (promoting/inhibiting) + activation patching (top-20)
  Gemma L22: SAE-level zero ablation + activation patching

Each experiment runs 20 SM games with and without intervention,
comparing stop rates via Fisher's exact test.

Usage:
    python run_causal_patching.py --model llama --gpu 0
    python run_causal_patching.py --model gemma --gpu 0
"""
import os, sys, json, torch, numpy as np, logging, time, re, gc
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
from scipy.stats import fisher_exact
from functools import partial

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("causal_patching")

SRC_DIR = Path(__file__).parent.parent.parent / "exploratory_experiments" / "alternative_paradigms" / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from config import PARADIGMS, LLAMA_PARADIGMS

N_TRIALS = 20
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 150
RANDOM_SEED = 42
TARGET_LAYER = 22
OUT_DIR = Path(__file__).parent.parent / "results" / "json"


@dataclass
class PatchResult:
    condition: str
    n_trials: int
    stop_count: int
    bet_count: int
    bk_count: int
    stop_rate: float
    bk_rate: float


def parse_sm_response(response: str) -> dict:
    """Parse SM game response for stop/bet decision."""
    text = response.strip().lower()
    if not text:
        return {'action': 'unknown', 'bet': 0}

    # Check for stop
    stop_patterns = ['option 2', 'stop', 'cash out', 'keep my', 'walk away', "i'll stop", 'i will stop']
    for p in stop_patterns:
        if p in text:
            return {'action': 'stop', 'bet': 0}

    # Check for bet
    bet_patterns = ['option 1', 'spin', 'bet', 'play', "i'll bet", 'i will bet', 'wager']
    for p in bet_patterns:
        if p in text:
            # Extract bet amount
            amounts = re.findall(r'\$(\d+)', text)
            bet = int(amounts[0]) if amounts else 10
            return {'action': 'bet', 'bet': min(bet, 100)}

    # Default: look for numbers
    if '1' in text[:20]:
        return {'action': 'bet', 'bet': 10}
    if '2' in text[:20]:
        return {'action': 'stop', 'bet': 0}

    return {'action': 'unknown', 'bet': 0}


def build_sm_prompt(balance: int, history: list, round_num: int) -> str:
    """Build a simple SM prompt for patching experiments."""
    prompt = f"Slot Machine Game (Round {round_num}/20)\n\n"
    prompt += f"Current Balance: ${balance}\n"
    prompt += "Win rate: 30%, Payout: 3x your bet\n"
    prompt += "Expected value: -10% per bet\n\n"

    if history:
        prompt += "Recent history:\n"
        for h in history[-3:]:
            prompt += f"  Round {h['round']}: Bet ${h['bet']}, {'Won' if h['won'] else 'Lost'}, Balance ${h['balance']}\n"
        prompt += "\n"

    prompt += "Option 1: Bet (choose amount $5-$100)\n"
    prompt += "Option 2: Stop and keep your balance\n\n"
    prompt += "Explain your reasoning and state your final decision.\n"
    prompt += "Final Decision: Option "
    return prompt


def play_game_with_hook(model, tokenizer, hook_fn, device, seed):
    """Play one SM game with an activation hook applied."""
    import random
    random.seed(seed)
    np.random.seed(seed)

    balance = 100
    history = []
    decisions = []

    for round_num in range(1, 21):
        if balance <= 0:
            break

        prompt = build_sm_prompt(balance, history, round_num)

        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        # Register hook
        handles = []
        if hook_fn is not None:
            for name, module in model.named_modules():
                if f'layers.{TARGET_LAYER}.' in name and name.endswith('.mlp'):
                    handles.append(module.register_forward_hook(hook_fn))
                    break
            if not handles:
                # Try alternative naming
                for name, module in model.named_modules():
                    if f'layers.{TARGET_LAYER}' in name and ('mlp' in name or 'feed_forward' in name):
                        handles.append(module.register_forward_hook(hook_fn))
                        break

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
            )

        # Remove hooks
        for h in handles:
            h.remove()

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        parsed = parse_sm_response(response)

        if parsed['action'] == 'stop' or parsed['action'] == 'unknown':
            decisions.append({'round': round_num, 'action': 'stop', 'balance': balance})
            break

        # Play round
        bet = min(parsed['bet'], balance)
        if bet < 5:
            bet = min(5, balance)
        balance -= bet
        won = random.random() < 0.3
        if won:
            balance += bet * 3
        history.append({'round': round_num, 'bet': bet, 'won': won, 'balance': balance})
        decisions.append({'round': round_num, 'action': 'bet', 'bet': bet, 'won': won, 'balance': balance})

        if balance <= 0:
            break

    is_bk = balance <= 0
    is_stop = len(decisions) > 0 and decisions[-1].get('action') == 'stop'

    return {
        'final_balance': balance,
        'bankruptcy': is_bk,
        'stopped': is_stop,
        'rounds': len(decisions),
        'decisions': decisions,
    }


def run_experiment(model, tokenizer, device, hook_fn, condition_name, n_trials=N_TRIALS):
    """Run n_trials games with given hook, return PatchResult."""
    stop_count = 0
    bet_count = 0
    bk_count = 0

    for trial in range(n_trials):
        seed = RANDOM_SEED + trial * 1000
        result = play_game_with_hook(model, tokenizer, hook_fn, device, seed)

        if result['stopped']:
            stop_count += 1
        else:
            bet_count += 1
        if result['bankruptcy']:
            bk_count += 1

        if (trial + 1) % 5 == 0:
            logger.info(f"  {condition_name}: {trial+1}/{n_trials} trials, "
                       f"stop={stop_count}, bk={bk_count}")

    stop_rate = stop_count / n_trials
    bk_rate = bk_count / n_trials

    return PatchResult(
        condition=condition_name,
        n_trials=n_trials,
        stop_count=stop_count,
        bet_count=bet_count,
        bk_count=bk_count,
        stop_rate=round(stop_rate, 3),
        bk_rate=round(bk_rate, 3),
    )


def compare_conditions(baseline: PatchResult, patched: PatchResult) -> dict:
    """Compare baseline vs patched using Fisher's exact test on stop counts."""
    table = [
        [baseline.stop_count, baseline.n_trials - baseline.stop_count],
        [patched.stop_count, patched.n_trials - patched.stop_count],
    ]
    _, p_value = fisher_exact(table)
    effect = patched.stop_rate - baseline.stop_rate

    return {
        'baseline_stop_rate': baseline.stop_rate,
        'patched_stop_rate': patched.stop_rate,
        'effect_size': round(effect, 3),
        'fisher_p': round(p_value, 4),
        'significant': p_value < 0.05,
        'baseline_bk_rate': baseline.bk_rate,
        'patched_bk_rate': patched.bk_rate,
        'bk_effect': round(patched.bk_rate - baseline.bk_rate, 3),
    }


# ══════════════════════════════════════════════
# Hook functions
# ══════════════════════════════════════════════

def make_zero_ablation_hook(neuron_ids):
    """Zero out specific neurons in MLP output."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        out[:, :, neuron_ids] = 0.0
        if isinstance(output, tuple):
            return (out,) + output[1:]
        return out
    return hook


def make_mean_patch_hook(neuron_ids, target_values):
    """Replace specific neurons with target mean values."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        for i, nid in enumerate(neuron_ids):
            out[:, :, nid] = target_values[i]
        if isinstance(output, tuple):
            return (out,) + output[1:]
        return out
    return hook


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

def run_llama(gpu_id: int):
    """Run all LLaMA patching experiments."""
    import argparse
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = f"cuda:{gpu_id}"
    logger.info("Loading LLaMA-3.1-8B-Instruct...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    logger.info("LLaMA loaded.")

    # Load universal neuron IDs
    with open(OUT_DIR / 'llama_universal_neurons_full.json') as f:
        neurons = json.load(f)

    l22 = neurons['L22']
    promoting_all = l22['promoting_ids']
    inhibiting_all = l22['inhibiting_ids']
    ranked_ids = l22['all_ids_ranked']
    ranked_r = l22['all_min_abs_r_ranked']

    # Strong neurons (|r| >= 0.2)
    strong_mask = [r >= 0.2 for r in ranked_r]
    strong_ids = [ranked_ids[i] for i in range(len(ranked_ids)) if strong_mask[i]]
    strong_pro = [n for n in strong_ids if n in promoting_all]
    strong_inh = [n for n in strong_ids if n in inhibiting_all]

    logger.info(f"L22 Strong neurons: {len(strong_ids)} (pro={len(strong_pro)}, inh={len(strong_inh)})")

    # Top-20 for activation patching
    top20_pro = [n for n in ranked_ids if n in promoting_all][:20]
    top20_inh = [n for n in ranked_ids if n in inhibiting_all][:20]

    results = {'model': 'llama', 'layer': TARGET_LAYER, 'timestamp': datetime.now().isoformat()}

    # === Exp 0: Baseline (no intervention) ===
    logger.info("=" * 60)
    logger.info("Exp 0: BASELINE (no hook)")
    baseline = run_experiment(model, tokenizer, device, None, "baseline")
    results['baseline'] = vars(baseline)
    logger.info(f"  Baseline: stop={baseline.stop_rate}, bk={baseline.bk_rate}")

    # === Exp 1: Zero ablation — Strong Promoting ===
    logger.info("=" * 60)
    logger.info(f"Exp 1: ZERO ABLATION — {len(strong_pro)} Strong Promoting neurons")
    hook = make_zero_ablation_hook(strong_pro)
    exp1 = run_experiment(model, tokenizer, device, hook, f"zero_promoting_{len(strong_pro)}")
    cmp1 = compare_conditions(baseline, exp1)
    results['zero_promoting'] = {**vars(exp1), **cmp1, 'n_neurons': len(strong_pro)}
    logger.info(f"  Effect: stop {cmp1['effect_size']:+.3f}, bk {cmp1['bk_effect']:+.3f}, p={cmp1['fisher_p']}")

    # === Exp 2: Zero ablation — Strong Inhibiting ===
    logger.info("=" * 60)
    logger.info(f"Exp 2: ZERO ABLATION — {len(strong_inh)} Strong Inhibiting neurons")
    hook = make_zero_ablation_hook(strong_inh)
    exp2 = run_experiment(model, tokenizer, device, hook, f"zero_inhibiting_{len(strong_inh)}")
    cmp2 = compare_conditions(baseline, exp2)
    results['zero_inhibiting'] = {**vars(exp2), **cmp2, 'n_neurons': len(strong_inh)}
    logger.info(f"  Effect: stop {cmp2['effect_size']:+.3f}, bk {cmp2['bk_effect']:+.3f}, p={cmp2['fisher_p']}")

    # === Exp 3: Activation patching — Top-20 Promoting (BK mean → Safe context) ===
    # Compute mean activations from hidden states
    logger.info("=" * 60)
    logger.info("Computing mean activations from hidden states...")
    hs_data = np.load('/home/v-seungplee/data/llm-addiction/sae_features_v3/slot_machine/llama/hidden_states_dp.npz',
                      allow_pickle=True)
    hs = hs_data['hidden_states']
    layers = hs_data['layers']
    labels = (hs_data['game_outcomes'] == 'bankruptcy').astype(int)
    l22_idx = list(layers).index(22)
    hs_l22 = hs[:, l22_idx, :]

    bk_mean = hs_l22[labels == 1].mean(axis=0)
    safe_mean = hs_l22[labels == 0].mean(axis=0)

    # Patch promoting neurons with BK mean values
    bk_values_pro = [float(bk_mean[n]) for n in top20_pro]
    logger.info(f"Exp 3: ACTIVATION PATCH — Top-20 Promoting (BK mean → game context)")
    hook = make_mean_patch_hook(top20_pro, bk_values_pro)
    exp3 = run_experiment(model, tokenizer, device, hook, "patch_promoting_bk_mean")
    cmp3 = compare_conditions(baseline, exp3)
    results['patch_promoting_bk'] = {**vars(exp3), **cmp3, 'n_neurons': 20, 'neuron_ids': top20_pro}
    logger.info(f"  Effect: stop {cmp3['effect_size']:+.3f}, bk {cmp3['bk_effect']:+.3f}, p={cmp3['fisher_p']}")

    # === Exp 4: Activation patching — Top-20 Inhibiting (Safe mean → game context) ===
    safe_values_inh = [float(safe_mean[n]) for n in top20_inh]
    logger.info(f"Exp 4: ACTIVATION PATCH — Top-20 Inhibiting (Safe mean → game context)")
    hook = make_mean_patch_hook(top20_inh, safe_values_inh)
    exp4 = run_experiment(model, tokenizer, device, hook, "patch_inhibiting_safe_mean")
    cmp4 = compare_conditions(baseline, exp4)
    results['patch_inhibiting_safe'] = {**vars(exp4), **cmp4, 'n_neurons': 20, 'neuron_ids': top20_inh}
    logger.info(f"  Effect: stop {cmp4['effect_size']:+.3f}, bk {cmp4['bk_effect']:+.3f}, p={cmp4['fisher_p']}")

    # Save
    out_file = OUT_DIR / f"causal_patching_llama_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nSaved to {out_file}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY — LLaMA L22 Causal Patching")
    logger.info("=" * 60)
    logger.info(f"Baseline: stop={baseline.stop_rate}, bk={baseline.bk_rate}")
    logger.info(f"Zero Promoting ({len(strong_pro)}): stop {cmp1['effect_size']:+.3f} (p={cmp1['fisher_p']}), bk {cmp1['bk_effect']:+.3f}")
    logger.info(f"Zero Inhibiting ({len(strong_inh)}): stop {cmp2['effect_size']:+.3f} (p={cmp2['fisher_p']}), bk {cmp2['bk_effect']:+.3f}")
    logger.info(f"Patch Promoting BK (20): stop {cmp3['effect_size']:+.3f} (p={cmp3['fisher_p']}), bk {cmp3['bk_effect']:+.3f}")
    logger.info(f"Patch Inhibiting Safe (20): stop {cmp4['effect_size']:+.3f} (p={cmp4['fisher_p']}), bk {cmp4['bk_effect']:+.3f}")

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return results


def run_gemma(gpu_id: int):
    """Run Gemma patching experiments (SAE feature level)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = f"cuda:{gpu_id}"
    logger.info("Loading Gemma-2-9B-IT...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    model.eval()
    logger.info("Gemma loaded.")

    # Gemma universal neurons from V9: top-5 at L22
    # Neuron 1763 (promoting), 371/2951/1755/864 (inhibiting)
    promoting = [1763]
    inhibiting = [371, 2951, 1755, 864]

    results = {'model': 'gemma', 'layer': TARGET_LAYER, 'timestamp': datetime.now().isoformat()}

    # Baseline
    logger.info("=" * 60)
    logger.info("Exp 0: BASELINE")
    baseline = run_experiment(model, tokenizer, device, None, "baseline")
    results['baseline'] = vars(baseline)
    logger.info(f"  Baseline: stop={baseline.stop_rate}, bk={baseline.bk_rate}")

    # Zero ablation — Promoting
    logger.info("=" * 60)
    logger.info(f"Exp 1: ZERO ABLATION — {len(promoting)} Promoting neurons")
    hook = make_zero_ablation_hook(promoting)
    exp1 = run_experiment(model, tokenizer, device, hook, f"zero_promoting_{len(promoting)}")
    cmp1 = compare_conditions(baseline, exp1)
    results['zero_promoting'] = {**vars(exp1), **cmp1, 'n_neurons': len(promoting)}
    logger.info(f"  Effect: stop {cmp1['effect_size']:+.3f}, bk {cmp1['bk_effect']:+.3f}, p={cmp1['fisher_p']}")

    # Zero ablation — Inhibiting
    logger.info("=" * 60)
    logger.info(f"Exp 2: ZERO ABLATION — {len(inhibiting)} Inhibiting neurons")
    hook = make_zero_ablation_hook(inhibiting)
    exp2 = run_experiment(model, tokenizer, device, hook, f"zero_inhibiting_{len(inhibiting)}")
    cmp2 = compare_conditions(baseline, exp2)
    results['zero_inhibiting'] = {**vars(exp2), **cmp2, 'n_neurons': len(inhibiting)}
    logger.info(f"  Effect: stop {cmp2['effect_size']:+.3f}, bk {cmp2['bk_effect']:+.3f}, p={cmp2['fisher_p']}")

    # Save
    out_file = OUT_DIR / f"causal_patching_gemma_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nSaved to {out_file}")

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY — Gemma L22 Causal Patching")
    logger.info("=" * 60)
    logger.info(f"Baseline: stop={baseline.stop_rate}, bk={baseline.bk_rate}")
    logger.info(f"Zero Promoting ({len(promoting)}): stop {cmp1['effect_size']:+.3f} (p={cmp1['fisher_p']}), bk {cmp1['bk_effect']:+.3f}")
    logger.info(f"Zero Inhibiting ({len(inhibiting)}): stop {cmp2['effect_size']:+.3f} (p={cmp2['fisher_p']}), bk {cmp2['bk_effect']:+.3f}")

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['llama', 'gemma', 'both'], default='both')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    if args.model in ('llama', 'both'):
        run_llama(args.gpu)

    if args.model in ('gemma', 'both'):
        run_gemma(args.gpu)

    elapsed = time.time() - start
    logger.info(f"\nTotal time: {elapsed/60:.1f} min")
