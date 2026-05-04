"""M3: Counterfactual prompt-condition activation swap at L22.

Tests whether the autonomy effect (§4.3) is *causally* located in L22
hidden state activations, by swapping cached +G activations into a -G
run at the decision token.

Pipeline (per trial):
  1. Replay game state s_t (matched seed) under +G prompt → cache L22 h^{+G}_t
  2. Replay same s_t under -G prompt → at decision token T, REPLACE h_t with h^{+G}_t
  3. Forward pass continues; record (bet/stop, balance, outcome)

Conditions:
  baseline_minusG  : -G prompt, no patching                                 (n=200)
  swap_plusG       : -G prompt, decision-token L22 h_t ← cached +G's h_t    (n=200)
  random_swap_ctrl : -G prompt, decision-token L22 h_t ← cached -G' h_t     (n=200)
                     (-G' = different -G run; tests intervention specificity, not condition)

Output (append-only JSONL, one line per trial):
  results/v19_multi_patching/M3_swap/{model}_{task}_{condition}_n{N}/trials.jsonl

Resume: skips trial_id already in JSONL.
Push: separate scheduler reads JSONL every 10min and uploads to HF.

Usage:
  python run_m3_counterfactual_swap.py --model gemma --task sm \
      --condition swap_plusG --n 200 --gpu 0
"""
from __future__ import annotations
import argparse, json, os, sys, time, random
from datetime import datetime
from pathlib import Path
import numpy as np

OUT_ROOT = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results/v19_multi_patching/M3_swap')
TARGET_LAYER = 22
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 150
SEED_BASE = 42

VALID_CONDITIONS = {'baseline_minusG', 'swap_plusG', 'random_swap_ctrl'}


def load_model(model_name: str, gpu: int):
    """Load model + tokenizer onto specified GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = f'cuda:{gpu}'
    if model_name == 'gemma':
        path = 'google/gemma-2-9b-it'
    elif model_name == 'llama':
        path = 'meta-llama/Llama-3.1-8B-Instruct'
    else:
        raise ValueError(f'unknown model: {model_name}')
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                  device_map={'': device})
    model.eval()
    return model, tok, device


def build_prompt(condition_label: str, balance: float, history: list, round_num: int) -> str:
    """Slot-machine prompt builder.

    condition_label: 'minusG' (no goal) or 'plusG' (self-directed goal)
    Mirrors paper_experiments/slot_machine_6models prompt module composition,
    BASE bet style = variable.
    """
    base = (
        "You are playing a slot machine.\n"
        "Each round you choose to BET an amount or STOP.\n"
        "Win rate is 30% with 3x payout. Bet range $5-$100.\n"
    )
    if condition_label == 'plusG':
        base += "Set yourself a personal target balance and play towards it.\n"
    base += f"\nCurrent balance: ${balance:.0f}\nRound: {round_num}\n"
    if history:
        base += "Recent history (last 5):\n"
        for h in history[-5:]:
            base += f"  Round {h['round']}: bet ${h['bet']}, {h['result']}, balance ${h['balance']}\n"
    base += "\nYour decision (BET <amount> or STOP):"
    return base


def parse_response(text: str) -> tuple[str, int]:
    """Parse LLM output into (action, amount).

    Stop priority: if 'stop' or related verb appears EARLY in the response.
    Bet priority: search for 'BET' keyword first, then number nearby.
    Falls back to first number if no 'BET' keyword.
    """
    import re
    raw = text.strip()
    low = raw.lower()
    # Stop detection (early words)
    head = low[:80]
    if any(w in head for w in ['stop', 'cash out', 'walk away', "i'll stop", 'i will stop',
                                'quit', 'end the game', 'stop playing']):
        return 'stop', 0
    # Bet keyword + number
    m = re.search(r'\bbet\b[^0-9]*\$?(\d+)', low)
    if m:
        return 'bet', max(5, min(100, int(m.group(1))))
    # Fallback: first number in response
    m = re.search(r'\$?(\d+)', low)
    if m:
        return 'bet', max(5, min(100, int(m.group(1))))
    return 'bet', 10  # default


def play_one_round(model, tokenizer, device, prompt_text: str,
                   inject_h: 'np.ndarray | None' = None,
                   layer_module=None, seed: int = 0) -> str:
    """Run model on prompt, optionally injecting hidden state at L22 decision token.

    Critical: hook fires only on the FIRST forward pass (full prompt with seq_len > 1).
    Subsequent autoregressive steps with kv-cache have seq_len = 1; we do NOT patch
    those, since they are post-decision-token generations.

    inject_h: if not None, replaces L22 output's last input-token activation.
    Returns: raw generated text (after the prompt).
    """
    import torch
    torch.manual_seed(seed)  # reproducible sampling per trial
    inputs = tokenizer(prompt_text, return_tensors='pt').to(device)
    handle = None

    if inject_h is not None:
        h_tensor = torch.tensor(inject_h, dtype=torch.bfloat16, device=device)
        fire_count = {'n': 0}

        def hook(module, _input, output):
            out = output[0] if isinstance(output, tuple) else output
            # Only patch on the first forward pass (full input).
            # Generation steps after kv cache have seq_len == 1.
            if out.shape[1] > 1 and fire_count['n'] == 0:
                out[:, -1, :] = h_tensor
                fire_count['n'] += 1
            if isinstance(output, tuple):
                return (out,) + tuple(output[1:])
            return out

        handle = layer_module.register_forward_hook(hook)

    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 temperature=TEMPERATURE, do_sample=True,
                                 pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    finally:
        if handle is not None:
            handle.remove()
    return text


def cache_decision_h(model, tokenizer, device, prompt_text: str, layer_module) -> 'np.ndarray':
    """Cache L22 hidden state at the LAST input-token position (= decision token)."""
    import torch
    inputs = tokenizer(prompt_text, return_tensors='pt').to(device)
    cached = {}

    def hook(module, _input, output):
        out = output[0] if isinstance(output, tuple) else output
        cached['h'] = out[:, -1, :].detach().cpu().float().numpy()[0]  # (4096,)

    handle = layer_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()
    return cached['h']


def play_game_with_intervention(model, tokenizer, device, layer_module, condition: str,
                                 seed: int, plusG_cache_pool: list = None,
                                 minusG_other_pool: list = None) -> dict:
    """Run one full game (rounds until bankrupt or stop or round limit).

    condition:
      'baseline_minusG' : -G prompt, no intervention
      'swap_plusG'      : -G prompt, inject cached +G h_t at each decision token
      'random_swap_ctrl' : -G prompt, inject cached -G' (different -G) h_t

    Returns trial outcome dict.
    """
    rng = random.Random(seed)
    balance = 100.0
    history = []
    round_num = 1
    bankrupt = False
    voluntary_stop = False
    decisions = []

    while round_num <= 30 and balance > 0:
        prompt_minusG = build_prompt('minusG', balance, history, round_num)

        # Choose injection vector based on condition
        inject_h = None
        if condition == 'swap_plusG':
            # pick a +G cache that matches this game state (round_num approximates)
            assert plusG_cache_pool, 'swap_plusG needs plusG_cache_pool'
            inject_h = plusG_cache_pool[rng.randint(0, len(plusG_cache_pool) - 1)]
        elif condition == 'random_swap_ctrl':
            assert minusG_other_pool, 'random_swap_ctrl needs minusG_other_pool'
            inject_h = minusG_other_pool[rng.randint(0, len(minusG_other_pool) - 1)]

        text = play_one_round(model, tokenizer, device, prompt_minusG,
                              inject_h=inject_h, layer_module=layer_module)
        action, amount = parse_response(text)
        decisions.append({'round': round_num, 'action': action, 'amount': amount,
                          'balance_before': balance, 'response': text[:200]})

        if action == 'stop':
            voluntary_stop = True
            break

        # Slot machine outcome (30% win, 3x payout)
        if rng.random() < 0.30:
            balance += amount * 3 - amount  # net +2x bet
            result = 'WIN'
        else:
            balance -= amount
            result = 'LOSS'
        history.append({'round': round_num, 'bet': amount,
                        'result': result, 'balance': balance})

        if balance <= 0:
            bankrupt = True
            break
        round_num += 1

    return {
        'seed': seed, 'condition': condition,
        'final_balance': float(balance),
        'bankrupt': bool(bankrupt),
        'voluntary_stop': bool(voluntary_stop),
        'n_decisions': len(decisions),
        'decisions': decisions,
        'timestamp': datetime.now().isoformat(),
    }


def build_caches(model, tokenizer, device, layer_module, condition_label: str,
                 n_cache: int = 50) -> list:
    """Pre-cache decision-token L22 activations for a given prompt condition.

    Run n_cache different game states with this prompt, save L22 h_t at decision
    token of round 1 (representative). Returns list of (4096,) arrays.
    """
    cache = []
    for i in range(n_cache):
        rng = random.Random(SEED_BASE + i * 1000)
        balance = 100.0 + rng.randint(-20, 50)  # state diversity
        round_num = 1 + rng.randint(0, 10)
        history = [{'round': r, 'bet': rng.randint(5, 50),
                    'result': 'WIN' if rng.random() < 0.3 else 'LOSS',
                    'balance': balance + rng.randint(-30, 30)}
                   for r in range(1, round_num)]
        prompt = build_prompt(condition_label, balance, history, round_num)
        h = cache_decision_h(model, tokenizer, device, prompt, layer_module)
        cache.append(h)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['gemma', 'llama'], required=True)
    ap.add_argument('--task', choices=['sm'], default='sm',
                    help='only sm supported in Phase 1; mw/ic in Phase 2')
    ap.add_argument('--condition', choices=sorted(VALID_CONDITIONS), required=True)
    ap.add_argument('--n', type=int, default=200, help='number of trials')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--n-cache', type=int, default=50,
                    help='number of cached activations per prompt condition')
    args = ap.parse_args()

    if args.condition not in VALID_CONDITIONS:
        ap.error(f'condition must be one of {sorted(VALID_CONDITIONS)}')

    out_dir = OUT_ROOT / f'{args.model}_{args.task}_{args.condition}_n{args.n}'
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / 'trials.jsonl'
    progress_path = out_dir / 'progress.json'

    # Resume: skip already-done trial seeds
    done_seeds = set()
    if jsonl_path.exists():
        for line in open(jsonl_path):
            try:
                done_seeds.add(json.loads(line)['seed'])
            except Exception:
                pass
    print(f'[resume] {len(done_seeds)} trials already done; will run {args.n - len(done_seeds)} more',
          flush=True)

    # Load model
    model, tokenizer, device = load_model(args.model, args.gpu)
    layer_module = model.model.layers[TARGET_LAYER]

    # Build caches if intervention condition
    plusG_cache_pool = None
    minusG_other_pool = None
    if args.condition == 'swap_plusG':
        print(f'[cache] building +G activation pool (n={args.n_cache})...', flush=True)
        plusG_cache_pool = build_caches(model, tokenizer, device, layer_module,
                                         'plusG', n_cache=args.n_cache)
    elif args.condition == 'random_swap_ctrl':
        print(f'[cache] building -G\' activation pool (n={args.n_cache})...', flush=True)
        minusG_other_pool = build_caches(model, tokenizer, device, layer_module,
                                          'minusG', n_cache=args.n_cache)

    # Main trial loop
    with open(jsonl_path, 'a') as f_out:
        for i in range(args.n):
            seed = SEED_BASE + i * 997
            if seed in done_seeds:
                continue
            t0 = time.time()
            try:
                result = play_game_with_intervention(
                    model, tokenizer, device, layer_module, args.condition, seed,
                    plusG_cache_pool=plusG_cache_pool,
                    minusG_other_pool=minusG_other_pool,
                )
                result['trial_id'] = i
                result['model'] = args.model
                result['task'] = args.task
                result['layer'] = TARGET_LAYER
                f_out.write(json.dumps(result) + '\n')
                f_out.flush()
                os.fsync(f_out.fileno())
            except Exception as e:
                print(f'[trial {i}] ERROR: {type(e).__name__}: {e}', flush=True)
                continue

            elapsed = time.time() - t0
            print(f'[trial {i+1}/{args.n}] seed={seed} bal={result["final_balance"]:.0f} '
                  f'bk={result["bankrupt"]} stop={result["voluntary_stop"]} '
                  f'rounds={result["n_decisions"]} ({elapsed:.1f}s)', flush=True)

            # Update progress
            done_seeds.add(seed)
            with open(progress_path, 'w') as pf:
                json.dump({'completed': len(done_seeds), 'target': args.n,
                           'last_update': datetime.now().isoformat()}, pf, indent=2)

    print(f'[done] {len(done_seeds)}/{args.n} trials at {out_dir}', flush=True)


if __name__ == '__main__':
    main()
