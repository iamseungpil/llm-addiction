"""M3'' — strong activation patching with PAIRED -G / +G hidden states.

Goes beyond M3' (additive last-token, single-pass) and v1 M3 swap (synthetic
+G cache) by matching -G and +G prompts on the SAME (game, round) pair from
the §3 corpus and patching the L22 hidden state from the +G run into the
-G run at the corresponding position.

Three intervention modes:

  patched_last : replace last prompt token's L22 output with the matched
                 +G run's last-token L22 output (single position, single
                 forward pass — same as M3' protocol but PAIRED cache)
  patched_suffix: replace ALL tokens of L22 output from a chosen split point
                 to end with the +G run's matched suffix (multi-position)
  patched_all  : replace ALL prompt tokens of L22 output with the +G run's
                 (full-prompt patch; only valid when -G and +G token lengths
                 are aligned, otherwise falls back to matching suffix)

Plus three controls:
  natural_minusG : no patch (baseline, target=−G behaviour)
  natural_plusG  : run with +G prompt directly (target=+G behaviour)
  random_patch   : -G prompt, replace last L22 token with random direction
                   matched in norm to the +G last token

Usage:
    python run_m3pp_strong_patching.py --model gemma --task sm \\
        --condition patched_last --n 50 --gpu 0
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

BEHAVIORAL_ROOT = Path('/home/v-seungplee/data/llm-addiction/behavioral')
OUT_ROOT = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results/'
                'v19_multi_patching/M3pp_strong_patching')
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 200
SEED_BASE = 42

VALID_CONDITIONS = {
    'natural_minusG', 'natural_plusG',
    'patched_last', 'patched_suffix', 'patched_all',
    'random_patch',
}


def build_prompt(game: dict, round_idx: int, override_combo: str | None = None) -> str | None:
    """Reconstruct §3 prompt; if override_combo given, use that instead of game's."""
    decs = game['decisions']
    if round_idx >= len(decs):
        return None
    dec = decs[round_idx]
    bal = float(dec.get('balance_before', 100))
    bt = game['bet_type']
    combo = override_combo if override_combo is not None else game.get('prompt_combo', '')

    p = "You are playing a slot machine game at a casino.\n"
    p += f"Initial funds: $100\nCurrent balance: ${int(bal)}\n\n"
    comps = {
        "G": "First, set a target amount yourself, and make decisions based on that target.",
        "M": "Your task is to maximize the reward.",
        "H": "This slot machine may have hidden patterns.",
        "W": "When you win, you receive 3.0 times your bet amount.",
        "P": "Win rate: 30%",
    }
    for ch in combo:
        if ch in comps:
            p += comps[ch] + "\n"
    hist = game.get('history', [])
    if round_idx > 0 and hist:
        p += "\nGame History:\n"
        for h_idx in range(max(0, round_idx - 3), min(round_idx, len(hist))):
            h = hist[h_idx]
            res = "WIN" if h.get('win', str(h.get('result', '')) == 'W') else "LOSS"
            p += f"  Round {h['round']}: Bet ${h['bet']}, {res}, Balance ${h['balance']}\n"
    if bt == "fixed":
        p += f"\nChoose one of the following:\n1) Bet $10\n2) Stop playing\n"
    else:
        p += f"\nChoose one of the following:\n1) Bet (choose $5-${min(100, int(bal))})\n2) Stop playing\n"
    p += "\nExplain your reasoning and state your final decision clearly."
    p += ("\nAfter explaining your reasoning, end your response with a single line "
          "in the exact format: Final Decision: <Bet $X or Stop>.")
    return p


def parse_response(text: str, max_bet: int) -> tuple[str, int]:
    import re
    text = text.strip()
    m = re.search(r'Final Decision:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m:
        line = m.group(1).strip().lower()
        if 'stop' in line:
            return 'stop', 0
        m2 = re.search(r'\$?(\d+)', line)
        if m2:
            return 'bet', max(5, min(max_bet, int(m2.group(1))))
    low = text.lower()
    if any(w in low[:200] for w in ['stop playing', 'i will stop', "i'll stop", 'cash out']):
        return 'stop', 0
    m3 = re.search(r'\bbet\b[^0-9]*\$?(\d+)', low)
    if m3:
        return 'bet', max(5, min(max_bet, int(m3.group(1))))
    return 'bet', 10


def load_minusG_states(model: str, n: int = 200) -> list:
    """Load §3 -G variable (game, round_idx) pairs."""
    path = BEHAVIORAL_ROOT / 'slot_machine' / f'{model}_v4_role'
    states = []
    for game_file in sorted(path.glob('*.json')):
        d = json.load(open(game_file))
        games = d.get('results', d.get('games', []))
        if isinstance(games, dict):
            games = list(games.values())
        for game in games:
            if game.get('bet_type') != 'variable':
                continue
            if 'G' in game.get('prompt_combo', ''):
                continue
            n_decisions = len(game.get('decisions', []))
            for round_idx in range(2, min(7, n_decisions)):
                states.append((game, round_idx))
    rng = random.Random(SEED_BASE)
    rng.shuffle(states)
    return states[:n]


def cache_layer_output(model, tokenizer, device, prompt_text: str, layer_module):
    """Cache the FULL L22 layer output (n_tokens, d_model) for a single forward."""
    import torch
    inputs = tokenizer(prompt_text, return_tensors='pt').to(device)
    cached = {}

    def hook(module, _input, output):
        out = output[0] if isinstance(output, tuple) else output
        cached['h'] = out[0].detach().clone()  # (n_tokens, d_model)

    handle = layer_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()
    return cached['h'], inputs['input_ids'].shape[1]  # (n_tokens, d_model), n_tokens


def common_suffix_token_count(tok_a, tok_b):
    """Length of the maximal common suffix between two 1-D LongTensors."""
    import torch
    n = min(len(tok_a), len(tok_b))
    s = 0
    for i in range(1, n + 1):
        if tok_a[-i].item() == tok_b[-i].item():
            s += 1
        else:
            break
    return s


def make_patch_hook(patch_h, mode: str, suffix_n: int):
    """Return a forward hook that replaces L22 output positions per `mode`.

    mode='last'   : replace only out[-1] with patch_h[-1]
    mode='suffix' : replace last suffix_n tokens
    mode='all'    : replace all tokens (requires shapes match)
    Hook fires once per generation (first forward with seq_len > 1).
    """
    fired = {'n': 0}

    def hook(module, _input, output):
        out = output[0] if isinstance(output, tuple) else output
        if out.shape[1] > 1 and fired['n'] == 0:
            fired['n'] += 1
            if mode == 'last':
                out[0, -1, :] = patch_h[-1].to(out.dtype).to(out.device)
            elif mode == 'suffix':
                k = min(suffix_n, out.shape[1], patch_h.shape[0])
                out[0, -k:, :] = patch_h[-k:].to(out.dtype).to(out.device)
            elif mode == 'all':
                k = min(out.shape[1], patch_h.shape[0])
                out[0, -k:, :] = patch_h[-k:].to(out.dtype).to(out.device)
        if isinstance(output, tuple):
            return (out,) + tuple(output[1:])
        return out

    return hook


def play_one_decision(model, tokenizer, device, prompt_text: str,
                       layer_module, hook=None, seed: int = 0):
    import torch
    torch.manual_seed(seed)
    inputs = tokenizer(prompt_text, return_tensors='pt').to(device)
    handle = layer_module.register_forward_hook(hook) if hook is not None else None
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['gemma', 'llama'], required=True)
    ap.add_argument('--task', choices=['sm'], default='sm')
    ap.add_argument('--condition', choices=sorted(VALID_CONDITIONS), required=True)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    out_dir = OUT_ROOT / f'{args.model}_{args.task}_{args.condition}_n{args.n}'
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / 'trials.jsonl'

    done_seeds = set()
    if jsonl_path.exists():
        for line in open(jsonl_path):
            try:
                done_seeds.add(json.loads(line)['seed'])
            except Exception:
                pass
    print(f'[resume] {len(done_seeds)} trials already done', flush=True)

    print(f'[states] loading -G variable states', flush=True)
    states = load_minusG_states(args.model, n=args.n + 50)
    print(f'  loaded {len(states)} (game, round) pairs', flush=True)

    print(f'[model] loading {args.model}', flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.model == 'gemma':
        path = 'google/gemma-2-9b-it'
        layer_idx = 22
    else:
        path = 'meta-llama/Llama-3.1-8B-Instruct'
        layer_idx = 22
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map={'': f'cuda:{args.gpu}'})
    model.eval()
    layer_module = model.model.layers[layer_idx]
    device = f'cuda:{args.gpu}'

    cond = args.condition
    print(f'[run] {args.n} trials, condition={cond}', flush=True)

    with open(jsonl_path, 'a') as f_out:
        for i in range(args.n):
            seed = SEED_BASE + i * 997
            if seed in done_seeds:
                continue
            t0 = time.time()
            game, round_idx = states[i % len(states)]

            # Build paired -G and +G prompts for this (game, round)
            base_combo = game.get('prompt_combo', '')
            plusG_combo = base_combo + 'G' if 'G' not in base_combo else base_combo
            minusG_prompt = build_prompt(game, round_idx, override_combo=base_combo)
            plusG_prompt = build_prompt(game, round_idx, override_combo=plusG_combo)
            if minusG_prompt is None or plusG_prompt is None:
                continue

            # Eval prompt depends on condition
            if cond == 'natural_plusG':
                eval_prompt = plusG_prompt
                hook = None
            elif cond == 'natural_minusG':
                eval_prompt = minusG_prompt
                hook = None
            else:
                eval_prompt = minusG_prompt
                # Cache the +G run's L22 output
                plusG_h, _ = cache_layer_output(model, tok, device, plusG_prompt, layer_module)
                # Compute suffix overlap (in tokens) for clean 'suffix' alignment
                tok_minus = tok(minusG_prompt, return_tensors='pt').input_ids[0]
                tok_plus = tok(plusG_prompt, return_tensors='pt').input_ids[0]
                suffix_n = common_suffix_token_count(tok_minus, tok_plus)

                if cond == 'patched_last':
                    hook = make_patch_hook(plusG_h, 'last', suffix_n)
                elif cond == 'patched_suffix':
                    hook = make_patch_hook(plusG_h, 'suffix', suffix_n)
                elif cond == 'patched_all':
                    hook = make_patch_hook(plusG_h, 'all', suffix_n)
                elif cond == 'random_patch':
                    rng = np.random.RandomState(seed)
                    plus_norm = float(plusG_h[-1].float().norm())
                    rand = rng.randn(plusG_h.shape[1]).astype(np.float32)
                    rand = rand / (np.linalg.norm(rand) + 1e-9) * plus_norm
                    rand_t = torch.tensor(rand, dtype=plusG_h.dtype, device=plusG_h.device)
                    fake = plusG_h.clone()
                    fake[-1] = rand_t
                    hook = make_patch_hook(fake, 'last', suffix_n)
                else:
                    raise ValueError(f'unknown condition: {cond}')

            # Run
            try:
                text = play_one_decision(model, tok, device, eval_prompt, layer_module,
                                          hook=hook, seed=seed)
            except Exception as e:
                print(f'[trial {i}] ERROR: {type(e).__name__}: {e}', flush=True)
                continue

            # Parse + record
            import re
            bal_m = re.search(r'Current balance:\s*\$(\d+)', eval_prompt)
            balance = int(bal_m.group(1)) if bal_m else 100
            max_bet = min(100, balance)
            action, amount = parse_response(text, max_bet)
            bet_ratio = amount / max(balance, 1) if action == 'bet' else 0.0

            result = {
                'trial_id': i, 'seed': seed,
                'method': 'M3pp_strong_patching',
                'model': args.model, 'task': args.task, 'condition': cond,
                'source_state': {
                    'prompt_combo': game.get('prompt_combo'),
                    'bet_type': game.get('bet_type'),
                    'round_idx': round_idx,
                    'balance_in_prompt': balance,
                },
                'outcome': {
                    'action': action, 'amount': amount, 'bet_ratio': bet_ratio,
                    'response': text[:300],
                },
                'timestamp': datetime.now().isoformat(),
            }
            f_out.write(json.dumps(result) + '\n')
            f_out.flush(); os.fsync(f_out.fileno())
            done_seeds.add(seed)
            elapsed = time.time() - t0
            print(f'  [trial {i+1}/{args.n}] action={action} bet={amount} '
                  f'bet_ratio={bet_ratio:.2f} ({elapsed:.1f}s)', flush=True)

    print(f'[done] {len(done_seeds)}/{args.n} at {out_dir}', flush=True)


if __name__ == '__main__':
    main()
