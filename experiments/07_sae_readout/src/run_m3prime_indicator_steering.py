"""M3': §4.1 indicator-direction steering experiment.

Tests whether the §4.1 Ridge readout direction (predictor) also CONTROLS
behaviour at L22, by additively steering the residual stream along it.

Pipeline (per trial):
  1. Sample a random §3 -G + variable game state from existing corpus
  2. Build prompt using EXACT §3 build_prompt() (extract_all_layers_dp.py)
     — same format as §4 SAE feature extraction
  3. Forward pass with steering hook adding α·σ·d_unit to L22 last token
  4. Parse decision, simulate one round outcome, log to JSONL

Conditions:
  alpha-2  : direction=I_BA, α=-2σ, layer=22
  alpha-1  : direction=I_BA, α=-1σ, layer=22
  alpha0   : no patch                     (baseline)
  alpha+1  : direction=I_BA, α=+1σ, layer=22
  alpha+2  : direction=I_BA, α=+2σ, layer=22
  alpha+3  : direction=I_BA, α=+3σ, layer=22
  random   : random Gaussian, norm-matched, α=+2σ, layer=22  (specificity)
  L8       : direction=I_BA, α=+2σ, layer=8                  (layer specificity)
  ILC      : direction=I_LC, α=+2σ, layer=22                 (indicator specificity)

Output: results/v19_multi_patching/M3prime_indicator_steering/{cell}/trials.jsonl

Usage:
    python run_m3prime_indicator_steering.py \
        --model gemma --task sm \
        --condition alpha+2 --alpha 2.0 --direction i_ba --layer 22 \
        --n 50 --gpu 0
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

# Paths
DIRECTION_DIR = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results/'
                     'v19_multi_patching/M3prime_indicator_steering/direction_metadata')
OUT_ROOT = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results/'
                'v19_multi_patching/M3prime_indicator_steering')
BEHAVIORAL_ROOT = Path('/home/v-seungplee/data/llm-addiction/behavioral')

TEMPERATURE = 0.7
MAX_NEW_TOKENS = 200
SEED_BASE = 42

# §3 PROMPT BUILDER — copied from extract_all_layers_dp.py for exact consistency
def build_prompt_section3(game: dict, round_idx: int) -> str | None:
    """Reconstruct §3 prompt for a specific round (matches §4 extraction)."""
    decs = game['decisions']
    if round_idx >= len(decs):
        return None
    dec = decs[round_idx]
    bal = float(dec.get('balance_before', 100))
    bt = game['bet_type']
    combo = game['prompt_combo']

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


def parse_section3_response(text: str, max_bet: int) -> tuple[str, int]:
    """Parse §3 response: looks for 'Final Decision: <Bet $X or Stop>'."""
    import re
    text = text.strip()
    # Look for Final Decision line first
    m = re.search(r'Final Decision:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m:
        line = m.group(1).strip().lower()
        if 'stop' in line:
            return 'stop', 0
        m2 = re.search(r'\$?(\d+)', line)
        if m2:
            return 'bet', max(5, min(max_bet, int(m2.group(1))))
    # Fallback: search whole text
    low = text.lower()
    if any(w in low[:200] for w in ['stop playing', 'i will stop', "i'll stop", 'cash out']):
        return 'stop', 0
    m3 = re.search(r'\bbet\b[^0-9]*\$?(\d+)', low)
    if m3:
        return 'bet', max(5, min(max_bet, int(m3.group(1))))
    return 'bet', 10


def load_section3_minusG_variable_states(model: str, task: str = 'sm', n: int = 200) -> list:
    """Load §3 -G variable game states for steering experiment.

    Returns list of (game, round_idx) tuples.
    Filter: bet_type='variable', prompt_combo doesn't contain 'G'.
    Sample uniformly across all such (game, round) pairs.
    """
    if task == 'sm':
        path = BEHAVIORAL_ROOT / 'slot_machine' / f'{model}_v4_role'
    elif task == 'ic':
        path = BEHAVIORAL_ROOT / 'investment_choice' / f'v2_role_{model}'
    else:
        raise ValueError(f'unsupported task: {task}')

    states = []
    for game_file in sorted(path.glob('*.json')):
        d = json.load(open(game_file))
        if 'results' in d:
            results = d['results']
            games = list(results.values()) if isinstance(results, dict) else results
        else:
            games = d if isinstance(d, list) else [d]

        for game in games:
            if game.get('bet_type') != 'variable':
                continue
            if 'G' in game.get('prompt_combo', ''):
                continue
            n_decisions = len(game.get('decisions', []))
            # Sample mid-game (rounds 2-7) for steering test
            for round_idx in range(2, min(7, n_decisions)):
                states.append((game, round_idx))

    rng = random.Random(SEED_BASE)
    rng.shuffle(states)
    return states[:n]


def make_steering_hook(d_unit_tensor, alpha_sigma_value):
    """Create forward hook that adds α·σ·d_unit to last input token at first pass only."""
    fire_count = {'n': 0}

    def hook(module, _input, output):
        out = output[0] if isinstance(output, tuple) else output
        if out.shape[1] > 1 and fire_count['n'] == 0:
            out[:, -1, :] += alpha_sigma_value * d_unit_tensor
            fire_count['n'] += 1
        return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out
    return hook


def play_one_decision(model, tokenizer, device, prompt_text: str,
                      hook_fn=None, layer_module=None,
                      seed: int = 42) -> tuple[str, str, int, float, int]:
    """Run model on prompt with optional hook. Returns (raw_text, action, amount,
    bet_ratio, balance_in_prompt)."""
    import torch
    torch.manual_seed(seed)
    inputs = tokenizer(prompt_text, return_tensors='pt').to(device)

    handle = layer_module.register_forward_hook(hook_fn) if hook_fn is not None else None

    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 temperature=TEMPERATURE, do_sample=True,
                                 pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    finally:
        if handle is not None:
            handle.remove()

    # Extract balance from prompt to compute bet_ratio
    import re
    bal_m = re.search(r'Current balance:\s*\$(\d+)', prompt_text)
    balance = int(bal_m.group(1)) if bal_m else 100
    max_bet = min(100, balance)
    action, amount = parse_section3_response(text, max_bet)
    bet_ratio = amount / max(balance, 1) if action == 'bet' else 0.0
    return text, action, amount, bet_ratio, balance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['gemma', 'llama'], required=True)
    ap.add_argument('--task', choices=['sm'], default='sm')
    ap.add_argument('--condition', required=True,
                    help='label like alpha-2 / alpha+0 / random / L8 / ILC')
    ap.add_argument('--alpha', type=float, required=True,
                    help='steering coefficient in σ units (0 = no steering)')
    ap.add_argument('--direction', choices=['i_ba', 'i_lc', 'random'], default='i_ba')
    ap.add_argument('--layer', type=int, default=22)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    out_dir = OUT_ROOT / f'{args.model}_{args.task}_{args.condition}_n{args.n}'
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / 'trials.jsonl'
    progress_path = out_dir / 'progress.json'

    # Resume support
    done_seeds = set()
    if jsonl_path.exists():
        for line in open(jsonl_path):
            try:
                done_seeds.add(json.loads(line)['seed'])
            except Exception:
                pass
    print(f'[resume] {len(done_seeds)} trials already done', flush=True)

    # Load §3 -G variable states
    print(f'[states] loading §3 -G variable states', flush=True)
    states = load_section3_minusG_variable_states(args.model, args.task, n=args.n + 50)
    print(f'  loaded {len(states)} (game, round) pairs', flush=True)

    # Load steering direction (if not baseline α=0 with no direction)
    import torch
    use_direction = (args.alpha != 0.0 or args.direction == 'random')
    d_unit = None
    sigma = None
    if use_direction:
        if args.direction == 'random':
            # Random Gaussian, norm matched to I_BA d_pre_norm
            ref_meta = json.load(open(DIRECTION_DIR / f'{args.model}_{args.task}_i_ba_L22_steering.json'))
            d_dim = len(ref_meta['d_unit'])
            rng = np.random.RandomState(42 + args.layer * 1000)
            d_unit_np = rng.randn(d_dim)
            d_unit_np = d_unit_np / np.linalg.norm(d_unit_np)
            sigma = ref_meta['sigma_baseline']
        else:
            steering_file = DIRECTION_DIR / f'{args.model}_{args.task}_{args.direction}_L22_steering.json'
            sm = json.load(open(steering_file))
            d_unit_np = np.array(sm['d_unit'])  # always L22 d_unit
            sigma = sm['sigma_baseline']
        d_unit = torch.tensor(d_unit_np, dtype=torch.bfloat16, device=f'cuda:{args.gpu}')

    # Load model
    print(f'[model] loading {args.model}', flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.model == 'gemma':
        path = 'google/gemma-2-9b-it'
    else:
        path = 'meta-llama/Llama-3.1-8B-Instruct'
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map={'': f'cuda:{args.gpu}'})
    model.eval()
    layer_module = model.model.layers[args.layer]

    # Trial loop
    print(f'[run] {args.n} trials at α={args.alpha}σ, direction={args.direction}, layer={args.layer}',
          flush=True)
    with open(jsonl_path, 'a') as f_out:
        for i in range(args.n):
            seed = SEED_BASE + i * 997
            if seed in done_seeds:
                continue
            t0 = time.time()
            game, round_idx = states[i % len(states)]
            prompt_text = build_prompt_section3(game, round_idx)
            if prompt_text is None:
                continue

            hook_fn = (make_steering_hook(d_unit, args.alpha * sigma)
                       if use_direction else None)
            try:
                text, action, amount, bet_ratio, balance_in_prompt = play_one_decision(
                    model, tok, f'cuda:{args.gpu}', prompt_text, hook_fn, layer_module,
                    seed=seed)
            except Exception as e:
                print(f'[trial {i}] ERROR: {type(e).__name__}: {e}', flush=True)
                continue

            result = {
                'trial_id': i,
                'seed': seed,
                'method': 'M3prime_steering',
                'model': args.model,
                'task': args.task,
                'condition': args.condition,
                'intervention': {
                    'alpha_sigma': args.alpha,
                    'direction': args.direction,
                    'layer': args.layer,
                    'sigma_used': sigma if use_direction else None,
                },
                'source_state': {
                    'prompt_combo': game.get('prompt_combo'),
                    'bet_type': game.get('bet_type'),
                    'round_idx': round_idx,
                    'balance_in_prompt': balance_in_prompt,
                },
                'outcome': {
                    'action': action,
                    'amount': amount,
                    'bet_ratio': bet_ratio,
                    'response': text[:300],
                },
                'timestamp': datetime.now().isoformat(),
            }
            f_out.write(json.dumps(result) + '\n')
            f_out.flush()
            os.fsync(f_out.fileno())

            done_seeds.add(seed)
            with open(progress_path, 'w') as pf:
                json.dump({'completed': len(done_seeds), 'target': args.n,
                           'last_update': datetime.now().isoformat()}, pf, indent=2)

            elapsed = time.time() - t0
            print(f'  [trial {i+1}/{args.n}] action={action} bet={amount} bet_ratio={bet_ratio:.2f} '
                  f'({elapsed:.1f}s)', flush=True)

    print(f'[done] {len(done_seeds)}/{args.n} at {out_dir}', flush=True)


if __name__ == '__main__':
    main()
