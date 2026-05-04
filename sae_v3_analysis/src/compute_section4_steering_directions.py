"""Compute steering direction (4096-dim) from §4.1 Ridge weights.

Pipeline:
  1. Load direction_metadata JSON (Ridge w + feature indices + scaler params)
  2. Load SAE decoder for the matching layer (Gemma-Scope or Llama-Scope)
  3. Project w via SAE decoder cols → d_4096
  4. Compute σ from baseline projection distribution
  5. Save direction + sigma to {cell}_steering.json

Usage:
    python compute_section4_steering_directions.py --model gemma --task sm --indicator i_ba

Notes:
  - For Gemma: feature_indices in direction_metadata are in active_subset (489)
    space; chain through active_feature_subset to get 131072 SAE space.
  - Ridge was fit on STANDARDIZED features, so steering must use w/scaler_scale.
  - Direction is normalized to unit norm; α applied as multiple of σ.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis')
DIRECTION_DIR = ROOT / 'results/v19_multi_patching/M3prime_indicator_steering/direction_metadata'

# Gemma-Scope L22 SAE: BASE pretrained (canonical IT release lacks L22)
# Note: §4 features were extracted from gemma-2-9b-IT, but pt-res SAE works
# as a residual-stream decomposition since architecture is shared.
GEMMA_SAE_RELEASE = 'gemma-scope-9b-pt-res-canonical'
LLAMA_SAE_RELEASE = 'llama_scope_lxr_8x'


def load_sae_decoder(model: str, layer: int):
    """Load SAE decoder weights W_dec of shape (n_features, d_model).

    Gemma: 131072 features × 3584 d_model (gemma-2-9b)
    LLaMA: 4096 d_model (llama-3.1-8b)
    """
    from sae_lens import SAE
    if model == 'gemma':
        sae = SAE.from_pretrained(
            release=GEMMA_SAE_RELEASE,
            sae_id=f'layer_{layer}/width_131k/canonical',
        )
    elif model == 'llama':
        sae = SAE.from_pretrained(
            release=LLAMA_SAE_RELEASE,
            sae_id=f'l{layer}r_8x',
        )
    else:
        raise ValueError(f'unknown model: {model}')
    W_dec = sae.W_dec.detach().cpu().float().numpy()
    return W_dec, sae


def compute_baseline_sigma(d_unit: np.ndarray, model: str, task: str, layer: int,
                           n_baseline: int = 200) -> float:
    """Compute σ = std(h · d_unit) across baseline rounds.

    Uses sae_features_v3/{task_dir}/{model}/hidden_states_dp.npz
    Filter to variable-betting baseline rounds (no intervention condition).
    """
    task_dirs = {'sm': 'slot_machine', 'ic': 'investment_choice', 'mw': 'mystery_wheel'}
    npz_path = (Path('/home/v-seungplee/data/llm-addiction/sae_features_v3')
                / task_dirs[task] / model / 'hidden_states_dp.npz')
    d = np.load(npz_path, allow_pickle=False)
    layers = list(d['layers'])
    li = layers.index(layer)
    H = d['hidden_states'][:, li, :]  # (n_rounds, d_model)

    # Use first n_baseline rounds (random subset)
    n = min(n_baseline, H.shape[0])
    idx = np.random.RandomState(42).choice(H.shape[0], size=n, replace=False)
    h_sample = H[idx]  # (n, 4096)

    projection = h_sample @ d_unit  # (n,)
    return float(projection.std()), float(projection.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['gemma', 'llama'], required=True)
    ap.add_argument('--task', choices=['sm', 'ic', 'mw'], required=True)
    ap.add_argument('--indicator', choices=['i_lc', 'i_ba', 'i_ec'], required=True)
    ap.add_argument('--layer', type=int, default=22)
    args = ap.parse_args()

    cell = f'{args.model}_{args.task}_{args.indicator}_L{args.layer}'
    meta_path = DIRECTION_DIR / f'{cell}.json'
    if not meta_path.exists():
        print(f'  metadata not found: {meta_path}; run extract_section4_ridge_weights.py first',
              file=sys.stderr)
        sys.exit(1)

    print(f'[load] {meta_path}')
    meta = json.load(open(meta_path))
    w_200 = np.array(meta['ridge_coef'])                          # (200,)
    feature_indices_active = np.array(meta['feature_indices'])    # into active subset (489)
    active_subset = np.array(meta['active_feature_subset'])       # active subset → full SAE space
    scaler_mean = np.array(meta['scaler_mean'])                   # (200,)
    scaler_scale = np.array(meta['scaler_scale'])                 # (200,)

    # Map active-subset indices back to full 131072-dim SAE space
    feature_indices_full = active_subset[feature_indices_active]  # (200,) into 131072

    print(f'  w shape: {w_200.shape}, scaler_scale shape: {scaler_scale.shape}')
    print(f'  feature_indices_full range: {feature_indices_full.min()}..{feature_indices_full.max()}')

    print(f'[load SAE decoder] {args.model} L{args.layer}')
    W_dec, sae = load_sae_decoder(args.model, args.layer)
    print(f'  W_dec shape: {W_dec.shape}')

    # Project w through SAE decoder
    # w is in standardized feature space; un-standardize then project.
    # h ↦ z = (φ(h)[idx] - mean) / scale ↦ pred = w · z
    # Inverse: contribution to h = w / scale @ W_dec[idx]  (decoder cols at our indices)
    w_unstd = w_200 / scaler_scale                                 # (200,)
    selected_dec_cols = W_dec[feature_indices_full]                # (200, 4096)
    d_pre_norm = w_unstd @ selected_dec_cols                       # (4096,)
    norm = np.linalg.norm(d_pre_norm)
    d_unit = d_pre_norm / norm                                     # (4096,)

    print(f'  d_pre_norm: norm={norm:.4f}')
    print(f'  d_unit: norm={np.linalg.norm(d_unit):.6f} (should be ~1)')

    # Compute σ from baseline projection
    print(f'[sigma] computing from baseline rounds')
    sigma, mu = compute_baseline_sigma(d_unit, args.model, args.task, args.layer)
    print(f'  baseline projection: μ={mu:.4f}, σ={sigma:.4f}')

    # Save
    out_path = DIRECTION_DIR / f'{cell}_steering.json'
    out = {
        'model': args.model,
        'task': args.task,
        'indicator': args.indicator,
        'layer': args.layer,
        'feature_indices_full': feature_indices_full.tolist(),
        'd_unit': d_unit.tolist(),       # (4096,) unit vector
        'd_norm_pre_unit': float(norm),  # for reproducibility
        'sigma_baseline': sigma,
        'mu_baseline': mu,
        'sae_release': GEMMA_SAE_RELEASE if args.model == 'gemma' else LLAMA_SAE_RELEASE,
        'sae_id_layer': args.layer,
        'source_metadata': str(meta_path.name),
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[save] {out_path}')


if __name__ == '__main__':
    main()
