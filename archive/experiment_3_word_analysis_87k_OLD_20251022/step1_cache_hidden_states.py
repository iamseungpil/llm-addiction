#!/usr/bin/env python3
"""
Experiment 3 Step 1: Cache Hidden States
Extract and cache hidden states for all 6,400 responses once
This avoids redundant forward passes (556M → 198K)
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

class HiddenStateCache:
    """Extract and cache hidden states for all responses"""

    def __init__(self, gpu_id: int = 0, layer_range=(1, 31)):
        self.device = f'cuda:{gpu_id}'
        self.layer_start, self.layer_end = layer_range

        self.cache_dir = Path('/data/llm_addiction/experiment_3_hidden_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"🎯 Caching hidden states for layers {self.layer_start}-{self.layer_end}")

    def load_model(self):
        """Load LLaMA model"""
        print("🚀 Loading LLaMA model...")

        model_name = "meta-llama/Llama-3.1-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={'': self.device},
            low_cpu_mem_usage=True,
            use_cache=False
        )
        self.model.eval()

        print("✅ LLaMA loaded")

    def load_exp1_data(self):
        """Load 6,400 Exp1 responses"""
        print("📂 Loading Exp1 data (6,400 games)...")

        file1 = '/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json'
        file2 = '/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json'

        with open(file1, 'r') as f:
            data1 = json.load(f)

        with open(file2, 'r') as f:
            data2 = json.load(f)

        exp1_responses = []

        for entry in data1.get('results', []):
            if 'round_features' in entry and entry['round_features']:
                last_round = entry['round_features'][-1]
                exp1_responses.append({
                    'prompt': last_round.get('prompt', ''),
                    'response': last_round.get('response', ''),
                    'is_bankrupt': entry.get('is_bankrupt', False),
                    'voluntary_stop': entry.get('voluntary_stop', False),
                })

        for entry in data2.get('results', []):
            if 'round_features' in entry and entry['round_features']:
                last_round = entry['round_features'][-1]
                exp1_responses.append({
                    'prompt': last_round.get('prompt', ''),
                    'response': last_round.get('response', ''),
                    'is_bankrupt': entry.get('is_bankrupt', False),
                    'voluntary_stop': entry.get('voluntary_stop', False),
                })

        print(f"✅ Loaded {len(exp1_responses)} responses")
        return exp1_responses

    def extract_and_cache(self, responses: list, batch_size: int = 32):
        """Extract hidden states and cache by layer"""

        print(f"\n{'='*80}")
        print("EXTRACTING HIDDEN STATES")
        print(f"{'='*80}")

        # Create layer-specific cache files
        layer_caches = {}
        for layer in range(self.layer_start, self.layer_end + 1):
            layer_caches[layer] = {
                'hidden_states': [],
                'response_ids': []
            }

        n_batches = (len(responses) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(responses))
            batch_responses = responses[start_idx:end_idx]

            # Extract prompts
            prompts = [r['prompt'] for r in batch_responses if r['prompt']]
            if not prompts:
                continue

            # Tokenize batch
            inputs = self.tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )

            # Extract hidden states for each layer
            hidden_states = outputs.hidden_states

            for i, prompt_idx in enumerate(range(start_idx, end_idx)):
                if not responses[prompt_idx]['prompt']:
                    continue

                # Get last token hidden state for each layer
                for layer in range(self.layer_start, self.layer_end + 1):
                    # Get hidden state: (batch, seq_len, hidden_dim)
                    layer_hidden = hidden_states[layer][i, -1, :].float().cpu().numpy()

                    layer_caches[layer]['hidden_states'].append(layer_hidden)
                    layer_caches[layer]['response_ids'].append(prompt_idx)

            # Clear GPU memory
            del outputs, hidden_states, inputs
            torch.cuda.empty_cache()

            # Save checkpoint every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"\n  Processed {end_idx}/{len(responses)} responses")

        # Save cached hidden states to disk (NPZ format)
        print(f"\n💾 Saving cached hidden states...")

        for layer in range(self.layer_start, self.layer_end + 1):
            cache_file = self.cache_dir / f'layer_{layer}_hidden_states.npz'

            np.savez_compressed(
                cache_file,
                hidden_states=np.array(layer_caches[layer]['hidden_states']),
                response_ids=np.array(layer_caches[layer]['response_ids'])
            )

            size_mb = cache_file.stat().st_size / 1024 / 1024
            print(f"  Layer {layer}: {size_mb:.1f} MB ({len(layer_caches[layer]['hidden_states'])} responses)")

        # Save metadata
        metadata = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'n_responses': len(responses),
            'layers': list(range(self.layer_start, self.layer_end + 1)),
            'model': 'meta-llama/Llama-3.1-8B'
        }

        with open(self.cache_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Caching complete!")
        print(f"   Cache directory: {self.cache_dir}")

    def run(self):
        """Main caching workflow"""
        print("="*80)
        print("EXPERIMENT 3 STEP 1: HIDDEN STATE CACHING")
        print("="*80)

        self.load_model()
        responses = self.load_exp1_data()
        self.extract_and_cache(responses, batch_size=32)

        print("\n" + "="*80)
        print("✅ STEP 1 COMPLETE - Ready for Step 2!")
        print("="*80)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--layer-start', type=int, default=1)
    parser.add_argument('--layer-end', type=int, default=31)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    cache = HiddenStateCache(
        gpu_id=0,
        layer_range=(args.layer_start, args.layer_end)
    )
    cache.run()
