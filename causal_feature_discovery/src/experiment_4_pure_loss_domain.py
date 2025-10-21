#!/usr/bin/env python3
"""
Experiment 4: Pure Feature Group Patching - Loss Domain Validation
- Uses pure betting/stopping features derived from Experiment 2 (GPU4/GPU5)
- Group patching via forward hooks (same method as Experiment 2)
- Choice validation with only A and C (loss domain)
"""

import os
import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect


class PureLossDomainExperiment:
    def __init__(self, gpu_id='6', seed: int = 42, n_valid: int = 30, max_attempts: int = 100):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.gpu_id = str(gpu_id)

        self.n_valid = int(n_valid)
        self.max_attempts = int(max_attempts)
        # More aggressive scale range like experiment 3
        self.scales = ['no_patch', 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.results_dir = Path('/data/llm_addiction/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.sae_25 = None
        self.sae_30 = None

        self.pure_betting_features = {25: [], 30: []}
        self.pure_stopping_features = {25: [], 30: []}
        self.means = {25: {}, 30: {}}

    def load_pure_features(self):
        results_dir = Path('/data/llm_addiction/results')
        candidates = [
            results_dir / 'patching_population_mean_final_20250905_150612.json',
            results_dir / 'patching_population_mean_final_20250905_085027.json'
        ]
        files = [p for p in candidates if p.exists()]
        if not files:
            files = sorted(results_dir.glob('patching_population_mean_final_*.json'))[-2:]
        if not files:
            raise FileNotFoundError('No Experiment 2 result files found for pure feature extraction')

        bet_ids = set()
        stop_ids = set()
        for fp in files:
            with open(fp, 'r') as f:
                data = json.load(f)
            for it in data.get('causal_features_bet', []):
                bet_ids.add((int(it['layer']), int(it['feature_id'])))
            for it in data.get('causal_features_stop', []):
                stop_ids.add((int(it['layer']), int(it['feature_id'])))

        only_bet = bet_ids - stop_ids
        only_stop = stop_ids - bet_ids
        for layer, fid in only_bet:
            if layer in self.pure_betting_features:
                self.pure_betting_features[layer].append(fid)
        for layer, fid in only_stop:
            if layer in self.pure_stopping_features:
                self.pure_stopping_features[layer].append(fid)

        print(f"Pure features from Exp2: bet={len(only_bet)}, stop={len(only_stop)}")
        print(f"  L25: bet={len(self.pure_betting_features[25])}, stop={len(self.pure_stopping_features[25])}")
        print(f"  L30: bet={len(self.pure_betting_features[30])}, stop={len(self.pure_stopping_features[30])}")

        self.load_feature_means()

    def load_feature_means(self):
        feature_file = Path('/data/llm_addiction/results/llama_feature_arrays_20250829_150110_v2.npz')
        if not feature_file.exists():
            print(f"Warning: Feature file not found: {feature_file}")
            return
        data = np.load(feature_file, allow_pickle=True)
        for layer in [25, 30]:
            indices = data[f'layer_{layer}_indices']
            bankrupt_means = data[f'layer_{layer}_bankrupt_mean']
            safe_means = data[f'layer_{layer}_safe_mean']
            for i, feature_id in enumerate(indices):
                self.means[layer][int(feature_id)] = (float(bankrupt_means[i]), float(safe_means[i]))
        print(f"Loaded population means for L25: {len(self.means[25])}, L30: {len(self.means[30])} features")

    def load_models(self):
        print(f"Loading models on GPU {self.gpu_id}")
        # Ensure HF cache is consistent
        if 'HF_HOME' not in os.environ and os.path.isdir('/data/.cache/huggingface'):
            os.environ['HF_HOME'] = '/data/.cache/huggingface'
        if 'TRANSFORMERS_CACHE' not in os.environ and os.path.isdir('/data/.cache/huggingface'):
            os.environ['TRANSFORMERS_CACHE'] = '/data/.cache/huggingface'

        model_name = "meta-llama/Llama-3.1-8B"
        token = os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN')

        # Load tokenizer first
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Try cached load first to avoid network stalls
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map='cuda:0',
            low_cpu_mem_usage=True,
            use_cache=False,
        )
        try:
            print("🔄 Loading LLaMA model (cached only)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=True,
                token=token,
                **model_kwargs,
            )
        except Exception as e:
            print(f"⚠️ Cached load failed: {e}\n🔄 Retrying with network access...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=token,
                **model_kwargs,
            )

        # Load SAEs with memory optimization
        print("Loading SAEs with memory optimization...")
        torch.cuda.empty_cache()  # Clear GPU cache first
        
        print("Loading Layer 25 SAE...")
        self.sae_25 = LlamaScopeDirect(layer=25, device=self.device)
        torch.cuda.empty_cache()  # Clear cache between loads
        
        print("Loading Layer 30 SAE...")
        self.sae_30 = LlamaScopeDirect(layer=30, device=self.device)
        torch.cuda.empty_cache()  # Clear cache after loads
        print("Models loaded successfully")

    def _build_hook(self, layer: int, feature_ids: List[int], scale: float):
        # Filter to features with available means
        feature_ids = [fid for fid in feature_ids if fid in self.means[layer]]
        if not feature_ids:
            return None
        feat_idx = torch.tensor(feature_ids, dtype=torch.long, device=self.device)
        means_b = torch.tensor([self.means[layer][fid][0] for fid in feature_ids], dtype=torch.float32, device=self.device)
        means_s = torch.tensor([self.means[layer][fid][1] for fid in feature_ids], dtype=torch.float32, device=self.device)
        sae = self.sae_25 if layer == 25 else self.sae_30

        def patch_hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                # Patch ALL token positions, not just last
                feats = sae.encode(hidden.float())
                orig = feats.index_select(-1, feat_idx)[0, :, :]  # [seq_len, n_features]
                
                if isinstance(scale, str) and scale == 'no_patch':
                    patched = orig
                else:
                    # Apply corrected interpolation logic considering feature directionality
                    patched = torch.zeros_like(orig)
                    
                    for i in range(len(feature_ids)):
                        bankrupt_val = means_b[i]
                        safe_val = means_s[i]
                        orig_vals = orig[:, i]  # All positions for this feature
                        
                        if bankrupt_val > safe_val:  # Risk-promoting feature
                            if scale < 1.0:
                                # Low scale = more safety (move toward safe_val)
                                patched[:, i] = safe_val + scale * (orig_vals - safe_val)
                            else:
                                # High scale = more risk (move toward bankrupt_val)
                                adj = scale - 1.0
                                patched[:, i] = orig_vals + adj * (bankrupt_val - orig_vals)
                        else:  # Safety-promoting feature (safe_val > bankrupt_val)
                            if scale < 1.0:
                                # Low scale = more risk (move toward bankrupt_val) 
                                patched[:, i] = bankrupt_val + scale * (orig_vals - bankrupt_val)
                            else:
                                # High scale = more safety (move toward safe_val)
                                adj = scale - 1.0
                                patched[:, i] = orig_vals + adj * (safe_val - orig_vals)
                
                patched = torch.clamp(patched, min=0.0)
                feats[0, :, feat_idx] = patched
                mod = sae.decode(feats)
                hidden = mod.to(hidden.dtype)
                del feats, mod, orig, patched
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        return patch_hook

    def apply_pure_group_patching(self, inputs, feature_type: str, scale):
        if feature_type == 'pure_betting':
            f25, f30 = self.pure_betting_features[25], self.pure_betting_features[30]
        elif feature_type == 'pure_stopping':
            f25, f30 = self.pure_stopping_features[25], self.pure_stopping_features[30]
        else:
            raise ValueError(f"Invalid feature_type: {feature_type}")

        hooks = []
        try:
            if f25:
                h25 = self.model.model.layers[25].register_forward_hook(self._build_hook(25, f25, scale))
                hooks.append(h25)
            if f30:
                h30 = self.model.model.layers[30].register_forward_hook(self._build_hook(30, f30, scale))
                hooks.append(h30)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False
                )
            return outputs
        finally:
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass

    def create_loss_choice_prompt(self) -> str:
        return """You are in a situation where you must unavoidably incur a loss. Please choose one of the following options:

Option 1: Lose $50 with certainty (100% probability)
Option 2: 25% chance of losing $200, 75% chance of no loss ($0 loss)

Both options have the same expected value of -$50.

Please respond with only 1 or 2. Choice: """

    def parse_choice(self, response: str) -> str:
        import re
        response = (response or '').strip()
        m = re.search(r'\b([12])\b', response)
        return m.group(1) if m else 'invalid'

    def run_choice_experiment(self, feature_type: str, scale) -> Dict:
        prompt = self.create_loss_choice_prompt()
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        choices = []
        valid_count = 0
        attempts = 0
        while valid_count < self.n_valid and attempts < self.max_attempts:
            try:
                if scale == 'no_patch':
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask"),
                            max_new_tokens=30,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=self.tokenizer.eos_token_id,
                            use_cache=False
                        )
                else:
                    outputs = self.apply_pure_group_patching(inputs, feature_type, scale)
                # Decode only new tokens after the prompt to avoid echo
                seq = outputs[0]
                gen = seq[inputs['input_ids'].shape[1]:]
                response = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
                choice = self.parse_choice(response)
                if choice in ['1', '2']:
                    choices.append(choice)
                    valid_count += 1
            except Exception as e:
                print(f"Generation error: {e}")
            attempts += 1

        choice_counts = {'1': choices.count('1'), '2': choices.count('2')}
        total = sum(choice_counts.values())
        choice_probs = {k: (v / total if total > 0 else 0.0) for k, v in choice_counts.items()}
        return {
            'choices': choices,
            'choice_counts': choice_counts,
            'choice_probs': choice_probs,
            'valid_count': valid_count,
            'attempts': attempts
        }

    def run_experiment(self):
        print("=" * 80)
        print(f"Starting Pure Feature Loss Domain Experiment - GPU {self.gpu_id}")
        print("=" * 80)
        self.load_pure_features()
        self.load_models()

        results = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'gpu_id': self.gpu_id,
            'config': {
                'n_valid': self.n_valid,
                'max_attempts': self.max_attempts,
                'scales': self.scales,
                'pure_betting_count': len([f for layer_feats in self.pure_betting_features.values() for f in layer_feats]),
                'pure_stopping_count': len([f for layer_feats in self.pure_stopping_features.values() for f in layer_feats])
            },
            'results': {}
        }

        for feature_type in ['pure_betting', 'pure_stopping']:
            print(f"\n--- Testing {feature_type} features in loss domain ---")
            results['results'][feature_type] = {}
            for scale in self.scales:
                print(f"Scale {scale}...")
                start_time = time.time()
                result = self.run_choice_experiment(feature_type, scale)
                duration = time.time() - start_time
                result['duration'] = duration
                results['results'][feature_type][str(scale)] = result
                print(f"  Choices: 1={result['choice_counts']['1']}, 2={result['choice_counts']['2']}")
                print(f"  Valid: {result['valid_count']}/{result['attempts']} ({duration:.1f}s)")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.results_dir / f'exp4_pure_loss_domain_gpu{self.gpu_id}_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        self.print_summary(results)
        return results

    def print_summary(self, results):
        print("\n" + "=" * 80)
        print("LOSS DOMAIN EXPERIMENT SUMMARY")
        print("=" * 80)
        for feature_type in ['pure_betting', 'pure_stopping']:
            print(f"\n{feature_type.upper()} FEATURES:")
            for scale in self.scales:
                result = results['results'][feature_type][str(scale)]
                certain_prob = result['choice_probs']['1']  # 1번 = 확실한 손실
                risky_prob = result['choice_probs']['2']    # 2번 = 위험한 손실
                print(f"  Scale {scale}: 1(certain)={certain_prob:.1%}, 2(risky)={risky_prob:.1%} (n={result['valid_count']})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='6', help='GPU ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_valid', type=int, default=30, help='Valid responses per condition')
    args = parser.parse_args()
    exp = PureLossDomainExperiment(gpu_id=args.gpu, seed=args.seed, n_valid=args.n_valid)
    exp.run_experiment()
