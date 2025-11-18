#!/usr/bin/env python3
"""
Experiment 1 CORRECTED: Extract SAE Features from L1-31
Uses the SAME setup as Experiment 2 (proven to work)
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime
from tqdm import tqdm
import torch
import gc
from statsmodels.stats.multitest import multipletests
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

# Add SAE loader path
sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect

class SAEFeatureExtractor:
    def __init__(self, gpu_id=0, layers_start=1, layers_end=31):
        """
        Initialize SAE feature extractor
        Uses same setup as Experiment 2 (proven working)
        """
        self.gpu_id = gpu_id
        self.device = f'cuda:{gpu_id}'
        self.layers_start = layers_start
        self.layers_end = layers_end

        self.model = None
        self.tokenizer = None
        self.sae_cache = {}

        print(f"🚀 SAE Feature Extractor Initialized")
        print(f"   GPU: {self.gpu_id}")
        print(f"   Layers: {layers_start}-{layers_end}")
        print(f"   Device: {self.device}")

    def load_models(self):
        """Load LLaMA model (same as Experiment 2)"""
        print(f"\n🔧 Loading models on GPU {self.gpu_id}...")

        torch.cuda.empty_cache()
        gc.collect()

        model_name = "meta-llama/Llama-3.1-8B"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={'': 0},  # After CUDA_VISIBLE_DEVICES, visible GPU is always cuda:0
            low_cpu_mem_usage=True,
            use_cache=False
        )
        self.model.eval()

        print("✅ LLaMA loaded successfully")
        print("🔧 SAEs will be loaded on-demand")

    def load_sae(self, layer):
        """Load SAE for specific layer (same as Experiment 2)"""
        if layer not in self.sae_cache:
            print(f"   Loading SAE Layer {layer}...")
            self.sae_cache[layer] = LlamaScopeDirect(layer=layer)
            torch.cuda.empty_cache()
        return self.sae_cache[layer]

    def extract_sae_features_single(self, prompt, layer):
        """Extract SAE features for a single prompt and layer"""
        sae = self.load_sae(layer)

        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states

            # Get layer hidden states (hidden_states[0] is embeddings, hidden_states[layer] is layer output)
            if layer < len(hidden_states):
                layer_hidden = hidden_states[layer][0, -1:, :]  # [1, 4096]
            else:
                layer_hidden = hidden_states[-1][0, -1:, :]

            # Encode with SAE
            sae_features = sae.encode(layer_hidden.float())  # [1, 32768]

            return sae_features[0].cpu().numpy()  # [32768]

    def load_experiments(self):
        """Load all 6400 experiments"""
        print("\n📂 Loading experiments...")

        results_dir = Path('/data/llm_addiction/results')
        main_file = results_dir / "exp1_multiround_intermediate_20250819_140040.json"
        missing_file = results_dir / "exp1_missing_complete_20250820_090040.json"

        all_experiments = []

        # Main file
        print(f"   Loading {main_file.name}...")
        with open(main_file, 'r') as f:
            data = json.load(f)
        experiments = data['results'] if isinstance(data, dict) and 'results' in data else data
        all_experiments.extend(experiments)
        print(f"   ✅ {len(experiments)} from main")
        del data, experiments
        gc.collect()

        # Missing file
        print(f"   Loading {missing_file.name}...")
        with open(missing_file, 'r') as f:
            data = json.load(f)
        experiments = data['results'] if isinstance(data, dict) and 'results' in data else data
        all_experiments.extend(experiments)
        print(f"   ✅ {len(experiments)} from missing")
        del data, experiments
        gc.collect()

        print(f"   ✅ Total: {len(all_experiments)} experiments")
        return all_experiments

    def process_layer_batch(self, experiments, target_layers):
        """Process a batch of layers"""
        print(f"\n{'='*80}")
        print(f"Processing Layers {target_layers[0]}-{target_layers[-1]}")
        print(f"{'='*80}")

        # Initialize storage
        layer_data = {
            layer: {'bankrupt': [], 'safe': []}
            for layer in target_layers
        }

        # Extract features
        for exp_idx, exp in enumerate(tqdm(experiments, desc=f"Extracting L{target_layers[0]}-L{target_layers[-1]}")):
            try:
                # Get group
                is_bankrupt = exp.get('is_bankrupt', False)

                # Get final round prompt
                round_features = exp.get('round_features', [])
                if not round_features:
                    continue

                final_round = round_features[-1]
                prompt = final_round.get('prompt', '')
                if not prompt:
                    continue

                # Extract SAE features for each layer
                for layer in target_layers:
                    sae_features = self.extract_sae_features_single(prompt, layer)

                    if is_bankrupt:
                        layer_data[layer]['bankrupt'].append(sae_features)
                    else:
                        layer_data[layer]['safe'].append(sae_features)

                # Memory cleanup
                if (exp_idx + 1) % 100 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                print(f"   ❌ Error on exp {exp_idx}: {e}")
                continue

        # Convert to numpy arrays
        for layer in target_layers:
            layer_data[layer]['bankrupt'] = np.array(layer_data[layer]['bankrupt'])
            layer_data[layer]['safe'] = np.array(layer_data[layer]['safe'])

            print(f"   Layer {layer}: Bankrupt={len(layer_data[layer]['bankrupt'])}, Safe={len(layer_data[layer]['safe'])}")

        return layer_data

    def analyze_layer(self, bankrupt_data, safe_data, layer, p_threshold=0.01, cohen_d_threshold=0.3):
        """Analyze single layer with statistical tests"""
        print(f"\n📊 Analyzing Layer {layer}...")
        print(f"   Bankrupt samples: {len(bankrupt_data)}")
        print(f"   Safe samples: {len(safe_data)}")
        print(f"   Features: {bankrupt_data.shape[1]}")

        n_features = bankrupt_data.shape[1]
        significant_features = []
        p_values = []
        cohen_d_values = []

        for feature_idx in tqdm(range(n_features), desc=f"Testing Layer {layer}"):
            bankrupt_vals = bankrupt_data[:, feature_idx]
            safe_vals = safe_data[:, feature_idx]

            # Skip if no variance
            if np.std(bankrupt_vals) == 0 and np.std(safe_vals) == 0:
                p_values.append(1.0)
                cohen_d_values.append(0.0)
                continue

            try:
                # t-test
                t_stat, p_value = stats.ttest_ind(bankrupt_vals, safe_vals)

                # Cohen's d
                pooled_std = np.sqrt(
                    ((len(bankrupt_vals) - 1) * np.var(bankrupt_vals, ddof=1) +
                     (len(safe_vals) - 1) * np.var(safe_vals, ddof=1)) /
                    (len(bankrupt_vals) + len(safe_vals) - 2)
                )

                cohen_d = (np.mean(bankrupt_vals) - np.mean(safe_vals)) / pooled_std if pooled_std > 0 else 0.0

                p_values.append(p_value)
                cohen_d_values.append(cohen_d)

                # Check significance
                if p_value < p_threshold and abs(cohen_d) > cohen_d_threshold:
                    significant_features.append({
                        'feature_idx': feature_idx,
                        'p_value': float(p_value),
                        'cohen_d': float(cohen_d),
                        'bankrupt_mean': float(np.mean(bankrupt_vals)),
                        'safe_mean': float(np.mean(safe_vals)),
                        'bankrupt_std': float(np.std(bankrupt_vals)),
                        'safe_std': float(np.std(safe_vals))
                    })

            except Exception as e:
                p_values.append(1.0)
                cohen_d_values.append(0.0)

        # FDR correction
        if p_values:
            rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')

            corrected_significant = []
            for feature in significant_features:
                feature_idx = feature['feature_idx']
                if p_corrected[feature_idx] < p_threshold:
                    feature['p_corrected'] = float(p_corrected[feature_idx])
                    corrected_significant.append(feature)

            significant_features = corrected_significant

        print(f"   ✅ {len(significant_features)} significant features (after FDR)")

        return {
            'layer': layer,
            'n_features': n_features,
            'n_bankrupt': len(bankrupt_data),
            'n_safe': len(safe_data),
            'significant_features': significant_features,
            'n_significant': len(significant_features)
        }

    def run(self):
        """Main execution"""
        print("\n" + "="*80)
        print("SAE FEATURE EXTRACTION L1-31")
        print("="*80)

        # Output directory
        output_dir = Path('/data/llm_addiction/experiment_1_L1_31_SAE_extraction')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load experiments
        experiments = self.load_experiments()

        # Load model
        self.load_models()

        # Process in batches
        layer_batches = [
            list(range(1, 11)),   # Batch 1: L1-L10
            list(range(11, 21)),  # Batch 2: L11-L20
            list(range(21, 32)),  # Batch 3: L21-L31
        ]

        all_results = {}

        for batch_num, target_layers in enumerate(layer_batches, 1):
            print(f"\n{'='*80}")
            print(f"BATCH {batch_num}/3: Layers {target_layers[0]}-{target_layers[-1]}")
            print(f"{'='*80}")

            # Extract features
            layer_data = self.process_layer_batch(experiments, target_layers)

            # Analyze each layer
            for layer in target_layers:
                result = self.analyze_layer(
                    layer_data[layer]['bankrupt'],
                    layer_data[layer]['safe'],
                    layer
                )
                all_results[layer] = result

            # Clean up
            del layer_data
            # Don't delete SAE cache yet - will reuse across batches
            torch.cuda.empty_cache()
            gc.collect()

            # Save intermediate
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = output_dir / f'L1_31_SAE_checkpoint_batch{batch_num}_{timestamp}.json'

            total_significant = sum(r['n_significant'] for r in all_results.values())

            checkpoint = {
                'timestamp': timestamp,
                'batch': batch_num,
                'feature_type': 'SAE_features_32768_per_layer',
                'sae_source': 'fnlp/Llama3_1-8B-Base-LXR-8x',
                'total_experiments': len(experiments),
                'layers_completed': sorted(all_results.keys()),
                'total_significant_features': total_significant,
                'layer_results': {str(k): v for k, v in all_results.items()}
            }

            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)

            print(f"\n💾 Checkpoint saved: {checkpoint_file}")
            print(f"   Total significant so far: {total_significant}")

        # Final save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = output_dir / f'L1_31_SAE_features_FINAL_{timestamp}.json'

        total_significant = sum(r['n_significant'] for r in all_results.values())

        final_data = {
            'timestamp': timestamp,
            'feature_type': 'SAE_features_32768_per_layer',
            'sae_source': 'fnlp/Llama3_1-8B-Base-LXR-8x',
            'total_experiments_processed': len(experiments),
            'layers_analyzed': sorted(all_results.keys()),
            'total_layers': len(all_results),
            'total_significant_features': total_significant,
            'significant_features_by_layer': {str(layer): result['n_significant'] for layer, result in all_results.items()},
            'layer_results': {str(k): v for k, v in all_results.items()}
        }

        with open(final_file, 'w') as f:
            json.dump(final_data, f, indent=2)

        print("\n" + "="*80)
        print("✅ EXTRACTION COMPLETE!")
        print("="*80)
        print(f"Layers analyzed: {len(all_results)}")
        print(f"Total significant SAE features: {total_significant}")
        print(f"Results saved: {final_file}")

        # Summary by layer
        print(f"\n📊 Significant SAE features by layer:")
        for layer in sorted(all_results.keys()):
            result = all_results[layer]
            print(f"   Layer {layer:2d}: {result['n_significant']:5d} features")

        return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (after CUDA_VISIBLE_DEVICES)')
    args = parser.parse_args()

    extractor = SAEFeatureExtractor(gpu_id=args.gpu)
    extractor.run()

if __name__ == '__main__':
    main()
