#!/usr/bin/env python3
"""
Analyze pathway tracking for full 6400 experiments
Extract L1-31 feature activations for all gambling decisions
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM

class PathwayAnalyzer6400:
    def __init__(self):
        self.device = 'cuda:0'
        self.results_dir = Path('/data/llm_addiction/experiment_1_pathway_L1_31')

        # Load 6400 experiments
        print("📂 Loading 6400 experiments...")
        exp1_main = json.load(open('/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json'))
        exp1_add = json.load(open('/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json'))

        self.experiments = []
        if isinstance(exp1_main, dict) and 'results' in exp1_main:
            self.experiments.extend(exp1_main['results'])
        elif isinstance(exp1_main, list):
            self.experiments.extend(exp1_main)

        if isinstance(exp1_add, dict) and 'results' in exp1_add:
            self.experiments.extend(exp1_add['results'])
        elif isinstance(exp1_add, list):
            self.experiments.extend(exp1_add)

        print(f"✅ Loaded {len(self.experiments)} experiments")

        # Load models
        print("🚀 Loading LLaMA model...")
        model_name = "meta-llama/Llama-3.1-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={'': 0}
        )
        self.model.eval()
        print("✅ LLaMA loaded")

        # Load SAEs for L1-31
        print("🚀 Loading SAEs for L1-31...")
        self.saes = {}
        for layer in tqdm(range(1, 32), desc="Loading SAEs"):
            self.saes[layer] = LlamaScopeDirect(layer=layer, device=self.device)
        print("✅ All SAEs loaded")

    def extract_features_for_prompt(self, prompt: str) -> dict:
        """Extract L1-31 features for a single prompt"""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        # Extract last token features for all layers
        features = {}
        for layer in range(1, 32):
            hidden = hidden_states[layer][:, -1, :]  # Last token
            sae_output = self.saes[layer].encode(hidden)
            feature_acts = sae_output['feature_acts'][0].cpu().numpy()
            features[f'L{layer}'] = feature_acts.tolist()

        return features

    def analyze_all_experiments(self):
        """Analyze all 6400 experiments"""
        print(f"\n🧪 Analyzing {len(self.experiments)} experiments...")

        results = []

        for exp_idx, exp in enumerate(tqdm(self.experiments, desc="Processing experiments")):
            # Get final decision prompt
            prompt = exp.get('prompt', '')
            response = exp.get('response', '')
            outcome = exp.get('outcome', '')

            if not prompt:
                continue

            # Extract features
            try:
                features = self.extract_features_for_prompt(prompt)

                result = {
                    'experiment_id': exp_idx,
                    'prompt': prompt,
                    'response': response,
                    'outcome': outcome,
                    'features': features
                }
                results.append(result)

                # Save checkpoint every 100
                if (exp_idx + 1) % 100 == 0:
                    checkpoint_file = self.results_dir / f'pathway_6400_checkpoint_{exp_idx+1}.json'
                    with open(checkpoint_file, 'w') as f:
                        json.dump({
                            'total_experiments': len(self.experiments),
                            'processed': exp_idx + 1,
                            'results': results
                        }, f, indent=2)
                    print(f"\n💾 Checkpoint: {exp_idx+1}/{len(self.experiments)}")

            except Exception as e:
                print(f"\n❌ Error processing experiment {exp_idx}: {e}")
                continue

        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = self.results_dir / f'pathway_6400_final_{timestamp}.json'

        with open(final_file, 'w') as f:
            json.dump({
                'total_experiments': len(self.experiments),
                'analyzed': len(results),
                'timestamp': timestamp,
                'results': results
            }, f, indent=2)

        print(f"\n✅ Analysis complete!")
        print(f"   Analyzed: {len(results)}/{len(self.experiments)}")
        print(f"   Saved to: {final_file}")

if __name__ == '__main__':
    analyzer = PathwayAnalyzer6400()
    analyzer.analyze_all_experiments()
