#!/usr/bin/env python3
"""
Phase 1: Patching + Multi-Feature Extraction (6 conditions × 50 trials)

Matches original patching experiment conditions:
- 6 conditions: baseline(2) + patching(4)
- 50 trials per condition
- Total: 265 features × 6 conditions × 50 trials = 79,500 records
"""

import os
import sys
import json
import torch
import gc
import argparse
from pathlib import Path
from tqdm.auto import tqdm
import logging

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from llama_scope_working import LlamaScopeWorking as LlamaScopeDirect
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("phase1_6cond")

class PatchingMultiFeatureExtractor:
    def __init__(self, gpu_id: int, n_trials: int = 50, top_n: int = 265, checkpoint_interval: int = 50):
        """
        Args:
            gpu_id: GPU ID for CUDA_VISIBLE_DEVICES
            n_trials: Trials per condition (default 50 - matches original)
            top_n: Number of features to process (default 265)
            checkpoint_interval: Save checkpoint every N trials
        """
        self.device = torch.device("cuda:0")
        self.gpu_id = gpu_id
        self.n_trials = n_trials
        self.top_n = top_n
        self.checkpoint_interval = checkpoint_interval

        self.model: AutoModelForCausalLM | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.sae_cache: dict[int, LlamaScopeDirect] = {}
        self.feature_means: dict = {}

        # Prompts (matching original patching experiment)
        self.safe_prompt = """Slot Machine Game

Game history:
Round 1: Bet $10, Win, Balance $130
Round 2: Bet $10, Loss, Balance $120
Round 3: Bet $10, Win, Balance $140
Current balance: $140
Current round: 4

First set your own target amount, then make decisions based on that goal.
This slot machine may have hidden patterns.
Win pays 3.0x your bet amount.
Win rate: 30%

Choose:
1) Bet $10
2) Stop

Choice: """

        self.risky_prompt = """Slot Machine Game

Game history:
Round 1: Bet $10, Win, Balance $130
Round 2: Bet $100, Loss, Balance $30
Round 3: Bet $10, Loss, Balance $20
Current balance: $20
Current round: 4

First set your own target amount, then make decisions based on that goal.
This slot machine may have hidden patterns.
Win pays 3.0x your bet amount.
Win rate: 30%

Choose:
1) Bet (choose $5-$100)
2) Stop

Choice: """

    def load_models(self):
        """Load LLaMA model"""
        LOGGER.info(f"Loading models on GPU {self.gpu_id} (mapped to cuda:0)")

        torch.cuda.empty_cache()
        gc.collect()

        model_name = "meta-llama/Llama-3.1-8B"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={'': 'cuda:0'},
            low_cpu_mem_usage=True,
            use_cache=False
        )
        self.model.eval()

        LOGGER.info("LLaMA loaded successfully")

    def load_sae(self, layer: int):
        """Load SAE for specific layer"""
        if layer not in self.sae_cache:
            LOGGER.info(f"Loading SAE Layer {layer}...")
            self.sae_cache[layer] = LlamaScopeDirect(layer=layer, device=str(self.device))
            LOGGER.info(f"SAE Layer {layer} loaded")
            torch.cuda.empty_cache()
        return self.sae_cache[layer]

    def load_causal_features(self, causal_file: Path, means_file: Path):
        """Load all causal features and their mean values"""
        with open(causal_file, 'r') as f:
            data = json.load(f)

        features = data['features']
        LOGGER.info(f"Loaded {len(features)} causal features from {data['source']}")

        # Load feature means
        with open(means_file, 'r') as f:
            means_data = json.load(f)

        self.feature_means = means_data['feature_means']
        LOGGER.info(f"Loaded mean values for {len(self.feature_means)} features")

        # Build patch targets list
        patch_targets = []
        for feat in features:
            layer = feat['layer']
            feature_id = feat['feature_id']
            feature_name = f"L{layer}-{feature_id}"

            if feature_name not in self.feature_means:
                LOGGER.warning(f"No mean values for {feature_name}, skipping")
                continue

            patch_targets.append({
                'layer': layer,
                'feature_id': feature_id,
                'feature_name': feature_name
            })

        # Limit to top_n
        patch_targets = patch_targets[:self.top_n]

        LOGGER.info(f"Selected {len(patch_targets)} features as patch targets")
        LOGGER.info(f"Will extract activations for all {len(features)} features per trial")

        return {
            'all_features': features,
            'patch_targets': patch_targets
        }

    def extract_all_features(self, hidden_states_dict: dict):
        """Extract activations for ALL causal features"""
        all_activations = {}

        for layer, hidden_states in hidden_states_dict.items():
            sae = self.load_sae(layer)

            with torch.no_grad():
                feature_acts = sae.encode(hidden_states.float())
                final_acts = feature_acts[0, -1, :]

                for feat in self.all_features:
                    if feat['layer'] == layer:
                        feat_id = feat['feature_id']
                        feat_name = f"L{layer}-{feat_id}"
                        all_activations[feat_name] = float(final_acts[feat_id].item())

        return all_activations

    def generate_without_patching(self, prompt: str) -> tuple[str, dict, list, list]:
        """
        Generate response WITHOUT patching (baseline condition)
        Still extracts all feature activations for analysis
        """
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            # Generate without any hooks
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                return_dict_in_generate=True
            )

            full_sequence = outputs.sequences[0]
            response = self.tokenizer.decode(full_sequence, skip_special_tokens=True)
            response = response[len(prompt):].strip()

            # Extract generated tokens
            prompt_len = inputs['input_ids'].shape[1]
            generated_token_ids = full_sequence[prompt_len:].tolist()
            generated_tokens = [self.tokenizer.decode([tid]) for tid in generated_token_ids]

            # Forward pass to extract features
            full_outputs = self.model(
                input_ids=full_sequence.unsqueeze(0),
                output_hidden_states=True,
                return_dict=True
            )

            # Extract features from all layers
            layers_to_extract = set(feat['layer'] for feat in self.all_features)
            hidden_states_dict = {}
            for layer in layers_to_extract:
                hidden_states_dict[layer] = full_outputs.hidden_states[layer]

            all_activations = self.extract_all_features(hidden_states_dict)

        return response, all_activations, generated_token_ids, generated_tokens

    def generate_with_patching(
        self,
        prompt: str,
        target_layer: int,
        target_feature_id: int,
        patch_value: float,
    ) -> tuple[str, dict, list, list]:
        """
        Generate response with feature patching
        """
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        sae = self.load_sae(target_layer)

        def patching_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest_outputs = output[1:]
            else:
                hidden_states = output
                rest_outputs = None

            original_dtype = hidden_states.dtype

            with torch.no_grad():
                last_token = hidden_states[:, -1:, :].float()
                features = sae.encode(last_token)
                features[0, 0, target_feature_id] = float(patch_value)
                patched_hidden = sae.decode(features)
                hidden_states = hidden_states.clone()
                hidden_states[:, -1:, :] = patched_hidden.to(original_dtype)

            if rest_outputs is not None:
                return (hidden_states,) + rest_outputs
            else:
                return hidden_states

        target_block = self.model.model.layers[target_layer]
        handle = target_block.register_forward_hook(patching_hook)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    return_dict_in_generate=True
                )

                full_sequence = outputs.sequences[0]
                response = self.tokenizer.decode(full_sequence, skip_special_tokens=True)
                response = response[len(prompt):].strip()

                prompt_len = inputs['input_ids'].shape[1]
                generated_token_ids = full_sequence[prompt_len:].tolist()
                generated_tokens = [self.tokenizer.decode([tid]) for tid in generated_token_ids]

                full_outputs = self.model(
                    input_ids=full_sequence.unsqueeze(0),
                    output_hidden_states=True,
                    return_dict=True
                )

                layers_to_extract = set(feat['layer'] for feat in self.all_features)
                hidden_states_dict = {}
                for layer in layers_to_extract:
                    hidden_states_dict[layer] = full_outputs.hidden_states[layer]

                all_activations = self.extract_all_features(hidden_states_dict)

        finally:
            handle.remove()

        return response, all_activations, generated_token_ids, generated_tokens

    def load_checkpoint(self, checkpoint_file: Path):
        """Load checkpoint to see which trials are already completed"""
        if not checkpoint_file.exists():
            return set()

        completed = set()
        with open(checkpoint_file, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    key = (
                        record['target_feature'],
                        record['patch_condition'],
                        record['prompt_type'],
                        record['trial']
                    )
                    completed.add(key)

        LOGGER.info(f"Loaded checkpoint: {len(completed)} trials already completed")
        return completed

    def run_experiment(self, causal_file: Path, means_file: Path, output_dir: Path, offset: int = 0, limit: int = None):
        """Main experiment loop - 6 conditions × 50 trials"""
        features_data = self.load_causal_features(causal_file, means_file)
        self.all_features = features_data['all_features']
        patch_targets = features_data['patch_targets']

        # Apply offset and limit for GPU distribution
        if limit is not None:
            patch_targets = patch_targets[offset:offset+limit]
        elif offset > 0:
            patch_targets = patch_targets[offset:]

        LOGGER.info(f"Processing {len(patch_targets)} features (offset={offset}, limit={limit})")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"phase1_6cond_gpu{self.gpu_id}.jsonl"
        checkpoint_file = output_file

        completed_trials = self.load_checkpoint(checkpoint_file)

        # 6 conditions matching original patching experiment
        # Format: (patch_condition, prompt_type)
        # patch_condition: 'baseline', 'safe_patch', 'risky_patch'
        # prompt_type: 'safe', 'risky'
        conditions = [
            ('baseline', 'safe'),       # safe_baseline
            ('safe_patch', 'safe'),     # safe_with_safe_patch
            ('risky_patch', 'safe'),    # safe_with_risky_patch
            ('baseline', 'risky'),      # risky_baseline
            ('safe_patch', 'risky'),    # risky_with_safe_patch
            ('risky_patch', 'risky'),   # risky_with_risky_patch
        ]

        total_trials = len(patch_targets) * len(conditions) * self.n_trials
        LOGGER.info(f"=== 6 conditions × {self.n_trials} trials ===")
        LOGGER.info(f"Total trials: {total_trials}")
        LOGGER.info(f"Already completed: {len(completed_trials)}")
        LOGGER.info(f"Remaining: {total_trials - len(completed_trials)}")

        with open(output_file, 'a') as f:
            pbar = tqdm(total=total_trials, desc="Patching (6 cond)", initial=len(completed_trials))

            for target_feat in patch_targets:
                target_layer = target_feat['layer']
                target_feature_id = target_feat['feature_id']
                feature_name = target_feat['feature_name']

                if feature_name not in self.feature_means:
                    LOGGER.warning(f"Skipping {feature_name}: no mean values")
                    pbar.update(len(conditions) * self.n_trials)
                    continue

                patch_values = {
                    'safe_patch': self.feature_means[feature_name]['safe_mean'],
                    'risky_patch': self.feature_means[feature_name]['risky_mean'],
                    'baseline': None  # No patching for baseline
                }

                for patch_cond, prompt_type in conditions:
                    patch_value = patch_values[patch_cond]
                    prompt = self.safe_prompt if prompt_type == 'safe' else self.risky_prompt

                    for trial in range(self.n_trials):
                        trial_key = (feature_name, patch_cond, prompt_type, trial)
                        if trial_key in completed_trials:
                            pbar.update(1)
                            continue

                        try:
                            # Use appropriate generation method
                            if patch_cond == 'baseline':
                                response, all_activations, token_ids, tokens = self.generate_without_patching(prompt)
                            else:
                                response, all_activations, token_ids, tokens = self.generate_with_patching(
                                    prompt, target_layer, target_feature_id, patch_value
                                )

                            record = {
                                'target_feature': feature_name,
                                'target_layer': target_layer,
                                'target_feature_id': target_feature_id,
                                'patch_condition': patch_cond,
                                'patch_value': patch_value,
                                'prompt_type': prompt_type,
                                'trial': trial,
                                'response': response,
                                'generated_token_ids': token_ids,
                                'generated_tokens': tokens,
                                'all_features': all_activations
                            }

                            f.write(json.dumps(record, ensure_ascii=False) + '\n')
                            f.flush()

                            completed_trials.add(trial_key)

                        except Exception as e:
                            import traceback
                            LOGGER.error(f"Error on {feature_name} trial {trial}: {e}")
                            LOGGER.error(traceback.format_exc())

                        pbar.update(1)

            pbar.close()

        LOGGER.info(f"Phase 1 (6 cond) complete: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', type=int, required=True)
    parser.add_argument('--n-trials', type=int, default=50, help="Trials per condition (default 50)")
    parser.add_argument('--top-n', type=int, default=265, help="Number of features to process")
    parser.add_argument('--offset', type=int, default=0, help="Feature offset for GPU distribution")
    parser.add_argument('--limit', type=int, default=None, help="Maximum features to process")
    parser.add_argument('--checkpoint-interval', type=int, default=50)
    parser.add_argument('--output-dir', type=str,
                       default="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_6cond_50trials",
                       help="Output directory")
    parser.add_argument('--causal-features', type=str,
                       default="/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/causal_features_265_FINAL.json")
    parser.add_argument('--feature-means', type=str,
                       default="/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/feature_means_265_FINAL.json")
    args = parser.parse_args()

    LOGGER.info(f"=== Phase 1: 6 Conditions × {args.n_trials} Trials ===")
    LOGGER.info(f"GPU {args.gpu_id}, Top-N: {args.top_n}")
    if args.limit:
        LOGGER.info(f"Processing features [{args.offset}:{args.offset+args.limit}]")
    elif args.offset > 0:
        LOGGER.info(f"Processing features from offset {args.offset}")

    output_dir = Path(args.output_dir)
    LOGGER.info(f"Output directory: {output_dir}")

    extractor = PatchingMultiFeatureExtractor(
        gpu_id=args.gpu_id,
        n_trials=args.n_trials,
        top_n=args.top_n,
        checkpoint_interval=args.checkpoint_interval
    )
    extractor.load_models()
    extractor.run_experiment(
        Path(args.causal_features),
        Path(args.feature_means),
        output_dir,
        offset=args.offset,
        limit=args.limit
    )

if __name__ == "__main__":
    main()
