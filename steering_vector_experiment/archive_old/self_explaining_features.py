#!/usr/bin/env python3
"""
Self-Explaining SAE Features

Implements the SelfIE/Patchscopes methodology for feature interpretation:
- Patch SAE decoder vectors into the model's residual stream
- Generate natural language explanations from the model itself

Based on:
- SelfIE (Luo et al., 2024): Self-Interpretation of Large Language Model Embeddings
- Patchscopes (Ghandeharioun et al., 2024): Inspecting hidden representations
- Self-explaining SAE features (Kharlapenko, 2024)

Usage:
    python self_explaining_features.py --model llama --gpu 0 --circuit circuit_llama.json
    python self_explaining_features.py --model gemma --gpu 1 --circuit circuit_gemma.json
"""

import os
import sys
import argparse
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    setup_logging,
    ModelRegistry,
    load_model_and_tokenizer,
    get_gpu_memory_info,
    clear_gpu_memory,
    get_causal_feature_src,
    get_default_config_path
)
from sparse_feature_circuits import SAEWrapper

# Add LlamaScope path dynamically
sys.path.insert(0, str(get_causal_feature_src()))


@dataclass
class FeatureExplanation:
    """Explanation for a single SAE feature."""
    layer: int
    feature_id: int
    explanation: str
    prompt_template: str
    scale: float
    raw_generation: str
    confidence: float  # Based on generation quality


class SelfExplainingFeatures:
    """
    Generate natural language explanations for SAE features.

    Method: Patch the SAE decoder direction into the model's residual stream
    at a placeholder token position, then generate an explanation.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        device: str = 'cuda:0',
        logger=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device
        self.logger = logger
        self.model_config = ModelRegistry.get(model_name)

        # SAE wrapper
        self.sae = SAEWrapper(model_name, device, logger)

        # Prompt templates
        self._setup_templates()

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _setup_templates(self):
        """Setup prompt templates for different models."""
        # Base templates (for LLaMA base model)
        self.base_templates = [
            'The word "X" means "',
            'The concept of X refers to ',
            '"X" represents the idea of ',
            'In this context, X indicates ',
        ]

        # Chat templates (for instruction-tuned models like Gemma-it)
        self.chat_templates = [
            {
                'messages': [
                    {'role': 'user', 'content': 'What does the word "X" mean? Give a brief explanation.'}
                ],
                'suffix': 'The word "X" means '
            },
            {
                'messages': [
                    {'role': 'user', 'content': 'Explain what "X" represents in one sentence.'}
                ],
                'suffix': '"X" represents '
            }
        ]

    def _get_prompt_and_position(
        self,
        template_idx: int = 0
    ) -> Tuple[str, int]:
        """
        Get prompt and the position of placeholder token.

        Returns:
            Tuple of (formatted_prompt, placeholder_position)
        """
        if self.model_config.use_chat_template:
            # Instruction-tuned model (Gemma-it)
            template = self.chat_templates[template_idx % len(self.chat_templates)]
            formatted = self.tokenizer.apply_chat_template(
                template['messages'],
                tokenize=False,
                add_generation_prompt=True
            )
            formatted += template['suffix']
        else:
            # Base model (LLaMA)
            formatted = self.base_templates[template_idx % len(self.base_templates)]

        # Tokenize to find X position
        tokens = self.tokenizer.encode(formatted, add_special_tokens=False)
        token_strs = [self.tokenizer.decode([t]) for t in tokens]

        # Find "X" token position
        x_position = None
        for i, t_str in enumerate(token_strs):
            if 'X' in t_str:
                x_position = i
                break

        if x_position is None:
            self._log(f"Warning: Could not find X in template, using last position")
            x_position = len(tokens) - 1

        return formatted, x_position

    def explain_feature(
        self,
        layer: int,
        feature_id: int,
        scale: float = 5.0,
        template_idx: int = 0,
        max_new_tokens: int = 30
    ) -> FeatureExplanation:
        """
        Generate explanation for a single feature.

        Args:
            layer: Layer of the feature
            feature_id: Feature ID in the SAE
            scale: Scale factor for decoder vector (sensitive hyperparameter)
            template_idx: Which template to use
            max_new_tokens: Max tokens to generate

        Returns:
            FeatureExplanation dataclass
        """
        # Load SAE if needed
        if self.sae.current_layer != layer:
            self.sae.load(layer)

        # Get decoder vector for this feature
        decoder = self.sae.decoder
        if decoder is None:
            raise RuntimeError("SAE has no decoder")

        feature_vec = decoder[feature_id].detach().float().to(self.device)
        feature_vec = feature_vec * scale

        # Get prompt and position
        prompt, x_position = self._get_prompt_and_position(template_idx)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            add_special_tokens=True
        ).to(self.device)

        # Adjust position for special tokens
        # Most tokenizers add 1 special token at the beginning
        actual_position = x_position + 1

        # Create hook to patch activation
        hook_handle = None
        patching_layer = min(layer, 2)  # Patch at early layer (0-2)

        def patch_hook(module, input, output):
            hidden_states = output[0]
            # Patch at the X position
            if actual_position < hidden_states.shape[1]:
                hidden_states[0, actual_position, :] = feature_vec
            return (hidden_states,) + output[1:]

        # Register hook
        target_module = self.model.model.layers[patching_layer]
        hook_handle = target_module.register_forward_hook(patch_hook)

        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            generated = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Extract explanation (after the prompt)
            explanation = generated[len(prompt):].strip()

            # Clean up explanation
            explanation = self._clean_explanation(explanation)

            # Estimate confidence based on generation quality
            confidence = self._estimate_confidence(explanation)

        finally:
            if hook_handle:
                hook_handle.remove()

        return FeatureExplanation(
            layer=layer,
            feature_id=feature_id,
            explanation=explanation,
            prompt_template=prompt,
            scale=scale,
            raw_generation=generated,
            confidence=confidence
        )

    def _clean_explanation(self, explanation: str) -> str:
        """Clean up generated explanation."""
        # Remove trailing incomplete sentences
        if explanation:
            # Find last complete sentence
            for end_char in ['.', '!', '?']:
                last_pos = explanation.rfind(end_char)
                if last_pos > 0:
                    explanation = explanation[:last_pos + 1]
                    break

            # Remove quotes if unbalanced
            if explanation.count('"') % 2 == 1:
                explanation = explanation.replace('"', '')

            # Truncate if too long
            words = explanation.split()
            if len(words) > 20:
                explanation = ' '.join(words[:20]) + '...'

        return explanation.strip()

    def _estimate_confidence(self, explanation: str) -> float:
        """
        Estimate confidence in explanation quality.

        Higher confidence for:
        - Longer, more specific explanations
        - Complete sentences
        - No repetition
        """
        if not explanation:
            return 0.0

        score = 0.5  # Base score

        # Length bonus
        words = explanation.split()
        if len(words) >= 5:
            score += 0.1
        if len(words) >= 10:
            score += 0.1

        # Complete sentence bonus
        if explanation.endswith(('.', '!', '?')):
            score += 0.1

        # Specificity bonus (contains concrete words)
        concrete_words = ['risk', 'gambl', 'bet', 'stop', 'continu', 'decision',
                         'money', 'loss', 'win', 'safe', 'danger', 'caution']
        for word in concrete_words:
            if word in explanation.lower():
                score += 0.05
                break

        # Repetition penalty
        unique_words = set(words)
        if len(unique_words) < len(words) * 0.7:
            score -= 0.2

        return min(max(score, 0.0), 1.0)

    def explain_features_batch(
        self,
        features: List[Dict],
        scales: List[float] = [3.0, 5.0, 8.0],
        n_templates: int = 2
    ) -> List[FeatureExplanation]:
        """
        Generate explanations for multiple features.

        Tries multiple scales and templates, returns best explanation.

        Args:
            features: List of dicts with 'layer' and 'feature_id'
            scales: Scale values to try
            n_templates: Number of templates to try

        Returns:
            List of best FeatureExplanation for each feature
        """
        explanations = []

        for feat in tqdm(features, desc="Explaining features"):
            layer = feat['layer']
            feature_id = feat['feature_id']

            best_explanation = None
            best_confidence = -1

            # Try different scales and templates
            for scale in scales:
                for template_idx in range(n_templates):
                    try:
                        exp = self.explain_feature(
                            layer=layer,
                            feature_id=feature_id,
                            scale=scale,
                            template_idx=template_idx
                        )

                        if exp.confidence > best_confidence:
                            best_confidence = exp.confidence
                            best_explanation = exp

                    except Exception as e:
                        self._log(f"Error explaining L{layer}-{feature_id}: {e}")
                        continue

            if best_explanation:
                explanations.append(best_explanation)
            else:
                # Fallback
                explanations.append(FeatureExplanation(
                    layer=layer,
                    feature_id=feature_id,
                    explanation="[Failed to generate explanation]",
                    prompt_template="",
                    scale=0,
                    raw_generation="",
                    confidence=0.0
                ))

            # Clear cache periodically
            if len(explanations) % 10 == 0:
                clear_gpu_memory(self.device)

        return explanations


def main():
    parser = argparse.ArgumentParser(description='Generate self-explanations for SAE features')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'])
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--circuit', type=str, required=True,
                       help='Path to circuit JSON from sparse_feature_circuits.py')
    parser.add_argument('--config', type=str,
                       default=None,
                       help='Path to config file (default: auto-detect)')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Number of top features to explain')
    parser.add_argument('--scales', type=str, default='3.0,5.0,8.0',
                       help='Comma-separated scale values to try')

    args = parser.parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Parse scales
    scales = [float(s) for s in args.scales.split(',')]

    # Load config
    config_path = args.config or str(get_default_config_path())
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(f'explain_{args.model}', output_dir / 'logs')

    logger.info("=" * 60)
    logger.info(f"SELF-EXPLAINING FEATURES - {args.model.upper()}")
    logger.info("=" * 60)

    # Load circuit results
    circuit_path = Path(args.circuit)
    if not circuit_path.is_absolute():
        circuit_path = output_dir / circuit_path

    with open(circuit_path, 'r') as f:
        circuit = json.load(f)

    logger.info(f"Loaded circuit with {circuit['summary']['total_causal_features']} features")

    # Get top features to explain
    features_to_explain = []

    # Add top safe features
    for f in circuit['top_safe_features'][:args.top_k // 2]:
        features_to_explain.append({
            'layer': f['layer'],
            'feature_id': f['feature_id'],
            'direction': 'safe',
            'indirect_effect': f['indirect_effect']
        })

    # Add top risky features
    for f in circuit['top_risky_features'][:args.top_k // 2]:
        features_to_explain.append({
            'layer': f['layer'],
            'feature_id': f['feature_id'],
            'direction': 'risky',
            'indirect_effect': f['indirect_effect']
        })

    logger.info(f"Will explain {len(features_to_explain)} features")

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        args.model, 'cuda:0', torch.bfloat16, logger
    )

    # Initialize explainer
    explainer = SelfExplainingFeatures(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model,
        device='cuda:0',
        logger=logger
    )

    # Generate explanations
    logger.info("Generating explanations...")
    explanations = explainer.explain_features_batch(
        features=features_to_explain,
        scales=scales,
        n_templates=2
    )

    # Combine with feature info
    results = []
    for feat, exp in zip(features_to_explain, explanations):
        results.append({
            **feat,
            'explanation': exp.explanation,
            'confidence': exp.confidence,
            'scale': exp.scale,
            'raw_generation': exp.raw_generation
        })

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f'explanations_{args.model}_{timestamp}.json'

    output_data = {
        'model': args.model,
        'n_features': len(results),
        'scales_tried': scales,
        'explanations': results
    }

    with open(results_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE EXPLANATIONS SUMMARY")
    logger.info("=" * 60)

    # Safe features
    logger.info("\nTop Safe Features:")
    for r in results:
        if r['direction'] == 'safe':
            logger.info(f"  L{r['layer']}-{r['feature_id']} (IE={r['indirect_effect']:.4f}):")
            logger.info(f"    \"{r['explanation']}\" [conf={r['confidence']:.2f}]")

    # Risky features
    logger.info("\nTop Risky Features:")
    for r in results:
        if r['direction'] == 'risky':
            logger.info(f"  L{r['layer']}-{r['feature_id']} (IE={r['indirect_effect']:.4f}):")
            logger.info(f"    \"{r['explanation']}\" [conf={r['confidence']:.2f}]")

    logger.info("\nSelf-explanation complete!")


if __name__ == '__main__':
    main()
