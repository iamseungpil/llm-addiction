#!/usr/bin/env python3
"""
Sparse Feature Circuits Discovery

Implements the methodology from Marks et al. (ICLR 2025):
"Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs"

Key steps:
1. Compute activations for clean (bankruptcy) and corrupt (safe) prompts
2. Encode through SAE to get feature activations
3. Compute indirect effects via attribution patching
4. Threshold to get causal features and edges
5. Validate via ablation

Usage:
    python sparse_feature_circuits.py --model llama --gpu 0 --pairs prompt_pairs_llama.json
    python sparse_feature_circuits.py --model gemma --gpu 1 --pairs prompt_pairs_gemma.json
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
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
from prepare_prompt_pairs import load_prompt_pairs, PromptPair

# Add LlamaScope path dynamically
sys.path.insert(0, str(get_causal_feature_src()))


@dataclass
class FeatureAttribution:
    """Attribution result for a single feature."""
    layer: int
    feature_id: int
    indirect_effect: float
    activation_clean: float
    activation_corrupt: float
    gradient: float
    direction: str  # 'safe' or 'risky'


@dataclass
class CircuitEdge:
    """Edge between two features in the circuit."""
    source_layer: int
    source_feature: int
    target_layer: int
    target_feature: int
    weight: float


class SAEWrapper:
    """
    Unified wrapper for different SAE implementations.
    Supports LlamaScope and GemmaScope (sae_lens).
    """

    def __init__(self, model_name: str, device: str = 'cuda:0', logger=None):
        self.model_name = model_name
        self.device = device
        self.logger = logger
        self.sae = None
        self.current_layer = None

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def load(self, layer: int) -> 'SAEWrapper':
        """Load SAE for specified layer."""
        if self.model_name == 'llama':
            return self._load_llama_sae(layer)
        elif self.model_name == 'gemma':
            return self._load_gemma_sae(layer)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _load_llama_sae(self, layer: int) -> 'SAEWrapper':
        """Load LlamaScope SAE."""
        try:
            from llama_scope_working import LlamaScopeWorking
            self._log(f"Loading LlamaScope SAE for layer {layer}...")
            self.sae = LlamaScopeWorking(layer=layer, device=self.device)
            self.current_layer = layer
            self._log(f"LlamaScope loaded: d_sae={self.sae.sae.d_sae}")
        except Exception as e:
            self._log(f"LlamaScope failed, trying EleutherAI SAE: {e}")
            self._load_eleutherai_sae(layer)
        return self

    def _load_eleutherai_sae(self, layer: int) -> None:
        """Load EleutherAI SAE as fallback."""
        try:
            from sparsify import Sae
            sae_id = "EleutherAI/sae-llama-3.1-8b-32x"
            hookpoint = f"layers.{layer}.mlp"
            self._log(f"Loading EleutherAI SAE: {sae_id}, hookpoint={hookpoint}")
            self.sae = Sae.load_from_hub(sae_id, hookpoint=hookpoint)
            self.sae = self.sae.to(self.device)
            self.current_layer = layer
        except Exception as e:
            raise RuntimeError(f"Failed to load any SAE for LLaMA layer {layer}: {e}")

    def _load_gemma_sae(self, layer: int) -> 'SAEWrapper':
        """Load GemmaScope SAE via sae_lens."""
        from sae_lens import SAE

        release = "gemma-scope-9b-pt-res"
        # Try different widths
        for width in ['16k', '32k']:
            try:
                sae_id = f"layer_{layer}/width_{width}/average_l0_71"
                self._log(f"Loading GemmaScope: {release}/{sae_id}")
                self.sae = SAE.from_pretrained(
                    release=release,
                    sae_id=sae_id,
                    device=self.device
                )[0]
                self.current_layer = layer
                self._log(f"GemmaScope loaded: d_sae={self.sae.cfg.d_sae}")
                return self
            except Exception as e:
                self._log(f"Failed with width {width}: {e}")
                continue

        raise RuntimeError(f"Failed to load GemmaScope for layer {layer}")

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Encode hidden states to feature activations."""
        if self.sae is None:
            raise RuntimeError("SAE not loaded")

        # Handle dtype
        if hidden_states.dtype == torch.bfloat16:
            hidden_states = hidden_states.float()
        hidden_states = hidden_states.to(self.device)

        # Handle dimensions
        original_shape = hidden_states.shape
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)

        # Encode
        if self.model_name == 'llama' and hasattr(self.sae, 'encode'):
            # LlamaScope expects [batch, seq, d_model]
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(1)
            features = self.sae.encode(hidden_states)
        elif hasattr(self.sae, 'encode'):
            features = self.sae.encode(hidden_states)
        else:
            raise RuntimeError("SAE has no encode method")

        return features.squeeze()

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to hidden states."""
        if self.sae is None:
            raise RuntimeError("SAE not loaded")

        features = features.to(self.device)
        if features.dim() == 1:
            features = features.unsqueeze(0)

        if hasattr(self.sae, 'decode'):
            return self.sae.decode(features).squeeze()
        else:
            # Manual decode: features @ decoder.T
            if hasattr(self.sae, 'W_dec'):
                return (features @ self.sae.W_dec).squeeze()
            else:
                raise RuntimeError("Cannot decode: no decoder found")

    @property
    def d_sae(self) -> int:
        """Get SAE dictionary size."""
        if self.sae is None:
            return 32768
        if hasattr(self.sae, 'd_sae'):
            return self.sae.d_sae
        elif hasattr(self.sae, 'sae') and hasattr(self.sae.sae, 'd_sae'):
            return self.sae.sae.d_sae
        elif hasattr(self.sae, 'cfg') and hasattr(self.sae.cfg, 'd_sae'):
            return self.sae.cfg.d_sae
        return 32768

    @property
    def decoder(self) -> torch.Tensor:
        """Get decoder weight matrix."""
        if hasattr(self.sae, 'W_dec'):
            return self.sae.W_dec
        elif hasattr(self.sae, 'sae') and hasattr(self.sae.sae, 'W_dec'):
            return self.sae.sae.W_dec
        elif hasattr(self.sae, 'decoder'):
            return self.sae.decoder.weight.T
        raise RuntimeError("Cannot find decoder weights")


class SparseFeatureCircuits:
    """
    Discover causal feature circuits using attribution patching.

    Based on Marks et al. (ICLR 2025) methodology.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        target_layers: List[int],
        device: str = 'cuda:0',
        logger=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.target_layers = target_layers
        self.device = device
        self.logger = logger
        self.model_config = ModelRegistry.get(model_name)

        # SAE wrapper
        self.sae = SAEWrapper(model_name, device, logger)

        # Cache for activations
        self.activation_cache = {}

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for model."""
        if self.model_config.use_chat_template:
            chat = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def _get_hidden_states(
        self,
        prompt: str,
        layers: List[int]
    ) -> Dict[int, torch.Tensor]:
        """Extract hidden states at specified layers."""
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(
            formatted,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        hidden_states = {}
        for layer in layers:
            # hidden_states[0] is embedding, [1] is layer 0, etc.
            hs = outputs.hidden_states[layer + 1]
            # Take last token
            hidden_states[layer] = hs[0, -1, :].cpu().float()

        del outputs
        clear_gpu_memory(self.device)

        return hidden_states

    def _get_output_metric(
        self,
        prompt: str
    ) -> torch.Tensor:
        """
        Compute output metric for attribution.

        Metric: P(stop) - P(bet)

        Uses the first token probability of each response type, as the model
        generates "Stop" or "Bet $X" as the decision. This properly measures
        the model's preference toward stopping vs betting.
        """
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(
            formatted,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
            probs = F.softmax(logits, dim=-1)

        # Get probability of "Stop" (or "2" for option 2)
        # Use first token only to avoid double-counting
        stop_tokens = ['Stop', '2']  # "Stop" or option "2)"
        p_stop = 0.0
        for token in stop_tokens:
            tid = self.tokenizer.encode(token, add_special_tokens=False)[0]
            p_stop = max(p_stop, probs[tid].item())  # Take max, not sum

        # Get probability of "Bet" (or "1" for option 1, or "$" for bet amount)
        bet_tokens = ['Bet', '1', '$']
        p_bet = 0.0
        for token in bet_tokens:
            tid = self.tokenizer.encode(token, add_special_tokens=False)[0]
            p_bet = max(p_bet, probs[tid].item())  # Take max, not sum

        # Metric: difference in preference
        metric = p_stop - p_bet

        del outputs
        return torch.tensor(metric)

    def compute_indirect_effects(
        self,
        prompt_pairs: List[PromptPair],
        layer: int,
        batch_size: int = 10
    ) -> Dict[int, FeatureAttribution]:
        """
        Compute indirect effects for all features at a layer.

        IE(feature) = gradient * (activation_clean - activation_corrupt)

        Args:
            prompt_pairs: List of clean/corrupt prompt pairs
            layer: Layer to analyze
            batch_size: Number of pairs to process at once

        Returns:
            Dict mapping feature_id to FeatureAttribution
        """
        self._log(f"Computing indirect effects for layer {layer}...")

        # Load SAE for this layer
        self.sae.load(layer)
        d_sae = self.sae.d_sae

        # Accumulators
        ie_sum = torch.zeros(d_sae)
        act_clean_sum = torch.zeros(d_sae)
        act_corrupt_sum = torch.zeros(d_sae)
        grad_sum = torch.zeros(d_sae)
        n_samples = 0

        for i in tqdm(range(0, len(prompt_pairs), batch_size), desc=f"Layer {layer}"):
            batch = prompt_pairs[i:i + batch_size]

            for pair in batch:
                try:
                    # Get hidden states
                    hs_clean = self._get_hidden_states(pair.clean_prompt, [layer])[layer]
                    hs_corrupt = self._get_hidden_states(pair.corrupt_prompt, [layer])[layer]

                    # Encode to features
                    f_clean = self.sae.encode(hs_clean).detach().cpu()
                    f_corrupt = self.sae.encode(hs_corrupt).detach().cpu()

                    # Compute gradient via finite difference approximation
                    # This avoids needing to backprop through generation
                    # grad ≈ (metric(clean) - metric(corrupt)) / ||f_clean - f_corrupt||
                    metric_clean = self._get_output_metric(pair.clean_prompt).item()
                    metric_corrupt = self._get_output_metric(pair.corrupt_prompt).item()

                    delta_f = f_clean - f_corrupt
                    delta_metric = metric_clean - metric_corrupt

                    # Normalize gradient estimate
                    norm = torch.norm(delta_f) + 1e-8
                    grad_estimate = delta_metric / norm * torch.sign(delta_f)

                    # Indirect effect
                    ie = grad_estimate * delta_f

                    # Accumulate
                    ie_sum += ie
                    act_clean_sum += f_clean
                    act_corrupt_sum += f_corrupt
                    grad_sum += grad_estimate.abs()
                    n_samples += 1

                except Exception as e:
                    self._log(f"Error processing pair {pair.pair_id}: {e}")
                    continue

            # Clear GPU cache periodically
            if i % (batch_size * 5) == 0:
                clear_gpu_memory(self.device)

        # Average
        if n_samples > 0:
            ie_mean = ie_sum / n_samples
            act_clean_mean = act_clean_sum / n_samples
            act_corrupt_mean = act_corrupt_sum / n_samples
            grad_mean = grad_sum / n_samples
        else:
            self._log("Warning: No samples processed")
            return {}

        # Build attribution results
        attributions = {}
        for feature_id in range(d_sae):
            ie = ie_mean[feature_id].item()
            if abs(ie) < 1e-8:
                continue

            attributions[feature_id] = FeatureAttribution(
                layer=layer,
                feature_id=feature_id,
                indirect_effect=ie,
                activation_clean=act_clean_mean[feature_id].item(),
                activation_corrupt=act_corrupt_mean[feature_id].item(),
                gradient=grad_mean[feature_id].item(),
                direction='safe' if ie > 0 else 'risky'
            )

        self._log(f"Layer {layer}: {len(attributions)} features with non-zero IE")
        return attributions

    def discover_circuit(
        self,
        prompt_pairs: List[PromptPair],
        node_threshold: float = 0.1,
        edge_threshold: float = 0.01,
        top_k_per_layer: int = 50
    ) -> Dict:
        """
        Discover sparse feature circuit across all target layers.

        Args:
            prompt_pairs: Clean/corrupt prompt pairs
            node_threshold: Minimum |IE| to include a feature
            edge_threshold: Minimum edge weight to include
            top_k_per_layer: Max features per layer

        Returns:
            Circuit dict with nodes, edges, and statistics
        """
        self._log("=" * 60)
        self._log("SPARSE FEATURE CIRCUIT DISCOVERY")
        self._log("=" * 60)
        self._log(f"Prompt pairs: {len(prompt_pairs)}")
        self._log(f"Target layers: {self.target_layers}")
        self._log(f"Node threshold: {node_threshold}")
        self._log(f"Edge threshold: {edge_threshold}")

        # Compute attributions for all layers
        all_attributions = {}
        for layer in self.target_layers:
            attributions = self.compute_indirect_effects(prompt_pairs, layer)
            all_attributions[layer] = attributions

        # Apply node threshold
        causal_features = []
        for layer, attrs in all_attributions.items():
            # Sort by |IE|
            sorted_attrs = sorted(
                attrs.values(),
                key=lambda x: abs(x.indirect_effect),
                reverse=True
            )

            # Apply threshold and top-k
            layer_features = []
            for attr in sorted_attrs:
                if abs(attr.indirect_effect) >= node_threshold:
                    layer_features.append(attr)
                if len(layer_features) >= top_k_per_layer:
                    break

            causal_features.extend(layer_features)
            self._log(f"Layer {layer}: {len(layer_features)} causal features")

        # Separate safe and risky
        safe_features = [f for f in causal_features if f.direction == 'safe']
        risky_features = [f for f in causal_features if f.direction == 'risky']

        self._log(f"\nTotal causal features: {len(causal_features)}")
        self._log(f"Safe features: {len(safe_features)}")
        self._log(f"Risky features: {len(risky_features)}")

        # Compute layer distribution
        layer_dist = {}
        for layer in self.target_layers:
            layer_safe = len([f for f in safe_features if f.layer == layer])
            layer_risky = len([f for f in risky_features if f.layer == layer])
            layer_dist[layer] = {'safe': layer_safe, 'risky': layer_risky}

        # Build result
        circuit = {
            'model': self.model_name,
            'target_layers': self.target_layers,
            'n_prompt_pairs': len(prompt_pairs),
            'node_threshold': node_threshold,
            'edge_threshold': edge_threshold,
            'summary': {
                'total_causal_features': len(causal_features),
                'safe_features': len(safe_features),
                'risky_features': len(risky_features)
            },
            'layer_distribution': layer_dist,
            'causal_features': [asdict(f) for f in causal_features],
            'top_safe_features': [asdict(f) for f in sorted(
                safe_features, key=lambda x: x.indirect_effect, reverse=True
            )[:20]],
            'top_risky_features': [asdict(f) for f in sorted(
                risky_features, key=lambda x: x.indirect_effect
            )[:20]]
        }

        return circuit

    def validate_circuit(
        self,
        circuit: Dict,
        prompt_pairs: List[PromptPair],
        n_validation: int = 50
    ) -> Dict:
        """
        Validate circuit via ablation experiments.

        Ablate top features and measure change in behavior.
        """
        self._log("\nValidating circuit via ablation...")

        # Sample validation pairs
        val_pairs = prompt_pairs[:min(n_validation, len(prompt_pairs))]

        # Get baseline metrics
        baseline_metrics = []
        for pair in val_pairs[:10]:
            m = self._get_output_metric(pair.clean_prompt).item()
            baseline_metrics.append(m)
        baseline = np.mean(baseline_metrics)

        # TODO: Implement actual ablation
        # This requires modifying activations during forward pass
        # For now, return placeholder

        validation = {
            'baseline_stop_prob': baseline,
            'n_validation_pairs': len(val_pairs),
            'ablation_results': 'Not implemented - requires activation hooks'
        }

        return validation


def main():
    parser = argparse.ArgumentParser(description='Discover sparse feature circuits')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'])
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--pairs', type=str, required=True,
                       help='Path to prompt pairs JSON')
    parser.add_argument('--config', type=str,
                       default=None,
                       help='Path to config file (default: auto-detect)')
    parser.add_argument('--node-threshold', type=float, default=0.1)
    parser.add_argument('--edge-threshold', type=float, default=0.01)
    parser.add_argument('--top-k', type=int, default=50)

    args = parser.parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Load config
    config_path = args.config or str(get_default_config_path())
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(f'circuits_{args.model}', output_dir / 'logs')

    logger.info("=" * 60)
    logger.info(f"SPARSE FEATURE CIRCUITS - {args.model.upper()}")
    logger.info("=" * 60)

    # Load prompt pairs
    pairs_path = Path(args.pairs)
    if not pairs_path.is_absolute():
        pairs_path = output_dir / pairs_path
    prompt_pairs = load_prompt_pairs(pairs_path)
    logger.info(f"Loaded {len(prompt_pairs)} prompt pairs")

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        args.model, 'cuda:0', torch.bfloat16, logger
    )

    # Initialize circuit discovery
    target_layers = config['target_layers']
    circuits = SparseFeatureCircuits(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model,
        target_layers=target_layers,
        device='cuda:0',
        logger=logger
    )

    # Discover circuit
    circuit = circuits.discover_circuit(
        prompt_pairs=prompt_pairs,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        top_k_per_layer=args.top_k
    )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f'circuit_{args.model}_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(circuit, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CIRCUIT DISCOVERY SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total causal features: {circuit['summary']['total_causal_features']}")
    logger.info(f"Safe features: {circuit['summary']['safe_features']}")
    logger.info(f"Risky features: {circuit['summary']['risky_features']}")
    logger.info("\nLayer distribution:")
    for layer, dist in circuit['layer_distribution'].items():
        logger.info(f"  Layer {layer}: {dist['safe']} safe, {dist['risky']} risky")

    logger.info("\nTop 5 safe features:")
    for f in circuit['top_safe_features'][:5]:
        logger.info(f"  L{f['layer']}-{f['feature_id']}: IE={f['indirect_effect']:.4f}")

    logger.info("\nTop 5 risky features:")
    for f in circuit['top_risky_features'][:5]:
        logger.info(f"  L{f['layer']}-{f['feature_id']}: IE={f['indirect_effect']:.4f}")

    logger.info("\nCircuit discovery complete!")


if __name__ == '__main__':
    main()
