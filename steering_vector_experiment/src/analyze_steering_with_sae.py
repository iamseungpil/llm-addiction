#!/usr/bin/env python3
"""
Phase 3: SAE Interpretation of Steering Vectors

Analyzes steering vectors using Sparse Autoencoders (SAEs) to identify
which interpretable features contribute most to the behavioral difference:
1. Load steering vectors
2. For LLaMA: use LlamaScope to encode steering vector to feature space
3. For Gemma: use GemmaScope (sae_lens) to encode
4. Find top 50 features by absolute contribution
5. Save feature analysis results

Usage:
    python analyze_steering_with_sae.py --model llama --gpu 0 --vectors steering_vectors_llama_20251216.npz
    python analyze_steering_with_sae.py --model gemma --gpu 1 --vectors steering_vectors_gemma_20251216.npz

Design: Model-specific SAE loaders with unified analysis interface.
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import yaml
from abc import ABC, abstractmethod

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    setup_logging,
    ModelRegistry,
    get_gpu_memory_info,
    clear_gpu_memory
)
from extract_steering_vectors import load_steering_vectors

# Add LlamaScope path
sys.path.insert(0, '/home/ubuntu/llm_addiction/causal_feature_discovery/src')


# =============================================================================
# SAE Loader Registry (Extensible design for new models)
# =============================================================================

class SAELoaderRegistry:
    """Registry for model-specific SAE loaders."""
    _loaders: Dict[str, type] = {}

    @classmethod
    def register(cls, model_name: str):
        """Decorator to register an SAE loader."""
        def decorator(loader_cls):
            cls._loaders[model_name] = loader_cls
            return loader_cls
        return decorator

    @classmethod
    def get(cls, model_name: str) -> type:
        """Get registered SAE loader class."""
        if model_name not in cls._loaders:
            raise ValueError(f"No SAE loader for model '{model_name}'. "
                           f"Available: {list(cls._loaders.keys())}")
        return cls._loaders[model_name]


class BaseSAELoader(ABC):
    """Abstract base class for SAE loaders."""

    @abstractmethod
    def load(self, layer: int) -> Any:
        """Load SAE for specified layer."""
        pass

    @abstractmethod
    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Encode hidden states to feature space."""
        pass

    @abstractmethod
    def get_feature_ids(self) -> List[int]:
        """Get list of feature indices."""
        pass

    @abstractmethod
    def get_d_sae(self) -> int:
        """Get SAE dictionary size."""
        pass


@SAELoaderRegistry.register('llama')
class LlamaSAELoader(BaseSAELoader):
    """SAE loader for LLaMA using LlamaScope."""

    def __init__(self, device: str = 'cuda:0', logger=None):
        self.device = device
        self.logger = logger
        self.sae = None
        self.current_layer = None

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def load(self, layer: int) -> 'LlamaSAELoader':
        """Load LlamaScope SAE for specified layer."""
        from llama_scope_working import LlamaScopeWorking

        self._log(f"Loading LlamaScope SAE for layer {layer}...")
        self.sae = LlamaScopeWorking(layer=layer, device=self.device)
        self.current_layer = layer
        self._log(f"LlamaScope SAE loaded for layer {layer}")

        return self

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Encode hidden states using LlamaScope."""
        if self.sae is None:
            raise RuntimeError("SAE not loaded. Call load() first.")

        # Ensure correct dtype and device
        if hidden_states.dtype == torch.bfloat16:
            hidden_states = hidden_states.float()
        hidden_states = hidden_states.to(self.device)

        # Add batch dimension if needed
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
        elif hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)  # [batch, 1, d_model]

        return self.sae.encode(hidden_states)

    def get_feature_ids(self) -> List[int]:
        """Get list of feature indices."""
        return list(range(self.get_d_sae()))

    def get_d_sae(self) -> int:
        """Get SAE dictionary size."""
        if self.sae is None:
            return 32768  # Default for LlamaScope
        return self.sae.sae.d_sae


@SAELoaderRegistry.register('gemma')
class GemmaSAELoader(BaseSAELoader):
    """SAE loader for Gemma using GemmaScope (sae_lens)."""

    def __init__(self, device: str = 'cuda:0', logger=None):
        self.device = device
        self.logger = logger
        self.sae = None
        self.current_layer = None

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def load(self, layer: int) -> 'GemmaSAELoader':
        """Load GemmaScope SAE for specified layer."""
        from sae_lens import SAE

        # GemmaScope naming convention
        # See: https://huggingface.co/google/gemma-scope-9b-pt-res
        release = "gemma-scope-9b-pt-res"
        sae_id = f"layer_{layer}/width_16k/average_l0_71"

        self._log(f"Loading GemmaScope SAE: {release}/{sae_id}...")

        try:
            self.sae = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=self.device
            )[0]  # Returns (sae, config, sparsity)
            self.current_layer = layer
            self._log(f"GemmaScope SAE loaded for layer {layer}")
        except Exception as e:
            self._log(f"Error loading GemmaScope: {e}")
            self._log("Trying alternative width...")
            # Try 32k width as fallback
            sae_id = f"layer_{layer}/width_32k/average_l0_72"
            self.sae = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=self.device
            )[0]
            self.current_layer = layer

        return self

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Encode hidden states using GemmaScope."""
        if self.sae is None:
            raise RuntimeError("SAE not loaded. Call load() first.")

        # Ensure correct dtype and device
        hidden_states = hidden_states.to(self.device)

        # sae_lens expects [batch, d_model] or [batch, seq, d_model]
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)  # [1, d_model]

        return self.sae.encode(hidden_states)

    def get_feature_ids(self) -> List[int]:
        """Get list of feature indices."""
        return list(range(self.get_d_sae()))

    def get_d_sae(self) -> int:
        """Get SAE dictionary size."""
        if self.sae is None:
            return 16384  # Default for GemmaScope 16k
        return self.sae.cfg.d_sae


# =============================================================================
# Steering Vector Analysis
# =============================================================================

class SteeringVectorAnalyzer:
    """
    Analyze steering vectors using SAEs to find interpretable features.
    """

    def __init__(
        self,
        steering_vectors: Dict[int, Dict],
        model_name: str,
        device: str = 'cuda:0',
        logger=None
    ):
        """
        Initialize analyzer.

        Args:
            steering_vectors: Dict mapping layer to steering vector info
            model_name: Name of the model
            device: Device for computation
            logger: Optional logger
        """
        self.steering_vectors = steering_vectors
        self.model_name = model_name
        self.device = device
        self.logger = logger

        # Get SAE loader
        loader_cls = SAELoaderRegistry.get(model_name)
        self.sae_loader = loader_cls(device=device, logger=logger)

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def analyze_layer(
        self,
        layer: int,
        top_k: int = 50,
        min_magnitude: float = 0.1
    ) -> Dict:
        """
        Analyze steering vector for a single layer.

        Args:
            layer: Layer index
            top_k: Number of top features to return
            min_magnitude: Minimum feature magnitude to include

        Returns:
            Analysis results dict
        """
        if layer not in self.steering_vectors:
            self._log(f"No steering vector for layer {layer}")
            return {}

        # Load SAE for this layer
        self.sae_loader.load(layer)

        # Get steering vector
        sv_info = self.steering_vectors[layer]
        steering_vector = sv_info['vector'].to(self.device)

        # Encode steering vector to feature space
        self._log(f"Encoding steering vector for layer {layer}...")
        with torch.no_grad():
            features = self.sae_loader.encode(steering_vector)

        # Remove batch dimensions if present
        features = features.squeeze()

        # Find top features by absolute contribution
        feature_magnitudes = features.abs().cpu().numpy()
        feature_values = features.cpu().numpy()

        # Sort by magnitude
        sorted_indices = np.argsort(feature_magnitudes)[::-1]

        # Get top k features above threshold
        top_features = []
        for idx in sorted_indices[:top_k * 2]:  # Get extra in case some filtered
            idx = int(idx)
            magnitude = float(feature_magnitudes[idx])
            value = float(feature_values[idx])

            if magnitude < min_magnitude:
                continue

            top_features.append({
                'feature_id': idx,
                'magnitude': magnitude,
                'value': value,
                'direction': 'risky' if value > 0 else 'safe'
            })

            if len(top_features) >= top_k:
                break

        # Compute summary statistics
        active_features = np.sum(feature_magnitudes > min_magnitude)
        total_energy = np.sum(feature_magnitudes ** 2)
        top_k_energy = sum(f['magnitude'] ** 2 for f in top_features)

        results = {
            'layer': layer,
            'steering_magnitude': float(sv_info['magnitude']),
            'n_samples': {
                'bankrupt': sv_info['n_bankrupt'],
                'safe': sv_info['n_safe']
            },
            'sae_stats': {
                'd_sae': self.sae_loader.get_d_sae(),
                'active_features': int(active_features),
                'total_energy': float(total_energy),
                'top_k_energy': float(top_k_energy),
                'top_k_energy_fraction': float(top_k_energy / total_energy) if total_energy > 0 else 0
            },
            'top_features': top_features
        }

        # Clean up
        clear_gpu_memory(self.device)

        return results

    def analyze_all_layers(
        self,
        top_k: int = 50,
        min_magnitude: float = 0.1
    ) -> Dict:
        """
        Analyze steering vectors for all available layers.

        Args:
            top_k: Number of top features per layer
            min_magnitude: Minimum feature magnitude

        Returns:
            Complete analysis results
        """
        results = {
            'model': self.model_name,
            'layers': {},
            'cross_layer_summary': {}
        }

        for layer in sorted(self.steering_vectors.keys()):
            self._log(f"\nAnalyzing layer {layer}...")
            layer_results = self.analyze_layer(layer, top_k, min_magnitude)
            if layer_results:
                results['layers'][layer] = layer_results

        # Compute cross-layer summary
        if results['layers']:
            results['cross_layer_summary'] = self._compute_cross_layer_summary(results['layers'])

        return results

    def _compute_cross_layer_summary(self, layer_results: Dict) -> Dict:
        """Compute summary statistics across layers."""
        summary = {
            'layer_magnitudes': {},
            'active_features_per_layer': {},
            'top_feature_overlap': {}
        }

        # Collect per-layer stats
        for layer, data in layer_results.items():
            summary['layer_magnitudes'][layer] = data['steering_magnitude']
            summary['active_features_per_layer'][layer] = data['sae_stats']['active_features']

        # Find feature overlap between adjacent layers
        layers = sorted(layer_results.keys())
        for i in range(len(layers) - 1):
            l1, l2 = layers[i], layers[i + 1]
            top1 = set(f['feature_id'] for f in layer_results[l1]['top_features'][:20])
            top2 = set(f['feature_id'] for f in layer_results[l2]['top_features'][:20])
            overlap = len(top1 & top2)
            summary['top_feature_overlap'][f"{l1}-{l2}"] = overlap

        return summary


def main():
    parser = argparse.ArgumentParser(description='Analyze steering vectors with SAE')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'],
                       help='Model to analyze')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID')
    parser.add_argument('--vectors', type=str, required=True,
                       help='Path to steering vectors .npz file')
    parser.add_argument('--config', type=str,
                       default='/home/ubuntu/llm_addiction/steering_vector_experiment/configs/experiment_config.yaml',
                       help='Path to config file')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Number of top features to analyze (overrides config)')
    parser.add_argument('--min-magnitude', type=float, default=None,
                       help='Minimum feature magnitude (overrides config)')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated list of layers to analyze (default: all available)')

    args = parser.parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup paths
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / 'logs'

    # Setup logging
    logger = setup_logging(f'sae_analysis_{args.model}', log_dir)
    logger.info("=" * 80)
    logger.info(f"Starting SAE analysis for {args.model.upper()}")
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Vectors: {args.vectors}")
    logger.info("=" * 80)

    # Load steering vectors
    vectors_path = Path(args.vectors)
    if not vectors_path.is_absolute():
        vectors_path = output_dir / vectors_path

    logger.info(f"Loading steering vectors from {vectors_path}")
    steering_vectors = load_steering_vectors(vectors_path)
    logger.info(f"Loaded vectors for layers: {list(steering_vectors.keys())}")

    # Filter layers if specified
    if args.layers:
        filter_layers = [int(l) for l in args.layers.split(',')]
        steering_vectors = {l: v for l, v in steering_vectors.items() if l in filter_layers}
        logger.info(f"Filtered to layers: {list(steering_vectors.keys())}")

    # Get analysis parameters
    top_k = args.top_k or config.get('top_k_features', 50)
    min_magnitude = args.min_magnitude or config.get('min_feature_magnitude', 0.1)

    logger.info(f"Analysis parameters: top_k={top_k}, min_magnitude={min_magnitude}")

    # Initialize analyzer
    analyzer = SteeringVectorAnalyzer(
        steering_vectors=steering_vectors,
        model_name=args.model,
        device='cuda:0',
        logger=logger
    )

    # Run analysis
    results = analyzer.analyze_all_layers(top_k=top_k, min_magnitude=min_magnitude)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f'sae_analysis_{args.model}_{timestamp}.json'

    # Add metadata
    results['metadata'] = {
        'model': args.model,
        'vectors_file': str(vectors_path),
        'timestamp': timestamp,
        'top_k': top_k,
        'min_magnitude': min_magnitude
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    # Print summary
    logger.info("=" * 80)
    logger.info("SAE ANALYSIS SUMMARY")
    logger.info("=" * 80)

    for layer, data in results['layers'].items():
        logger.info(f"\nLayer {layer}:")
        logger.info(f"  Steering magnitude: {data['steering_magnitude']:.4f}")
        logger.info(f"  Active features: {data['sae_stats']['active_features']}")
        logger.info(f"  Top-{top_k} energy fraction: {data['sae_stats']['top_k_energy_fraction']:.2%}")

        # Show top 5 features
        logger.info(f"  Top 5 features:")
        for f in data['top_features'][:5]:
            logger.info(f"    Feature {f['feature_id']}: {f['value']:+.4f} ({f['direction']})")

    if results['cross_layer_summary']:
        logger.info("\nCross-layer summary:")
        logger.info(f"  Layer magnitudes: {results['cross_layer_summary']['layer_magnitudes']}")

    logger.info("\nSAE analysis complete!")


if __name__ == '__main__':
    main()
