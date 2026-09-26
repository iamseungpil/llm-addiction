#!/usr/bin/env python3
"""
Phase 3: SAE Interpretation (Improved)

Two-pronged analysis for robust feature discovery:
1. Steering Vector Projection: Project steering vectors onto SAE feature space
2. Sample-Based Validation: Compare actual bankrupt vs safe sample activations

Key insight:
  - SAE.encode(steering_vector) gives feature contributions from the steering direction
  - SAE.encode(sample_activations) gives actual feature activations per sample
  - Cross-validation: features significant in BOTH analyses are most reliable

This avoids reconstruction error by using encode-only (one-way projection).

Input:
  - Steering vectors from Phase 1
  - Validated causal layers from Phase 2
  - Original experiment data for sample activations

Output:
  - Top contributing features per layer
  - Cross-validated features (significant in both analyses)
  - Feature statistics for paper

Usage:
    python phase3_sae_interpretation.py --model llama --gpu 0
    python phase3_sae_interpretation.py --model llama --gpu 0 --layers 13,18
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from tqdm import tqdm
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class FeatureContribution:
    """A single feature's contribution to steering vector."""
    layer: int
    feature_id: int
    contribution: float  # Raw contribution (can be negative)
    abs_contribution: float  # Absolute value
    direction: str  # "risky" if positive, "safe" if negative


@dataclass
class FeatureStatistics:
    """Statistics for a feature from sample-based analysis."""
    layer: int
    feature_id: int
    bankrupt_mean: float
    safe_mean: float
    difference: float  # bankrupt - safe
    cohen_d: float
    p_value: float
    significant: bool


@dataclass
class CrossValidatedFeature:
    """Feature validated through both steering vector and sample analysis."""
    layer: int
    feature_id: int
    steering_contribution: float  # From steering vector projection
    sample_difference: float  # From sample-based analysis
    cohen_d: float
    p_value: float
    direction: str  # Consistent direction


@dataclass
class LayerInterpretation:
    """Interpretation results for a layer."""
    layer: int
    n_total_features: int

    # Steering vector analysis
    top_risky_features: List[Dict]  # Push toward risky
    top_safe_features: List[Dict]  # Push toward safe
    steering_variance_explained: float

    # Sample-based analysis (if available)
    sample_based_features: List[Dict] = field(default_factory=list)
    n_significant_features: int = 0

    # Cross-validated features
    cross_validated_features: List[Dict] = field(default_factory=list)


class Phase3SAEInterpretation:
    """SAE-based interpretation of steering vectors with sample validation."""

    def __init__(
        self,
        model_name: str = "llama",
        gpu_id: int = 0,
        config_path: str = None
    ):
        self.model_name = model_name
        self.gpu_id = gpu_id

        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "experiment_config_direct_steering.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.device = 'cuda:0'

        self.output_dir = Path(self.config['output_dir']) / "phase3_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.steering_vectors = {}
        self.sae_models = {}
        self.model = None
        self.tokenizer = None

    def load_steering_vectors(self, vectors_path: str = None):
        """Load steering vectors from Phase 1."""
        if vectors_path is None:
            output_dir = Path(self.config['output_dir'])

            # Try .npz first (Phase 1 default)
            vector_files = sorted(
                output_dir.glob(f"steering_vectors_{self.model_name}*.npz"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            # Fallback to .pt
            if not vector_files:
                vector_files = sorted(
                    output_dir.glob(f"steering_vectors_{self.model_name}*.pt"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )

            if not vector_files:
                raise FileNotFoundError(
                    f"No steering vectors found in {output_dir}. "
                    f"Run Phase 1 first."
                )
            vectors_path = vector_files[0]

        print(f"Loading steering vectors: {vectors_path}")

        vectors_path = Path(vectors_path)
        if vectors_path.suffix == '.npz':
            data = np.load(vectors_path, allow_pickle=True)

            # Format: layer_{layer}_vector (from extract_steering_vectors.py)
            for key in data.files:
                if key.endswith('_vector') and key.startswith('layer_'):
                    parts = key.split('_')
                    layer = int(parts[1])
                    self.steering_vectors[layer] = torch.from_numpy(data[key])
        else:
            data = torch.load(vectors_path, map_location='cpu')

            if isinstance(data, dict) and 'vectors' in data:
                vectors = data['vectors']
            else:
                vectors = data

            for layer, vec in vectors.items():
                layer_int = int(layer) if isinstance(layer, str) else layer
                self.steering_vectors[layer_int] = vec

        print(f"Loaded {len(self.steering_vectors)} layer vectors")

    def load_llama_sae(self, layer: int):
        """Load LlamaScope SAE for a layer."""
        import safetensors.torch

        if layer in self.sae_models:
            return self.sae_models[layer]

        # LlamaScope SAEs are stored locally
        local_path = (
            f"/data/.cache/huggingface/hub/models--fnlp--Llama3_1-8B-Base-LXR-8x/"
            f"snapshots/8dbc1d85edfced43081c03c38b05514dbab1368b/"
            f"Llama3_1-8B-Base-L{layer}R-8x/checkpoints/final.safetensors"
        )
        print(f"Loading SAE from local: Llama3_1-8B-Base-L{layer}R-8x")

        try:
            state_dict = safetensors.torch.load_file(local_path)

            # Extract encoder weight (W_enc)
            if 'encoder.weight' in state_dict:
                W_enc = state_dict['encoder.weight']
            elif 'W_enc' in state_dict:
                W_enc = state_dict['W_enc']
            else:
                for key in state_dict:
                    if 'enc' in key.lower() and 'weight' in key.lower():
                        W_enc = state_dict[key]
                        break
                else:
                    raise KeyError(f"Cannot find encoder weight in {list(state_dict.keys())}")

            # Also get bias if available
            if 'encoder.bias' in state_dict:
                b_enc = state_dict['encoder.bias']
            elif 'b_enc' in state_dict:
                b_enc = state_dict['b_enc']
            else:
                b_enc = None

            sae = {
                'W_enc': W_enc.to(self.device),
                'b_enc': b_enc.to(self.device) if b_enc is not None else None,
                'd_model': W_enc.shape[1],
                'd_sae': W_enc.shape[0]
            }

            self.sae_models[layer] = sae
            print(f"  SAE loaded: d_model={sae['d_model']}, d_sae={sae['d_sae']}")
            return sae

        except Exception as e:
            print(f"  Error loading SAE for layer {layer}: {e}")
            return None

    def load_gemma_sae(self, layer: int):
        """Load GemmaScope SAE for a layer."""
        if layer in self.sae_models:
            return self.sae_models[layer]

        try:
            from sae_lens import SAE

            release = "gemma-scope-9b-pt-res-canonical"
            width = self.config['sae']['gemma']['width']
            sae_id = f"layer_{layer}/width_{width}/canonical"

            print(f"Loading GemmaScope: {release}/{sae_id}")

            sae_lens = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=self.device
            )[0]

            W_enc = sae_lens.W_enc

            sae = {
                'W_enc': W_enc,
                'b_enc': sae_lens.b_enc if hasattr(sae_lens, 'b_enc') else None,
                'd_model': W_enc.shape[0],
                'd_sae': W_enc.shape[1],
                'sae_lens': sae_lens
            }

            self.sae_models[layer] = sae
            print(f"  SAE loaded: d_model={sae['d_model']}, d_sae={sae['d_sae']}")
            return sae

        except Exception as e:
            print(f"  Error loading SAE for layer {layer}: {e}")
            return None

    def encode_vector(
        self,
        layer: int,
        vector: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Encode a vector into SAE feature space (one-way projection)."""
        if self.model_name == 'llama':
            sae = self.load_llama_sae(layer)
        else:
            sae = self.load_gemma_sae(layer)

        if sae is None:
            return None

        W_enc = sae['W_enc']
        vec = vector.to(device=self.device, dtype=W_enc.dtype)

        # Project vector onto encoder
        if W_enc.shape[1] == vec.shape[-1]:
            # W_enc is (d_sae, d_model)
            if vec.dim() == 1:
                features = W_enc @ vec
            else:
                features = vec @ W_enc.T
        else:
            # W_enc is (d_model, d_sae)
            if vec.dim() == 1:
                features = vec @ W_enc
            else:
                features = vec @ W_enc

        if sae['b_enc'] is not None:
            features = features + sae['b_enc']

        return features

    def get_top_features_from_steering(
        self,
        features: torch.Tensor,
        layer: int,
        top_k: int = 50
    ) -> List[FeatureContribution]:
        """Get top contributing features from steering vector projection."""
        features_np = features.float().detach().cpu().numpy()
        abs_features = np.abs(features_np)

        # Get top-k by absolute value
        top_indices = np.argsort(abs_features)[-top_k:][::-1]

        contributions = []
        for idx in top_indices:
            contrib = float(features_np[idx])
            contributions.append(FeatureContribution(
                layer=layer,
                feature_id=int(idx),
                contribution=contrib,
                abs_contribution=abs(contrib),
                direction="risky" if contrib > 0 else "safe"
            ))

        return contributions

    def analyze_sample_features(
        self,
        layer: int,
        bankrupt_activations: torch.Tensor,
        safe_activations: torch.Tensor,
        p_threshold: float = 0.05,
        cohen_d_threshold: float = 0.3
    ) -> List[FeatureStatistics]:
        """Analyze features from actual sample activations.

        Args:
            layer: Target layer
            bankrupt_activations: Shape (n_bankrupt, d_model)
            safe_activations: Shape (n_safe, d_model)

        Returns:
            List of significant features with statistics
        """
        # Encode all samples
        bankrupt_features = self.encode_vector(layer, bankrupt_activations)
        safe_features = self.encode_vector(layer, safe_activations)

        if bankrupt_features is None or safe_features is None:
            return []

        bankrupt_np = bankrupt_features.float().detach().cpu().numpy()
        safe_np = safe_features.float().detach().cpu().numpy()

        n_features = bankrupt_np.shape[1]
        significant_features = []

        for feat_id in range(n_features):
            bankrupt_vals = bankrupt_np[:, feat_id]
            safe_vals = safe_np[:, feat_id]

            # Skip if all zeros
            if np.std(bankrupt_vals) == 0 and np.std(safe_vals) == 0:
                continue

            # Calculate statistics
            bankrupt_mean = np.mean(bankrupt_vals)
            safe_mean = np.mean(safe_vals)
            difference = bankrupt_mean - safe_mean

            # Cohen's d
            pooled_std = np.sqrt(
                (np.var(bankrupt_vals) + np.var(safe_vals)) / 2
            )
            cohen_d = difference / pooled_std if pooled_std > 0 else 0

            # T-test
            t_stat, p_value = stats.ttest_ind(bankrupt_vals, safe_vals)

            # Check significance
            significant = p_value < p_threshold and abs(cohen_d) > cohen_d_threshold

            if significant:
                significant_features.append(FeatureStatistics(
                    layer=layer,
                    feature_id=feat_id,
                    bankrupt_mean=float(bankrupt_mean),
                    safe_mean=float(safe_mean),
                    difference=float(difference),
                    cohen_d=float(cohen_d),
                    p_value=float(p_value),
                    significant=True
                ))

        # Sort by absolute Cohen's d
        significant_features.sort(key=lambda x: abs(x.cohen_d), reverse=True)

        return significant_features

    def cross_validate_features(
        self,
        steering_features: List[FeatureContribution],
        sample_features: List[FeatureStatistics]
    ) -> List[CrossValidatedFeature]:
        """Find features significant in both steering vector and sample analysis."""

        # Create lookup for sample features
        sample_lookup = {f.feature_id: f for f in sample_features}
        steering_lookup = {f.feature_id: f for f in steering_features}

        cross_validated = []

        for feat_id in steering_lookup:
            if feat_id in sample_lookup:
                steer = steering_lookup[feat_id]
                sample = sample_lookup[feat_id]

                # Check direction consistency
                steer_direction = "risky" if steer.contribution > 0 else "safe"
                sample_direction = "risky" if sample.difference > 0 else "safe"

                if steer_direction == sample_direction:
                    cross_validated.append(CrossValidatedFeature(
                        layer=steer.layer,
                        feature_id=feat_id,
                        steering_contribution=steer.contribution,
                        sample_difference=sample.difference,
                        cohen_d=sample.cohen_d,
                        p_value=sample.p_value,
                        direction=steer_direction
                    ))

        # Sort by Cohen's d
        cross_validated.sort(key=lambda x: abs(x.cohen_d), reverse=True)

        return cross_validated

    def interpret_layer(
        self,
        layer: int,
        bankrupt_activations: Optional[torch.Tensor] = None,
        safe_activations: Optional[torch.Tensor] = None
    ) -> Optional[LayerInterpretation]:
        """Interpret steering vector for a layer with optional sample validation."""

        if layer not in self.steering_vectors:
            print(f"No steering vector for layer {layer}")
            return None

        # === PART 1: Steering Vector Projection ===
        steering_vec = self.steering_vectors[layer]
        features = self.encode_vector(layer, steering_vec)

        if features is None:
            return None

        top_k = self.config.get('phase3_sae_interpretation', {}).get('top_k_per_layer', 50)
        contributions = self.get_top_features_from_steering(features, layer, top_k)

        # Split by direction
        risky_features = [c for c in contributions if c.direction == "risky"]
        safe_features = [c for c in contributions if c.direction == "safe"]

        # Calculate variance explained
        features_np = features.float().detach().cpu().numpy()
        total_var = np.sum(features_np ** 2)
        top_var = sum(c.abs_contribution ** 2 for c in contributions)
        var_explained = top_var / total_var if total_var > 0 else 0

        # === PART 2: Sample-Based Analysis (if activations provided) ===
        sample_based_features = []
        cross_validated_features = []

        if bankrupt_activations is not None and safe_activations is not None:
            sample_based_features = self.analyze_sample_features(
                layer, bankrupt_activations, safe_activations
            )

            # Cross-validate
            cross_validated_features = self.cross_validate_features(
                contributions, sample_based_features
            )

        return LayerInterpretation(
            layer=layer,
            n_total_features=len(features),
            top_risky_features=[asdict(f) for f in risky_features[:25]],
            top_safe_features=[asdict(f) for f in safe_features[:25]],
            steering_variance_explained=round(var_explained, 4),
            sample_based_features=[asdict(f) for f in sample_based_features[:50]],
            n_significant_features=len(sample_based_features),
            cross_validated_features=[asdict(f) for f in cross_validated_features[:30]]
        )

    def run(
        self,
        vectors_path: str = None,
        target_layers: List[int] = None,
        validated_layers: List[int] = None,
        sample_activations: Dict[int, Dict[str, torch.Tensor]] = None
    ):
        """Run Phase 3 interpretation.

        Args:
            vectors_path: Path to steering vectors
            target_layers: Specific layers to analyze
            validated_layers: Causal layers from Phase 2
            sample_activations: Dict of {layer: {'bankrupt': tensor, 'safe': tensor}}
        """
        print("=" * 70)
        print("PHASE 3: SAE INTERPRETATION (IMPROVED)")
        print("=" * 70)
        print(f"Model: {self.model_name}")
        print("Method: One-way SAE projection (no reconstruction error)")

        # Load steering vectors
        self.load_steering_vectors(vectors_path)

        # Determine target layers
        phase3_config = self.config.get('phase3_sae_interpretation', {})

        if target_layers is not None:
            pass
        elif validated_layers:
            target_layers = validated_layers
            print(f"Using {len(validated_layers)} validated causal layers from Phase 2")
        elif phase3_config.get('default_layers'):
            target_layers = phase3_config['default_layers']
            print(f"Using default layers: {target_layers}")
        else:
            target_layers = sorted(self.steering_vectors.keys())

        available_layers = [l for l in target_layers if l in self.steering_vectors]
        print(f"\nTarget layers: {available_layers}")

        # Interpret each layer
        all_interpretations = []

        for layer in available_layers:
            print(f"\n{'='*60}")
            print(f"Interpreting Layer {layer}")
            print(f"{'='*60}")

            # Get sample activations if available
            bankrupt_act = None
            safe_act = None
            if sample_activations and layer in sample_activations:
                bankrupt_act = sample_activations[layer].get('bankrupt')
                safe_act = sample_activations[layer].get('safe')
                print(f"  Using sample activations: {bankrupt_act.shape[0]} bankrupt, {safe_act.shape[0]} safe")

            interpretation = self.interpret_layer(
                layer,
                bankrupt_activations=bankrupt_act,
                safe_activations=safe_act
            )

            if interpretation is None:
                continue

            all_interpretations.append(interpretation)

            # Print summary
            print(f"\n  STEERING VECTOR ANALYSIS:")
            print(f"    Top risky features: {len(interpretation.top_risky_features)}")
            print(f"    Top safe features: {len(interpretation.top_safe_features)}")
            print(f"    Variance explained: {interpretation.steering_variance_explained:.1%}")

            # Show top 5 each
            print("\n    Top 5 RISKY features (push toward gambling):")
            for f in interpretation.top_risky_features[:5]:
                print(f"      Feature {f['feature_id']}: contribution={f['contribution']:.4f}")

            print("\n    Top 5 SAFE features (push toward stopping):")
            for f in interpretation.top_safe_features[:5]:
                print(f"      Feature {f['feature_id']}: contribution={f['contribution']:.4f}")

            if interpretation.n_significant_features > 0:
                print(f"\n  SAMPLE-BASED ANALYSIS:")
                print(f"    Significant features: {interpretation.n_significant_features}")
                print(f"    Cross-validated: {len(interpretation.cross_validated_features)}")

                if interpretation.cross_validated_features:
                    print("\n    Top 5 CROSS-VALIDATED features:")
                    for f in interpretation.cross_validated_features[:5]:
                        print(f"      Feature {f['feature_id']}: Cohen's d={f['cohen_d']:.3f}, direction={f['direction']}")

        # Save results
        self.save_results(all_interpretations)

        return all_interpretations

    def save_results(self, interpretations: List[LayerInterpretation]):
        """Save interpretation results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        output = {
            'model': self.model_name,
            'timestamp': timestamp,
            'method': 'one_way_projection_with_sample_validation',
            'note': 'SAE encode only - no reconstruction error',
            'n_layers': len(interpretations),
            'interpretations': [asdict(i) for i in interpretations]
        }

        output_path = self.output_dir / f"sae_interpretation_{self.model_name}_{timestamp}.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)

        print(f"\nResults saved: {output_path}")

        # Summary for paper
        summary = {
            'model': self.model_name,
            'timestamp': timestamp,
            'layers_analyzed': [i.layer for i in interpretations],
            'steering_features': {},
            'cross_validated_features': {},
            'statistics': {}
        }

        for interp in interpretations:
            layer = interp.layer
            summary['steering_features'][layer] = {
                'n_risky': len(interp.top_risky_features),
                'n_safe': len(interp.top_safe_features),
                'top_risky_ids': [f['feature_id'] for f in interp.top_risky_features[:10]],
                'top_safe_ids': [f['feature_id'] for f in interp.top_safe_features[:10]]
            }
            summary['cross_validated_features'][layer] = {
                'n_features': len(interp.cross_validated_features),
                'top_ids': [f['feature_id'] for f in interp.cross_validated_features[:10]]
            }
            summary['statistics'][layer] = {
                'variance_explained': interp.steering_variance_explained,
                'n_significant_sample_features': interp.n_significant_features
            }

        summary_path = self.output_dir / f"interpretation_summary_{self.model_name}_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)

        print(f"Summary saved: {summary_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Phase 3: SAE Interpretation (Improved)')
    parser.add_argument('--model', type=str, default='llama',
                       choices=['llama', 'gemma', 'gemma_base'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--vectors', type=str, default=None,
                       help='Path to steering vectors')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated layer numbers (e.g., "13,18")')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--phase2-results', type=str, default=None,
                       help='Path to Phase 2 layer validations JSON for causal layer selection')

    args = parser.parse_args()

    target_layers = None
    if args.layers:
        target_layers = [int(l) for l in args.layers.split(',')]

    phase3 = Phase3SAEInterpretation(
        model_name=args.model,
        gpu_id=args.gpu,
        config_path=args.config
    )

    # Load validated layers from Phase 2 results if provided
    validated_layers = None
    if args.phase2_results:
        try:
            import json
            with open(args.phase2_results, 'r') as f:
                phase2_data = json.load(f)
            validations = phase2_data.get('validations', [])
            validated_layers = [v['layer'] for v in validations if v.get('is_causal')]
            if validated_layers:
                print(f"Using {len(validated_layers)} validated causal layers from Phase 2")
        except Exception as e:
            print(f"Warning: Failed to load Phase 2 results ({e}); falling back to defaults.")

    phase3.run(
        vectors_path=args.vectors,
        target_layers=target_layers,
        validated_layers=validated_layers
    )


if __name__ == '__main__':
    main()
