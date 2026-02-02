#!/usr/bin/env python3
"""
Phase 4: SAE Interpretation

Project steering vectors onto SAE feature space and cross-validate with Phase 2 results.

Input:
    - Phase 2: significant_features.json
    - Phase 3: steering_vectors.npz

Output:
    - phase4_interpretation/sae_features.json
    - phase4_interpretation/cross_validated_features.json

Usage:
    python phase4_sae_interpretation.py --gpu 0
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, get_phase_output_dir
from utils import GemmaSAE, logger


@dataclass
class FeatureContribution:
    """Feature contribution from steering vector projection."""
    layer: int
    feature_id: int
    contribution: float
    abs_contribution: float
    direction: str  # "risky" or "safe"


@dataclass
class CrossValidatedFeature:
    """Feature validated in both Phase 2 and Phase 4."""
    layer: int
    feature_id: int
    steering_contribution: float
    cohens_d: float
    p_value: float
    direction: str


class Phase4SAEInterpretation:
    """Interpret steering vectors through SAE feature space."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        gpu_id: int = 0,
        target_layers: Optional[List[int]] = None
    ):
        self.config = load_config(config_path)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.device = 'cuda:0'

        if target_layers:
            self.target_layers = target_layers
        else:
            self.target_layers = self.config.get('target_layers', list(range(20, 42)))

        # Directories
        self.phase2_dir = get_phase_output_dir(self.config, 2)
        self.phase3_dir = get_phase_output_dir(self.config, 3)
        self.output_dir = get_phase_output_dir(self.config, 4)

        self.top_k = self.config.get('phase4', {}).get('top_k_features', 50)

        self.sae = None

    def load_steering_vectors(self) -> Dict[int, np.ndarray]:
        """Load steering vectors from Phase 3."""
        vectors_file = self.phase3_dir / "steering_vectors.npz"

        if not vectors_file.exists():
            raise FileNotFoundError(f"Steering vectors not found: {vectors_file}")

        data = np.load(vectors_file, allow_pickle=True)

        vectors = {}
        for layer in self.target_layers:
            key = f'layer_{layer}_vector'
            if key in data:
                vectors[layer] = data[key]

        logger.info(f"Loaded steering vectors for {len(vectors)} layers")
        return vectors

    def load_phase2_features(self) -> Dict[int, Dict[int, Dict]]:
        """
        Load significant features from Phase 2.

        Returns:
            Dict mapping layer -> feature_id -> feature_stats
        """
        sig_file = self.phase2_dir / "significant_features.json"

        if not sig_file.exists():
            logger.warning(f"Phase 2 results not found: {sig_file}")
            return {}

        with open(sig_file, 'r') as f:
            data = json.load(f)

        # Build lookup
        features = {}
        for f in data.get('risky_features', []) + data.get('safe_features', []):
            layer = f['layer']
            fid = f['feature_id']

            if layer not in features:
                features[layer] = {}
            features[layer][fid] = f

        logger.info(f"Loaded {sum(len(v) for v in features.values())} significant features from Phase 2")
        return features

    def project_steering_vector(
        self,
        steering_vec: np.ndarray,
        layer: int
    ) -> Optional[List[FeatureContribution]]:
        """
        Project steering vector onto SAE feature space.

        Returns:
            List of top-k feature contributions
        """
        if self.sae is None:
            sae_config = self.config.get('sae', {})
            self.sae = GemmaSAE(device=self.device, width=sae_config.get('width', '16k'))

        # Convert to tensor
        sv_tensor = torch.from_numpy(steering_vec).unsqueeze(0).float()

        # Encode through SAE
        features = self.sae.encode(sv_tensor, layer)
        if features is None:
            return None

        features_np = features.squeeze(0).cpu().numpy()

        # Get top-k by absolute value
        abs_features = np.abs(features_np)
        top_indices = np.argsort(abs_features)[-self.top_k:][::-1]

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

    def cross_validate(
        self,
        steering_features: List[FeatureContribution],
        phase2_features: Dict[int, Dict]
    ) -> List[CrossValidatedFeature]:
        """Find features significant in both steering projection and Phase 2."""
        cross_validated = []

        for sf in steering_features:
            layer_features = phase2_features.get(sf.layer, {})
            p2_feature = layer_features.get(sf.feature_id)

            if p2_feature is None:
                continue

            # Check direction consistency
            p2_direction = p2_feature['direction']
            if sf.direction == p2_direction:
                cross_validated.append(CrossValidatedFeature(
                    layer=sf.layer,
                    feature_id=sf.feature_id,
                    steering_contribution=sf.contribution,
                    cohens_d=p2_feature['cohens_d'],
                    p_value=p2_feature['p_value'],
                    direction=sf.direction
                ))

        # Sort by Cohen's d
        cross_validated.sort(key=lambda x: abs(x.cohens_d), reverse=True)

        return cross_validated

    def run(self):
        """Run Phase 4 SAE interpretation."""
        logger.info("=" * 70)
        logger.info("PHASE 4: SAE INTERPRETATION")
        logger.info("=" * 70)
        logger.info(f"Top-k features per layer: {self.top_k}")

        # Load inputs
        steering_vectors = self.load_steering_vectors()
        phase2_features = self.load_phase2_features()

        all_steering_features = []
        all_cross_validated = []

        # Process each layer
        for layer in self.target_layers:
            if layer not in steering_vectors:
                continue

            logger.info(f"\nProcessing Layer {layer}")

            # Project steering vector
            contributions = self.project_steering_vector(steering_vectors[layer], layer)
            if contributions is None:
                continue

            all_steering_features.extend(contributions)

            # Count directions
            n_risky = sum(1 for c in contributions if c.direction == "risky")
            n_safe = len(contributions) - n_risky
            logger.info(f"  Top-{self.top_k}: {n_risky} risky, {n_safe} safe")

            # Cross-validate
            cv_features = self.cross_validate(contributions, phase2_features)
            all_cross_validated.extend(cv_features)
            logger.info(f"  Cross-validated: {len(cv_features)}")

        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total steering features: {len(all_steering_features)}")
        logger.info(f"Cross-validated features: {len(all_cross_validated)}")

        if all_cross_validated:
            cv_risky = sum(1 for f in all_cross_validated if f.direction == "risky")
            cv_safe = len(all_cross_validated) - cv_risky
            logger.info(f"  Risky: {cv_risky}, Safe: {cv_safe}")

            logger.info("\nTop 10 Cross-Validated Features:")
            for f in all_cross_validated[:10]:
                logger.info(f"  L{f.layer}:F{f.feature_id} d={f.cohens_d:.3f} contrib={f.steering_contribution:.4f} [{f.direction}]")

        # Save results
        self._save_results(all_steering_features, all_cross_validated)

        logger.info("\nPhase 4 complete!")

    def _save_results(
        self,
        steering_features: List[FeatureContribution],
        cross_validated: List[CrossValidatedFeature]
    ):
        """Save interpretation results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save steering features
        steering_output = {
            'timestamp': timestamp,
            'top_k': self.top_k,
            'n_features': len(steering_features),
            'features': [asdict(f) for f in steering_features]
        }

        steering_file = self.output_dir / "sae_features.json"
        with open(steering_file, 'w') as f:
            json.dump(steering_output, f, indent=2)
        logger.info(f"Saved: {steering_file}")

        # Save cross-validated features
        cv_output = {
            'timestamp': timestamp,
            'n_features': len(cross_validated),
            'risky_features': [asdict(f) for f in cross_validated if f.direction == "risky"],
            'safe_features': [asdict(f) for f in cross_validated if f.direction == "safe"]
        }

        cv_file = self.output_dir / "cross_validated_features.json"
        with open(cv_file, 'w') as f:
            json.dump(cv_output, f, indent=2)
        logger.info(f"Saved: {cv_file}")


def main():
    parser = argparse.ArgumentParser(description='Phase 4: SAE Interpretation')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--layers', type=str, default=None)

    args = parser.parse_args()

    target_layers = None
    if args.layers:
        target_layers = [int(l) for l in args.layers.split(',')]

    phase4 = Phase4SAEInterpretation(
        config_path=args.config,
        gpu_id=args.gpu,
        target_layers=target_layers
    )
    phase4.run()


if __name__ == '__main__':
    main()
