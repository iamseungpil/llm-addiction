#!/usr/bin/env python3
"""
Phase 5: Gambling-Context Feature Interpretation

Generate gambling-specific explanations for validated features:
1. Use gambling-context SelfIE templates (not generic "X means")
2. Patch decoder vector at the SAME layer as the feature
3. Cluster features by semantic similarity
4. Generate risk-seeking vs loss-aversion narrative

This implements enhanced interpretation for the A+B+C+D causal analysis strategy.

Usage:
    python gambling_interpretation.py --model llama --gpu 0 \\
        --validated validated_features_llama.json

Design: Context-specific SelfIE with feature clustering and narrative generation.

References:
- SelfIE (Luo et al., 2024): Self-Interpretation of Large Language Model Embeddings
- Patchscopes (Ghandeharioun et al., 2024): Inspecting hidden representations
"""

# CRITICAL: Set CUDA_VISIBLE_DEVICES before torch import
# This ensures the --gpu argument maps to the correct physical GPU
import os
import sys
import argparse

def _set_gpu_before_torch():
    """Parse --gpu argument and set CUDA_VISIBLE_DEVICES before torch import."""
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            gpu_id = sys.argv[i + 1]
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
            return gpu_id
    return None

_gpu_set = _set_gpu_before_torch()

import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from tqdm import tqdm
import yaml
from collections import defaultdict

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    setup_logging,
    ModelRegistry,
    load_model_and_tokenizer,
    load_config,
    get_gpu_memory_info,
    clear_gpu_memory,
    get_default_config_path,
    get_causal_feature_src,
    CheckpointManager
)

# Add LlamaScope path dynamically
sys.path.insert(0, str(get_causal_feature_src()))


# =============================================================================
# Gambling-Context Templates
# =============================================================================

# English templates for gambling context
GAMBLING_TEMPLATES_EN = [
    # Risk behavior templates
    {
        'template': 'When gambling, if this pattern activates, the player tends to {X}',
        'placeholder_pos': -1,  # X at end
        'context': 'behavior'
    },
    {
        'template': 'This pattern represents the concept of "{X}" in risk-taking decisions',
        'placeholder_pos': -1,
        'context': 'concept'
    },
    {
        'template': 'In a betting scenario, this feature indicates {X}',
        'placeholder_pos': -1,
        'context': 'indication'
    },
    {
        'template': 'When deciding to bet or stop, this pattern signals {X}',
        'placeholder_pos': -1,
        'context': 'decision'
    },
    {
        'template': 'This gambling-related feature captures the tendency to {X}',
        'placeholder_pos': -1,
        'context': 'tendency'
    }
]

# Korean templates for gambling context
GAMBLING_TEMPLATES_KO = [
    {
        'template': '도박 상황에서 이 패턴이 활성화되면, 플레이어는 {X}',
        'placeholder_pos': -1,
        'context': 'behavior'
    },
    {
        'template': '이 패턴은 위험 감수 결정에서 "{X}"의 개념을 나타냅니다',
        'placeholder_pos': -1,
        'context': 'concept'
    },
    {
        'template': '베팅 시나리오에서 이 feature는 {X}를 나타냅니다',
        'placeholder_pos': -1,
        'context': 'indication'
    }
]

# Completion suffixes for generating explanations
COMPLETION_SUFFIXES = {
    'behavior': '',
    'concept': '',
    'indication': '',
    'decision': '',
    'tendency': ''
}


@dataclass
class FeatureExplanation:
    """Explanation for a validated feature."""
    layer: int
    feature_id: int
    direction: str  # 'safe' or 'risky'
    explanation: str
    template_used: str
    scale_used: float
    confidence: float
    raw_generation: str
    gambling_keywords: List[str]


@dataclass
class FeatureCluster:
    """Cluster of semantically similar features."""
    cluster_id: int
    theme: str
    features: List[FeatureExplanation]
    representative_explanation: str


@dataclass
class InterpretationResults:
    """Complete interpretation results."""
    model: str
    n_features: int
    explanations: List[FeatureExplanation]
    clusters: List[FeatureCluster]
    safe_narrative: str
    risky_narrative: str
    summary_stats: Dict


class SAEDecoder:
    """
    Access SAE decoder vectors for feature interpretation.
    """

    def __init__(self, model_name: str, device: str = 'cuda:0', logger=None):
        """Initialize SAE decoder access."""
        self.model_name = model_name
        self.device = device
        self.logger = logger
        self.saes = {}

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def load_sae(self, layer: int) -> Any:
        """Load SAE for specified layer."""
        if layer in self.saes:
            return self.saes[layer]

        if self.model_name == 'llama':
            try:
                from llama_scope_working import LlamaScopeWorking
                self._log(f"Loading LlamaScope SAE for layer {layer}...")
                sae = LlamaScopeWorking(layer=layer, device=self.device)
                self.saes[layer] = sae
                return sae
            except Exception as e:
                self._log(f"Error loading LlamaScope: {e}")
                raise

        elif self.model_name == 'gemma':
            from sae_lens import SAE
            # Try canonical release first (more reliable)
            try:
                release = "gemma-scope-9b-pt-res-canonical"
                sae_id = f"layer_{layer}/width_16k/canonical"
                self._log(f"Loading GemmaScope: {release}/{sae_id}")
                sae = SAE.from_pretrained(
                    release=release,
                    sae_id=sae_id,
                    device=self.device
                )[0]
                self.saes[layer] = sae
                return sae
            except Exception as e:
                self._log(f"Canonical release failed: {e}, trying fallback...")
                # Fallback to non-canonical release
                release = "gemma-scope-9b-pt-res"
                for width in ['16k', '32k']:
                    try:
                        sae_id = f"layer_{layer}/width_{width}/average_l0_71"
                        self._log(f"Loading GemmaScope: {release}/{sae_id}")
                        sae = SAE.from_pretrained(
                            release=release,
                            sae_id=sae_id,
                            device=self.device
                        )[0]
                        self.saes[layer] = sae
                        return sae
                    except Exception:
                        continue
                raise RuntimeError(f"Failed to load GemmaScope for layer {layer}")

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def get_decoder_vector(self, layer: int, feature_id: int) -> torch.Tensor:
        """Get decoder vector for a specific feature."""
        sae = self.load_sae(layer)

        # Try different decoder attribute names
        if hasattr(sae, 'sae') and hasattr(sae.sae, 'W_D'):
            # LlamaScopeWorking wrapper: sae.sae.W_D
            decoder = sae.sae.W_D
        elif hasattr(sae, 'W_D'):
            # Direct W_D access
            decoder = sae.W_D
        elif hasattr(sae, 'sae') and hasattr(sae.sae, 'W_dec'):
            # Some LlamaScope versions
            decoder = sae.sae.W_dec
        elif hasattr(sae, 'W_dec'):
            # sae_lens style
            decoder = sae.W_dec
        elif hasattr(sae, 'decoder'):
            decoder = sae.decoder.weight.T
        else:
            # Debug: print available attributes
            attrs = [a for a in dir(sae) if not a.startswith('_')]
            sae_attrs = [a for a in dir(sae.sae) if not a.startswith('_')] if hasattr(sae, 'sae') else []
            raise RuntimeError(f"Cannot find decoder for layer {layer}. sae attrs: {attrs[:10]}, sae.sae attrs: {sae_attrs[:10]}")

        return decoder[feature_id].detach().float()


class GamblingInterpreter:
    """
    Generate gambling-context explanations for validated features.

    Uses SelfIE methodology with gambling-specific templates.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        config: Dict,
        device: str = 'cuda:0',
        logger=None
    ):
        """
        Initialize gambling interpreter.

        Args:
            model: Loaded transformer model
            tokenizer: Corresponding tokenizer
            model_name: 'llama' or 'gemma'
            config: Configuration dictionary
            device: Device for computation
            logger: Optional logger
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.config = config
        self.device = device
        self.logger = logger
        self.model_config = ModelRegistry.get(model_name)

        # SAE decoder access
        self.sae_decoder = SAEDecoder(model_name, device, logger)

        # Get config values
        interp_config = config.get('interpretation', {})
        self.use_gambling_context = interp_config.get('use_gambling_context', True)
        self.scales = interp_config.get('scales', [3.0, 5.0, 8.0])
        self.n_templates = interp_config.get('n_templates', 3)
        self.max_new_tokens = interp_config.get('max_new_tokens', 50)
        self.enable_clustering = interp_config.get('enable_clustering', True)
        self.min_cluster_size = interp_config.get('min_cluster_size', 3)

        # Select templates based on model
        if self.model_config.use_chat_template:
            self.templates = GAMBLING_TEMPLATES_EN[:self.n_templates]
        else:
            self.templates = GAMBLING_TEMPLATES_EN[:self.n_templates]

        # Gambling keywords for relevance scoring
        self.gambling_keywords = [
            'risk', 'bet', 'gambl', 'stop', 'continu', 'win', 'loss', 'lose',
            'money', 'reward', 'caution', 'safe', 'danger', 'chance', 'luck',
            'decision', 'uncertain', 'profit', 'balance', 'stake', 'wager',
            'conservative', 'aggressive', 'cautious', 'bold', 'quit', 'play'
        ]

    def _log(self, msg: str):
        """Log a message."""
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _format_prompt(self, template: str) -> str:
        """Format prompt for model."""
        if self.model_config.use_chat_template:
            # For chat models, wrap in chat format
            chat = [{"role": "user", "content": f"Complete this sentence: {template}"}]
            return self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        return template

    def _find_placeholder_position(self, prompt: str) -> int:
        """Find the position to patch in the tokenized prompt.

        Looks for '___' placeholder (which replaced {X}) or falls back to
        a sensible default position.
        """
        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        token_strs = [self.tokenizer.decode([t]) for t in tokens]

        # Look for ___ placeholder (which replaced {X})
        for i, t_str in enumerate(token_strs):
            if '___' in t_str or '_' in t_str:
                return i

        # Also try looking for patterns that might contain the placeholder
        for i, t_str in enumerate(token_strs):
            # Common patterns after placeholder: quotes, colons, periods
            if t_str.strip() in ['"', "'", ':', '.', '은', '는', 'is', 'means']:
                # Return position before this token
                return max(0, i - 1)

        # Default to position before last token
        return max(0, len(tokens) - 2)

    def explain_feature(
        self,
        layer: int,
        feature_id: int,
        direction: str = 'unknown',
        scale: float = 5.0,
        template_idx: int = 0
    ) -> FeatureExplanation:
        """
        Generate explanation for a single feature.

        IMPORTANT: Patches decoder vector at the SAME layer as the feature,
        not at an early layer. This ensures we capture the feature's
        semantics as they exist at that layer.

        Args:
            layer: Layer of the feature
            feature_id: Feature ID in SAE
            direction: 'safe' or 'risky'
            scale: Scale factor for decoder vector
            template_idx: Which template to use

        Returns:
            FeatureExplanation dataclass
        """
        # Get decoder vector
        feature_vec = self.sae_decoder.get_decoder_vector(layer, feature_id)
        feature_vec = feature_vec.to(self.device) * scale

        # Select template
        template_info = self.templates[template_idx % len(self.templates)]
        template = template_info['template']

        # Format prompt (replace {X} with placeholder)
        prompt = template.replace('{X}', '___')
        formatted_prompt = self._format_prompt(prompt)

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors='pt',
            add_special_tokens=True
        ).to(self.device)

        # Find position to patch
        patch_position = self._find_placeholder_position(formatted_prompt)

        # Create patching hook - PATCHES AT SAME LAYER as feature
        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            # Patch at the placeholder position
            if patch_position < hidden.shape[1]:
                hidden[0, patch_position, :] = feature_vec.to(hidden.dtype)

            if rest is not None:
                return (hidden,) + rest
            return hidden

        # Register hook at the FEATURE'S layer (not early layer)
        target_layer = layer
        if self.model_name == 'llama':
            layer_module = self.model.model.layers[target_layer]
        else:
            layer_module = self.model.model.layers[target_layer]

        hook = layer_module.register_forward_hook(patch_hook)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract explanation (after the prompt)
            explanation = generated[len(formatted_prompt):].strip()
            explanation = self._clean_explanation(explanation)

            # Find gambling keywords
            found_keywords = self._find_gambling_keywords(explanation)

            # Compute confidence
            confidence = self._compute_confidence(explanation, found_keywords)

        finally:
            hook.remove()
            del inputs

        return FeatureExplanation(
            layer=layer,
            feature_id=feature_id,
            direction=direction,
            explanation=explanation,
            template_used=template,
            scale_used=scale,
            confidence=confidence,
            raw_generation=generated,
            gambling_keywords=found_keywords
        )

    def _clean_explanation(self, explanation: str) -> str:
        """Clean up generated explanation."""
        if not explanation:
            return ""

        # Remove trailing incomplete sentences
        for end_char in ['.', '!', '?']:
            last_pos = explanation.rfind(end_char)
            if last_pos > 10:  # At least some content
                explanation = explanation[:last_pos + 1]
                break

        # Remove quotes if unbalanced
        if explanation.count('"') % 2 == 1:
            explanation = explanation.replace('"', '')

        # Truncate if too long
        words = explanation.split()
        if len(words) > 30:
            explanation = ' '.join(words[:30]) + '...'

        return explanation.strip()

    def _find_gambling_keywords(self, text: str) -> List[str]:
        """Find gambling-related keywords in text."""
        text_lower = text.lower()
        found = []
        for keyword in self.gambling_keywords:
            if keyword in text_lower:
                found.append(keyword)
        return found

    def _compute_confidence(self, explanation: str, keywords: List[str]) -> float:
        """Compute confidence score for explanation."""
        if not explanation:
            return 0.0

        score = 0.3  # Base score

        # Length bonus
        words = explanation.split()
        if len(words) >= 5:
            score += 0.1
        if len(words) >= 10:
            score += 0.1

        # Complete sentence bonus
        if explanation.endswith(('.', '!', '?')):
            score += 0.1

        # Gambling keyword bonus
        score += min(len(keywords) * 0.1, 0.3)

        # Specificity bonus
        specific_words = ['bet', 'stop', 'risk', 'caution', 'aggressive']
        for word in specific_words:
            if word in explanation.lower():
                score += 0.05

        return min(max(score, 0.0), 1.0)

    def explain_features_batch(
        self,
        features: List[Dict]
    ) -> List[FeatureExplanation]:
        """
        Generate explanations for multiple features.

        Tries multiple scales and templates, returns best explanation.

        Args:
            features: List of feature dicts with layer, feature_id, direction

        Returns:
            List of best explanations
        """
        explanations = []

        for feat in tqdm(features, desc="Generating explanations"):
            layer = feat['layer']
            feature_id = feat['feature_id']
            direction = feat.get('direction', 'unknown')

            best_explanation = None
            best_confidence = -1

            # Try different scales and templates
            for scale in self.scales:
                for template_idx in range(len(self.templates)):
                    try:
                        exp = self.explain_feature(
                            layer=layer,
                            feature_id=feature_id,
                            direction=direction,
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
                    direction=direction,
                    explanation="[Failed to generate explanation]",
                    template_used="",
                    scale_used=0,
                    confidence=0.0,
                    raw_generation="",
                    gambling_keywords=[]
                ))

            # Clear cache periodically
            if len(explanations) % 5 == 0:
                clear_gpu_memory(self.device)

        return explanations

    def cluster_features(
        self,
        explanations: List[FeatureExplanation]
    ) -> List[FeatureCluster]:
        """
        Cluster features by semantic similarity.

        Uses keyword overlap as similarity metric.

        Args:
            explanations: List of feature explanations

        Returns:
            List of feature clusters
        """
        if not self.enable_clustering or len(explanations) < self.min_cluster_size:
            return []

        # Simple keyword-based clustering
        clusters_dict = defaultdict(list)

        for exp in explanations:
            if not exp.gambling_keywords:
                clusters_dict['other'].append(exp)
            else:
                # Use most prominent keyword as cluster key
                primary_keyword = exp.gambling_keywords[0]
                clusters_dict[primary_keyword].append(exp)

        # Convert to cluster objects
        clusters = []
        cluster_id = 0
        for theme, features in clusters_dict.items():
            if len(features) >= self.min_cluster_size:
                # Get representative explanation (highest confidence)
                representative = max(features, key=lambda x: x.confidence)
                clusters.append(FeatureCluster(
                    cluster_id=cluster_id,
                    theme=theme,
                    features=features,
                    representative_explanation=representative.explanation
                ))
                cluster_id += 1

        return clusters

    def generate_narrative(
        self,
        explanations: List[FeatureExplanation]
    ) -> Tuple[str, str]:
        """
        Generate narratives for safe and risky features.

        Args:
            explanations: List of feature explanations

        Returns:
            Tuple of (safe_narrative, risky_narrative)
        """
        safe_features = [e for e in explanations if e.direction == 'safe']
        risky_features = [e for e in explanations if e.direction == 'risky']

        # Build safe narrative
        if safe_features:
            safe_themes = set()
            for f in safe_features[:5]:  # Top 5
                for kw in f.gambling_keywords[:2]:
                    safe_themes.add(kw)

            safe_narrative = (
                f"The model's safe/conservative behavior is associated with {len(safe_features)} "
                f"validated features. Key themes include: {', '.join(safe_themes) if safe_themes else 'general caution'}. "
                f"Top explanation: \"{safe_features[0].explanation if safe_features else 'N/A'}\""
            )
        else:
            safe_narrative = "No safe features identified."

        # Build risky narrative
        if risky_features:
            risky_themes = set()
            for f in risky_features[:5]:
                for kw in f.gambling_keywords[:2]:
                    risky_themes.add(kw)

            risky_narrative = (
                f"The model's risky/aggressive behavior is associated with {len(risky_features)} "
                f"validated features. Key themes include: {', '.join(risky_themes) if risky_themes else 'risk-taking'}. "
                f"Top explanation: \"{risky_features[0].explanation if risky_features else 'N/A'}\""
            )
        else:
            risky_narrative = "No risky features identified."

        return safe_narrative, risky_narrative

    def run_full_interpretation(
        self,
        validated_features: List[Dict],
        checkpoint_mgr: Optional[CheckpointManager] = None
    ) -> InterpretationResults:
        """
        Run complete interpretation pipeline.

        Args:
            validated_features: List of validated feature dicts
            checkpoint_mgr: Optional checkpoint manager

        Returns:
            InterpretationResults with all analysis
        """
        self._log("=" * 60)
        self._log("GAMBLING-CONTEXT FEATURE INTERPRETATION")
        self._log("=" * 60)
        self._log(f"Features to interpret: {len(validated_features)}")
        self._log(f"Scales: {self.scales}")
        self._log(f"Templates: {len(self.templates)}")

        # Generate explanations
        explanations = self.explain_features_batch(validated_features)

        # Cluster features
        clusters = self.cluster_features(explanations)

        # Generate narratives
        safe_narrative, risky_narrative = self.generate_narrative(explanations)

        # Compute summary stats
        n_safe = sum(1 for e in explanations if e.direction == 'safe')
        n_risky = sum(1 for e in explanations if e.direction == 'risky')
        avg_confidence = np.mean([e.confidence for e in explanations])
        n_with_keywords = sum(1 for e in explanations if e.gambling_keywords)

        summary_stats = {
            'n_safe_features': n_safe,
            'n_risky_features': n_risky,
            'avg_confidence': avg_confidence,
            'n_with_gambling_keywords': n_with_keywords,
            'n_clusters': len(clusters)
        }

        results = InterpretationResults(
            model=self.model_name,
            n_features=len(validated_features),
            explanations=explanations,
            clusters=clusters,
            safe_narrative=safe_narrative,
            risky_narrative=risky_narrative,
            summary_stats=summary_stats
        )

        # Print summary
        self._log("\n" + "=" * 60)
        self._log("INTERPRETATION SUMMARY")
        self._log("=" * 60)
        self._log(f"Safe features: {n_safe}")
        self._log(f"Risky features: {n_risky}")
        self._log(f"Avg confidence: {avg_confidence:.3f}")
        self._log(f"Features with gambling keywords: {n_with_keywords}")
        self._log(f"Clusters identified: {len(clusters)}")

        return results


def load_validated_features(filepath: Path) -> List[Dict]:
    """Load validated features from JSON file.

    Supports multiple input formats:
    1. Phase 3 output: {'interpretations': [{'top_risky_features': [...], 'top_safe_features': [...]}]}
    2. Legacy format: {'validated_features': [...]}
    3. Simple feature list: {'features': [...]}
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Format 1: Phase 3 SAE interpretation output
    if 'interpretations' in data:
        features = []
        for interp in data['interpretations']:
            # Add risky features
            for f in interp.get('top_risky_features', []):
                features.append(f)
            # Add safe features
            for f in interp.get('top_safe_features', []):
                features.append(f)
        return features

    # Format 2: Legacy validated_features format
    if 'validated_features' in data:
        return data['validated_features']

    # Format 3: Simple features list (from top_features_*.json)
    if 'features' in data:
        return data['features']

    return []


def save_interpretation_results(results: InterpretationResults, output_path: Path) -> None:
    """Save interpretation results to JSON file."""
    output_data = {
        'model': results.model,
        'n_features': results.n_features,
        'summary_stats': results.summary_stats,
        'safe_narrative': results.safe_narrative,
        'risky_narrative': results.risky_narrative,
        'explanations': [asdict(e) for e in results.explanations],
        'clusters': [
            {
                'cluster_id': c.cluster_id,
                'theme': c.theme,
                'n_features': len(c.features),
                'representative_explanation': c.representative_explanation,
                'feature_ids': [f"L{f.layer}-{f.feature_id}" for f in c.features]
            }
            for c in results.clusters
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Generate gambling-context feature interpretations')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'],
                       help='Model to use')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID')
    parser.add_argument('--validated', type=str, required=True,
                       help='Path to validated features JSON')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--scales', type=str, default=None,
                       help='Comma-separated scale values (e.g., "3.0,5.0,8.0")')
    parser.add_argument('--no-clustering', action='store_true',
                       help='Disable feature clustering')

    args = parser.parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    if args.scales:
        config.setdefault('interpretation', {})['scales'] = [
            float(s) for s in args.scales.split(',')
        ]
    if args.no_clustering:
        config.setdefault('interpretation', {})['enable_clustering'] = False

    # Setup paths
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / 'logs'
    checkpoint_dir = output_dir / 'checkpoints'

    # Setup logging
    logger = setup_logging(f'interpretation_{args.model}', log_dir)
    logger.info("=" * 70)
    logger.info(f"GAMBLING-CONTEXT INTERPRETATION - {args.model.upper()}")
    logger.info("=" * 70)
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Validated features: {args.validated}")

    # Setup checkpoint manager
    checkpoint_mgr = CheckpointManager(checkpoint_dir, f'interp_{args.model}')

    # Load validated features
    validated_path = Path(args.validated)
    if not validated_path.is_absolute():
        validated_path = output_dir / validated_path

    logger.info(f"Loading validated features from {validated_path}")
    validated_features = load_validated_features(validated_path)
    logger.info(f"Loaded {len(validated_features)} validated features")

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        args.model, 'cuda:0', torch.bfloat16, logger
    )

    # Initialize interpreter
    interpreter = GamblingInterpreter(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model,
        config=config,
        device='cuda:0',
        logger=logger
    )

    # Run interpretation
    results = interpreter.run_full_interpretation(
        validated_features=validated_features,
        checkpoint_mgr=checkpoint_mgr
    )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f'interpretations_{args.model}_{timestamp}.json'
    save_interpretation_results(results, results_path)
    logger.info(f"\nResults saved to {results_path}")

    # Print narratives
    logger.info("\n" + "=" * 70)
    logger.info("BEHAVIORAL NARRATIVES")
    logger.info("=" * 70)
    logger.info("\n[SAFE BEHAVIOR NARRATIVE]")
    logger.info(results.safe_narrative)
    logger.info("\n[RISKY BEHAVIOR NARRATIVE]")
    logger.info(results.risky_narrative)

    # Print top explanations
    logger.info("\n" + "=" * 70)
    logger.info("TOP FEATURE EXPLANATIONS")
    logger.info("=" * 70)

    sorted_explanations = sorted(
        results.explanations,
        key=lambda x: x.confidence,
        reverse=True
    )

    for exp in sorted_explanations[:10]:
        logger.info(f"\nL{exp.layer}-{exp.feature_id} ({exp.direction}):")
        logger.info(f"  Explanation: \"{exp.explanation}\"")
        logger.info(f"  Confidence: {exp.confidence:.3f}")
        logger.info(f"  Keywords: {exp.gambling_keywords}")

    logger.info("\nGambling-context interpretation complete!")


if __name__ == '__main__':
    main()
