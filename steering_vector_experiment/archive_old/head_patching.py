#!/usr/bin/env python3
"""
Phase 4: Head Patching for Attention Mechanism Analysis

Identifies causally important attention heads in layers with validated features:
1. For layers with validated features, patch each attention head
2. Measure behavioral change from head patching
3. Cross-reference with feature locations
4. Output causal head rankings and feature-head correspondence

This implements the "D" component of the A+B+C+D causal analysis strategy.

Usage:
    python head_patching.py --model llama --gpu 0 \\
        --validated validated_features_llama.json

Design: Hook-based attention output patching with per-head granularity.

References:
- Attention Head Patching: Wang et al. (2022) "Interpretability in the Wild"
- Causal Tracing: Meng et al. (2022) "Locating and Editing Factual Associations"
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
from typing import Dict, List, Tuple, Optional, Any, Callable
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
    BehavioralMetrics,
    compute_effect_size,
    TestPromptGenerator,
    CheckpointManager
)


@dataclass
class HeadPatchResult:
    """Result of patching a single attention head."""
    layer: int
    head_idx: int
    baseline_metric: float
    patched_metric: float
    effect: float  # patched - baseline
    effect_size: float  # Cohen's d if multiple samples
    is_causal: bool


@dataclass
class LayerHeadAnalysis:
    """Analysis of all heads in a layer."""
    layer: int
    n_heads: int
    causal_heads: List[HeadPatchResult]
    non_causal_heads: List[HeadPatchResult]
    total_layer_effect: float


@dataclass
class HeadPatchingResults:
    """Complete head patching results."""
    model: str
    n_layers_tested: int
    n_heads_per_layer: int
    effect_threshold: float
    layer_analyses: Dict[int, LayerHeadAnalysis]
    top_causal_heads: List[HeadPatchResult]
    feature_head_correspondence: Dict[str, List[int]]  # "L{layer}-{feat}" -> [heads]


class AttentionHeadPatcher:
    """
    Patch individual attention heads during model forward pass.

    Intercepts attention outputs and replaces specific head outputs
    with values from a different (corrupted) context.
    """

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str,
        device: str = 'cuda:0',
        logger=None
    ):
        """
        Initialize the attention head patcher.

        Args:
            model: Loaded transformer model
            tokenizer: Corresponding tokenizer
            model_name: 'llama' or 'gemma'
            device: Device for computation
            logger: Optional logger
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device
        self.logger = logger
        self.model_config = ModelRegistry.get(model_name)

        # Model architecture info
        self.n_layers = self.model_config.n_layers
        self.n_heads = self._get_n_heads()
        self.head_dim = self._get_head_dim()

        # Hook management
        self.hooks = []

        # Cached attention outputs
        self.clean_attn_cache = {}
        self.corrupt_attn_cache = {}

    def _log(self, msg: str):
        """Log a message."""
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def _get_n_heads(self) -> int:
        """Get number of attention heads."""
        if self.model_name == 'llama':
            return self.model.config.num_attention_heads
        elif self.model_name == 'gemma':
            return self.model.config.num_attention_heads
        else:
            return 32  # Default

    def _get_head_dim(self) -> int:
        """Get dimension per attention head."""
        d_model = self.model_config.d_model
        return d_model // self.n_heads

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for model."""
        if self.model_config.use_chat_template:
            chat = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def get_attention_module(self, layer: int) -> torch.nn.Module:
        """Get the attention module for a layer."""
        if self.model_name == 'llama':
            return self.model.model.layers[layer].self_attn
        elif self.model_name == 'gemma':
            return self.model.model.layers[layer].self_attn
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def get_o_proj_module(self, layer: int) -> torch.nn.Module:
        """Get the output projection module (o_proj) for a layer.

        This is where we hook to capture/modify per-head outputs BEFORE mixing.
        """
        attn = self.get_attention_module(layer)
        if hasattr(attn, 'o_proj'):
            return attn.o_proj
        elif hasattr(attn, 'out_proj'):
            return attn.out_proj
        else:
            raise ValueError(f"Cannot find o_proj in {type(attn)}")

    def get_layer_module(self, layer: int) -> torch.nn.Module:
        """Get the full layer module."""
        if self.model_name == 'llama':
            return self.model.model.layers[layer]
        elif self.model_name == 'gemma':
            return self.model.model.layers[layer]
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def capture_attention_output(
        self,
        prompt: str,
        layer: int
    ) -> torch.Tensor:
        """
        Capture per-head attention output BEFORE o_proj mixing.

        Uses forward_pre_hook on o_proj to capture the input (concatenated heads).

        Args:
            prompt: Input prompt
            layer: Layer index

        Returns:
            Attention output tensor [batch, seq, n_heads, head_dim]
        """
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(
            formatted,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Storage for captured output
        captured = {}

        def capture_pre_hook(module, args):
            """Capture INPUT to o_proj (before heads are mixed)."""
            # args[0] is the input tensor: [batch, seq, n_heads * head_dim]
            if isinstance(args, tuple) and len(args) > 0:
                attn_out = args[0]
            else:
                attn_out = args

            # Reshape to per-head: [batch, seq, n_heads, head_dim]
            batch, seq, _ = attn_out.shape
            attn_out_heads = attn_out.view(batch, seq, self.n_heads, self.head_dim)
            captured['attn_out'] = attn_out_heads.detach().clone()

        # Register PRE-hook on o_proj to capture input BEFORE mixing
        o_proj_module = self.get_o_proj_module(layer)
        hook = o_proj_module.register_forward_pre_hook(capture_pre_hook)

        try:
            with torch.no_grad():
                self.model(**inputs, output_hidden_states=False)
        finally:
            hook.remove()

        del inputs
        return captured.get('attn_out')

    def create_head_patch_hook(
        self,
        layer: int,
        head_idx: int,
        replacement_output: torch.Tensor
    ) -> Callable:
        """
        Create a PRE-hook that patches a specific attention head BEFORE o_proj.

        This modifies the input to o_proj, which is the concatenated per-head
        outputs BEFORE they get mixed by the output projection.

        Args:
            layer: Layer index
            head_idx: Head index to patch
            replacement_output: Tensor to use for replacement [batch, seq, head_dim]

        Returns:
            Pre-hook function that modifies o_proj input
        """
        def patch_pre_hook(module, args):
            """Modify INPUT to o_proj (before heads are mixed)."""
            if isinstance(args, tuple) and len(args) > 0:
                attn_out = args[0]
            else:
                return args  # Can't modify

            batch, seq, d_concat = attn_out.shape

            # Reshape to per-head: [batch, seq, n_heads, head_dim]
            attn_out_heads = attn_out.view(batch, seq, self.n_heads, self.head_dim)

            # Replace specific head
            repl_seq = min(replacement_output.shape[1], seq)
            attn_out_heads = attn_out_heads.clone()  # Don't modify in-place
            attn_out_heads[:, -repl_seq:, head_idx, :] = replacement_output[:, -repl_seq:, :].to(attn_out.dtype)

            # Reshape back to concatenated form
            patched = attn_out_heads.view(batch, seq, d_concat)

            # Return modified args tuple
            return (patched,) + args[1:] if len(args) > 1 else (patched,)

        return patch_pre_hook

    def patch_attention_head(
        self,
        prompt: str,
        layer: int,
        head_idx: int,
        corrupt_prompt: str
    ) -> Tuple[float, float]:
        """
        Run forward pass with attention head patched.

        Args:
            prompt: Clean prompt for forward pass
            layer: Layer to patch
            head_idx: Head index to patch
            corrupt_prompt: Prompt to get replacement attention from

        Returns:
            Tuple of (baseline_metric, patched_metric)
        """
        # Get corrupt attention output
        corrupt_attn = self.capture_attention_output(corrupt_prompt, layer)
        if corrupt_attn is None:
            return 0.0, 0.0

        # Get specific head's output [batch, seq, head_dim]
        corrupt_head_out = corrupt_attn[:, :, head_idx, :]

        # Create patch hook
        patch_hook = self.create_head_patch_hook(layer, head_idx, corrupt_head_out)

        # Compute baseline (unpatched)
        baseline = self._compute_metric(prompt, patch_hook=None, layer=None)

        # Compute patched
        patched = self._compute_metric(prompt, patch_hook=patch_hook, layer=layer)

        return baseline, patched

    def _compute_metric(
        self,
        prompt: str,
        patch_hook: Optional[Callable],
        layer: Optional[int]
    ) -> float:
        """Compute behavioral metric with optional patching."""
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(
            formatted,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        ).to(self.device)

        hook_handle = None
        if patch_hook and layer is not None:
            # Use o_proj pre-hook to patch BEFORE heads are mixed
            o_proj_module = self.get_o_proj_module(layer)
            hook_handle = o_proj_module.register_forward_pre_hook(patch_hook)

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]

            # Compute stop probability
            import torch.nn.functional as F
            probs = F.softmax(logits, dim=-1)

            # Get stop/bet token probabilities
            p_stop = 0.0
            p_bet = 0.0
            for token in ['Stop', '2', 'stop']:
                try:
                    tid = self.tokenizer.encode(token, add_special_tokens=False)[0]
                    p_stop = max(p_stop, probs[tid].item())
                except Exception:
                    pass

            for token in ['Bet', '1', '$']:
                try:
                    tid = self.tokenizer.encode(token, add_special_tokens=False)[0]
                    p_bet = max(p_bet, probs[tid].item())
                except Exception:
                    pass

            # Normalize
            total = p_stop + p_bet
            metric = p_stop / total if total > 0 else 0.5

        finally:
            if hook_handle:
                hook_handle.remove()
            del inputs

        return metric

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class HeadPatcher:
    """
    Main class for Phase 4: Head Patching Analysis.

    Identifies causally important attention heads and cross-references
    with validated features.
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
        Initialize head patcher.

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

        # Attention patcher
        self.patcher = AttentionHeadPatcher(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            device=device,
            logger=logger
        )

        # Get config values
        hp_config = config.get('head_patching', {})
        self.effect_threshold = hp_config.get('effect_threshold', 0.1)
        self.n_test_prompts = hp_config.get('n_test_prompts', 50)
        self.patching_direction = hp_config.get('patching_direction', 'corrupt_to_clean')

        # Get n_heads from the actual model (via patcher), not config
        # This ensures consistency with the model's actual architecture
        self.n_heads = self.patcher.n_heads

    def _log(self, msg: str):
        """Log a message."""
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def measure_head_effect(
        self,
        layer: int,
        head_idx: int,
        test_prompts: List[Dict]
    ) -> HeadPatchResult:
        """
        Measure behavioral effect of patching a single head.

        Args:
            layer: Layer index
            head_idx: Head index
            test_prompts: List of test prompt dicts

        Returns:
            HeadPatchResult with effect measurements
        """
        baseline_metrics = []
        patched_metrics = []

        # Get fallback prompts for contrastive patching
        risky_fallback = TestPromptGenerator.get_risky_scenario()['prompt']
        safe_fallback = TestPromptGenerator.get_safe_scenario()['prompt']

        for prompt_info in test_prompts[:10]:  # Limit for efficiency
            prompt = prompt_info['prompt']
            # Use the test prompt as the main prompt, with contrastive fallback
            is_risky_prompt = prompt_info.get('is_risky', True)

            try:
                if self.patching_direction == 'corrupt_to_clean':
                    # Patch with corrupt (safe) attention into clean (risky) prompt
                    baseline, patched = self.patcher.patch_attention_head(
                        prompt=prompt,  # Use actual test prompt
                        layer=layer,
                        head_idx=head_idx,
                        corrupt_prompt=safe_fallback if is_risky_prompt else risky_fallback
                    )
                else:
                    # Patch with clean (risky) attention into corrupt (safe) prompt
                    baseline, patched = self.patcher.patch_attention_head(
                        prompt=prompt,  # Use actual test prompt
                        layer=layer,
                        head_idx=head_idx,
                        corrupt_prompt=risky_fallback if not is_risky_prompt else safe_fallback
                    )

                baseline_metrics.append(baseline)
                patched_metrics.append(patched)

            except Exception as e:
                self._log(f"Error patching L{layer}-H{head_idx}: {e}")
                continue

        if not baseline_metrics:
            return HeadPatchResult(
                layer=layer,
                head_idx=head_idx,
                baseline_metric=0.0,
                patched_metric=0.0,
                effect=0.0,
                effect_size=0.0,
                is_causal=False
            )

        # Compute statistics
        mean_baseline = np.mean(baseline_metrics)
        mean_patched = np.mean(patched_metrics)
        effect = mean_patched - mean_baseline
        effect_size = compute_effect_size(patched_metrics, baseline_metrics)

        is_causal = abs(effect) >= self.effect_threshold

        return HeadPatchResult(
            layer=layer,
            head_idx=head_idx,
            baseline_metric=mean_baseline,
            patched_metric=mean_patched,
            effect=effect,
            effect_size=effect_size,
            is_causal=is_causal
        )

    def find_causal_heads(
        self,
        layers_to_test: List[int],
        test_prompts: List[Dict],
        checkpoint_mgr: Optional[CheckpointManager] = None
    ) -> Dict[int, LayerHeadAnalysis]:
        """
        Find all causally important heads in specified layers.

        Args:
            layers_to_test: List of layer indices to test
            test_prompts: Test prompts for evaluation
            checkpoint_mgr: Optional checkpoint manager

        Returns:
            Dict mapping layer to LayerHeadAnalysis
        """
        self._log("=" * 60)
        self._log("HEAD PATCHING ANALYSIS")
        self._log("=" * 60)
        self._log(f"Layers to test: {layers_to_test}")
        self._log(f"Heads per layer: {self.n_heads}")
        self._log(f"Effect threshold: {self.effect_threshold}")

        layer_analyses = {}

        for layer in tqdm(layers_to_test, desc="Analyzing layers"):
            self._log(f"\nAnalyzing layer {layer}...")

            causal_heads = []
            non_causal_heads = []

            for head_idx in tqdm(range(self.n_heads), desc=f"Layer {layer} heads", leave=False):
                result = self.measure_head_effect(
                    layer=layer,
                    head_idx=head_idx,
                    test_prompts=test_prompts
                )

                if result.is_causal:
                    causal_heads.append(result)
                else:
                    non_causal_heads.append(result)

                # Clear GPU memory periodically
                if (head_idx + 1) % 8 == 0:
                    clear_gpu_memory(self.device)

            # Compute total layer effect
            total_effect = sum(abs(h.effect) for h in causal_heads)

            analysis = LayerHeadAnalysis(
                layer=layer,
                n_heads=self.n_heads,
                causal_heads=sorted(causal_heads, key=lambda x: abs(x.effect), reverse=True),
                non_causal_heads=non_causal_heads,
                total_layer_effect=total_effect
            )

            layer_analyses[layer] = analysis

            self._log(f"  Layer {layer}: {len(causal_heads)} causal heads, "
                     f"total effect={total_effect:.4f}")

            # Checkpoint
            if checkpoint_mgr:
                checkpoint_data = {
                    'progress_layer': layer,
                    'analyses': {
                        l: {
                            'layer': a.layer,
                            'n_causal': len(a.causal_heads),
                            'total_effect': a.total_layer_effect
                        }
                        for l, a in layer_analyses.items()
                    }
                }
                checkpoint_mgr.save(checkpoint_data, 'head_patching')

        return layer_analyses

    def cross_reference_features(
        self,
        layer_analyses: Dict[int, LayerHeadAnalysis],
        validated_features: List[Dict]
    ) -> Dict[str, List[int]]:
        """
        Cross-reference validated features with causal heads.

        Match features to heads by layer co-occurrence.

        Args:
            layer_analyses: Head analysis results per layer
            validated_features: List of validated feature dicts

        Returns:
            Dict mapping "L{layer}-{feat_id}" to list of causal head indices
        """
        correspondence = {}

        for feat in validated_features:
            layer = feat['layer']
            feat_id = feat['feature_id']
            key = f"L{layer}-{feat_id}"

            if layer in layer_analyses:
                analysis = layer_analyses[layer]
                # Get causal heads in same layer
                causal_head_indices = [h.head_idx for h in analysis.causal_heads]
                correspondence[key] = causal_head_indices
            else:
                correspondence[key] = []

        return correspondence

    def run_full_analysis(
        self,
        validated_features: List[Dict],
        test_prompts: Optional[List[Dict]] = None,
        checkpoint_mgr: Optional[CheckpointManager] = None
    ) -> HeadPatchingResults:
        """
        Run complete head patching analysis.

        Args:
            validated_features: List of validated feature dicts from Phase 3
            test_prompts: Optional custom test prompts
            checkpoint_mgr: Optional checkpoint manager

        Returns:
            HeadPatchingResults with all analysis
        """
        # Get unique layers from validated features
        layers_to_test = sorted(set(f['layer'] for f in validated_features))
        self._log(f"Layers with validated features: {layers_to_test}")

        # Generate test prompts if not provided
        if test_prompts is None:
            test_prompts = TestPromptGenerator.generate_balanced_test_prompts(
                n_prompts=min(self.n_test_prompts, 20),
                include_variations=True
            )

        # Find causal heads
        layer_analyses = self.find_causal_heads(
            layers_to_test=layers_to_test,
            test_prompts=test_prompts,
            checkpoint_mgr=checkpoint_mgr
        )

        # Cross-reference with features
        correspondence = self.cross_reference_features(layer_analyses, validated_features)

        # Collect top causal heads across all layers
        all_causal = []
        for analysis in layer_analyses.values():
            all_causal.extend(analysis.causal_heads)
        top_causal = sorted(all_causal, key=lambda x: abs(x.effect), reverse=True)[:20]

        results = HeadPatchingResults(
            model=self.model_name,
            n_layers_tested=len(layers_to_test),
            n_heads_per_layer=self.n_heads,
            effect_threshold=self.effect_threshold,
            layer_analyses=layer_analyses,
            top_causal_heads=top_causal,
            feature_head_correspondence=correspondence
        )

        # Print summary
        self._log("\n" + "=" * 60)
        self._log("HEAD PATCHING SUMMARY")
        self._log("=" * 60)

        for layer in sorted(layer_analyses.keys()):
            analysis = layer_analyses[layer]
            self._log(f"\nLayer {layer}:")
            self._log(f"  Causal heads: {len(analysis.causal_heads)} / {self.n_heads}")
            self._log(f"  Total layer effect: {analysis.total_layer_effect:.4f}")
            if analysis.causal_heads:
                top_head = analysis.causal_heads[0]
                self._log(f"  Top head: H{top_head.head_idx} (effect={top_head.effect:.4f})")

        return results


def load_validated_features(filepath: Path) -> List[Dict]:
    """Load validated features from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    return data.get('validated_features', [])


def save_head_patching_results(results: HeadPatchingResults, output_path: Path) -> None:
    """Save head patching results to JSON file."""
    # Convert layer analyses to serializable format
    layer_analyses_dict = {}
    for layer, analysis in results.layer_analyses.items():
        layer_analyses_dict[str(layer)] = {
            'layer': analysis.layer,
            'n_heads': analysis.n_heads,
            'n_causal_heads': len(analysis.causal_heads),
            'total_layer_effect': analysis.total_layer_effect,
            'causal_heads': [asdict(h) for h in analysis.causal_heads],
            'non_causal_heads': [asdict(h) for h in analysis.non_causal_heads[:10]]  # Limit
        }

    output_data = {
        'model': results.model,
        'n_layers_tested': results.n_layers_tested,
        'n_heads_per_layer': results.n_heads_per_layer,
        'effect_threshold': results.effect_threshold,
        'layer_analyses': layer_analyses_dict,
        'top_causal_heads': [asdict(h) for h in results.top_causal_heads],
        'feature_head_correspondence': results.feature_head_correspondence
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Head patching analysis for causal attention heads')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'],
                       help='Model to use')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID')
    parser.add_argument('--validated', type=str, required=True,
                       help='Path to validated features JSON from Phase 3')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--n-prompts', type=int, default=None,
                       help='Number of test prompts')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Effect threshold for causal heads')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated layers to test (overrides feature layers)')

    args = parser.parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    if args.n_prompts:
        config.setdefault('head_patching', {})['n_test_prompts'] = args.n_prompts
    if args.threshold:
        config.setdefault('head_patching', {})['effect_threshold'] = args.threshold

    # Setup paths
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / 'logs'
    checkpoint_dir = output_dir / 'checkpoints'

    # Setup logging
    logger = setup_logging(f'head_patching_{args.model}', log_dir)
    logger.info("=" * 70)
    logger.info(f"HEAD PATCHING ANALYSIS - {args.model.upper()}")
    logger.info("=" * 70)
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Validated features: {args.validated}")

    # Setup checkpoint manager
    checkpoint_mgr = CheckpointManager(checkpoint_dir, f'head_patch_{args.model}')

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

    # Initialize head patcher
    patcher = HeadPatcher(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model,
        config=config,
        device='cuda:0',
        logger=logger
    )

    # Run analysis
    results = patcher.run_full_analysis(
        validated_features=validated_features,
        checkpoint_mgr=checkpoint_mgr
    )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f'causal_heads_{args.model}_{timestamp}.json'
    save_head_patching_results(results, results_path)
    logger.info(f"\nResults saved to {results_path}")

    # Print top causal heads
    logger.info("\n" + "=" * 70)
    logger.info("TOP CAUSAL ATTENTION HEADS")
    logger.info("=" * 70)

    for head in results.top_causal_heads[:10]:
        logger.info(f"Layer {head.layer}, Head {head.head_idx}:")
        logger.info(f"  Effect: {head.effect:+.4f}")
        logger.info(f"  Effect size: {head.effect_size:.3f}")
        logger.info(f"  Baseline P(stop): {head.baseline_metric:.3f}")
        logger.info(f"  Patched P(stop): {head.patched_metric:.3f}")

    # Print feature-head correspondence summary
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE-HEAD CORRESPONDENCE")
    logger.info("=" * 70)

    for feat_key, heads in list(results.feature_head_correspondence.items())[:10]:
        logger.info(f"{feat_key}: {len(heads)} causal heads in layer")

    logger.info("\nHead patching analysis complete!")


if __name__ == '__main__':
    main()
