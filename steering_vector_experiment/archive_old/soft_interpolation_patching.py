#!/usr/bin/env python3
"""
Phase 3: Soft Interpolation Patching for Feature Validation

Validates causal features via dose-response patching experiments:
1. For each candidate feature, test alpha values [0.0, 0.25, 0.5, 0.75, 1.0]
2. Patch: interpolated_activation = alpha * clean + (1-alpha) * corrupt
3. Measure P(stop) - P(bet) at each alpha level
4. Filter: Keep features with |Spearman rho| > 0.8 (monotonic dose-response)

This implements the "B" component of the A+B+C+D causal analysis strategy.

Usage:
    python soft_interpolation_patching.py --model llama --gpu 0 \\
        --candidates candidate_features_llama.json

Design: Hook-based activation patching with configurable interpolation levels.

References:
- Activation Patching: Wang et al. (2022) "Interpretability in the Wild"
- Causal Mediation Analysis: Vig et al. (2020) "Causal Mediation Analysis"
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
import torch.nn.functional as F
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
    get_causal_feature_src,
    BehavioralMetrics,
    compute_spearman_correlation,
    compute_effect_size,
    ActivationHook,
    TestPromptGenerator,
    CheckpointManager,
    # FDR and group-level testing
    benjamini_hochberg,
    apply_fdr_correction,
    compute_fdr_summary,
    permutation_test,
    bootstrap_ci,
    group_causal_test,
    GroupCausalTestResult
)

# Add LlamaScope path dynamically
sys.path.insert(0, str(get_causal_feature_src()))


def load_real_prompts(prompts_file: str, model_name: str) -> List[Dict]:
    """
    Load real prompts from condition prompts JSON file.

    This loads prompts extracted from actual experimental data (3,200 experiments)
    by real_prompt_extractor.py. These prompts represent the 4 conditions per model:
    - safe_fixed: voluntary stop + fixed betting
    - safe_variable: voluntary stop + variable betting
    - risky_fixed: bankruptcy + fixed betting (using pre-fatal bet state)
    - risky_variable: bankruptcy + variable betting (using pre-fatal bet state)

    Args:
        prompts_file: Path to condition_prompts.json
        model_name: 'llama' or 'gemma'

    Returns:
        List of prompt dictionaries with 'prompt', 'condition', 'is_safe' keys
    """
    with open(prompts_file, 'r') as f:
        data = json.load(f)

    # Extract prompts for the specified model
    model_conditions = data.get('conditions', {}).get(model_name, [])

    test_prompts = []
    for cond in model_conditions:
        test_prompts.append({
            'prompt': cond['prompt'],
            'condition': cond['condition_name'],
            'is_safe': cond['outcome'] == 'voluntary_stop',
            'bet_type': cond['bet_type'],
            'balance': cond['balance'],
            'rounds_played': cond['rounds_played']
        })

    return test_prompts


@dataclass
class FeatureCandidate:
    """A candidate feature to validate."""
    layer: int
    feature_id: int
    contribution: float
    direction: str  # 'safe' or 'risky'


@dataclass
class DoseResponseResult:
    """Result of dose-response validation for a feature."""
    layer: int
    feature_id: int
    alpha_values: List[float]
    behavioral_responses: List[float]
    spearman_rho: float
    spearman_p: float
    q_value: float = 1.0  # FDR-adjusted p-value
    is_monotonic: bool = False
    is_fdr_significant: bool = False  # Significant after FDR correction
    effect_size: float = 0.0
    effect_ci_lower: float = 0.0  # 95% CI lower bound
    effect_ci_upper: float = 0.0  # 95% CI upper bound
    direction: str = ''
    validated: bool = False


@dataclass
class BidirectionalResult:
    """Result of bidirectional causal testing for a feature.

    True causality requires consistent effects in BOTH directions:
    - safe_to_risky: Patching safe prompts with risky values → increases P(bet)
    - risky_to_safe: Patching risky prompts with safe values → increases P(stop)

    Uses permutation test for p-values and bootstrap for confidence intervals.
    """
    layer: int
    feature_id: int
    direction: str

    # Safe → Risky test (patch voluntary-stop prompts with risky feature values)
    safe_to_risky_effect: float  # Increase in P(bet) (Cohen's d)
    safe_to_risky_ci_lower: float = 0.0  # 95% CI
    safe_to_risky_ci_upper: float = 0.0
    safe_to_risky_p_value: float = 1.0  # Permutation test
    safe_to_risky_significant: bool = False

    # Risky → Safe test (patch bankruptcy prompts with safe feature values)
    risky_to_safe_effect: float = 0.0  # Increase in P(stop) (Cohen's d)
    risky_to_safe_ci_lower: float = 0.0
    risky_to_safe_ci_upper: float = 0.0
    risky_to_safe_p_value: float = 1.0
    risky_to_safe_significant: bool = False

    # Sample sizes
    n_safe_prompts: int = 0
    n_risky_prompts: int = 0

    # Combined result
    is_bidirectional_causal: bool = False  # True only if BOTH directions are significant
    combined_effect: float = 0.0  # Average effect size
    combined_q_value: float = 1.0  # FDR-corrected (worst of two q-values)


@dataclass
class ValidationResults:
    """Complete validation results."""
    model: str
    n_candidates: int
    n_validated: int  # Nominally significant (monotonic + effect size, BEFORE FDR)
    n_fdr_validated: int = 0  # After FDR correction (q < fdr_alpha)
    n_bidirectional_causal: int = 0  # Features passing bidirectional test
    monotonicity_threshold: float = 0.8
    fdr_alpha: float = 0.05  # FDR significance level
    validated_features: List[DoseResponseResult] = field(default_factory=list)
    rejected_features: List[DoseResponseResult] = field(default_factory=list)
    bidirectional_results: List[BidirectionalResult] = field(default_factory=list)
    layer_summary: Dict[int, Dict] = field(default_factory=dict)
    fdr_summary: Dict = field(default_factory=dict)  # FDR correction statistics


class SAEFeaturePatcher:
    """
    Patch individual SAE features during model forward pass.

    Uses hooks to intercept hidden states, encode through SAE,
    modify specific features, then decode back.
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
        Initialize the feature patcher.

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

        # SAE instances per layer (loaded on demand)
        self.saes = {}

        # Hook management
        self.hooks = []

        # Cached clean/corrupt activations
        self.clean_cache = {}
        self.corrupt_cache = {}

    def _log(self, msg: str):
        """Log a message."""
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
            errors = []

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
                self._log(f"  Canonical failed: {e}")
                errors.append(f"canonical: {e}")

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
                except Exception as e:
                    self._log(f"  Failed to load width {width}: {e}")
                    errors.append(f"{width}: {e}")
                    continue
            raise RuntimeError(f"Failed to load GemmaScope for layer {layer}. Errors: {errors}")

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def get_layer_module(self, layer: int) -> torch.nn.Module:
        """Get the transformer layer module."""
        if self.model_name == 'llama':
            return self.model.model.layers[layer]
        elif self.model_name == 'gemma':
            return self.model.model.layers[layer]
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for model."""
        if self.model_config.use_chat_template:
            chat = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def get_hidden_state(
        self,
        prompt: str,
        layer: int,
        position: str = 'last'
    ) -> torch.Tensor:
        """
        Get hidden state at specified layer for a prompt.

        Args:
            prompt: Input prompt
            layer: Layer index
            position: Token position ('last' or int)

        Returns:
            Hidden state tensor
        """
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(
            formatted,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Capture activation
        activation = None

        def capture_hook(module, input, output):
            nonlocal activation
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            if position == 'last':
                activation = hidden[0, -1, :].detach().clone()
            elif isinstance(position, int):
                activation = hidden[0, position, :].detach().clone()

        # Register hook
        layer_module = self.get_layer_module(layer)
        hook = layer_module.register_forward_hook(capture_hook)

        try:
            with torch.no_grad():
                self.model(**inputs, output_hidden_states=False)
        finally:
            hook.remove()

        del inputs
        return activation

    def get_feature_activation(
        self,
        hidden_state: torch.Tensor,
        layer: int,
        feature_id: int
    ) -> float:
        """
        Get activation of specific feature from hidden state.

        Args:
            hidden_state: Hidden state tensor [d_model]
            layer: Layer index
            feature_id: Feature ID in SAE

        Returns:
            Feature activation value
        """
        sae = self.load_sae(layer)

        # Handle dtype
        if hidden_state.dtype == torch.bfloat16:
            hidden_state = hidden_state.float()
        hidden_state = hidden_state.to(self.device)

        # Encode
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
        elif hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(1)  # [batch, 1, d_model]

        with torch.no_grad():
            features = sae.encode(hidden_state)

        return features.squeeze()[feature_id].item()

    def create_feature_patch_hook(
        self,
        layer: int,
        feature_id: int,
        target_value: float
    ) -> Callable:
        """
        Create a hook that patches a specific feature to target value.

        Args:
            layer: Layer index
            feature_id: Feature ID to patch
            target_value: Value to set the feature to

        Returns:
            Hook function
        """
        sae = self.load_sae(layer)

        def patch_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            # Get last token position
            batch_size, seq_len, d_model = hidden.shape

            # Encode to features (last token only for efficiency)
            last_hidden = hidden[:, -1:, :].float()
            features = sae.encode(last_hidden)  # [batch, 1, d_sae]

            # Patch the specific feature
            features[:, :, feature_id] = target_value

            # Decode back
            patched_hidden = sae.decode(features)

            # Replace last token's hidden state
            new_hidden = hidden.clone()
            new_hidden[:, -1:, :] = patched_hidden.to(hidden.dtype)

            if rest is not None:
                return (new_hidden,) + rest
            return new_hidden

        return patch_hook

    def patch_feature_activation(
        self,
        layer: int,
        feature_id: int,
        alpha: float,
        clean_value: float,
        corrupt_value: float
    ) -> Callable:
        """
        Create hook for interpolated feature patching.

        Args:
            layer: Layer index
            feature_id: Feature ID to patch
            alpha: Interpolation coefficient (0=corrupt, 1=clean)
            clean_value: Clean (risky) feature value
            corrupt_value: Corrupt (safe) feature value

        Returns:
            Hook function that patches feature to interpolated value
        """
        interpolated_value = alpha * clean_value + (1 - alpha) * corrupt_value
        return self.create_feature_patch_hook(layer, feature_id, interpolated_value)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_behavioral_metric(
        self,
        prompt: str,
        behavioral_metric: str = 'stop_probability'
    ) -> float:
        """
        Compute behavioral metric for a prompt WITHOUT patching.

        Args:
            prompt: Input prompt
            behavioral_metric: Type of metric ('stop_probability' or 'decision_logit_diff')

        Returns:
            Behavioral metric value (e.g., P(stop))
        """
        # Format prompt
        if self.model_config.use_chat_template:
            chat = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        inputs = self.tokenizer(
            formatted,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits

        # Import BehavioralMetrics for computation
        metrics = BehavioralMetrics(self.tokenizer, self.device)

        if behavioral_metric == 'stop_probability':
            result = metrics.compute_stop_probability(logits)
        elif behavioral_metric == 'decision_logit_diff':
            result = metrics.compute_decision_logit_diff(logits)
        else:
            result = metrics.compute_stop_probability(logits)

        del inputs, outputs
        return result

    def compute_behavioral_metric_with_patch(
        self,
        prompt: str,
        layer: int,
        feature_id: int,
        target_value: float,
        behavioral_metric: str = 'stop_probability'
    ) -> float:
        """
        Compute behavioral metric with feature patching applied.

        Args:
            prompt: Input prompt
            layer: Layer to apply patch
            feature_id: Feature ID to patch
            target_value: Value to set the feature to
            behavioral_metric: Type of metric

        Returns:
            Behavioral metric value with patching applied
        """
        # Format prompt
        if self.model_config.use_chat_template:
            chat = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        inputs = self.tokenizer(
            formatted,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Create and register patch hook
        patch_hook = self.create_feature_patch_hook(layer, feature_id, target_value)
        layer_module = self.get_layer_module(layer)
        hook_handle = layer_module.register_forward_hook(patch_hook)

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]  # Last token logits

            # Import BehavioralMetrics for computation
            metrics = BehavioralMetrics(self.tokenizer, self.device)

            if behavioral_metric == 'stop_probability':
                result = metrics.compute_stop_probability(logits)
            elif behavioral_metric == 'decision_logit_diff':
                result = metrics.compute_decision_logit_diff(logits)
            else:
                result = metrics.compute_stop_probability(logits)

        finally:
            hook_handle.remove()
            del inputs, outputs

        return result


class SoftInterpolationPatcher:
    """
    Validate feature causality via soft interpolation patching.

    Main class for Phase 3 of the pipeline.
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
        Initialize the soft interpolation patcher.

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

        # Feature patcher
        self.patcher = SAEFeaturePatcher(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            device=device,
            logger=logger
        )

        # Behavioral metrics
        self.metrics = BehavioralMetrics(tokenizer, device)

        # Get config values
        si_config = config.get('soft_interpolation', {})
        self.alpha_values = si_config.get('alpha_values', [0.0, 0.25, 0.5, 0.75, 1.0])
        self.n_test_prompts = si_config.get('n_test_prompts', 100)
        self.monotonicity_threshold = si_config.get('monotonicity_threshold', 0.8)
        self.min_effect_size = si_config.get('min_effect_size', 0.3)
        # Separate threshold for bidirectional tests (typically higher bar for causal claims)
        self.min_bidirectional_effect = si_config.get('min_bidirectional_effect', 0.5)
        self.fdr_alpha = si_config.get('fdr_alpha', 0.05)  # FDR significance threshold
        self.behavioral_metric = si_config.get('behavioral_metric', 'stop_probability')

    def _log(self, msg: str):
        """Log a message."""
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

    def compute_behavioral_metric(
        self,
        prompt: str,
        patch_hook: Optional[Callable] = None,
        layer: Optional[int] = None
    ) -> float:
        """
        Compute behavioral metric for a prompt with optional patching.

        Args:
            prompt: Input prompt
            patch_hook: Optional hook for patching
            layer: Layer to apply patch hook

        Returns:
            Behavioral metric value
        """
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(
            formatted,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        ).to(self.device)

        hook_handle = None
        if patch_hook and layer is not None:
            layer_module = self.patcher.get_layer_module(layer)
            hook_handle = layer_module.register_forward_hook(patch_hook)

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]  # Last token logits

            # Compute metric
            if self.behavioral_metric == 'stop_probability':
                metric = self.metrics.compute_stop_probability(logits)
            elif self.behavioral_metric == 'decision_logit_diff':
                metric = self.metrics.compute_decision_logit_diff(logits)
            else:
                metric = self.metrics.compute_stop_probability(logits)

        finally:
            if hook_handle:
                hook_handle.remove()
            del inputs, outputs

        return metric

    def compute_dose_response(
        self,
        feature: FeatureCandidate,
        test_prompts: List[Dict],
        clean_values: Dict[int, float],
        corrupt_values: Dict[int, float]
    ) -> DoseResponseResult:
        """
        Compute dose-response curve for a candidate feature.

        Args:
            feature: Feature to test
            test_prompts: List of test prompt dicts
            clean_values: Dict mapping prompt_id to clean feature value
            corrupt_values: Dict mapping prompt_id to corrupt feature value

        Returns:
            DoseResponseResult with validation outcome
        """
        self._log(f"Testing L{feature.layer}-{feature.feature_id}...")

        # Collect responses at each alpha level
        alpha_responses = {alpha: [] for alpha in self.alpha_values}

        for i, prompt_info in enumerate(test_prompts):
            prompt = prompt_info['prompt']
            # Use enumerate index as fallback if prompt_id not specified
            prompt_id = prompt_info.get('prompt_id', i)

            # Get clean/corrupt values for this prompt
            clean_val = clean_values.get(prompt_id, 0.0)
            corrupt_val = corrupt_values.get(prompt_id, 0.0)

            for alpha in self.alpha_values:
                # Create patching hook
                patch_hook = self.patcher.patch_feature_activation(
                    layer=feature.layer,
                    feature_id=feature.feature_id,
                    alpha=alpha,
                    clean_value=clean_val,
                    corrupt_value=corrupt_val
                )

                # Compute behavioral metric with patching
                try:
                    metric = self.compute_behavioral_metric(
                        prompt=prompt,
                        patch_hook=patch_hook,
                        layer=feature.layer
                    )
                    alpha_responses[alpha].append(metric)
                except Exception as e:
                    self._log(f"Error at alpha={alpha}: {e}")
                    continue

            # Clear cache periodically
            clear_gpu_memory(self.device)

        # Aggregate responses per alpha level
        behavioral_responses = []
        for alpha in self.alpha_values:
            if alpha_responses[alpha]:
                behavioral_responses.append(np.mean(alpha_responses[alpha]))
            else:
                behavioral_responses.append(0.0)

        # Compute Spearman correlation (monotonicity test)
        rho, p_value = compute_spearman_correlation(
            self.alpha_values,
            behavioral_responses
        )

        # Determine if monotonic
        is_monotonic = bool(abs(rho) >= self.monotonicity_threshold)

        # Compute effect size (alpha=0 vs alpha=1)
        responses_0 = alpha_responses.get(0.0, [])
        responses_1 = alpha_responses.get(1.0, [])
        effect_size = compute_effect_size(responses_1, responses_0) if responses_0 and responses_1 else 0.0

        # Compute bootstrap CI for effect size
        if responses_0 and responses_1 and len(responses_0) >= 2 and len(responses_1) >= 2:
            # Bootstrap the effect size difference
            diffs = [r1 - r0 for r1, r0 in zip(responses_1, responses_0)]
            if len(diffs) >= 2:
                _, ci_lower, ci_upper = bootstrap_ci(diffs, confidence=0.95, n_bootstrap=1000)
            else:
                ci_lower, ci_upper = effect_size, effect_size
        else:
            ci_lower, ci_upper = 0.0, 0.0

        # Determine validation status
        validated = bool(is_monotonic and abs(effect_size) >= self.min_effect_size)

        return DoseResponseResult(
            layer=feature.layer,
            feature_id=feature.feature_id,
            alpha_values=self.alpha_values,
            behavioral_responses=behavioral_responses,
            spearman_rho=rho,
            spearman_p=p_value,
            is_monotonic=is_monotonic,
            effect_size=effect_size,
            effect_ci_lower=ci_lower,
            effect_ci_upper=ci_upper,
            direction=feature.direction,
            validated=validated
        )

    def validate_monotonicity(
        self,
        responses: List[float],
        alpha_values: List[float]
    ) -> bool:
        """
        Check if responses show monotonic relationship with alpha.

        Args:
            responses: Behavioral responses at each alpha
            alpha_values: Alpha values tested

        Returns:
            True if |Spearman rho| > threshold
        """
        rho, _ = compute_spearman_correlation(alpha_values, responses)
        return abs(rho) >= self.monotonicity_threshold

    def get_feature_values_for_prompts(
        self,
        test_prompts: List[Dict],
        layer: int,
        feature_id: int
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Get clean and corrupt feature values for each test prompt.

        Strategy: Extract steering direction from reference prompts, then
        apply to each test prompt's natural feature value. This respects
        prompt-specific baselines while applying consistent steering.

        Clean = natural + steering toward risky
        Corrupt = natural (baseline)

        Args:
            test_prompts: List of test prompt dicts
            layer: Layer of the feature
            feature_id: Feature ID

        Returns:
            Tuple of (clean_values, corrupt_values) dicts
        """
        clean_values = {}
        corrupt_values = {}

        # Get steering direction from reference prompts
        risky_ref = TestPromptGenerator.get_risky_scenario()
        safe_ref = TestPromptGenerator.get_safe_scenario()

        risky_hidden = self.patcher.get_hidden_state(risky_ref['prompt'], layer)
        safe_hidden = self.patcher.get_hidden_state(safe_ref['prompt'], layer)

        risky_feat_val = self.patcher.get_feature_activation(risky_hidden, layer, feature_id)
        safe_feat_val = self.patcher.get_feature_activation(safe_hidden, layer, feature_id)

        # Compute steering magnitude (difference between risky and safe)
        steering_magnitude = risky_feat_val - safe_feat_val

        # For each test prompt, get its natural value and apply steering
        for i, prompt_info in enumerate(test_prompts):
            prompt = prompt_info['prompt']
            prompt_id = prompt_info.get('prompt_id', i)

            try:
                # Get this prompt's natural feature value
                natural_hidden = self.patcher.get_hidden_state(prompt, layer)
                natural_feat_val = self.patcher.get_feature_activation(
                    natural_hidden, layer, feature_id
                )

                # Clean = natural + steering toward risky
                # Corrupt = natural (baseline, no steering)
                clean_values[prompt_id] = natural_feat_val + steering_magnitude
                corrupt_values[prompt_id] = natural_feat_val

            except Exception as e:
                self._log(f"Error getting feature for prompt {prompt_id}: {e}")
                # Fallback to reference values
                clean_values[prompt_id] = risky_feat_val
                corrupt_values[prompt_id] = safe_feat_val

        return clean_values, corrupt_values

    def test_bidirectional_causality(
        self,
        feature: FeatureCandidate,
        safe_prompts: List[Dict],
        risky_prompts: List[Dict],
        min_effect: float = 0.1,
        n_permutations: int = 1000
    ) -> BidirectionalResult:
        """
        Test bidirectional causal effect of a feature using proper group-level statistics.

        True causality requires consistent effects in BOTH directions:
        1. Safe → Risky: Patching safe-context prompts with risky values → increases P(bet)
        2. Risky → Safe: Patching risky-context prompts with safe values → increases P(stop)

        Uses permutation tests for p-values and bootstrap for 95% confidence intervals.

        Args:
            feature: Feature to test
            safe_prompts: Prompts from voluntary-stop scenarios
            risky_prompts: Prompts from bankruptcy scenarios
            min_effect: Minimum effect size to consider significant
            n_permutations: Number of permutations for significance testing

        Returns:
            BidirectionalResult with both direction tests and proper statistics
        """
        layer = feature.layer
        feature_id = feature.feature_id

        # Get reference feature values
        risky_ref = TestPromptGenerator.get_risky_scenario()
        safe_ref = TestPromptGenerator.get_safe_scenario()

        risky_hidden = self.patcher.get_hidden_state(risky_ref['prompt'], layer)
        safe_hidden = self.patcher.get_hidden_state(safe_ref['prompt'], layer)

        risky_feat_val = self.patcher.get_feature_activation(risky_hidden, layer, feature_id)
        safe_feat_val = self.patcher.get_feature_activation(safe_hidden, layer, feature_id)

        # =========================================================================
        # Test 1: Safe → Risky (patch safe prompts with risky values)
        # Expect: P(bet) increases (P(stop) decreases)
        # =========================================================================
        safe_baseline_metrics = []
        safe_patched_metrics = []

        for prompt_info in safe_prompts[:10]:  # Limit for efficiency
            prompt = prompt_info['prompt']
            try:
                baseline = self.patcher.compute_behavioral_metric(prompt)
                patched = self.patcher.compute_behavioral_metric_with_patch(
                    prompt, layer, feature_id, risky_feat_val
                )
                safe_baseline_metrics.append(baseline)
                safe_patched_metrics.append(patched)
            except Exception:
                continue

        # Group-level statistical test for Safe → Risky
        if len(safe_baseline_metrics) == 0:
            self._log(f"  WARNING: No valid samples for safe→risky test on L{layer}-{feature_id}")
            safe_to_risky_effect = 0.0
            safe_to_risky_ci_lower = 0.0
            safe_to_risky_ci_upper = 0.0
            safe_to_risky_p = 1.0
            safe_to_risky_significant = False
        elif len(safe_baseline_metrics) >= 3:
            s2r_test = group_causal_test(
                baseline_values=safe_baseline_metrics,
                patched_values=safe_patched_metrics,
                test_name=f"L{layer}-{feature_id}_safe_to_risky",
                method='permutation',
                n_permutations=n_permutations
            )
            safe_to_risky_effect = -s2r_test.effect_size  # Negative because P(stop) should decrease
            safe_to_risky_ci_lower = -s2r_test.effect_ci_upper
            safe_to_risky_ci_upper = -s2r_test.effect_ci_lower
            safe_to_risky_p = s2r_test.p_value
            safe_to_risky_significant = bool(
                s2r_test.p_value < 0.05 and abs(s2r_test.effect_size) >= min_effect
            )
        else:
            self._log(f"  WARNING: Too few samples ({len(safe_baseline_metrics)}) for safe→risky test on L{layer}-{feature_id}")
            safe_to_risky_effect = 0.0
            safe_to_risky_ci_lower = 0.0
            safe_to_risky_ci_upper = 0.0
            safe_to_risky_p = 1.0
            safe_to_risky_significant = False

        # =========================================================================
        # Test 2: Risky → Safe (patch risky prompts with safe values)
        # Expect: P(stop) increases
        # =========================================================================
        risky_baseline_metrics = []
        risky_patched_metrics = []

        for prompt_info in risky_prompts[:10]:
            prompt = prompt_info['prompt']
            try:
                baseline = self.patcher.compute_behavioral_metric(prompt)
                patched = self.patcher.compute_behavioral_metric_with_patch(
                    prompt, layer, feature_id, safe_feat_val
                )
                risky_baseline_metrics.append(baseline)
                risky_patched_metrics.append(patched)
            except Exception:
                continue

        # Group-level statistical test for Risky → Safe
        if len(risky_baseline_metrics) == 0:
            self._log(f"  WARNING: No valid samples for risky→safe test on L{layer}-{feature_id}")
            risky_to_safe_effect = 0.0
            risky_to_safe_ci_lower = 0.0
            risky_to_safe_ci_upper = 0.0
            risky_to_safe_p = 1.0
            risky_to_safe_significant = False
        elif len(risky_baseline_metrics) >= 3:
            r2s_test = group_causal_test(
                baseline_values=risky_baseline_metrics,
                patched_values=risky_patched_metrics,
                test_name=f"L{layer}-{feature_id}_risky_to_safe",
                method='permutation',
                n_permutations=n_permutations
            )
            risky_to_safe_effect = r2s_test.effect_size  # Positive because P(stop) should increase
            risky_to_safe_ci_lower = r2s_test.effect_ci_lower
            risky_to_safe_ci_upper = r2s_test.effect_ci_upper
            risky_to_safe_p = r2s_test.p_value
            risky_to_safe_significant = bool(
                r2s_test.p_value < 0.05 and abs(r2s_test.effect_size) >= min_effect
            )
        else:
            self._log(f"  WARNING: Too few samples ({len(risky_baseline_metrics)}) for risky→safe test on L{layer}-{feature_id}")
            risky_to_safe_effect = 0.0
            risky_to_safe_ci_lower = 0.0
            risky_to_safe_ci_upper = 0.0
            risky_to_safe_p = 1.0
            risky_to_safe_significant = False

        # =========================================================================
        # Combined result: BOTH must be significant
        # =========================================================================
        is_bidirectional_causal = safe_to_risky_significant and risky_to_safe_significant
        combined_effect = (abs(safe_to_risky_effect) + abs(risky_to_safe_effect)) / 2

        return BidirectionalResult(
            layer=layer,
            feature_id=feature_id,
            direction=feature.direction,
            safe_to_risky_effect=safe_to_risky_effect,
            safe_to_risky_ci_lower=safe_to_risky_ci_lower,
            safe_to_risky_ci_upper=safe_to_risky_ci_upper,
            safe_to_risky_p_value=safe_to_risky_p,
            safe_to_risky_significant=safe_to_risky_significant,
            risky_to_safe_effect=risky_to_safe_effect,
            risky_to_safe_ci_lower=risky_to_safe_ci_lower,
            risky_to_safe_ci_upper=risky_to_safe_ci_upper,
            risky_to_safe_p_value=risky_to_safe_p,
            risky_to_safe_significant=risky_to_safe_significant,
            n_safe_prompts=len(safe_baseline_metrics),
            n_risky_prompts=len(risky_baseline_metrics),
            is_bidirectional_causal=is_bidirectional_causal,
            combined_effect=combined_effect
        )

    def validate_features(
        self,
        candidate_features: List[FeatureCandidate],
        test_prompts: Optional[List[Dict]] = None,
        checkpoint_mgr: Optional[CheckpointManager] = None
    ) -> ValidationResults:
        """
        Validate all candidate features.

        Args:
            candidate_features: List of candidate features from Phase 2
            test_prompts: Optional custom test prompts
            checkpoint_mgr: Optional checkpoint manager

        Returns:
            ValidationResults with validated and rejected features
        """
        self._log("=" * 60)
        self._log("SOFT INTERPOLATION PATCHING VALIDATION")
        self._log("=" * 60)
        self._log(f"Candidates: {len(candidate_features)}")
        self._log(f"Alpha values: {self.alpha_values}")
        self._log(f"Monotonicity threshold: {self.monotonicity_threshold}")
        self._log(f"Min effect size: {self.min_effect_size}")

        # Generate test prompts if not provided
        if test_prompts is None:
            test_prompts = TestPromptGenerator.generate_balanced_test_prompts(
                n_prompts=min(self.n_test_prompts, 20),  # Limit for efficiency
                include_variations=True
            )
        self._log(f"Test prompts: {len(test_prompts)}")

        # Pre-FDR categorization (before multiple comparison correction)
        nominal_validated = []  # Nominally significant: monotonic + effect size threshold
        nominal_rejected = []   # Did not meet initial criteria
        layer_summary = defaultdict(lambda: {'validated': 0, 'rejected': 0})

        for i, feature in enumerate(tqdm(candidate_features, desc="Validating features")):
            # Get feature values for prompts
            clean_values, corrupt_values = self.get_feature_values_for_prompts(
                test_prompts=test_prompts,
                layer=feature.layer,
                feature_id=feature.feature_id
            )

            # Run dose-response test
            result = self.compute_dose_response(
                feature=feature,
                test_prompts=test_prompts,
                clean_values=clean_values,
                corrupt_values=corrupt_values
            )

            # Sort by nominal validation status (pre-FDR)
            if result.validated:
                nominal_validated.append(result)
                layer_summary[feature.layer]['validated'] += 1
                self._log(f"  NOMINAL: L{feature.layer}-{feature.feature_id} "
                         f"(rho={result.spearman_rho:.3f}, d={result.effect_size:.3f})")
            else:
                nominal_rejected.append(result)
                layer_summary[feature.layer]['rejected'] += 1

            # Checkpoint periodically
            if checkpoint_mgr and (i + 1) % 10 == 0:
                checkpoint_data = {
                    'progress': i + 1,
                    'total': len(candidate_features),
                    'nominal_validated': [asdict(f) for f in nominal_validated],
                    'nominal_rejected': [asdict(f) for f in nominal_rejected]
                }
                checkpoint_mgr.save(checkpoint_data, 'soft_interpolation')

            # Clear GPU memory
            if (i + 1) % 5 == 0:
                clear_gpu_memory(self.device)

        # =====================================================================
        # FDR CORRECTION FOR MONOTONICITY TESTS
        # =====================================================================
        self._log("\n" + "=" * 60)
        self._log("FDR CORRECTION (Benjamini-Hochberg)")
        self._log("=" * 60)

        # Collect all p-values from monotonicity tests
        all_dose_response = nominal_validated + nominal_rejected
        monotonicity_p_values = [r.spearman_p for r in all_dose_response]

        # Apply FDR correction (using configurable alpha)
        fdr_significant, q_values = benjamini_hochberg(monotonicity_p_values, alpha=self.fdr_alpha)

        # Update results with q-values and FDR significance
        # Also recompute layer_summary based on FDR-corrected validation
        layer_summary_fdr = defaultdict(lambda: {'validated': 0, 'rejected': 0})

        for i, result in enumerate(all_dose_response):
            result.q_value = float(q_values[i])
            result.is_fdr_significant = bool(fdr_significant[i])
            # Validated requires: monotonic AND FDR significant AND sufficient effect size
            result.validated = bool(
                result.is_monotonic and
                result.is_fdr_significant and
                abs(result.effect_size) >= self.min_effect_size
            )

            # Update layer summary with FDR-corrected status
            if result.validated:
                layer_summary_fdr[result.layer]['validated'] += 1
            else:
                layer_summary_fdr[result.layer]['rejected'] += 1

        # Re-categorize based on FDR correction
        fdr_validated = [r for r in all_dose_response if r.validated]
        fdr_rejected = [r for r in all_dose_response if not r.validated]

        # Use FDR-corrected layer summary
        layer_summary = dict(layer_summary_fdr)

        n_fdr_validated = len(fdr_validated)
        fdr_summary = compute_fdr_summary(
            [{'p_value': r.spearman_p, 'fdr_significant': r.is_fdr_significant}
             for r in all_dose_response]
        )

        self._log(f"Before FDR: {len(nominal_validated)} nominally significant "
                 f"(|rho| >= {self.monotonicity_threshold}, |d| >= {self.min_effect_size})")
        self._log(f"After FDR:  {n_fdr_validated} significant (q < {self.fdr_alpha})")
        self._log(f"Reduction:  {fdr_summary['reduction_rate']*100:.1f}%")

        # =====================================================================
        # BIDIRECTIONAL CAUSAL TESTING (on FDR-validated features)
        # =====================================================================
        self._log("\n" + "=" * 60)
        self._log("BIDIRECTIONAL CAUSAL TESTING")
        self._log("=" * 60)

        # Generate safe and risky context prompts
        safe_prompts = TestPromptGenerator.generate_safe_context_prompts(n_prompts=10)
        risky_prompts = TestPromptGenerator.generate_risky_context_prompts(n_prompts=10)

        bidirectional_results = []

        # Only test FDR-validated features
        for result in tqdm(fdr_validated, desc="Bidirectional testing"):
            feature = FeatureCandidate(
                layer=result.layer,
                feature_id=result.feature_id,
                contribution=result.effect_size,
                direction=result.direction
            )

            try:
                bidir_result = self.test_bidirectional_causality(
                    feature=feature,
                    safe_prompts=safe_prompts,
                    risky_prompts=risky_prompts,
                    min_effect=self.min_bidirectional_effect  # Higher bar for causal claims
                )
                bidirectional_results.append(bidir_result)
            except Exception as e:
                self._log(f"  Error testing L{feature.layer}-{feature.feature_id}: {e}")

            clear_gpu_memory(self.device)

        # =====================================================================
        # FDR CORRECTION FOR BIDIRECTIONAL TESTS
        # =====================================================================
        if bidirectional_results:
            # Collect p-values from bidirectional tests (both directions)
            # Apply Bonferroni correction for testing 2 directions: p_combined = min(1.0, 2 * max(p1, p2))
            bidir_p_values = []
            for r in bidirectional_results:
                # Bonferroni-corrected combined p-value for joint hypothesis
                combined_p = min(1.0, 2 * max(r.safe_to_risky_p_value, r.risky_to_safe_p_value))
                bidir_p_values.append(combined_p)

            # Apply FDR correction (using configurable alpha)
            bidir_fdr_sig, bidir_q_values = benjamini_hochberg(bidir_p_values, alpha=self.fdr_alpha)

            # Update bidirectional results with FDR-corrected significance
            n_bidirectional_causal = 0
            for i, result in enumerate(bidirectional_results):
                result.combined_q_value = bidir_q_values[i]
                # Bidirectional causal only if FDR significant AND both directions significant
                result.is_bidirectional_causal = (
                    bidir_fdr_sig[i] and
                    result.safe_to_risky_significant and
                    result.risky_to_safe_significant
                )
                if result.is_bidirectional_causal:
                    n_bidirectional_causal += 1
                    self._log(f"  BIDIRECTIONAL: L{result.layer}-{result.feature_id} "
                             f"(S→R={result.safe_to_risky_effect:.3f} [{result.safe_to_risky_ci_lower:.2f}, {result.safe_to_risky_ci_upper:.2f}], "
                             f"R→S={result.risky_to_safe_effect:.3f} [{result.risky_to_safe_ci_lower:.2f}, {result.risky_to_safe_ci_upper:.2f}], "
                             f"q={result.combined_q_value:.4f})")
        else:
            n_bidirectional_causal = 0

        # =====================================================================
        # BUILD FINAL RESULTS
        # =====================================================================
        results = ValidationResults(
            model=self.model_name,
            n_candidates=len(candidate_features),
            n_validated=len(nominal_validated),  # Nominally significant (before FDR)
            n_fdr_validated=n_fdr_validated,  # After FDR correction
            n_bidirectional_causal=n_bidirectional_causal,
            monotonicity_threshold=self.monotonicity_threshold,
            fdr_alpha=self.fdr_alpha,
            validated_features=fdr_validated,  # Final FDR-validated features
            rejected_features=fdr_rejected,
            bidirectional_results=bidirectional_results,
            layer_summary=dict(layer_summary),
            fdr_summary=fdr_summary
        )

        # =====================================================================
        # VALIDATION SUMMARY
        # =====================================================================
        self._log("\n" + "=" * 60)
        self._log("VALIDATION SUMMARY")
        self._log("=" * 60)
        self._log(f"Total candidates:       {len(candidate_features)}")
        self._log(f"Nominally significant:  {len(nominal_validated)} "
                 f"({len(nominal_validated)/len(candidate_features)*100:.1f}%)")
        self._log(f"FDR-corrected (q<{self.fdr_alpha}): {n_fdr_validated} "
                 f"({n_fdr_validated/len(candidate_features)*100:.1f}%)")
        if n_fdr_validated > 0:
            self._log(f"Bidirectional causal:   {n_bidirectional_causal} "
                     f"({n_bidirectional_causal/n_fdr_validated*100:.1f}% of FDR-validated)")
        self._log("\nPer-layer breakdown (FDR-validated):")
        layer_fdr_counts = defaultdict(int)
        for f in fdr_validated:
            layer_fdr_counts[f.layer] += 1
        for layer in sorted(layer_fdr_counts.keys()):
            self._log(f"  Layer {layer}: {layer_fdr_counts[layer]} features")

        return results


def load_candidate_features(filepath: Path) -> List[FeatureCandidate]:
    """Load candidate features from JSON file.

    Supports two formats:
    1. Original format with 'layers' dict containing 'top_features'
    2. Merged format with 'candidates' list
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    candidates = []

    # Check for merged format with 'candidates' list
    if 'candidates' in data:
        for feat in data['candidates']:
            candidates.append(FeatureCandidate(
                layer=feat['layer'],
                feature_id=feat['feature_id'],
                contribution=feat.get('contribution', 0.0),
                direction=feat.get('direction', 'risky')
            ))
    # Original format with 'layers' dict
    elif 'layers' in data:
        for layer_data in data.get('layers', {}).values():
            for feat in layer_data.get('top_features', []):
                candidates.append(FeatureCandidate(
                    layer=feat['layer'] if 'layer' in feat else layer_data.get('layer', 0),
                    feature_id=feat['feature_id'],
                    contribution=feat.get('magnitude', feat.get('contribution', 0.0)),
                    direction=feat.get('direction', 'risky' if feat.get('value', 0) > 0 else 'safe')
                ))

    return candidates


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int_, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types."""
    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int_, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, (bool, int, float, str)):
        return obj
    # Fallback: try to convert using numpy's item() method
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj


def save_validation_results(results: ValidationResults, output_path: Path) -> None:
    """Save validation results to JSON file."""
    output_data = {
        'model': results.model,
        'n_candidates': results.n_candidates,
        'n_validated': results.n_validated,
        'n_bidirectional_causal': results.n_bidirectional_causal,
        'validation_rate': results.n_validated / results.n_candidates if results.n_candidates > 0 else 0,
        'bidirectional_rate': (
            results.n_bidirectional_causal / results.n_validated
            if results.n_validated > 0 else 0
        ),
        'monotonicity_threshold': results.monotonicity_threshold,
        'layer_summary': results.layer_summary,
        'validated_features': [asdict(f) for f in results.validated_features],
        'rejected_features': [asdict(f) for f in results.rejected_features[:50]],  # Limit rejected
        'bidirectional_results': [asdict(f) for f in results.bidirectional_results],
        # Summary of bidirectional causal features (the truly validated ones)
        'bidirectional_causal_features': [
            {
                'layer': r.layer,
                'feature_id': r.feature_id,
                'direction': r.direction,
                'safe_to_risky_effect': r.safe_to_risky_effect,
                'risky_to_safe_effect': r.risky_to_safe_effect,
                'combined_effect': r.combined_effect
            }
            for r in results.bidirectional_results if r.is_bidirectional_causal
        ]
    }

    # Convert all numpy types to Python native types
    output_data = convert_numpy_types(output_data)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, cls=NumpyEncoder)


def main():
    parser = argparse.ArgumentParser(description='Validate features via soft interpolation patching')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'],
                       help='Model to use')
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID')
    parser.add_argument('--candidates', type=str, required=True,
                       help='Path to candidate features JSON')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--n-prompts', type=int, default=None,
                       help='Number of test prompts (overrides config)')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Monotonicity threshold (overrides config)')
    parser.add_argument('--alpha-values', type=str, default=None,
                       help='Comma-separated alpha values (e.g., "0.0,0.5,1.0")')
    parser.add_argument('--prompts', type=str, default=None,
                       help='Path to condition prompts JSON (from real_prompt_extractor.py)')

    args = parser.parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    if args.n_prompts:
        config.setdefault('soft_interpolation', {})['n_test_prompts'] = args.n_prompts
    if args.threshold:
        config.setdefault('soft_interpolation', {})['monotonicity_threshold'] = args.threshold
    if args.alpha_values:
        config.setdefault('soft_interpolation', {})['alpha_values'] = [
            float(a) for a in args.alpha_values.split(',')
        ]

    # Setup paths
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / 'logs'
    checkpoint_dir = output_dir / 'checkpoints'

    # Setup logging
    logger = setup_logging(f'soft_interpolation_{args.model}', log_dir)
    logger.info("=" * 70)
    logger.info(f"SOFT INTERPOLATION PATCHING - {args.model.upper()}")
    logger.info("=" * 70)
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Candidates: {args.candidates}")

    # Setup checkpoint manager
    checkpoint_mgr = CheckpointManager(checkpoint_dir, f'soft_interp_{args.model}')

    # Load candidate features
    candidates_path = Path(args.candidates)
    if not candidates_path.is_absolute():
        candidates_path = output_dir / candidates_path

    logger.info(f"Loading candidates from {candidates_path}")
    candidate_features = load_candidate_features(candidates_path)
    logger.info(f"Loaded {len(candidate_features)} candidate features")

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        args.model, 'cuda:0', torch.bfloat16, logger
    )

    # Load real prompts if provided
    test_prompts = None
    if args.prompts:
        prompts_path = Path(args.prompts)
        if not prompts_path.is_absolute():
            prompts_path = output_dir / prompts_path
        if prompts_path.exists():
            logger.info(f"Loading real prompts from {prompts_path}")
            test_prompts = load_real_prompts(str(prompts_path), args.model)
            logger.info(f"Loaded {len(test_prompts)} real condition prompts:")
            for p in test_prompts:
                logger.info(f"  - {p['condition']}: balance=${p['balance']}, {p['rounds_played']} rounds")
        else:
            logger.warning(f"Prompts file not found: {prompts_path}, using synthetic prompts")

    # Initialize patcher
    patcher = SoftInterpolationPatcher(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model,
        config=config,
        device='cuda:0',
        logger=logger
    )

    # Run validation with real prompts if available
    results = patcher.validate_features(
        candidate_features=candidate_features,
        test_prompts=test_prompts,  # Pass real prompts if loaded
        checkpoint_mgr=checkpoint_mgr
    )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f'validated_features_{args.model}_{timestamp}.json'
    save_validation_results(results, results_path)
    logger.info(f"\nResults saved to {results_path}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("TOP VALIDATED FEATURES")
    logger.info("=" * 70)

    # Sort by effect size
    sorted_validated = sorted(
        results.validated_features,
        key=lambda x: abs(x.effect_size),
        reverse=True
    )

    for feat in sorted_validated[:10]:
        logger.info(f"L{feat.layer}-{feat.feature_id}:")
        logger.info(f"  Direction: {feat.direction}")
        logger.info(f"  Spearman rho: {feat.spearman_rho:.3f} (p={feat.spearman_p:.4f})")
        logger.info(f"  Effect size: {feat.effect_size:.3f}")
        logger.info(f"  Response curve: {[f'{r:.3f}' for r in feat.behavioral_responses]}")

    logger.info("\nSoft interpolation patching complete!")


if __name__ == '__main__':
    main()
