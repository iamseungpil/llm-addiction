#!/usr/bin/env python3
"""
Phase 4: Head and Component Analysis

Decompose steering effects to attention head and component (Attention vs MLP) level.
This provides fine-grained mechanistic understanding of which model components
drive gambling behavior.

Analysis Components:
1. Head-Level Steering Vector Decomposition
   - Per-head contribution to steering vector
   - Identify heads that encode risky vs safe directions

2. Head-Level Causal Validation
   - Apply steering to individual heads
   - Measure behavioral impact per head

3. Component Attribution (Attention vs MLP)
   - Separate steering effects by component type
   - Compare attention-only vs MLP-only steering

Input:
  - Steering vectors from Phase 1
  - Validated causal layers from Phase 2

Output:
  - Per-head steering magnitudes and directions
  - Head-level causal effect sizes
  - Component attribution analysis
  - Top contributing heads for paper

Usage:
    # 1. Decompose ALL layers (no model needed, fast)
    python phase4_head_component_analysis.py --model llama --decompose-all --vectors PATH

    # 2. Validate multiple layers
    python phase4_head_component_analysis.py --model llama --gpu 0 --validate-layers 23,9,17 --vectors PATH

    # 3. Cumulative head steering test
    python phase4_head_component_analysis.py --model llama --gpu 0 --layer 23 --cumulative --vectors PATH

    # 4. Single layer analysis (original behavior)
    python phase4_head_component_analysis.py --model llama --gpu 0 --layer 23 --validate --components --vectors PATH
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


# =============================================================================
# Custom JSON Encoder
# =============================================================================

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


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class HeadContribution:
    """Contribution of a single attention head to the steering vector."""
    layer: int
    head_id: int
    contribution_magnitude: float  # L2 norm of head's portion
    contribution_fraction: float  # Fraction of total layer magnitude
    direction: str  # "risky" or "safe" based on projection
    mean_value: float  # Mean value of head dimensions
    max_abs_value: float  # Max absolute value in head dimensions
    variance: float  # Variance within head dimensions


@dataclass
class HeadCausalResult:
    """Causal validation result for a single head."""
    layer: int
    head_id: int
    stop_rate_baseline: float
    stop_rate_steered: float
    effect_size: float  # Cohen's d
    p_value: float
    is_causal: bool
    direction_correct: bool  # +alpha increases risky?
    n_trials: int


@dataclass
class ComponentAttribution:
    """Attribution of steering effect to model components."""
    layer: int
    attention_effect: float  # Effect size when steering attention only
    mlp_effect: float  # Effect size when steering MLP only
    full_layer_effect: float  # Effect size when steering full layer
    attention_fraction: float  # attention_effect / full_layer_effect
    mlp_fraction: float  # mlp_effect / full_layer_effect
    interaction: float  # full - (attention + mlp), captures non-additivity
    dominant_component: str  # "attention", "mlp", or "balanced"


@dataclass
class Phase4Results:
    """Complete results from Phase 4 analysis."""
    model: str
    layer: int
    timestamp: str

    # Head-level analysis
    n_heads: int
    head_dim: int
    head_contributions: List[Dict]
    top_risky_heads: List[Dict]
    top_safe_heads: List[Dict]

    # Causal validation (if run)
    head_causal_results: List[Dict] = field(default_factory=list)
    n_causal_heads: int = 0

    # Component attribution (if run)
    component_attribution: Optional[Dict] = None


# =============================================================================
# Model Architecture Constants
# =============================================================================

MODEL_ARCHITECTURE = {
    'llama': {
        'n_heads': 32,
        'head_dim': 128,
        'd_model': 4096,
        'n_layers': 32,
        # Attention output: model.model.layers[L].self_attn.o_proj
        # MLP output: model.model.layers[L].mlp.down_proj
    },
    'gemma': {
        'n_heads': 16,
        'head_dim': 256,  # gemma-2-9b: 3584 / 16 = 224 (with GQA adjustments)
        'd_model': 3584,
        'n_layers': 42,
    }
}


# =============================================================================
# Phase 4: Head and Component Analysis
# =============================================================================

class Phase4HeadComponentAnalysis:
    """Head-level and component-level steering vector analysis."""

    def __init__(
        self,
        model_name: str = "llama",
        gpu_id: int = 0,
        config_path: str = None
    ):
        self.model_name = model_name
        self.gpu_id = gpu_id

        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "experiment_config_direct_steering.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.device = 'cuda:0'

        self.output_dir = Path(self.config['output_dir']) / "phase4_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Model architecture - prefer config values, fall back to defaults
        model_config = self.config.get('models', {}).get(model_name, {})
        self.arch = MODEL_ARCHITECTURE.get(model_name, MODEL_ARCHITECTURE['llama'])

        self.n_heads = model_config.get('n_heads', self.arch['n_heads'])
        self.d_model = model_config.get('d_model', self.arch['d_model'])
        self.head_dim = self.d_model // self.n_heads

        # Phase 4 specific config
        self.phase4_config = self.config.get('phase4_head_component', {})

        # Steering vectors
        self.steering_vectors = {}

        # Model (only loaded if validation requested)
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

    def load_model(self):
        """Load model for causal validation."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_config = self.config['models'].get(self.model_name)
        if model_config is None:
            raise ValueError(f"Unknown model: {self.model_name}")

        model_id = model_config['model_id']
        print(f"Loading {model_id} on GPU {self.gpu_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={'': 0},
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )
        self.model.eval()
        print(f"Model loaded: {model_id}")

    def load_test_prompts(self, prompts_path: str = None) -> List[Dict]:
        """Load test prompts for causal validation (balanced across conditions)."""
        if prompts_path is None:
            prompts_path = self.config.get('prompts_path')

            if prompts_path is None:
                prompts_path = Path(self.config['output_dir']).parent / "steering_vector_experiment_full" / "condition_prompts.json"

        print(f"Loading prompts: {prompts_path}")
        with open(prompts_path) as f:
            data = json.load(f)

        conditions = data['conditions'].get(self.model_name, [])
        if not conditions and self.model_name == 'gemma_base':
            conditions = data['conditions'].get('gemma', [])

        # Group by condition for balanced sampling
        from collections import defaultdict
        by_condition = defaultdict(list)
        for c in conditions:
            by_condition[c['condition_name']].append(c)

        n_total = self.config.get('phase2_direct_steering', {}).get('n_test_prompts', 20)
        n_conditions = len(by_condition)
        n_per_condition = n_total // n_conditions if n_conditions > 0 else n_total

        prompts = []
        for cond_name in sorted(by_condition.keys()):
            cond_prompts = by_condition[cond_name]
            for i, c in enumerate(cond_prompts[:n_per_condition]):
                prompts.append({
                    'id': f"{c['condition_name']}_{i}",
                    'condition': c['condition_name'],
                    'prompt': c['prompt']
                })

        print(f"Loaded {len(prompts)} test prompts (balanced across {n_conditions} conditions)")
        for cond_name in sorted(by_condition.keys()):
            count = sum(1 for p in prompts if p['condition'] == cond_name)
            print(f"  {cond_name}: {count}")

        return prompts

    def load_validated_layers(self, phase2_results_path: str) -> List[int]:
        """Load causal layers from Phase 2 results JSON."""
        if not phase2_results_path:
            return []
        try:
            with open(phase2_results_path, 'r') as f:
                data = json.load(f)
            validations = data.get('validations', [])
            return [v['layer'] for v in validations if v.get('is_causal')]
        except Exception as e:
            print(f"Warning: Failed to load Phase 2 results ({e})")
            return []

    # =========================================================================
    # Head-Level Steering Vector Decomposition
    # =========================================================================

    def decompose_steering_vector_by_head(
        self,
        layer: int
    ) -> List[HeadContribution]:
        """
        Decompose layer steering vector into per-head contributions.

        The steering vector has shape (d_model,) = (n_heads * head_dim,).
        We reshape to (n_heads, head_dim) and analyze each head's portion.

        IMPORTANT LIMITATION:
        This is an APPROXIMATION. The steering vector is computed at the
        residual stream level (after O_proj transformation), so individual
        head contributions are NOT truly separable. The O_proj matrix
        combines all heads: hidden = (concat(head_outputs) @ W_O) + residual.

        This analysis provides a rough heuristic for identifying potentially
        important heads, but the actual causal contributions may differ.
        Use --validate mode for empirical causal validation of specific heads.

        Args:
            layer: Target layer

        Returns:
            List of HeadContribution objects, sorted by magnitude (approximate)
        """
        if layer not in self.steering_vectors:
            raise ValueError(f"No steering vector for layer {layer}")

        steering_vec = self.steering_vectors[layer].numpy()

        # Validate dimensions
        if len(steering_vec) != self.d_model:
            print(f"Warning: steering vector dim {len(steering_vec)} != d_model {self.d_model}")
            # Adjust architecture based on actual vector
            self.d_model = len(steering_vec)
            self.head_dim = self.d_model // self.n_heads

        # Reshape to (n_heads, head_dim)
        # LLaMA stores heads consecutively: [head0_dim0, ..., head0_dim127, head1_dim0, ...]
        head_vectors = steering_vec.reshape(self.n_heads, self.head_dim)

        # Compute total layer magnitude for normalization
        total_magnitude = np.linalg.norm(steering_vec)

        contributions = []
        for head_id in range(self.n_heads):
            head_vec = head_vectors[head_id]

            magnitude = np.linalg.norm(head_vec)
            fraction = magnitude / total_magnitude if total_magnitude > 0 else 0

            # Direction: positive mean suggests risky direction
            # (since steering_vec = mean(bankrupt) - mean(safe))
            mean_val = np.mean(head_vec)
            direction = "risky" if mean_val > 0 else "safe"

            contributions.append(HeadContribution(
                layer=layer,
                head_id=head_id,
                contribution_magnitude=float(magnitude),
                contribution_fraction=float(fraction),
                direction=direction,
                mean_value=float(mean_val),
                max_abs_value=float(np.max(np.abs(head_vec))),
                variance=float(np.var(head_vec))
            ))

        # Sort by magnitude (descending)
        contributions.sort(key=lambda x: x.contribution_magnitude, reverse=True)

        return contributions

    # =========================================================================
    # Head-Level Causal Validation
    # =========================================================================

    def create_head_steering_hook(
        self,
        layer: int,
        head_id: int,
        alpha: float
    ):
        """
        Create hook that steers only a specific attention head.

        The hook modifies only the portion of the hidden state corresponding
        to the specified head.

        Args:
            layer: Target layer
            head_id: Head to steer (0 to n_heads-1)
            alpha: Steering strength

        Returns:
            Hook function
        """
        steering_vec = self.steering_vectors[layer].to(self.device)

        # Reshape to get per-head portions
        head_vectors = steering_vec.reshape(self.n_heads, self.head_dim)
        head_steering = head_vectors[head_id]  # (head_dim,)

        # Create mask for this head's dimensions
        start_idx = head_id * self.head_dim
        end_idx = start_idx + self.head_dim

        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                # Apply steering only to this head's dimensions
                modified = hidden.clone()
                modified[:, -1, start_idx:end_idx] = (
                    modified[:, -1, start_idx:end_idx] + alpha * head_steering
                )
                return (modified,) + output[1:]
            else:
                output[:, -1, start_idx:end_idx] = (
                    output[:, -1, start_idx:end_idx] + alpha * head_steering
                )
                return output

        return hook

    def validate_single_head(
        self,
        layer: int,
        head_id: int,
        test_prompts: List[Dict],
        alpha: float = 2.0
    ) -> HeadCausalResult:
        """
        Validate causal effect of steering a single head.

        Args:
            layer: Target layer
            head_id: Head to test
            test_prompts: List of test prompt dicts
            alpha: Steering strength

        Returns:
            HeadCausalResult with effect measurements
        """
        import re

        layer_module = self.model.model.layers[layer]

        baseline_stops = []
        steered_stops = []

        for prompt_info in test_prompts:
            prompt = prompt_info['prompt']

            # Apply chat template if needed
            if self.config['models'][self.model_name].get('use_chat_template', False):
                chat = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted_prompt = prompt

            inputs = self.tokenizer(formatted_prompt, return_tensors='pt').to(self.device)

            # Baseline (no steering)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            input_len = inputs['input_ids'].shape[1]
            response = self.tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True
            ).strip()

            baseline_stops.append(1 if 'stop' in response.lower() else 0)

            # Steered
            hook = self.create_head_steering_hook(layer, head_id, alpha)
            handle = layer_module.register_forward_hook(hook)

            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                response = self.tokenizer.decode(
                    outputs[0][input_len:], skip_special_tokens=True
                ).strip()

                steered_stops.append(1 if 'stop' in response.lower() else 0)
            finally:
                handle.remove()

        # Compute statistics
        baseline_rate = np.mean(baseline_stops)
        steered_rate = np.mean(steered_stops)

        # Cohen's d
        if len(baseline_stops) > 1 and len(steered_stops) > 1:
            pooled_std = np.sqrt(
                (np.var(baseline_stops) + np.var(steered_stops)) / 2
            )
            effect_size = (steered_rate - baseline_rate) / pooled_std if pooled_std > 0 else 0

            # t-test
            t_stat, p_value = stats.ttest_ind(steered_stops, baseline_stops)
        else:
            effect_size = 0
            p_value = 1.0

        # Direction check: +alpha should decrease stop rate (more risky)
        direction_correct = steered_rate < baseline_rate

        # Get thresholds from config
        causal_config = self.phase4_config.get('causal_validation', {})
        min_effect = causal_config.get('min_effect_size', 0.3)
        p_threshold = causal_config.get('p_value_threshold', 0.05)

        # Causal if significant effect in correct direction
        is_causal = (abs(effect_size) >= min_effect and
                    p_value < p_threshold and
                    direction_correct)

        return HeadCausalResult(
            layer=layer,
            head_id=head_id,
            stop_rate_baseline=float(baseline_rate),
            stop_rate_steered=float(steered_rate),
            effect_size=float(effect_size),
            p_value=float(p_value),
            is_causal=is_causal,
            direction_correct=direction_correct,
            n_trials=len(test_prompts)
        )

    def validate_all_heads(
        self,
        layer: int,
        test_prompts: List[Dict],
        alpha: float = 2.0,
        top_k: int = None
    ) -> List[HeadCausalResult]:
        """
        Validate causal effects for all (or top-k) heads.

        Args:
            layer: Target layer
            test_prompts: List of test prompts
            alpha: Steering strength
            top_k: If set, only validate top-k heads by magnitude

        Returns:
            List of HeadCausalResult objects
        """
        # First decompose to find top heads
        contributions = self.decompose_steering_vector_by_head(layer)

        if top_k is not None:
            heads_to_test = [c.head_id for c in contributions[:top_k]]
        else:
            heads_to_test = list(range(self.n_heads))

        results = []
        for i, head_id in enumerate(tqdm(heads_to_test, desc=f"Validating heads (Layer {layer})")):
            result = self.validate_single_head(layer, head_id, test_prompts, alpha)
            results.append(result)

            # GPU memory management: clear cache periodically
            if (i + 1) % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Monitor memory usage
                memory_gb = torch.cuda.memory_allocated() / 1e9
                if memory_gb > 30:
                    print(f"\nWARNING: High GPU memory usage: {memory_gb:.1f}GB")

        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Sort by effect size
        results.sort(key=lambda x: abs(x.effect_size), reverse=True)

        return results

    # =========================================================================
    # Component Attribution (Attention vs MLP)
    # =========================================================================

    def create_full_layer_hook(self, layer: int, alpha: float):
        """
        Create hook that steers the full layer output (after both Attention and MLP).

        This applies steering to the residual stream at the layer output,
        affecting all subsequent computations.
        """
        steering_vec = self.steering_vectors[layer].to(self.device)

        def hook(module, input, output):
            # Hook placed on full layer output
            if isinstance(output, tuple):
                hidden = output[0]
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * steering_vec
                return (hidden,) + output[1:]
            else:
                output[:, -1, :] = output[:, -1, :] + alpha * steering_vec
                return output

        return hook

    def create_attention_only_hook(self, layer: int, alpha: float):
        """Create hook that steers only the attention component."""
        steering_vec = self.steering_vectors[layer].to(self.device)

        def hook(module, input, output):
            # Hook placed on self_attn output (o_proj output)
            if isinstance(output, tuple):
                hidden = output[0]
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * steering_vec
                return (hidden,) + output[1:]
            else:
                output[:, -1, :] = output[:, -1, :] + alpha * steering_vec
                return output

        return hook

    def create_mlp_only_hook(self, layer: int, alpha: float):
        """Create hook that steers only the MLP component."""
        steering_vec = self.steering_vectors[layer].to(self.device)

        def hook(module, input, output):
            # Hook placed on MLP output (down_proj output)
            if isinstance(output, tuple):
                hidden = output[0]
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * steering_vec
                return (hidden,) + output[1:]
            else:
                output[:, -1, :] = output[:, -1, :] + alpha * steering_vec
                return output

        return hook

    def measure_component_effect(
        self,
        layer: int,
        test_prompts: List[Dict],
        alpha: float = 2.0,
        component: str = "full"
    ) -> float:
        """
        Measure behavioral effect of steering a specific component.

        Args:
            layer: Target layer
            test_prompts: Test prompts
            alpha: Steering strength
            component: "full", "attention", or "mlp"

        Returns:
            Stop rate under steering
        """
        layer_module = self.model.model.layers[layer]
        stops = []

        for prompt_info in test_prompts:
            prompt = prompt_info['prompt']

            if self.config['models'][self.model_name].get('use_chat_template', False):
                chat = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted_prompt = prompt

            inputs = self.tokenizer(formatted_prompt, return_tensors='pt').to(self.device)

            # Register appropriate hook
            if component == "full":
                hook = self.create_full_layer_hook(layer, alpha)
                handle = layer_module.register_forward_hook(hook)
            elif component == "attention":
                # Hook on attention output specifically
                hook = self.create_attention_only_hook(layer, alpha)
                handle = layer_module.self_attn.register_forward_hook(hook)
            elif component == "mlp":
                # Hook on MLP output specifically
                hook = self.create_mlp_only_hook(layer, alpha)
                handle = layer_module.mlp.register_forward_hook(hook)
            else:
                raise ValueError(f"Unknown component: {component}")

            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                input_len = inputs['input_ids'].shape[1]
                response = self.tokenizer.decode(
                    outputs[0][input_len:], skip_special_tokens=True
                ).strip()

                stops.append(1 if 'stop' in response.lower() else 0)
            finally:
                handle.remove()

        return np.mean(stops)

    def analyze_component_attribution(
        self,
        layer: int,
        test_prompts: List[Dict],
        alpha: float = 2.0
    ) -> ComponentAttribution:
        """
        Analyze contribution of attention vs MLP to steering effect.

        Args:
            layer: Target layer
            test_prompts: Test prompts
            alpha: Steering strength

        Returns:
            ComponentAttribution with effect decomposition
        """
        print(f"\nAnalyzing component attribution for Layer {layer}")

        # Baseline (no steering)
        baseline_rate = self.measure_component_effect(
            layer, test_prompts, alpha=0.0, component="full"
        )
        print(f"  Baseline stop rate: {baseline_rate:.3f}")

        # Full layer steering
        full_rate = self.measure_component_effect(
            layer, test_prompts, alpha=alpha, component="full"
        )
        full_effect = baseline_rate - full_rate  # Positive if steering reduces stops
        print(f"  Full layer effect: {full_effect:.3f}")

        # Attention-only steering
        attn_rate = self.measure_component_effect(
            layer, test_prompts, alpha=alpha, component="attention"
        )
        attn_effect = baseline_rate - attn_rate
        print(f"  Attention-only effect: {attn_effect:.3f}")

        # MLP-only steering
        mlp_rate = self.measure_component_effect(
            layer, test_prompts, alpha=alpha, component="mlp"
        )
        mlp_effect = baseline_rate - mlp_rate
        print(f"  MLP-only effect: {mlp_effect:.3f}")

        # Compute fractions and interaction
        if abs(full_effect) > 0.01:
            attn_fraction = attn_effect / full_effect
            mlp_fraction = mlp_effect / full_effect
        else:
            attn_fraction = 0.5
            mlp_fraction = 0.5

        interaction = full_effect - (attn_effect + mlp_effect)

        # Determine dominant component
        comp_config = self.phase4_config.get('component_attribution', {})
        dominance_threshold = comp_config.get('dominance_threshold', 1.5)

        if abs(attn_effect) > dominance_threshold * abs(mlp_effect):
            dominant = "attention"
        elif abs(mlp_effect) > dominance_threshold * abs(attn_effect):
            dominant = "mlp"
        else:
            dominant = "balanced"

        return ComponentAttribution(
            layer=layer,
            attention_effect=float(attn_effect),
            mlp_effect=float(mlp_effect),
            full_layer_effect=float(full_effect),
            attention_fraction=float(attn_fraction),
            mlp_fraction=float(mlp_fraction),
            interaction=float(interaction),
            dominant_component=dominant
        )

    # =========================================================================
    # Main Run Methods
    # =========================================================================

    def run_head_analysis(
        self,
        layer: int,
        validate: bool = False,
        top_k_validate: int = 10
    ) -> Phase4Results:
        """
        Run head-level analysis for a layer.

        Args:
            layer: Target layer
            validate: Whether to run causal validation
            top_k_validate: Number of top heads to validate

        Returns:
            Phase4Results with all analysis
        """
        print(f"\n{'='*70}")
        print(f"PHASE 4: HEAD-LEVEL ANALYSIS - Layer {layer}")
        print(f"{'='*70}")

        # Decompose steering vector by head
        print("\n1. Decomposing steering vector by head...")
        contributions = self.decompose_steering_vector_by_head(layer)

        # Separate by direction
        risky_heads = [c for c in contributions if c.direction == "risky"]
        safe_heads = [c for c in contributions if c.direction == "safe"]

        print(f"   Total heads: {self.n_heads}")
        print(f"   Head dimension: {self.head_dim}")
        print(f"   Risky-direction heads: {len(risky_heads)}")
        print(f"   Safe-direction heads: {len(safe_heads)}")

        # Top contributors
        print("\n   Top 5 contributing heads:")
        for i, c in enumerate(contributions[:5]):
            print(f"     Head {c.head_id}: mag={c.contribution_magnitude:.4f}, "
                  f"frac={c.contribution_fraction:.3f}, dir={c.direction}")

        # Causal validation (if requested)
        head_causal_results = []
        n_causal = 0

        if validate:
            print(f"\n2. Running causal validation for top {top_k_validate} heads...")

            if self.model is None:
                self.load_model()

            test_prompts = self.load_test_prompts()

            head_causal_results = self.validate_all_heads(
                layer, test_prompts, alpha=2.0, top_k=top_k_validate
            )

            n_causal = sum(1 for r in head_causal_results if r.is_causal)
            print(f"\n   Causal heads: {n_causal}/{len(head_causal_results)}")

            for r in head_causal_results[:5]:
                print(f"     Head {r.head_id}: effect={r.effect_size:.3f}, "
                      f"p={r.p_value:.4f}, causal={r.is_causal}")

        # Build results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        results = Phase4Results(
            model=self.model_name,
            layer=layer,
            timestamp=timestamp,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            head_contributions=[asdict(c) for c in contributions],
            top_risky_heads=[asdict(c) for c in risky_heads[:10]],
            top_safe_heads=[asdict(c) for c in safe_heads[:10]],
            head_causal_results=[asdict(r) for r in head_causal_results],
            n_causal_heads=n_causal
        )

        return results

    def run_component_analysis(
        self,
        layer: int,
        alpha: float = 2.0
    ) -> ComponentAttribution:
        """
        Run component attribution analysis.

        Args:
            layer: Target layer
            alpha: Steering strength

        Returns:
            ComponentAttribution results
        """
        print(f"\n{'='*70}")
        print(f"PHASE 4: COMPONENT ATTRIBUTION - Layer {layer}")
        print(f"{'='*70}")

        if self.model is None:
            self.load_model()

        test_prompts = self.load_test_prompts()

        attribution = self.analyze_component_attribution(layer, test_prompts, alpha)

        print(f"\n   Summary:")
        print(f"     Attention contribution: {attribution.attention_fraction:.1%}")
        print(f"     MLP contribution: {attribution.mlp_fraction:.1%}")
        print(f"     Interaction effect: {attribution.interaction:.3f}")
        print(f"     Dominant component: {attribution.dominant_component}")

        return attribution

    def run(
        self,
        vectors_path: str = None,
        layer: int = None,
        validate: bool = False,
        analyze_components: bool = False,
        top_k_validate: int = None,
        decompose_all: bool = False,
        validate_layers: List[int] = None,
        cumulative: bool = False,
        phase2_results_path: str = None
    ):
        """
        Run complete Phase 4 analysis.

        Args:
            vectors_path: Path to steering vectors
            layer: Target layer for single-layer analysis
            validate: Run causal validation on heads
            analyze_components: Run component attribution
            top_k_validate: Number of heads to validate
            decompose_all: Run decomposition on ALL layers
            validate_layers: List of layers to validate
            cumulative: Run cumulative head steering test
            phase2_results_path: Path to Phase 2 results for layer selection
        """
        if top_k_validate is None:
            causal_config = self.phase4_config.get('causal_validation', {})
            top_k_validate = causal_config.get('top_k_validate', 10)

        print("=" * 70)
        print("PHASE 4: HEAD AND COMPONENT ANALYSIS")
        print("=" * 70)
        print(f"Model: {self.model_name}")

        # Load steering vectors
        self.load_steering_vectors(vectors_path)

        # Mode 1: Decompose all layers
        if decompose_all:
            print(f"Mode: Decompose ALL layers")
            return self.run_all_layers_decomposition()

        # Mode 2: Validate multiple layers
        if validate_layers:
            print(f"Mode: Validate layers {validate_layers}")
            return self.run_multi_layer_validation(
                validate_layers, top_k_validate
            )

        # Mode 3: Cumulative head steering
        if cumulative:
            if layer is None:
                layer = 23  # Default to known causal layer
            print(f"Mode: Cumulative head steering (Layer {layer})")
            return self.run_cumulative_steering(layer)

        # Select layer from Phase 2 results if provided
        if phase2_results_path and layer is None and not validate_layers and not cumulative:
            validated_layers = self.load_validated_layers(phase2_results_path)
            if validated_layers:
                layer = validated_layers[0]
                print(f"Selected causal layer from Phase 2: {layer}")

        # Mode 4: Single layer analysis (original behavior)
        if layer is None:
            layer = self.phase4_config.get('default_layer', 23)

        print(f"Mode: Single layer analysis")
        print(f"Target layer: {layer}")
        print(f"Validate heads: {validate}")
        print(f"Component analysis: {analyze_components}")

        if layer not in self.steering_vectors:
            available = sorted(self.steering_vectors.keys())
            raise ValueError(
                f"Layer {layer} not in steering vectors. Available: {available}"
            )

        # Run head analysis
        results = self.run_head_analysis(
            layer,
            validate=validate,
            top_k_validate=top_k_validate
        )

        # Run component analysis if requested
        if analyze_components:
            attribution = self.run_component_analysis(layer)
            results.component_attribution = asdict(attribution)

        # Save results
        self.save_results(results)

        return results

    def run_all_layers_decomposition(self) -> Dict:
        """
        Run head decomposition for ALL layers.
        This doesn't require model loading - just analyzes steering vectors.
        """
        print(f"\n{'='*70}")
        print("ALL-LAYER HEAD DECOMPOSITION")
        print(f"{'='*70}")

        all_results = {}
        available_layers = sorted(self.steering_vectors.keys())
        print(f"Analyzing {len(available_layers)} layers: {available_layers}")

        for layer in tqdm(available_layers, desc="Decomposing layers"):
            contributions = self.decompose_steering_vector_by_head(layer)

            # Summarize
            risky_heads = [c for c in contributions if c.direction == "risky"]
            safe_heads = [c for c in contributions if c.direction == "safe"]

            all_results[layer] = {
                'n_risky': len(risky_heads),
                'n_safe': len(safe_heads),
                'top_5_heads': [
                    {
                        'head_id': c.head_id,
                        'magnitude': round(c.contribution_magnitude, 4),
                        'fraction': round(c.contribution_fraction, 3),
                        'direction': c.direction
                    }
                    for c in contributions[:5]
                ],
                'all_contributions': [asdict(c) for c in contributions]
            }

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = self.output_dir / f"head_decomposition_{self.model_name}_all_layers_{timestamp}.json"

        output_data = {
            'model': self.model_name,
            'timestamp': timestamp,
            'n_layers': len(available_layers),
            'n_heads': self.n_heads,
            'head_dim': self.head_dim,
            'layer_results': all_results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, cls=NumpyEncoder)
        print(f"\nResults saved: {output_path}")

        # Print summary
        print(f"\n{'='*70}")
        print("SUMMARY: Head Direction Distribution by Layer")
        print(f"{'='*70}")
        print(f"{'Layer':>6} | {'Risky':>6} | {'Safe':>6} | Top Head (dir, frac)")
        print("-" * 50)
        for layer in available_layers:
            r = all_results[layer]
            top = r['top_5_heads'][0]
            print(f"{layer:>6} | {r['n_risky']:>6} | {r['n_safe']:>6} | "
                  f"H{top['head_id']} ({top['direction']}, {top['fraction']:.1%})")

        return output_data

    def run_multi_layer_validation(
        self,
        layers: List[int],
        top_k: int = 10
    ) -> Dict:
        """
        Run head validation on multiple layers.
        """
        print(f"\n{'='*70}")
        print(f"MULTI-LAYER HEAD VALIDATION")
        print(f"{'='*70}")
        print(f"Layers to validate: {layers}")
        print(f"Top-k heads per layer: {top_k}")

        # Load model
        self.load_model()
        test_prompts = self.load_test_prompts()

        all_results = {}
        for layer in layers:
            if layer not in self.steering_vectors:
                print(f"Warning: Layer {layer} not in steering vectors, skipping")
                continue

            print(f"\n{'='*50}")
            print(f"Validating Layer {layer}")
            print(f"{'='*50}")

            results = self.run_head_analysis(
                layer,
                validate=True,
                top_k_validate=top_k
            )
            all_results[layer] = asdict(results)

            # GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save combined results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = self.output_dir / f"head_validation_{self.model_name}_L{'_'.join(map(str, layers))}_{timestamp}.json"

        output_data = {
            'model': self.model_name,
            'timestamp': timestamp,
            'layers_validated': layers,
            'top_k': top_k,
            'results': all_results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, cls=NumpyEncoder)
        print(f"\nResults saved: {output_path}")

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY: Causal Heads by Layer")
        print(f"{'='*70}")
        for layer, result in all_results.items():
            n_causal = result.get('n_causal_heads', 0)
            print(f"Layer {layer}: {n_causal}/{top_k} causal heads")

        return output_data

    def run_cumulative_steering(self, layer: int, alpha: float = 2.0) -> Dict:
        """
        Test cumulative head steering effect.
        Add heads progressively and measure behavioral change.
        """
        print(f"\n{'='*70}")
        print(f"CUMULATIVE HEAD STEERING TEST - Layer {layer}")
        print(f"{'='*70}")

        # Load model
        self.load_model()
        test_prompts = self.load_test_prompts()

        # Get head contributions sorted by magnitude
        contributions = self.decompose_steering_vector_by_head(layer)

        # Test configurations: 1, 2, 3, 5, 10, 15, 20, 25, 32 heads
        n_heads_to_test = [1, 2, 3, 5, 10, 15, 20, 25, self.n_heads]
        n_heads_to_test = [n for n in n_heads_to_test if n <= self.n_heads]

        results = []
        steering_vec = self.steering_vectors[layer].to(self.device)

        print(f"\nTesting cumulative steering with alpha={alpha}")
        print(f"Head configurations: {n_heads_to_test}")

        # Baseline
        baseline_stops = []
        for prompt_data in tqdm(test_prompts[:20], desc="Baseline"):
            prompt = prompt_data.get('prompt', prompt_data)
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                chat = [{"role": "user", "content": prompt}]
                formatted = self.tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted = prompt

            inputs = self.tokenizer(formatted, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=100, do_sample=True,
                    temperature=0.7, pad_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
            )
            baseline_stops.append(1 if 'stop' in response.lower() else 0)

        baseline_rate = np.mean(baseline_stops)
        print(f"Baseline stop rate: {baseline_rate:.3f}")

        for n_heads in tqdm(n_heads_to_test, desc="Testing head combinations"):
            # Get top-n heads
            top_n_heads = [c.head_id for c in contributions[:n_heads]]

            # Create cumulative steering vector (only for selected heads)
            cumulative_steering = torch.zeros_like(steering_vec)
            for head_id in top_n_heads:
                start_idx = head_id * self.head_dim
                end_idx = start_idx + self.head_dim
                cumulative_steering[start_idx:end_idx] = steering_vec[start_idx:end_idx]

            # Create hook
            def create_cumulative_hook(steering):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                        hidden[:, -1, :] = hidden[:, -1, :] + alpha * steering
                        return (hidden,) + output[1:]
                    else:
                        output[:, -1, :] = output[:, -1, :] + alpha * steering
                        return output
                return hook

            # Test
            steered_stops = []
            layer_module = self.model.model.layers[layer]

            for prompt_data in test_prompts[:20]:
                prompt = prompt_data.get('prompt', prompt_data)
                if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                    chat = [{"role": "user", "content": prompt}]
                    formatted = self.tokenizer.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=True
                    )
                else:
                    formatted = prompt

                inputs = self.tokenizer(formatted, return_tensors='pt').to(self.device)

                hook = create_cumulative_hook(cumulative_steering)
                handle = layer_module.register_forward_hook(hook)

                try:
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs, max_new_tokens=100, do_sample=True,
                            temperature=0.7, pad_token_id=self.tokenizer.eos_token_id
                        )
                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
                    )
                    steered_stops.append(1 if 'stop' in response.lower() else 0)
                finally:
                    handle.remove()

            stop_rate = np.mean(steered_stops)
            effect = baseline_rate - stop_rate

            results.append({
                'n_heads': n_heads,
                'head_ids': top_n_heads,
                'stop_rate': float(stop_rate),
                'effect': float(effect)
            })

            print(f"  {n_heads:2d} heads: stop_rate={stop_rate:.3f}, effect={effect:+.3f}")

            # GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = self.output_dir / f"cumulative_steering_{self.model_name}_L{layer}_{timestamp}.json"

        output_data = {
            'model': self.model_name,
            'layer': layer,
            'timestamp': timestamp,
            'alpha': alpha,
            'baseline_stop_rate': float(baseline_rate),
            'n_test_prompts': 20,
            'results': results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, cls=NumpyEncoder)
        print(f"\nResults saved: {output_path}")

        # Summary
        print(f"\n{'='*70}")
        print("CUMULATIVE STEERING SUMMARY")
        print(f"{'='*70}")
        print(f"{'N Heads':>8} | {'Stop Rate':>10} | {'Effect':>10}")
        print("-" * 35)
        for r in results:
            print(f"{r['n_heads']:>8} | {r['stop_rate']:>10.3f} | {r['effect']:>+10.3f}")

        return output_data

    def save_results(self, results: Phase4Results):
        """Save analysis results."""
        timestamp = results.timestamp
        layer = results.layer

        # Full results
        output_path = self.output_dir / f"head_analysis_{self.model_name}_L{layer}_{timestamp}.json"
        with open(output_path, 'w') as f:
            json.dump(asdict(results), f, indent=2, cls=NumpyEncoder)
        print(f"\nResults saved: {output_path}")

        # Summary for paper
        summary = {
            'model': results.model,
            'layer': results.layer,
            'timestamp': timestamp,
            'n_heads': results.n_heads,
            'head_dim': results.head_dim,
            'top_5_contributing_heads': [
                {
                    'head_id': h['head_id'],
                    'magnitude': round(h['contribution_magnitude'], 4),
                    'fraction': round(h['contribution_fraction'], 3),
                    'direction': h['direction']
                }
                for h in results.head_contributions[:5]
            ],
            'n_causal_heads': results.n_causal_heads,
            'component_attribution': results.component_attribution
        }

        summary_path = self.output_dir / f"head_summary_{self.model_name}_L{layer}_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        print(f"Summary saved: {summary_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Phase 4: Head and Component Analysis'
    )
    parser.add_argument('--model', type=str, default='llama',
                       choices=['llama', 'gemma', 'gemma_base'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--layer', type=int, default=None,
                       help='Target layer for single-layer analysis')
    parser.add_argument('--vectors', type=str, default=None,
                       help='Path to steering vectors from Phase 1')
    parser.add_argument('--validate', action='store_true',
                       help='Run causal validation on heads (requires model loading)')
    parser.add_argument('--components', action='store_true',
                       help='Run component attribution analysis (attention vs MLP)')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of top heads to validate')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')

    # New options for comprehensive analysis
    parser.add_argument('--decompose-all', action='store_true',
                       help='Run head decomposition for ALL layers')
    parser.add_argument('--validate-layers', type=str, default=None,
                       help='Comma-separated list of layers to validate (e.g., "23,9,17")')
    parser.add_argument('--cumulative', action='store_true',
                       help='Run cumulative head steering test')
    parser.add_argument('--phase2-results', type=str, default=None,
                       help='Path to Phase 2 results for automatic layer selection')

    args = parser.parse_args()

    phase4 = Phase4HeadComponentAnalysis(
        model_name=args.model,
        gpu_id=args.gpu,
        config_path=args.config
    )

    # Parse validate-layers if provided
    validate_layers = None
    if args.validate_layers:
        validate_layers = [int(l.strip()) for l in args.validate_layers.split(',')]

    phase4.run(
        vectors_path=args.vectors,
        layer=args.layer,
        validate=args.validate,
        analyze_components=args.components,
        top_k_validate=args.top_k,
        decompose_all=args.decompose_all,
        validate_layers=validate_layers,
        cumulative=args.cumulative,
        phase2_results_path=args.phase2_results
    )


if __name__ == '__main__':
    main()
