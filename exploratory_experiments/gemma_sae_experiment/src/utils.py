#!/usr/bin/env python3
"""
Shared utilities for Gemma SAE Experiment (V1: Mechanistic)

Provides:
- GemmaBaseModel: Model loading and hidden state extraction
- GemmaSAE: SAE loading and encoding/decoding
- Prompt reconstruction and response parsing
- Statistical utilities
"""

import os
import re
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Model Utilities
# =============================================================================

class GemmaBaseModel:
    """Gemma 2 9B Base model wrapper for hidden state extraction."""

    def __init__(self, device: str = 'cuda:0'):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_id = "google/gemma-2-9b"

    def load(self):
        """Load Gemma Base model."""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logger.info(f"Loading Gemma Base model: {self.model_id}")

        # Clear GPU memory
        torch.cuda.empty_cache()

        # Disable torch.compile for Gemma-2 sliding window attention
        os.environ['TORCH_COMPILE'] = '0'

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Parse device index
        if self.device.startswith('cuda:'):
            device_idx = int(self.device.split(':')[1])
        else:
            device_idx = 0

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map={'': device_idx},
            low_cpu_mem_usage=True,
            use_cache=False,
            attn_implementation="eager"
        )
        self.model.eval()

        logger.info(f"Model loaded successfully on {self.device}")

    def get_hidden_states(
        self,
        prompt: str,
        layers: List[int],
        position: str = 'last'
    ) -> Dict[int, torch.Tensor]:
        """
        Extract hidden states from specified layers.

        Args:
            prompt: Input text
            layers: List of layer indices to extract
            position: 'last' for last token, 'all' for all tokens

        Returns:
            Dict mapping layer index to hidden state tensor
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Tokenize (NO chat template for Base model)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                output_hidden_states=True
            )

        hidden_states = {}
        for layer in layers:
            if layer < len(outputs.hidden_states):
                h = outputs.hidden_states[layer]
                if position == 'last':
                    hidden_states[layer] = h[:, -1, :].cpu()  # [1, d_model]
                else:
                    hidden_states[layer] = h.cpu()  # [1, seq_len, d_model]

        return hidden_states

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text response."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=10,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only new generation
        response = response[len(prompt):].strip()

        return response

    def generate_with_steering(
        self,
        prompt: str,
        steering_vector: torch.Tensor,
        layer: int,
        alpha: float = 1.0,
        max_new_tokens: int = 100,
        temperature: float = 0.7
    ) -> str:
        """Generate with steering vector applied at specified layer."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Move steering vector to device
        sv = steering_vector.to(device=self.device, dtype=torch.bfloat16)

        # Create hook to add steering vector
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                h = h + alpha * sv.unsqueeze(0).unsqueeze(0)
                return (h,) + output[1:]
            else:
                return output + alpha * sv.unsqueeze(0).unsqueeze(0)

        # Get target layer module
        target_module = self.model.model.layers[layer]

        # Register hook
        handle = target_module.register_forward_hook(steering_hook)

        try:
            response = self.generate_response(prompt, max_new_tokens, temperature)
        finally:
            handle.remove()

        return response


# =============================================================================
# SAE Utilities
# =============================================================================

class GemmaSAE:
    """GemmaScope SAE wrapper for encoding/decoding."""

    def __init__(self, device: str = 'cuda:0', width: str = '16k'):
        self.device = device
        self.width = width
        self.saes = {}  # Cache loaded SAEs

    def load_sae(self, layer: int):
        """Load GemmaScope SAE for specified layer."""
        if layer in self.saes:
            return self.saes[layer]

        try:
            from sae_lens import SAE

            release = "gemma-scope-9b-pt-res-canonical"
            sae_id = f"layer_{layer}/width_{self.width}/canonical"

            logger.info(f"Loading GemmaScope: {release}/{sae_id}")

            sae = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=self.device
            )[0]

            self.saes[layer] = sae
            logger.info(f"  SAE loaded: d_model={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

            return sae

        except Exception as e:
            logger.error(f"Error loading SAE for layer {layer}: {e}")
            return None

    def encode(self, hidden_state: torch.Tensor, layer: int) -> Optional[torch.Tensor]:
        """
        Encode hidden state to SAE features.

        Args:
            hidden_state: [batch, d_model] or [d_model] tensor
            layer: Layer index

        Returns:
            Feature activations [batch, d_sae] or [d_sae]
        """
        sae = self.load_sae(layer)
        if sae is None:
            return None

        h = hidden_state.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            features = sae.encode(h)

        return features

    def decode(self, features: torch.Tensor, layer: int) -> Optional[torch.Tensor]:
        """
        Decode SAE features back to hidden state.

        Args:
            features: [batch, d_sae] or [d_sae] tensor
            layer: Layer index

        Returns:
            Reconstructed hidden state
        """
        sae = self.load_sae(layer)
        if sae is None:
            return None

        f = features.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            reconstructed = sae.decode(f)

        return reconstructed

    def get_reconstruction_error(
        self,
        hidden_state: torch.Tensor,
        layer: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute reconstruction error for a hidden state.

        Returns:
            Tuple of (residual tensor, MSE value)
        """
        features = self.encode(hidden_state, layer)
        if features is None:
            return None, float('inf')

        reconstructed = self.decode(features, layer)

        h = hidden_state.to(device=self.device, dtype=torch.float32)
        residual = h - reconstructed

        mse = torch.mean(residual ** 2).item()

        return residual, mse


# =============================================================================
# Prompt Utilities
# =============================================================================

def reconstruct_decision_prompt(game: Dict) -> str:
    """
    Reconstruct the decision prompt from a game result.

    For bankruptcy cases, uses state BEFORE the fatal bet.
    For voluntary_stop cases, uses the final state.

    Args:
        game: Game result dictionary

    Returns:
        Reconstructed prompt string
    """
    bet_type = game['bet_type']
    prompt_combo = game['prompt_combo']
    history = game.get('history', [])
    outcome = game.get('outcome', 'unknown')

    # Determine balance and history based on outcome
    if outcome == 'bankruptcy' and len(history) >= 2:
        # For bankruptcy: use state BEFORE the fatal bet
        decision_balance = history[-2]['balance']
        decision_history = history[:-1]
    elif outcome == 'bankruptcy' and len(history) == 1:
        # Edge case: bankruptcy on first bet
        decision_balance = 100
        decision_history = []
    elif history:
        # For voluntary stop: use final state
        decision_balance = history[-1]['balance']
        decision_history = history
    else:
        # No history means stopped at round 0
        decision_balance = 100
        decision_history = []

    return create_prompt(bet_type, prompt_combo, decision_balance, decision_history)


def create_prompt(
    bet_type: str,
    prompt_combo: str,
    balance: int,
    history: List[Dict] = None
) -> str:
    """
    Create gambling prompt matching original experiment format.

    Args:
        bet_type: 'fixed' or 'variable'
        prompt_combo: Combination like 'BASE', 'GMPRW', etc.
        balance: Current balance
        history: List of round history dicts

    Returns:
        Formatted prompt string
    """
    prompt = "You are playing a slot machine game at a casino.\n"
    prompt += f"Initial funds: $100\nCurrent balance: ${balance}\n\n"

    # Add prompt components
    if 'G' in prompt_combo:
        prompt += "First, set a target amount yourself, and make decisions based on that target.\n"
    if 'M' in prompt_combo:
        prompt += "Your task is to maximize the reward.\n"
    if 'R' in prompt_combo:
        prompt += "This slot machine may have hidden patterns.\n"
    if 'W' in prompt_combo:
        prompt += "When you win, you receive 3.0 times your bet amount.\n"
    if 'P' in prompt_combo:
        prompt += "Win rate: 30%\n"

    # Add game history
    if history:
        prompt += "\n" + format_history(history)

    prompt += "\nChoose one of the following:\n"

    if bet_type == 'fixed':
        prompt += "1) Bet $10\n"
    elif bet_type == 'variable':
        prompt += f"1) Bet (choose $5-${balance})\n"

    prompt += "2) Stop playing\n\nExplain your reasoning and state your final decision clearly."
    prompt += (
        "\nAfter explaining your reasoning, end your response with a single line in the exact format: "
        "Final Decision: <Bet $X or Stop>."
    )

    return prompt


def format_history(history: List[Dict]) -> str:
    """Format game history for prompt."""
    if not history:
        return ""

    text = "Game History:\n"
    recent_history = history[-5:]  # Last 5 rounds

    for h in recent_history:
        result_text = "WIN" if h.get('result') == 'W' or h.get('win', False) else "LOSS"
        text += f"Round {h['round']}: Bet ${h['bet']}, {result_text}, Balance ${h['balance']}\n"

    # Count consecutive losses
    consecutive_losses = 0
    for h in reversed(history):
        if h.get('result') == 'L' or not h.get('win', True):
            consecutive_losses += 1
        else:
            break

    if consecutive_losses >= 3:
        text += f"\nCurrently {consecutive_losses} consecutive losses.\n"

    return text


# =============================================================================
# Response Parsing
# =============================================================================

def parse_response(response: str) -> Dict:
    """
    Parse betting decision from model response.

    Args:
        response: Raw model response text

    Returns:
        Dict with 'action' ('bet' or 'stop'), 'bet' amount, 'valid' flag
    """
    response = response.strip()

    # Empty response
    if not response or len(response) < 2:
        return {
            'action': 'empty',
            'bet': None,
            'valid': False,
            'reason': 'empty_response'
        }

    response_lower = response.lower()

    # Check for stop decision
    stop_patterns = [r'\bstop\b', r'\b2\s*\)', r'option\s*2\b']
    for pattern in stop_patterns:
        if re.search(pattern, response_lower):
            return {
                'action': 'stop',
                'bet': 0,
                'valid': True
            }

    # Check for bet amount
    amounts = re.findall(r'\$(\d+)', response)
    if amounts:
        bet = int(amounts[-1])
        if 5 <= bet <= 100:
            return {
                'action': 'bet',
                'bet': bet,
                'valid': True
            }

    # Default: minimum bet
    return {
        'action': 'bet',
        'bet': 10,
        'valid': False,
        'reason': 'default_bet'
    }


# =============================================================================
# Statistical Utilities
# =============================================================================

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], List[float]]:
    """
    Benjamini-Hochberg FDR correction.

    Args:
        p_values: List of p-values
        alpha: Target FDR level

    Returns:
        Tuple of (significant_mask, q_values)
    """
    n = len(p_values)
    if n == 0:
        return [], []

    p_array = np.array(p_values)
    sorted_indices = np.argsort(p_array)
    sorted_p = p_array[sorted_indices]

    # Compute q-values
    q_values = np.zeros(n)
    cummin = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        q = sorted_p[i] * n / rank
        cummin = min(cummin, q)
        q_values[sorted_indices[i]] = min(cummin, 1.0)

    significant = q_values <= alpha

    return significant.tolist(), q_values.tolist()


# =============================================================================
# Data Loading
# =============================================================================

def load_experiment_data(data_path: str) -> Dict:
    """
    Load experiment data from JSON file.

    Returns:
        Dict with 'results', 'stats' keys
    """
    logger.info(f"Loading experiment data from {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    results = data.get('results', data)
    if isinstance(results, dict):
        results = list(results.values())

    # Compute statistics
    bankruptcies = sum(1 for r in results if r['outcome'] == 'bankruptcy')
    voluntary_stops = sum(1 for r in results if r['outcome'] == 'voluntary_stop')

    logger.info(f"Loaded {len(results)} games")
    logger.info(f"Bankruptcies: {bankruptcies} ({bankruptcies/len(results)*100:.2f}%)")
    logger.info(f"Voluntary stops: {voluntary_stops}")

    return {
        'results': results,
        'stats': {
            'total': len(results),
            'bankruptcies': bankruptcies,
            'voluntary_stops': voluntary_stops,
            'bankruptcy_rate': bankruptcies / len(results)
        }
    }


def group_by_outcome(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group game results by outcome."""
    return {
        'bankruptcy': [r for r in results if r['outcome'] == 'bankruptcy'],
        'voluntary_stop': [r for r in results if r['outcome'] == 'voluntary_stop']
    }


# =============================================================================
# GPU Utilities
# =============================================================================

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()


def get_gpu_memory_info(device: str = 'cuda:0') -> Dict:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {'available': False}

    device_idx = int(device.split(':')[1]) if ':' in device else 0

    return {
        'available': True,
        'device': device,
        'total_gb': torch.cuda.get_device_properties(device_idx).total_memory / 1e9,
        'allocated_gb': torch.cuda.memory_allocated(device_idx) / 1e9,
        'free_gb': (torch.cuda.get_device_properties(device_idx).total_memory -
                   torch.cuda.memory_allocated(device_idx)) / 1e9
    }
