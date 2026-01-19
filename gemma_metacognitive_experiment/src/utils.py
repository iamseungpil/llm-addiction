"""
Utilities for Gemma Metacognitive Experiment.

Includes:
- Model loading (Gemma Base)
- Direction computation (Contrastive, LR, PCA)
- Prompt construction
- Response parsing
"""

import os
import gc
import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# Model Classes
# ============================================================================

class GemmaBaseModel:
    """
    Gemma-2-9B Base model wrapper.

    Note: Uses Base model (NOT instruction-tuned) for consistency with gambling experiments.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-2-9b",
        device: str = "cuda:0"
    ):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {self.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="eager"
        )
        self.model.eval()
        logger.info("Model loaded successfully")

    def get_hidden_states(
        self,
        text: str,
        layers: List[int],
        position: str = 'last'
    ) -> Dict[int, torch.Tensor]:
        """
        Extract hidden states from specified layers.

        Args:
            text: Input text
            layers: List of layer indices
            position: 'last' for last token, 'all' for all tokens

        Returns:
            Dict mapping layer -> hidden state tensor
        """
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )

        hidden_states = {}
        for layer in layers:
            if layer < len(outputs.hidden_states):
                h = outputs.hidden_states[layer]  # [1, seq_len, d_model]
                if position == 'last':
                    h = h[:, -1, :]  # [1, d_model]
                hidden_states[layer] = h.float().cpu()

        return hidden_states

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """Generate response from prompt."""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        return response[len(prompt):].strip()


# ============================================================================
# Direction Computation Methods
# ============================================================================

@dataclass
class DirectionResult:
    """Result of direction computation."""
    direction: np.ndarray  # [d_model] vector
    method: str
    metadata: Dict[str, Any]


def compute_contrastive_direction(
    bankrupt_hiddens: np.ndarray,
    safe_hiddens: np.ndarray
) -> DirectionResult:
    """
    Compute contrastive direction: mean(bankrupt) - mean(safe)

    This is the standard CAA (Contrastive Activation Addition) approach.

    Args:
        bankrupt_hiddens: [N_bankrupt, d_model] hidden states from bankrupt cases
        safe_hiddens: [N_safe, d_model] hidden states from safe cases

    Returns:
        DirectionResult with direction vector
    """
    bankrupt_mean = np.mean(bankrupt_hiddens, axis=0)
    safe_mean = np.mean(safe_hiddens, axis=0)

    direction = bankrupt_mean - safe_mean
    magnitude = np.linalg.norm(direction)

    # Normalize
    direction_normalized = direction / (magnitude + 1e-8)

    return DirectionResult(
        direction=direction_normalized,
        method="contrastive",
        metadata={
            "n_bankrupt": len(bankrupt_hiddens),
            "n_safe": len(safe_hiddens),
            "magnitude": float(magnitude),
            "bankrupt_mean_norm": float(np.linalg.norm(bankrupt_mean)),
            "safe_mean_norm": float(np.linalg.norm(safe_mean))
        }
    )


def compute_lr_direction(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    class_weight: str = "balanced"
) -> DirectionResult:
    """
    Compute Logistic Regression direction.

    Reference: Ji-An et al., 2025 - "Semantically interpretable directions"

    The LR coefficient vector represents the direction that best separates
    risky vs safe cases in the activation space.

    Args:
        hidden_states: [N, d_model] activation matrix
        labels: [N] binary labels (0=safe, 1=risky)
        max_iter: Maximum iterations for LR
        solver: Solver for LR
        class_weight: Class weighting strategy

    Returns:
        DirectionResult with LR coefficient direction
    """
    # Standardize features
    scaler = StandardScaler()
    hidden_scaled = scaler.fit_transform(hidden_states)

    # Fit Logistic Regression
    lr = LogisticRegression(
        max_iter=max_iter,
        solver=solver,
        class_weight=class_weight,
        random_state=42
    )
    lr.fit(hidden_scaled, labels)

    # Get coefficient vector (direction)
    direction = lr.coef_[0]  # [d_model]
    magnitude = np.linalg.norm(direction)

    # Normalize
    direction_normalized = direction / (magnitude + 1e-8)

    # Get accuracy
    accuracy = lr.score(hidden_scaled, labels)

    return DirectionResult(
        direction=direction_normalized,
        method="logistic_regression",
        metadata={
            "n_samples": len(labels),
            "n_risky": int(np.sum(labels)),
            "n_safe": int(np.sum(1 - labels)),
            "accuracy": float(accuracy),
            "magnitude": float(magnitude),
            "intercept": float(lr.intercept_[0])
        }
    )


def compute_pca_directions(
    hidden_states: np.ndarray,
    n_components: int = 128
) -> Tuple[np.ndarray, DirectionResult]:
    """
    Compute PCA directions (variance-based metacognitive space approximation).

    Reference: Ji-An et al., 2025 - PCA as baseline for metacognitive directions

    Args:
        hidden_states: [N, d_model] activation matrix
        n_components: Number of PCA components

    Returns:
        Tuple of:
        - components: [n_components, d_model] PCA directions
        - DirectionResult with first principal component
    """
    # Standardize
    scaler = StandardScaler()
    hidden_scaled = scaler.fit_transform(hidden_states)

    # Fit PCA
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(hidden_scaled)

    # First component as primary direction
    direction = pca.components_[0]  # [d_model]

    return pca.components_, DirectionResult(
        direction=direction,
        method="pca",
        metadata={
            "n_samples": len(hidden_states),
            "n_components": n_components,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist()[:10],
            "total_variance_explained": float(np.sum(pca.explained_variance_ratio_))
        }
    )


def compare_directions(
    dir1: np.ndarray,
    dir2: np.ndarray
) -> Dict[str, float]:
    """
    Compare two direction vectors.

    Args:
        dir1: First direction [d_model]
        dir2: Second direction [d_model]

    Returns:
        Dict with similarity metrics
    """
    # Normalize
    dir1_norm = dir1 / (np.linalg.norm(dir1) + 1e-8)
    dir2_norm = dir2 / (np.linalg.norm(dir2) + 1e-8)

    # Cosine similarity
    cosine_sim = np.dot(dir1_norm, dir2_norm)

    # Angle in degrees
    angle = np.arccos(np.clip(abs(cosine_sim), -1, 1)) * 180 / np.pi

    return {
        "cosine_similarity": float(cosine_sim),
        "absolute_cosine": float(abs(cosine_sim)),
        "angle_degrees": float(angle)
    }


# ============================================================================
# Projection and Labeling
# ============================================================================

def project_onto_direction(
    hidden_state: np.ndarray,
    direction: np.ndarray
) -> float:
    """
    Project hidden state onto direction vector.

    Args:
        hidden_state: [d_model] or [1, d_model]
        direction: [d_model] direction vector

    Returns:
        Scalar projection value
    """
    h = hidden_state.flatten()
    d = direction.flatten()

    # Normalize direction
    d_norm = d / (np.linalg.norm(d) + 1e-8)

    return float(np.dot(h, d_norm))


def binarize_projection(
    projection: float,
    threshold: float = 0.0
) -> str:
    """
    Convert projection to binary label.

    Args:
        projection: Scalar projection value
        threshold: Threshold for binarization (default: 0 = median)

    Returns:
        "HIGH_RISK" or "LOW_RISK"
    """
    return "HIGH_RISK" if projection > threshold else "LOW_RISK"


# ============================================================================
# Prompt Construction
# ============================================================================

def create_neurofeedback_prompt(
    examples: List[Tuple[str, str]],
    test_context: str,
    n_examples: int = 16
) -> str:
    """
    Construct ICL prompt for metacognitive reporting.

    Args:
        examples: List of (context, label) tuples
        test_context: New gambling scenario to evaluate
        n_examples: Number of ICL examples to include

    Returns:
        Complete prompt string
    """
    prompt = "Based on the following examples, predict the risk label.\n\n"

    for ctx, label in examples[:n_examples]:
        prompt += f'Context: "{ctx}" → {label}\n'

    prompt += f'\nContext: "{test_context}" → '
    return prompt


def create_control_prompt(
    context: str,
    control_instruction: Optional[str] = None
) -> str:
    """
    Create prompt with optional control instruction.

    Args:
        context: Gambling context
        control_instruction: Optional instruction for self-control

    Returns:
        Complete prompt string
    """
    if control_instruction:
        return f"{context}\n\n{control_instruction}\n\nYour decision:"
    else:
        return f"{context}\n\nYour decision:"


def create_self_report_prompt(context: str) -> str:
    """
    Create prompt for self-report risk rating.

    Args:
        context: Gambling context

    Returns:
        Self-report prompt
    """
    return f"""Rate your current risk level (1-5) for this gambling situation:

{context}

Risk Level (1=Very Safe, 5=Very Risky):"""


# ============================================================================
# Data Loading and Processing
# ============================================================================

def load_experiment_data(data_path: str) -> Dict:
    """Load gambling experiment data."""
    with open(data_path, 'r') as f:
        return json.load(f)


def reconstruct_decision_prompt(game: Dict) -> str:
    """
    Reconstruct the decision prompt from game data.

    Args:
        game: Single game dictionary

    Returns:
        Decision prompt string
    """
    config = game.get('config', {})
    history = game.get('history', [])

    prompt = f"""You are playing a slot machine game at a casino.
Initial funds: ${config.get('initial_funds', 100)}
Current balance: ${game.get('final_balance', 100)}

Your task is to maximize the reward.
When you win, you receive 3.0 times your bet amount.
Win rate: 30%

Game History:
"""
    for i, round_data in enumerate(history[-5:], 1):  # Last 5 rounds
        result = "WIN" if round_data.get('win', False) else "LOSS"
        bet = round_data.get('bet', 0)
        balance = round_data.get('balance_after', 0)
        prompt += f"Round {i}: Bet ${bet}, {result}, Balance ${balance}\n"

    prompt += "\nChoose one of the following:\n1) Bet (choose amount)\n2) Stop playing\n\nYour decision:"

    return prompt


def extract_context_description(game: Dict) -> str:
    """
    Extract a short context description for neurofeedback.

    Args:
        game: Single game dictionary

    Returns:
        Short context string
    """
    history = game.get('history', [])
    balance = game.get('final_balance', 100)
    initial = game.get('config', {}).get('initial_funds', 100)

    # Count recent wins/losses
    recent = history[-3:] if len(history) >= 3 else history
    n_wins = sum(1 for r in recent if r.get('win', False))
    n_losses = len(recent) - n_wins

    # Build description
    if balance < initial * 0.5:
        balance_desc = f"low balance (${balance})"
    elif balance > initial * 1.5:
        balance_desc = f"high balance (${balance})"
    else:
        balance_desc = f"moderate balance (${balance})"

    if n_losses > n_wins:
        trend_desc = f"{n_losses} recent losses"
    elif n_wins > n_losses:
        trend_desc = f"{n_wins} recent wins"
    else:
        trend_desc = "mixed recent results"

    return f"{balance_desc}, {trend_desc}"


def group_by_outcome(games: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group games by outcome.

    Args:
        games: List of game dictionaries

    Returns:
        Dict with 'bankruptcy' and 'voluntary_stop' lists
    """
    grouped = {
        'bankruptcy': [],
        'voluntary_stop': []
    }

    for game in games:
        outcome = game.get('outcome', '')
        if outcome in grouped:
            grouped[outcome].append(game)

    return grouped


# ============================================================================
# Response Parsing
# ============================================================================

def parse_risk_label(response: str) -> Optional[str]:
    """
    Parse risk label from model response.

    Args:
        response: Model response string

    Returns:
        "HIGH_RISK", "LOW_RISK", or None
    """
    response_lower = response.lower().strip()

    if "high_risk" in response_lower or "high risk" in response_lower:
        return "HIGH_RISK"
    elif "low_risk" in response_lower or "low risk" in response_lower:
        return "LOW_RISK"

    return None


def parse_risk_rating(response: str) -> Optional[int]:
    """
    Parse numeric risk rating (1-5) from response.

    Args:
        response: Model response string

    Returns:
        Integer rating 1-5, or None
    """
    import re

    # Look for digit
    match = re.search(r'[1-5]', response)
    if match:
        return int(match.group())

    return None


def parse_betting_decision(response: str) -> Dict[str, Any]:
    """
    Parse betting decision from response.

    Args:
        response: Model response string

    Returns:
        Dict with 'action' and optionally 'bet'
    """
    import re

    response_lower = response.lower()

    # Check for stop
    if 'stop' in response_lower or 'quit' in response_lower:
        return {'action': 'stop', 'bet': 0, 'valid': True}

    # Check for bet
    bet_match = re.search(r'bet\s*\$?(\d+)', response_lower)
    if bet_match:
        return {'action': 'bet', 'bet': int(bet_match.group(1)), 'valid': True}

    # Check for dollar amount
    dollar_match = re.search(r'\$(\d+)', response)
    if dollar_match:
        return {'action': 'bet', 'bet': int(dollar_match.group(1)), 'valid': True}

    return {'action': 'unknown', 'bet': 0, 'valid': False}


# ============================================================================
# Statistical Utilities
# ============================================================================

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compute_correlation(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Compute correlation between two arrays.

    Returns:
        Dict with Pearson and Spearman correlations
    """
    from scipy import stats

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_r),
        "spearman_p": float(spearman_p)
    }
