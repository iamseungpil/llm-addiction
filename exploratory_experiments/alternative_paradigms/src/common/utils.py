"""
Utility functions for alternative paradigms experiments
"""

import os
import json
import logging
import random
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with consistent formatting.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def save_json(data: Dict[str, Any], filepath: str):
    """
    Save data to JSON file.

    Args:
        data: Data to save
        filepath: Output filepath
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file.

    Args:
        filepath: Input filepath

    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_timestamp() -> str:
    """
    Get current timestamp string.

    Returns:
        Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_bet_amount(response: str, min_bet: int = 5, max_bet: int = 100) -> Optional[int]:
    """
    Parse bet amount from model response.

    Tries multiple patterns:
    - "$XX" or "$ XX"
    - "bet XX" or "bet $XX"
    - Just numbers

    Args:
        response: Model response
        min_bet: Minimum bet allowed
        max_bet: Maximum bet allowed

    Returns:
        Parsed bet amount or None if parsing fails
    """
    import re

    response = response.lower().strip()

    # Pattern 1: "$XX" or "$ XX"
    match = re.search(r'\$\s*(\d+)', response)
    if match:
        amount = int(match.group(1))
        return max(min_bet, min(amount, max_bet))

    # Pattern 2: "bet XX" or "wager XX"
    match = re.search(r'(?:bet|wager)\s+\$?\s*(\d+)', response)
    if match:
        amount = int(match.group(1))
        return max(min_bet, min(amount, max_bet))

    # Pattern 3: Just numbers at start
    match = re.search(r'^(\d+)', response)
    if match:
        amount = int(match.group(1))
        return max(min_bet, min(amount, max_bet))

    # Pattern 4: Any number in response
    numbers = re.findall(r'\d+', response)
    if numbers:
        amount = int(numbers[0])
        return max(min_bet, min(amount, max_bet))

    return None


def parse_choice(response: str, valid_choices: list) -> Optional[str]:
    """
    Parse choice from model response.

    Args:
        response: Model response
        valid_choices: List of valid choices (e.g., ["A", "B", "C", "D"])

    Returns:
        Parsed choice or None if parsing fails
    """
    import re
    response = response.upper().strip()

    # Pattern 1: "Choice: X", "Deck X", "Select X", etc.
    for choice in valid_choices:
        pattern = rf'(?:choice|deck|select|option|choose)[:\s]+{choice}\b'
        if re.search(pattern, response, re.IGNORECASE):
            return choice

    # Pattern 2: Standalone letter with word boundary
    for choice in valid_choices:
        pattern = rf'\b{choice}\b'
        if re.search(pattern, response[:50]):  # Check first 50 chars
            return choice

    return None


def parse_stop_decision(response: str) -> bool:
    """
    Parse STOP decision from model response.

    Args:
        response: Model response

    Returns:
        True if model wants to stop, False otherwise
    """
    response = response.lower().strip()

    stop_keywords = [
        "stop",
        "quit",
        "exit",
        "cash out",
        "i'll stop",
        "i will stop",
        "i want to stop",
        "choose to stop",
        "decide to stop"
    ]

    return any(keyword in response[:50] for keyword in stop_keywords)
