"""
Configuration loading utilities for Metacognitive Experiment.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load experiment configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default.

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "experiment_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_output_dir(config: Dict[str, Any]) -> Path:
    """Get output directory from config."""
    output_dir = Path(config.get('data', {}).get('output_dir', './output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_experiment_output_dir(config: Dict[str, Any], experiment: str) -> Path:
    """
    Get output directory for a specific experiment.

    Args:
        config: Configuration dictionary
        experiment: Experiment name ('a', 'b', 'c')

    Returns:
        Path to experiment output directory
    """
    base_dir = get_output_dir(config)
    exp_dir = base_dir / f"experiment_{experiment}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir
