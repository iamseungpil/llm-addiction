#!/usr/bin/env python3
"""
Configuration loader for Gemma SAE Experiment (V1: Mechanistic)
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_config_path() -> Path:
    """Get default config file path."""
    return get_project_root() / "configs" / "experiment_config.yaml"


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load experiment configuration from YAML file.

    Args:
        config_path: Path to config file, or None for default

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = str(get_config_path())

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "gemma_base"
    model_id: str = "google/gemma-2-9b"
    d_model: int = 3584
    n_layers: int = 42
    use_chat_template: bool = False


@dataclass
class SAEConfig:
    """SAE configuration."""
    release: str = "gemma-scope-9b-pt-res-canonical"
    width: str = "16k"
    d_sae: int = 16384


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)

    # Data paths
    experiment_data: str = "/data/llm_addiction/experiment_0_gemma_corrected/final_gemma_20251004_172426.json"
    output_dir: str = "/data/llm_addiction/gemma_sae_experiment"

    # Target layers
    target_layers: List[int] = field(default_factory=lambda: list(range(20, 42)))

    # Phase settings
    phase1_checkpoint_frequency: int = 100
    phase2_fdr_alpha: float = 0.05
    phase2_min_cohens_d: float = 0.3
    phase3_max_samples: int = 500
    phase4_top_k: int = 50
    phase5_n_trials: int = 50
    phase5_alpha_values: List[float] = field(default_factory=lambda: [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

    # Technical
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        config_dict = load_config(yaml_path)

        model_config = ModelConfig(**config_dict.get('model', {}))
        sae_config = SAEConfig(**config_dict.get('sae', {}))

        return cls(
            model=model_config,
            sae=sae_config,
            experiment_data=config_dict.get('data', {}).get('experiment_data', cls.experiment_data),
            output_dir=config_dict.get('data', {}).get('output_dir', cls.output_dir),
            target_layers=config_dict.get('target_layers', list(range(20, 42))),
            phase1_checkpoint_frequency=config_dict.get('phase1', {}).get('checkpoint_frequency', 100),
            phase2_fdr_alpha=config_dict.get('phase2', {}).get('fdr_alpha', 0.05),
            phase2_min_cohens_d=config_dict.get('phase2', {}).get('min_cohens_d', 0.3),
            phase3_max_samples=config_dict.get('phase3', {}).get('max_samples_per_group', 500),
            phase4_top_k=config_dict.get('phase4', {}).get('top_k_features', 50),
            phase5_n_trials=config_dict.get('phase5', {}).get('n_trials', 50),
            phase5_alpha_values=config_dict.get('phase5', {}).get('alpha_values', [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]),
            random_seed=config_dict.get('random_seed', 42)
        )


def get_output_dir(config: Dict) -> Path:
    """Get output directory from config, creating if needed."""
    output_dir = Path(config.get('data', {}).get('output_dir', '/data/llm_addiction/gemma_sae_experiment'))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_phase_output_dir(config: Dict, phase: int) -> Path:
    """Get output directory for a specific phase."""
    base_dir = get_output_dir(config)
    phase_dir = base_dir / f"phase{phase}_results"
    phase_dir.mkdir(parents=True, exist_ok=True)
    return phase_dir
