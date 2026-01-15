"""
Steering Vector Experiment Module

This module provides tools for extracting and applying steering vectors
to control LLM behavior in slot machine experiments.

Phases:
    1. extract_steering_vectors: Extract steering vectors from behavioral data
    2. run_steering_experiment: Apply steering during generation
    3. analyze_steering_with_sae: Interpret steering vectors using SAEs

Example usage:
    # Phase 1: Extract vectors
    python -m steering_vector_experiment.src.extract_steering_vectors --model llama --gpu 0

    # Phase 2: Apply steering
    python -m steering_vector_experiment.src.run_steering_experiment --model llama --gpu 0 --vectors vectors.npz

    # Phase 3: SAE analysis
    python -m steering_vector_experiment.src.analyze_steering_with_sae --model llama --gpu 0 --vectors vectors.npz
"""

from .utils import (
    setup_logging,
    load_experiment_data,
    group_by_outcome,
    PromptBuilder,
    ResponseParser,
    ModelRegistry,
    CheckpointManager,
    load_model_and_tokenizer,
    ExperimentConfig,
    ModelConfig,
)

from .extract_steering_vectors import (
    HiddenStateExtractor,
    SteeringVectorComputer,
    load_steering_vectors,
    save_steering_vectors,
)

__version__ = "1.0.0"
__all__ = [
    # Utilities
    "setup_logging",
    "load_experiment_data",
    "group_by_outcome",
    "PromptBuilder",
    "ResponseParser",
    "ModelRegistry",
    "CheckpointManager",
    "load_model_and_tokenizer",
    "ExperimentConfig",
    "ModelConfig",
    # Steering vector extraction
    "HiddenStateExtractor",
    "SteeringVectorComputer",
    "load_steering_vectors",
    "save_steering_vectors",
]
