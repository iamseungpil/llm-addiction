#!/usr/bin/env python3
"""Centralized configuration for V3 SAE analysis (Azure VM)."""
from pathlib import Path

DATA_ROOT = Path("/home/v-seungplee/data/llm-addiction")
SAE_V3_ROOT = DATA_ROOT / "sae_features_v3"
REPO_ROOT = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis")
RESULTS_DIR = REPO_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
JSON_DIR = RESULTS_DIR / "json"
LOG_DIR = RESULTS_DIR / "logs"

PARADIGMS = {
    "ic": {
        "name": "Investment Choice V2role",
        "short": "IC",
        "sae_dir": SAE_V3_ROOT / "investment_choice" / "gemma",
        "behavioral": DATA_ROOT / "behavioral" / "investment_choice" / "v2_role_gemma",
        "n_games": 1600,
        "model": "gemma",
    },
    "sm": {
        "name": "Slot Machine V4role",
        "short": "SM",
        "sae_dir": SAE_V3_ROOT / "slot_machine" / "gemma",
        "behavioral": DATA_ROOT / "behavioral" / "slot_machine" / "gemma_v4_role",
        "n_games": 3200,
        "model": "gemma",
    },
    "mw": {
        "name": "Mystery Wheel V2role",
        "short": "MW",
        "sae_dir": SAE_V3_ROOT / "mystery_wheel" / "gemma",
        "behavioral": DATA_ROOT / "behavioral" / "mystery_wheel" / "gemma_v2_role",
        "n_games": 3200,
        "model": "gemma",
    },
}

# Llama paradigms (IC ready, SM/MW pending extraction)
LLAMA_PARADIGMS = {
    "ic": {
        "name": "Investment Choice V2role (LLaMA)",
        "short": "IC",
        "sae_dir": SAE_V3_ROOT / "investment_choice" / "llama",
        "behavioral": DATA_ROOT / "behavioral" / "investment_choice" / "v2_role_llama",
        "n_games": 1600,
        "model": "llama",
    },
}

N_LAYERS_GEMMA = 42
N_LAYERS_LLAMA = 32
N_SAE_FEATURES_GEMMA = 131072
N_SAE_FEATURES_LLAMA = 32768
HIDDEN_DIM_GEMMA = 3584
HIDDEN_DIM_LLAMA = 4096

MIN_ACTIVATION_RATE = 0.01
CLASSIFICATION_CV_FOLDS = 5
CLASSIFICATION_C = 1.0
N_PERMUTATIONS = 100  # Reduced for speed; increase for final
FDR_ALPHA = 0.05
RANDOM_SEED = 42

PARADIGM_COLORS = {"ic": "#2ecc71", "sm": "#e74c3c", "mw": "#3498db"}
PARADIGM_LABELS = {"ic": "Investment Choice", "sm": "Slot Machine", "mw": "Mystery Wheel"}
BET_COLORS = {"variable": "#e74c3c", "fixed": "#3498db"}
