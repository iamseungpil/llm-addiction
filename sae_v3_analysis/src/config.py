#!/usr/bin/env python3
"""
Centralized configuration for V3 SAE analysis.

All paths, constants, and paradigm definitions in one place.
"""

from pathlib import Path

# ===========================================================================
# Paths
# ===========================================================================
DATA_ROOT = Path("/home/jovyan/beomi/llm-addiction-data")
SAE_V3_ROOT = DATA_ROOT / "sae_features_v3"
REPO_ROOT = Path("/home/jovyan/llm-addiction/sae_v3_analysis")
RESULTS_DIR = REPO_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
JSON_DIR = RESULTS_DIR / "json"
LOG_DIR = RESULTS_DIR / "logs"

# ===========================================================================
# Paradigm configs — matches extract_all_rounds.py PARADIGM_CONFIG
# ===========================================================================
PARADIGMS = {
    "ic": {
        "name": "Investment Choice V2role",
        "short": "IC",
        "sae_dir": SAE_V3_ROOT / "investment_choice" / "gemma",
        "n_games": 1600,
        "n_bk": 172,
        "model": "gemma",
    },
    "sm": {
        "name": "Slot Machine V4role",
        "short": "SM",
        "sae_dir": SAE_V3_ROOT / "slot_machine" / "gemma",
        "n_games": 3200,
        "n_bk": 87,
        "model": "gemma",
    },
    "mw": {
        "name": "Mystery Wheel V2role",
        "short": "MW",
        "sae_dir": SAE_V3_ROOT / "mystery_wheel" / "gemma",
        "n_games": 3200,
        "n_bk": 54,
        "model": "gemma",
    },
}

# ===========================================================================
# Model constants
# ===========================================================================
N_LAYERS = 42
N_SAE_FEATURES = 131072
HIDDEN_DIM = 3584

# ===========================================================================
# Analysis parameters
# ===========================================================================
MIN_ACTIVATION_RATE = 0.01       # Filter features active in <1% of samples
CLASSIFICATION_CV_FOLDS = 5
CLASSIFICATION_C = 1.0
N_PERMUTATIONS = 1000
FDR_ALPHA = 0.05
RANDOM_SEED = 42

# ===========================================================================
# Visualization
# ===========================================================================
PARADIGM_COLORS = {"ic": "#2ecc71", "sm": "#e74c3c", "mw": "#3498db"}
PARADIGM_LABELS = {"ic": "Investment Choice", "sm": "Slot Machine", "mw": "Mystery Wheel"}
BET_COLORS = {"variable": "#e74c3c", "fixed": "#3498db"}
