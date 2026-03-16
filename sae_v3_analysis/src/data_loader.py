#!/usr/bin/env python3
"""
V3 SAE data loader.

Loads sparse COO features from sae_features_v3/ and provides:
- Per-game aggregation (mean/max over rounds)
- Decision-point extraction (last round only)
- Round-level access with metadata
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from config import PARADIGMS, N_SAE_FEATURES


def load_sparse_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    """Load a single layer's sparse SAE features + metadata."""
    data = np.load(npz_path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def sparse_to_dense(data: dict) -> np.ndarray:
    """Reconstruct dense array from sparse COO components."""
    shape = tuple(data["shape"])
    dense = np.zeros(shape, dtype=np.float32)
    dense[data["row_indices"], data["col_indices"]] = data["values"]
    return dense


def get_metadata(data: dict) -> Dict[str, np.ndarray]:
    """Extract metadata arrays from loaded NPZ."""
    meta = {
        "game_ids": data["game_ids"],
        "round_nums": data["round_nums"],
        "game_outcomes": data["game_outcomes"],
        "is_last_round": data["is_last_round"].astype(bool),
        "bet_types": data["bet_types"],
    }
    for field in ["bet_constraints", "prompt_conditions", "balances"]:
        if field in data:
            meta[field] = data[field]
    return meta


def load_layer_features(paradigm: str, layer: int,
                        mode: str = "decision_point",
                        dense: bool = True) -> Optional[Tuple[np.ndarray, Dict]]:
    """Load features for a paradigm/layer with specified aggregation.

    Args:
        paradigm: 'ic', 'sm', or 'mw'
        layer: 0-41
        mode: 'decision_point' (last round per game),
              'all_rounds' (every round),
              'game_mean' (mean activation per game),
              'game_max' (max activation per game)
        dense: If True, return dense matrix. If False, return sparse dict.

    Returns:
        (features, metadata) tuple, or None if file doesn't exist
    """
    sae_dir = PARADIGMS[paradigm]["sae_dir"]
    npz_path = sae_dir / f"sae_features_L{layer}.npz"

    if not npz_path.exists():
        return None

    raw = load_sparse_npz(npz_path)
    meta = get_metadata(raw)

    if mode == "all_rounds":
        if dense:
            features = sparse_to_dense(raw)
        else:
            return raw, meta
        return features, meta

    if mode == "decision_point":
        mask = meta["is_last_round"]
        if dense:
            full = sparse_to_dense(raw)
            features = full[mask]
        else:
            # For sparse, filter indices
            row_mask = np.isin(raw["row_indices"], np.where(mask)[0])
            # Remap row indices to compressed range
            old_to_new = np.full(len(mask), -1, dtype=np.int64)
            old_to_new[mask] = np.arange(mask.sum())
            return {
                "row_indices": old_to_new[raw["row_indices"][row_mask]],
                "col_indices": raw["col_indices"][row_mask],
                "values": raw["values"][row_mask],
                "shape": np.array([mask.sum(), raw["shape"][1]]),
            }, {k: v[mask] for k, v in meta.items()}

        filtered_meta = {k: v[mask] for k, v in meta.items()}
        return features, filtered_meta

    if mode in ("game_mean", "game_max"):
        full = sparse_to_dense(raw)
        game_ids = meta["game_ids"]
        unique_games = np.unique(game_ids)
        n_games = len(unique_games)

        agg = np.zeros((n_games, full.shape[1]), dtype=np.float32)
        game_meta = {
            "game_ids": unique_games,
            "game_outcomes": np.empty(n_games, dtype=meta["game_outcomes"].dtype),
            "bet_types": np.empty(n_games, dtype=meta["bet_types"].dtype),
        }

        for i, gid in enumerate(unique_games):
            gmask = game_ids == gid
            if mode == "game_mean":
                agg[i] = full[gmask].mean(axis=0)
            else:
                agg[i] = full[gmask].max(axis=0)
            # Get game-level metadata from any round of this game
            idx = np.where(gmask)[0][0]
            game_meta["game_outcomes"][i] = meta["game_outcomes"][idx]
            game_meta["bet_types"][i] = meta["bet_types"][idx]

        return agg, game_meta

    raise ValueError(f"Unknown mode: {mode}")


def filter_active_features(features: np.ndarray,
                           min_rate: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """Filter to features with activation rate >= min_rate.

    Returns:
        (filtered_features, active_indices)
    """
    rate = (features != 0).mean(axis=0)
    mask = rate >= min_rate
    return features[:, mask], np.where(mask)[0]


def get_labels(meta: Dict[str, np.ndarray]) -> np.ndarray:
    """Binary labels: 1=bankruptcy, 0=voluntary_stop."""
    return (meta["game_outcomes"] == "bankruptcy").astype(np.int32)


def load_hidden_states(paradigm: str, layer: int,
                       mode: str = "decision_point") -> Optional[Tuple[np.ndarray, Dict]]:
    """Load raw hidden states from Phase A checkpoint.

    Args:
        paradigm: 'ic', 'sm', or 'mw'
        layer: 0-41
        mode: 'decision_point', 'all_rounds', or 'game_mean'

    Returns:
        (hidden_states, metadata) tuple, or None if checkpoint doesn't exist
    """
    sae_dir = PARADIGMS[paradigm]["sae_dir"]
    ckpt_path = sae_dir / "checkpoint" / "phase_a_hidden_states.npz"

    if not ckpt_path.exists():
        return None

    # Load checkpoint (contains hidden_states: [n_rounds, n_layers, hidden_dim])
    ckpt = np.load(ckpt_path, allow_pickle=False)
    hidden_all = ckpt["hidden_states"]  # shape: (n_rounds, n_layers, hidden_dim)

    # Get metadata from any SAE feature file (they all share the same metadata)
    any_sae = list(sae_dir.glob("sae_features_L*.npz"))
    if not any_sae:
        return None
    raw = load_sparse_npz(any_sae[0])
    meta = get_metadata(raw)

    # Extract single layer
    features = hidden_all[:, layer, :]  # (n_rounds, hidden_dim)

    # Check for valid_mask
    if "valid_mask" in ckpt:
        valid = ckpt["valid_mask"].astype(bool)
        features = features[valid]
        meta = {k: v[valid] for k, v in meta.items()}

    if mode == "decision_point":
        mask = meta["is_last_round"]
        return features[mask], {k: v[mask] for k, v in meta.items()}

    if mode == "game_mean":
        game_ids = meta["game_ids"]
        unique_games = np.unique(game_ids)
        n_games = len(unique_games)
        agg = np.zeros((n_games, features.shape[1]), dtype=np.float32)
        game_meta = {
            "game_ids": unique_games,
            "game_outcomes": np.empty(n_games, dtype=meta["game_outcomes"].dtype),
            "bet_types": np.empty(n_games, dtype=meta["bet_types"].dtype),
        }
        for i, gid in enumerate(unique_games):
            gmask = game_ids == gid
            agg[i] = features[gmask].mean(axis=0)
            idx = np.where(gmask)[0][0]
            game_meta["game_outcomes"][i] = meta["game_outcomes"][idx]
            game_meta["bet_types"][i] = meta["bet_types"][idx]
        return agg, game_meta

    return features, meta


def check_paradigm_ready(paradigm: str) -> dict:
    """Check if extraction is complete for a paradigm."""
    sae_dir = PARADIGMS[paradigm]["sae_dir"]
    summary_file = sae_dir / "extraction_summary.json"
    n_layers = len(list(sae_dir.glob("sae_features_L*.npz")))

    return {
        "paradigm": paradigm,
        "sae_dir": str(sae_dir),
        "n_layers": n_layers,
        "complete": n_layers == 42,
        "has_summary": summary_file.exists(),
    }
