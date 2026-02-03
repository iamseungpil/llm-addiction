#!/usr/bin/env python3
"""
Utility functions for SAE Condition Comparison Analysis.
Handles data loading, statistics, and result saving.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy import stats
from statsmodels.stats.multitest import multipletests


@dataclass
class FeatureResult:
    """Result for a single feature comparison"""
    layer: int
    feature_id: int
    t_stat: float
    p_value: float
    cohens_d: float
    group1_mean: float
    group2_mean: float
    group1_std: float
    group2_std: float
    group1_n: int
    group2_n: int
    direction: str  # 'higher_in_variable' or 'higher_in_fixed'

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FourWayResult:
    """Result for 4-way ANOVA comparison"""
    layer: int
    feature_id: int
    f_stat: float
    p_value: float
    eta_squared: float
    group_means: Dict[str, float]
    group_stds: Dict[str, float]
    group_ns: Dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InteractionResult:
    """Result for interaction analysis"""
    layer: int
    feature_id: int
    # Main effects
    bet_type_f: float
    bet_type_p: float
    bet_type_eta: float
    outcome_f: float
    outcome_p: float
    outcome_eta: float
    # Interaction
    interaction_f: float
    interaction_p: float
    interaction_eta: float
    # Group means
    group_means: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


class DataLoader:
    """Load SAE features and map bet_type from original experiment data"""

    def __init__(self, config: dict, model_type: str):
        self.config = config
        self.model_type = model_type
        self.feature_dir = Path(config['data'][model_type]['feature_dir'])
        self.experiment_file = Path(config['data'][model_type]['experiment_file'])
        self._game_metadata = None

    def _load_experiment_metadata(self) -> Dict[str, dict]:
        """Load bet_type and outcome for each game from experiment JSON"""
        if self._game_metadata is not None:
            return self._game_metadata

        with open(self.experiment_file, 'r') as f:
            data = json.load(f)

        self._game_metadata = {}
        results = data.get('results', data)  # Handle both formats

        for i, game in enumerate(results):
            game_id = str(i)
            bet_type = game.get('bet_type', 'unknown')

            # Determine outcome
            if 'outcome' in game:
                outcome = game['outcome']
            elif 'is_bankrupt' in game:
                outcome = 'bankruptcy' if game['is_bankrupt'] else 'voluntary_stop'
            else:
                outcome = 'unknown'

            self._game_metadata[game_id] = {
                'bet_type': bet_type,
                'outcome': outcome,
                'final_balance': game.get('final_balance', 0),
                'total_rounds': game.get('total_rounds', 0)
            }

        return self._game_metadata

    def load_layer_features(self, layer: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load features for a specific layer and map bet_type labels.

        Returns:
            Tuple of (features, outcomes, bet_types) arrays, or None if file not found
        """
        feature_file = self.feature_dir / f'layer_{layer}_features.npz'

        if not feature_file.exists():
            return None

        data = np.load(feature_file, allow_pickle=True)
        features = data['features']
        outcomes = data['outcomes']
        game_ids = data['game_ids']

        # Map bet_type from experiment metadata
        metadata = self._load_experiment_metadata()
        bet_types = np.array([
            metadata.get(str(gid), {}).get('bet_type', 'unknown')
            for gid in game_ids
        ])

        return features, outcomes, bet_types

    def load_layer_features_grouped(self, layer: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Load features grouped by condition.

        Returns dict with keys:
            - 'variable': all variable condition features
            - 'fixed': all fixed condition features
            - 'variable_bankrupt': variable + bankrupt
            - 'variable_safe': variable + voluntary_stop
            - 'fixed_bankrupt': fixed + bankrupt
            - 'fixed_safe': fixed + voluntary_stop
        """
        result = self.load_layer_features(layer)
        if result is None:
            return None

        features, outcomes, bet_types = result

        # Create masks
        variable_mask = bet_types == 'variable'
        fixed_mask = bet_types == 'fixed'
        bankrupt_mask = outcomes == 'bankruptcy'
        safe_mask = outcomes == 'voluntary_stop'

        return {
            'variable': features[variable_mask],
            'fixed': features[fixed_mask],
            'variable_bankrupt': features[variable_mask & bankrupt_mask],
            'variable_safe': features[variable_mask & safe_mask],
            'fixed_bankrupt': features[fixed_mask & bankrupt_mask],
            'fixed_safe': features[fixed_mask & safe_mask],
            # Also return counts
            '_counts': {
                'variable': int(variable_mask.sum()),
                'fixed': int(fixed_mask.sum()),
                'variable_bankrupt': int((variable_mask & bankrupt_mask).sum()),
                'variable_safe': int((variable_mask & safe_mask).sum()),
                'fixed_bankrupt': int((fixed_mask & bankrupt_mask).sum()),
                'fixed_safe': int((fixed_mask & safe_mask).sum()),
            }
        }

    def get_data_summary(self) -> dict:
        """Get summary statistics about the data"""
        metadata = self._load_experiment_metadata()

        summary = {
            'total_games': len(metadata),
            'variable': 0,
            'fixed': 0,
            'variable_bankrupt': 0,
            'variable_safe': 0,
            'fixed_bankrupt': 0,
            'fixed_safe': 0,
        }

        for game in metadata.values():
            bt = game['bet_type']
            outcome = game['outcome']

            if bt == 'variable':
                summary['variable'] += 1
                if outcome == 'bankruptcy':
                    summary['variable_bankrupt'] += 1
                elif outcome == 'voluntary_stop':
                    summary['variable_safe'] += 1
            elif bt == 'fixed':
                summary['fixed'] += 1
                if outcome == 'bankruptcy':
                    summary['fixed_bankrupt'] += 1
                elif outcome == 'voluntary_stop':
                    summary['fixed_safe'] += 1

        # Calculate rates
        if summary['variable'] > 0:
            summary['variable_bankruptcy_rate'] = summary['variable_bankrupt'] / summary['variable']
        if summary['fixed'] > 0:
            summary['fixed_bankruptcy_rate'] = summary['fixed_bankrupt'] / summary['fixed']

        return summary


class StatisticalAnalyzer:
    """Statistical analysis functions for feature comparison"""

    @staticmethod
    def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0

        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (np.mean(group1) - np.mean(group2)) / pooled_std

    @staticmethod
    def compute_eta_squared(groups: List[np.ndarray]) -> float:
        """Compute eta-squared effect size for ANOVA"""
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)

        # Between-group sum of squares
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)

        # Total sum of squares
        ss_total = np.sum((all_data - grand_mean)**2)

        if ss_total == 0:
            return 0.0

        return ss_between / ss_total

    @staticmethod
    def filter_sparse_features(
        features: np.ndarray,
        min_activation_rate: float = 0.01,
        min_mean_activation: float = 0.001
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter out extremely sparse SAE features before statistical analysis.

        SAE features are inherently sparse (L1 penalty design), but features
        with <1% activation rate cause numerical instability in ANOVA.
        This filtering step removes features that are active in less than 1%
        of samples, which typically cause interaction_eta ≈ 1.0 artifacts.

        Args:
            features: (n_samples, n_features) array of SAE activations
            min_activation_rate: Minimum proportion of non-zero activations (default: 0.01 = 1%)
            min_mean_activation: Minimum mean activation value (default: 0.001)

        Returns:
            Tuple of (filtered_features, valid_feature_indices)
            - filtered_features: (n_samples, n_valid_features) array
            - valid_feature_indices: Array of indices of retained features

        Example:
            >>> features = np.random.rand(1000, 131072)  # 1000 samples, 131K features
            >>> filtered, indices = StatisticalAnalyzer.filter_sparse_features(features)
            >>> print(f"Retained {len(indices)}/{features.shape[1]} features")
        """
        n_samples = features.shape[0]

        # Calculate activation metrics
        activation_rate = np.count_nonzero(features, axis=0) / n_samples
        mean_activation = np.mean(features, axis=0)

        # Create validity mask
        valid_mask = (activation_rate >= min_activation_rate) & (mean_activation >= min_mean_activation)
        valid_indices = np.where(valid_mask)[0]

        return features[:, valid_mask], valid_indices

    @staticmethod
    def welch_ttest(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
        """Perform Welch's t-test"""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0, 1.0
        if np.std(group1) == 0 and np.std(group2) == 0:
            return 0.0, 1.0

        try:
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            if np.isnan(p_value):
                p_value = 1.0
            return float(t_stat), float(p_value)
        except Exception:
            return 0.0, 1.0

    @staticmethod
    def one_way_anova(groups: List[np.ndarray]) -> Tuple[float, float]:
        """Perform one-way ANOVA"""
        # Filter out empty groups
        valid_groups = [g for g in groups if len(g) > 0]
        if len(valid_groups) < 2:
            return 0.0, 1.0

        try:
            f_stat, p_value = stats.f_oneway(*valid_groups)
            if np.isnan(p_value):
                p_value = 1.0
            return float(f_stat), float(p_value)
        except Exception:
            return 0.0, 1.0

    @staticmethod
    def two_way_anova_simple(
        features: np.ndarray,
        factor1: np.ndarray,  # bet_type
        factor2: np.ndarray   # outcome
    ) -> Tuple[dict, dict, dict]:
        """
        Simplified 2x2 factorial analysis using separate one-way ANOVAs.
        Returns main effects and interaction estimates.

        Note: For proper 2-way ANOVA with interaction, use statsmodels.
        This is a simplified version for computational efficiency.
        """
        # Unique levels
        levels1 = np.unique(factor1)
        levels2 = np.unique(factor2)

        # Group means
        group_means = {}
        for l1 in levels1:
            for l2 in levels2:
                mask = (factor1 == l1) & (factor2 == l2)
                key = f"{l1}_{l2}"
                if mask.sum() > 0:
                    group_means[key] = float(np.mean(features[mask]))
                else:
                    group_means[key] = 0.0

        # Main effect of factor1 (bet_type)
        groups1 = [features[factor1 == l] for l in levels1]
        f1, p1 = StatisticalAnalyzer.one_way_anova(groups1)
        eta1 = StatisticalAnalyzer.compute_eta_squared(groups1) if len(groups1) >= 2 else 0.0

        # Main effect of factor2 (outcome)
        groups2 = [features[factor2 == l] for l in levels2]
        f2, p2 = StatisticalAnalyzer.one_way_anova(groups2)
        eta2 = StatisticalAnalyzer.compute_eta_squared(groups2) if len(groups2) >= 2 else 0.0

        # Interaction estimate (difference of differences)
        # For 2x2 design: interaction = (A1B1 - A1B2) - (A2B1 - A2B2)
        if len(levels1) == 2 and len(levels2) == 2:
            l1_0, l1_1 = levels1
            l2_0, l2_1 = levels2

            # Get cell means
            cell_means = {}
            for l1 in levels1:
                for l2 in levels2:
                    mask = (factor1 == l1) & (factor2 == l2)
                    if mask.sum() > 0:
                        cell_means[(l1, l2)] = np.mean(features[mask])
                    else:
                        cell_means[(l1, l2)] = 0.0

            # Compute interaction contrast
            interaction_effect = (
                (cell_means[(l1_0, l2_0)] - cell_means[(l1_0, l2_1)]) -
                (cell_means[(l1_1, l2_0)] - cell_means[(l1_1, l2_1)])
            )

            # Estimate interaction F-stat using residual approach
            # This is an approximation; for exact values use statsmodels
            grand_mean = np.mean(features)
            ss_total = np.sum((features - grand_mean)**2)

            # Compute predicted values under additive model
            marginal1 = {l: np.mean(features[factor1 == l]) for l in levels1}
            marginal2 = {l: np.mean(features[factor2 == l]) for l in levels2}

            predicted_additive = np.zeros_like(features, dtype=float)
            for i in range(len(features)):
                predicted_additive[i] = (
                    marginal1[factor1[i]] + marginal2[factor2[i]] - grand_mean
                )

            ss_interaction = np.sum((features - predicted_additive - grand_mean)**2)
            ss_residual = ss_total - ss_interaction

            n_groups = 4
            n_total = len(features)
            df_interaction = 1  # (2-1)(2-1) = 1
            df_residual = n_total - n_groups

            if df_residual > 0 and ss_residual > 0:
                ms_interaction = ss_interaction / df_interaction
                ms_residual = ss_residual / df_residual
                f_int = ms_interaction / ms_residual if ms_residual > 0 else 0.0
                p_int = 1 - stats.f.cdf(f_int, df_interaction, df_residual)
                eta_int = ss_interaction / ss_total if ss_total > 0 else 0.0
            else:
                f_int, p_int, eta_int = 0.0, 1.0, 0.0
        else:
            f_int, p_int, eta_int = 0.0, 1.0, 0.0

        main_effect1 = {'f': f1, 'p': p1, 'eta_squared': eta1}
        main_effect2 = {'f': f2, 'p': p2, 'eta_squared': eta2}
        interaction = {'f': f_int, 'p': p_int, 'eta_squared': eta_int}

        return main_effect1, main_effect2, interaction, group_means

    @staticmethod
    def apply_fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Apply FDR (Benjamini-Hochberg) correction"""
        p_values = np.nan_to_num(p_values, nan=1.0)
        rejected, corrected_p, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        return rejected, corrected_p

    @staticmethod
    def apply_bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Bonferroni correction"""
        p_values = np.nan_to_num(p_values, nan=1.0)
        rejected, corrected_p, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
        return rejected, corrected_p


def load_prompt_metadata(json_file: str, game_ids: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Load prompt_combo metadata and parse into component indicators.

    Args:
        json_file: Path to experiment JSON file
        game_ids: Array of game IDs (0-3199 for slot machine experiment)

    Returns:
        Dictionary with keys:
            - 'prompt_combos': array of full combo strings (e.g., 'BASE', 'GM', 'GMRWP')
            - 'has_G': binary array indicating Goal-setting component
            - 'has_M': binary array indicating Maximize component
            - 'has_R': binary array indicating Risk/patterns component
            - 'has_W': binary array indicating Win multiplier component
            - 'has_P': binary array indicating Probability/win rate component
            - 'complexity': int array (0-5) indicating number of components

    Example:
        >>> metadata = load_prompt_metadata(json_file, game_ids)
        >>> # Group by Goal-setting component
        >>> has_G = metadata['has_G']
        >>> group1 = features[has_G]  # Games with Goal-setting
        >>> group2 = features[~has_G]  # Games without Goal-setting
    """
    with open(json_file) as f:
        data = json.load(f)

    results = data.get('results', data)

    metadata = {
        'prompt_combos': [],
        'has_G': [],
        'has_M': [],
        'has_R': [],
        'has_W': [],
        'has_P': [],
        'complexity': []
    }

    for gid in game_ids:
        combo = results[int(gid)]['prompt_combo']
        metadata['prompt_combos'].append(combo)

        # Parse components (BASE has no components)
        # Note: 'G' appears in many combos, so we check it's not just 'G' substring
        metadata['has_G'].append('G' in combo and combo != 'BASE')
        metadata['has_M'].append('M' in combo)
        metadata['has_R'].append('R' in combo)
        metadata['has_W'].append('W' in combo)
        metadata['has_P'].append('P' in combo)

        # Calculate complexity
        if combo == 'BASE':
            complexity = 0
        else:
            # Count unique components
            complexity = sum([
                'G' in combo,
                'M' in combo,
                'R' in combo,
                'W' in combo,
                'P' in combo
            ])
        metadata['complexity'].append(complexity)

    # Convert to numpy arrays
    return {
        'prompt_combos': np.array(metadata['prompt_combos'], dtype=str),
        'has_G': np.array(metadata['has_G'], dtype=bool),
        'has_M': np.array(metadata['has_M'], dtype=bool),
        'has_R': np.array(metadata['has_R'], dtype=bool),
        'has_W': np.array(metadata['has_W'], dtype=bool),
        'has_P': np.array(metadata['has_P'], dtype=bool),
        'complexity': np.array(metadata['complexity'], dtype=int)
    }


def save_results(results: dict, output_path: Path):
    """Save results to JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
