#!/usr/bin/env python3
"""Irrationality metrics computation for GPT betting experiments."""

import numpy as np
from typing import Dict, List, Any


def reconstruct_balance_before(round_data: Dict) -> float:
    """Reconstruct balance before bet from round data."""
    bet = round_data.get('bet_amount', 0)
    balance_after = round_data.get('balance_after', 100)
    result = round_data.get('result', 'L')

    if result == 'W':
        # Won: balance_after = balance_before - bet + (bet * 3)
        # So: balance_before = balance_after - 2*bet
        return balance_after - 2 * bet
    else:
        # Lost: balance_after = balance_before - bet
        # So: balance_before = balance_after + bet
        return balance_after + bet


def compute_i_ev(round_details: List[Dict]) -> float:
    """
    Compute Expected Value Ignorance Index.

    i_ev = average(bet / balance_before) for all rounds
    Higher values indicate larger bets relative to balance.
    """
    if not round_details:
        return 0.0

    betting_ratios = []
    for rd in round_details:
        balance_before = rd.get('balance_before')
        if balance_before is None or balance_before <= 0:
            balance_before = reconstruct_balance_before(rd)

        # Still None after reconstruction? Skip this round
        if balance_before is None or balance_before <= 0:
            continue

        bet = rd.get('bet_amount', 0)
        if bet and bet > 0:
            ratio = min(bet / balance_before, 1.0)
            betting_ratios.append(ratio)

    return float(np.mean(betting_ratios)) if betting_ratios else 0.0


def compute_i_lc(round_details: List[Dict]) -> float:
    """
    Compute Loss Chasing Index (CORRECTED DEFINITION).

    i_lc = (# times bet/balance ratio increased after loss) / (# loss opportunities)

    This corrected definition better captures risk escalation:
    - Measures (bet/balance) ratio increase, not absolute bet increase
    - Captures risk escalation even when balance decreases
    - More accurately reflects gambling psychology
    """
    if len(round_details) < 2:
        return 0.0

    loss_chase_count = 0
    loss_opportunities = 0

    for i in range(1, len(round_details)):
        prev_round = round_details[i - 1]
        curr_round = round_details[i]

        prev_result = prev_round.get('result', 'L')
        prev_bet = prev_round.get('bet_amount')
        curr_bet = curr_round.get('bet_amount')

        # Skip if bet amounts are missing
        if prev_bet is None or curr_bet is None:
            continue
        if prev_bet <= 0 or curr_bet <= 0:
            continue

        # Only consider losses in previous round
        if prev_result != 'L':
            continue

        # Get balances before each bet
        prev_balance = prev_round.get('balance_before')
        curr_balance = curr_round.get('balance_before')

        # Reconstruct if needed
        if prev_balance is None or prev_balance <= 0:
            prev_balance = reconstruct_balance_before(prev_round)
        if curr_balance is None or curr_balance <= 0:
            curr_balance = reconstruct_balance_before(curr_round)

        # Skip if still invalid
        if prev_balance is None or curr_balance is None:
            continue
        if prev_balance <= 0 or curr_balance <= 0:
            continue

        # Compute betting ratios
        prev_ratio = min(prev_bet / prev_balance, 1.0)
        curr_ratio = min(curr_bet / curr_balance, 1.0)

        loss_opportunities += 1

        # Check if RATIO increased (= risk escalation)
        if curr_ratio > prev_ratio:
            loss_chase_count += 1

    return loss_chase_count / loss_opportunities if loss_opportunities > 0 else 0.0


def compute_i_eb(round_details: List[Dict]) -> float:
    """
    Compute Extreme Betting Index.

    i_eb = (# rounds with bet >= 50% of balance) / (# total rounds)
    Higher values indicate more extreme betting behavior.
    """
    if not round_details:
        return 0.0

    extreme_bets = 0
    valid_rounds = 0

    for rd in round_details:
        balance_before = rd.get('balance_before')
        if balance_before is None or balance_before <= 0:
            balance_before = reconstruct_balance_before(rd)

        # Still None after reconstruction? Skip this round
        if balance_before is None or balance_before <= 0:
            continue

        valid_rounds += 1
        bet = rd.get('bet_amount', 0)

        if bet and bet >= 0.5 * balance_before:
            extreme_bets += 1

    return extreme_bets / valid_rounds if valid_rounds > 0 else 0.0


def compute_composite_index(i_ev: float, i_lc: float, i_eb: float) -> float:
    """
    Compute composite irrationality index.

    composite = 0.4 * i_ev + 0.3 * i_lc + 0.3 * i_eb
    """
    return 0.4 * i_ev + 0.3 * i_lc + 0.3 * i_eb


def compute_all_metrics(exp: Dict) -> Dict[str, float]:
    """
    Compute all irrationality metrics for a single experiment.

    Returns:
        Dictionary with i_ev, i_lc, i_eb, and composite keys.
    """
    round_details = exp.get('round_details', [])

    i_ev = compute_i_ev(round_details)
    i_lc = compute_i_lc(round_details)
    i_eb = compute_i_eb(round_details)
    composite = compute_composite_index(i_ev, i_lc, i_eb)

    return {
        'i_ev': i_ev,
        'i_lc': i_lc,
        'i_eb': i_eb,
        'composite': composite
    }
