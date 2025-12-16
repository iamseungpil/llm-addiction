#!/usr/bin/env python3
"""
Option Variance and Risk Analysis

Calculates variance for each option to determine if Option 4
is truly the "riskiest" in terms of outcome variance.
"""

import numpy as np

def calculate_option_statistics(bet_amount=10):
    """Calculate EV, Variance, SD for each option"""

    print("="*80)
    print(f"Option Statistics (Bet Amount: ${bet_amount})")
    print("="*80)

    options = {
        1: {'prob': [1.0], 'outcomes': [bet_amount]},
        2: {'prob': [0.5, 0.5], 'outcomes': [bet_amount * 1.8, 0]},
        3: {'prob': [0.25, 0.75], 'outcomes': [bet_amount * 3.2, 0]},
        4: {'prob': [0.1, 0.9], 'outcomes': [bet_amount * 9.0, 0]},
    }

    results = {}

    for opt_num, opt_data in options.items():
        probs = opt_data['prob']
        outcomes = opt_data['outcomes']

        # Expected value
        ev = sum(p * o for p, o in zip(probs, outcomes))

        # Net EV (subtract initial bet)
        net_ev = ev - bet_amount

        # Variance
        variance = sum(p * (o - ev)**2 for p, o in zip(probs, outcomes))

        # Standard deviation
        std = np.sqrt(variance)

        # Loss probability
        loss_prob = sum(p for p, o in zip(probs, outcomes) if o < bet_amount)

        results[opt_num] = {
            'ev': ev,
            'net_ev': net_ev,
            'variance': variance,
            'std': std,
            'loss_prob': loss_prob
        }

        print(f"\nOption {opt_num}:")
        print(f"  Outcomes: {outcomes}")
        print(f"  Probabilities: {probs}")
        print(f"  Expected Value: ${ev:.2f}")
        print(f"  Net EV (profit): ${net_ev:.2f}")
        print(f"  Variance: {variance:.2f}")
        print(f"  Std Dev: ${std:.2f}")
        print(f"  Loss Probability: {loss_prob*100:.0f}%")

    return results

def compare_options_2_vs_4():
    """Compare Option 2 vs 4 with same negative EV"""

    print("\n" + "="*80)
    print("Option 2 vs Option 4 Comparison")
    print("="*80)

    opt2 = {'ev': 9, 'variance': 81, 'loss_prob': 0.5}
    opt4 = {'ev': 9, 'variance': 729, 'loss_prob': 0.9}

    print("\nKey Finding: Option 2 and 4 have IDENTICAL expected value!")
    print(f"  Option 2: EV = ${opt2['ev']}, Net = $-1")
    print(f"  Option 4: EV = ${opt4['ev']}, Net = $-1")

    print("\nBut different variance:")
    print(f"  Option 2: Var = {opt2['variance']}, SD = ${np.sqrt(opt2['variance']):.2f}")
    print(f"  Option 4: Var = {opt4['variance']}, SD = ${np.sqrt(opt4['variance']):.2f}")
    print(f"  Ratio: Option 4 variance is {opt4['variance']/opt2['variance']:.1f}× higher")

    print("\nRisk-averse decision maker with same EV:")
    print("  → Should choose LOWER variance (Option 2)")
    print("  → Choosing Option 4 is 'variance-seeking' behavior")
    print("  → IRRATIONAL under standard finance theory")

def multi_round_ruin_probability():
    """Calculate bankruptcy probability over multiple rounds"""

    print("\n" + "="*80)
    print("Multi-Round Bankruptcy Risk (Fixed Betting)")
    print("="*80)

    print("\nStarting balance: $100, Bet: $10 per round")
    print("\nProbability of consecutive losses:")

    for opt, loss_prob in [(2, 0.5), (3, 0.75), (4, 0.9)]:
        print(f"\nOption {opt} (Loss prob: {loss_prob*100:.0f}%):")
        for n in range(1, 6):
            consec_loss = loss_prob ** n
            print(f"  {n} consecutive losses: {consec_loss*100:5.1f}%")

def variable_betting_risk():
    """Analyze risk in variable betting context"""

    print("\n" + "="*80)
    print("Variable Betting: Risk Depends on BOTH Option AND Bet Amount")
    print("="*80)

    scenarios = [
        ("Option 4, $1 bet", 4, 1),
        ("Option 4, $50 bet", 4, 50),
        ("Option 2, $1 bet", 2, 1),
        ("Option 2, $50 bet", 2, 50),
    ]

    print("\nScenario Comparison:")
    print(f"{'Scenario':<20} {'Net EV':<10} {'Variance':<12} {'SD':<10} {'Max Loss'}")
    print("-"*80)

    for name, opt, bet in scenarios:
        if opt == 2:
            ev = bet * 0.9  # 0.5 * 1.8 * bet
            variance = 0.5 * (bet * 1.8 - ev)**2 + 0.5 * (0 - ev)**2
        elif opt == 4:
            ev = bet * 0.9  # 0.1 * 9.0 * bet
            variance = 0.1 * (bet * 9.0 - ev)**2 + 0.9 * (0 - ev)**2

        net_ev = ev - bet
        std = np.sqrt(variance)
        max_loss = bet

        print(f"{name:<20} ${net_ev:>7.2f}   {variance:>10.2f}   ${std:>7.2f}   ${max_loss:>7}")

    print("\nConclusion:")
    print("  In VARIABLE betting, risk = f(Option, Bet Amount)")
    print("  Option 4 + small bet < Option 2 + large bet (in terms of variance)")
    print("  Cannot say 'Option 4 is riskiest' without knowing bet amount!")

if __name__ == '__main__':
    # Analysis
    results = calculate_option_statistics(bet_amount=10)
    compare_options_2_vs_4()
    multi_round_ruin_probability()
    variable_betting_risk()

    # Final conclusion
    print("\n" + "="*80)
    print("FINAL CONCLUSION")
    print("="*80)

    print("\nQ: Is Option 4 the 'riskiest' (highest variance)?")
    print("\nA: It depends on the context:")

    print("\n1. FIXED BETTING:")
    print("   ✅ YES - Option 4 has highest variance (729 vs 192 vs 81)")
    print("   ✅ YES - Option 4 has highest loss probability (90%)")
    print("   ✅ YES - Option 4 has highest bankruptcy risk")

    print("\n2. VARIABLE BETTING:")
    print("   ⚠️  INCOMPLETE - Must consider bet amount")
    print("   ❌ NO - Option 4 + $1 bet < Option 2 + $100 bet (risk)")
    print("   ⚠️  BUT if comparing same bet amounts, then YES")

    print("\n3. RATIONALITY PERSPECTIVE:")
    print("   ✅ Option 2 vs 4: Same EV (-$1), but Option 4 has 9× variance")
    print("   ✅ Risk-averse agent should choose Option 2")
    print("   ✅ Choosing Option 4 = variance-seeking = IRRATIONAL")

    print("\n" + "="*80)
