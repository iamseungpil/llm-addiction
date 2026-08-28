#!/usr/bin/env python3
"""
Analysis of LLaMA-3.1-8B Slot Machine Experiment Data (V1 Original Run)
=========================================================================

Data: /home/jovyan/beomi/llm-addiction-data/hf-dataset/slot_machine/llama/final_llama_20251004_021106.json

LLaMA-3.1-8B is a BASE model (not instruction-tuned), using prefix-completion.
V1 parser: 'stop' in response_lower, max_new_tokens=100
Expected impact on LLaMA: LESS than Gemma (base model generates short responses)

Key questions:
1. Data integrity: Are fixed bets actually fixed? Any parser anomalies?
2. Behavioral profile: How does LLaMA gamble?
3. Condition effects: Which prompt components drive behavior?
4. Comparison baseline: Profile for later Gemma comparison
"""

import json
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================================
# 1. LOAD AND SUMMARIZE
# ============================================================================
DATA_PATH = "/home/jovyan/beomi/llm-addiction-data/hf-dataset/slot_machine/llama/final_llama_20251004_021106.json"

with open(DATA_PATH) as f:
    data = json.load(f)

results = data["results"]
total_games = len(results)

print("=" * 80)
print("SECTION 1: DATA SUMMARY")
print("=" * 80)
print(f"Model: {data['model']} (LLaMA-3.1-8B, BASE model)")
print(f"Timestamp: {data['timestamp']}")
print(f"Total games: {total_games}")
print(f"Conditions: {data['conditions']}")
print(f"Repetitions per condition: {data['repetitions_per_condition']}")
print()

# Bet types
bet_types = Counter(r["bet_type"] for r in results)
print(f"Bet types: {dict(bet_types)}")
print(f"  Fixed: {bet_types['fixed']} games ({bet_types['fixed']/total_games*100:.1f}%)")
print(f"  Variable: {bet_types['variable']} games ({bet_types['variable']/total_games*100:.1f}%)")

# Outcomes
outcomes = Counter(r["outcome"] for r in results)
print(f"\nOutcomes: {dict(outcomes)}")
for outcome, count in outcomes.most_common():
    print(f"  {outcome}: {count} ({count/total_games*100:.1f}%)")

# Prompt combos
combos = sorted(set(r["prompt_combo"] for r in results))
print(f"\nPrompt combinations: {len(combos)}")
print(f"  {', '.join(combos[:10])}{'...' if len(combos) > 10 else ''}")

# ============================================================================
# 2. DATA INTEGRITY: FIXED BET ANOMALIES
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: DATA INTEGRITY - V1 PARSER BUG ANALYSIS")
print("=" * 80)

fixed_games = [r for r in results if r["bet_type"] == "fixed"]
variable_games = [r for r in results if r["bet_type"] == "variable"]

# Fixed bet distribution
fixed_bets = []
for r in fixed_games:
    for h in r["history"]:
        fixed_bets.append(h["bet"])

fixed_bet_counter = Counter(fixed_bets)
total_fixed_bets = len(fixed_bets)
correct_bets = fixed_bet_counter.get(10, 0)
anomalous_bets = total_fixed_bets - correct_bets

print(f"Fixed bet games: {len(fixed_games)}")
print(f"Total fixed bets placed: {total_fixed_bets}")
print(f"  Correct ($10): {correct_bets} ({correct_bets/total_fixed_bets*100:.1f}%)")
print(f"  ANOMALOUS (not $10): {anomalous_bets} ({anomalous_bets/total_fixed_bets*100:.1f}%)")
print()

print("Fixed bet amount distribution:")
for bet, count in sorted(fixed_bet_counter.items()):
    marker = "" if bet == 10 else " <-- ANOMALOUS"
    print(f"  ${bet:3d}: {count:5d} ({count/total_fixed_bets*100:5.1f}%){marker}")

# Games with anomalous bets
anomalous_games = [r for r in fixed_games if any(h["bet"] != 10 for h in r["history"])]
print(f"\nFixed games containing anomalous bets: {anomalous_games.__len__()}/{len(fixed_games)} "
      f"({len(anomalous_games)/len(fixed_games)*100:.1f}%)")

# Impact assessment: do anomalous bets change outcomes?
anom_bankrupt = sum(1 for g in anomalous_games if g["outcome"] == "bankruptcy")
clean_fixed = [r for r in fixed_games if all(h["bet"] == 10 for h in r["history"])]
clean_bankrupt = sum(1 for g in clean_fixed if g["outcome"] == "bankruptcy")
print(f"\nBankruptcy rate comparison:")
print(f"  Anomalous-bet games: {anom_bankrupt}/{len(anomalous_games)} ({anom_bankrupt/len(anomalous_games)*100:.1f}%)" if anomalous_games else "  None")
print(f"  Clean fixed games: {clean_bankrupt}/{len(clean_fixed)} ({clean_bankrupt/len(clean_fixed)*100:.1f}%)" if clean_fixed else "  None")

# Severity: are anomalous bets concentrated in specific combos?
anom_by_combo = Counter(g["prompt_combo"] for g in anomalous_games)
print(f"\nAnomalous games by prompt combo (top 10):")
for combo, count in anom_by_combo.most_common(10):
    total_combo = sum(1 for g in fixed_games if g["prompt_combo"] == combo)
    print(f"  {combo:8s}: {count}/{total_combo} ({count/total_combo*100:.1f}%)")

print("\n** ASSESSMENT: LLaMA base model occasionally outputs bet amounts that the V1")
print("** parser extracted even in 'fixed' mode. The V1 code lacked enforcement of")
print("** bet_type='fixed' → bet=$10, so the parser's extracted amount was used.")
print("** However, 90.3% of fixed bets are correctly $10, so impact is moderate.")

# ============================================================================
# 3. RESPONSE LENGTH PROXY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: RESPONSE BEHAVIOR (Inferred from Data)")
print("=" * 80)
print()
print("NOTE: Raw responses are NOT stored in this data file.")
print("We can infer response characteristics from game behavior patterns.")

# Games with 0 rounds = immediate stop (LLaMA responded with stop immediately)
zero_round_games = [r for r in results if r["total_rounds"] == 0]
print(f"\nGames with 0 rounds (immediate stop): {len(zero_round_games)} ({len(zero_round_games)/total_games*100:.1f}%)")
print(f"  Fixed: {sum(1 for g in zero_round_games if g['bet_type'] == 'fixed')}")
print(f"  Variable: {sum(1 for g in zero_round_games if g['bet_type'] == 'variable')}")

# Since LLaMA is a base model, it likely generates:
# - Short prefix completions (not CoT reasoning)
# - Format: "1) Bet $10" or "2) Stop playing" or just a number
# - max_new_tokens=100 is sufficient for base model completions
# The V1 parser bug ('stop' in response_lower) could still match if the model
# generates text like "I'll stop" in its continuation

# Round distribution
round_counts = [r["total_rounds"] for r in results]
print(f"\nRound count distribution:")
print(f"  Mean: {np.mean(round_counts):.1f}")
print(f"  Median: {np.median(round_counts):.1f}")
print(f"  Std: {np.std(round_counts):.1f}")
print(f"  Min: {min(round_counts)}, Max: {max(round_counts)}")

# Histogram of round counts
round_counter = Counter(round_counts)
print(f"\nRound count histogram:")
for n_rounds in sorted(round_counter.keys())[:20]:
    count = round_counter[n_rounds]
    bar = "#" * min(60, int(count / 5))
    print(f"  {n_rounds:3d} rounds: {count:4d} ({count/total_games*100:5.1f}%) {bar}")
if max(round_counts) > 19:
    long_games = sum(1 for c in round_counts if c > 19)
    print(f"  20+ rounds: {long_games:4d} ({long_games/total_games*100:5.1f}%)")

# ============================================================================
# 4. BEHAVIORAL METRICS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: BEHAVIORAL METRICS")
print("=" * 80)

# 4a. Bankruptcy rate by bet type
print("\n--- 4a. Bankruptcy Rate ---")
for bt in ["fixed", "variable"]:
    subset = [r for r in results if r["bet_type"] == bt]
    bankrupt = sum(1 for r in subset if r["outcome"] == "bankruptcy")
    print(f"  {bt:8s}: {bankrupt}/{len(subset)} = {bankrupt/len(subset)*100:.1f}%")

# Overall
all_bankrupt = sum(1 for r in results if r["outcome"] == "bankruptcy")
print(f"  {'overall':8s}: {all_bankrupt}/{total_games} = {all_bankrupt/total_games*100:.1f}%")

# 4b. Voluntary stop rate
print("\n--- 4b. Voluntary Stop Rate ---")
for bt in ["fixed", "variable"]:
    subset = [r for r in results if r["bet_type"] == bt]
    stops = sum(1 for r in subset if r["outcome"] == "voluntary_stop")
    print(f"  {bt:8s}: {stops}/{len(subset)} = {stops/len(subset)*100:.1f}%")

# 4c. Average rounds played
print("\n--- 4c. Average Rounds Played ---")
for bt in ["fixed", "variable"]:
    subset = [r for r in results if r["bet_type"] == bt]
    rounds = [r["total_rounds"] for r in subset]
    print(f"  {bt:8s}: mean={np.mean(rounds):.2f}, median={np.median(rounds):.1f}, std={np.std(rounds):.2f}")

print(f"  {'overall':8s}: mean={np.mean(round_counts):.2f}, median={np.median(round_counts):.1f}")

# 4d. Average final balance
print("\n--- 4d. Average Final Balance ---")
for bt in ["fixed", "variable"]:
    subset = [r for r in results if r["bet_type"] == bt]
    balances = [r["final_balance"] for r in subset]
    print(f"  {bt:8s}: mean=${np.mean(balances):.1f}, median=${np.median(balances):.1f}, "
          f"std=${np.std(balances):.1f}")
    # Distribution
    below_100 = sum(1 for b in balances if b < 100)
    at_100 = sum(1 for b in balances if b == 100)
    above_100 = sum(1 for b in balances if b > 100)
    print(f"           Below $100: {below_100} ({below_100/len(balances)*100:.1f}%), "
          f"At $100: {at_100} ({at_100/len(balances)*100:.1f}%), "
          f"Above $100: {above_100} ({above_100/len(balances)*100:.1f}%)")

# 4e. Variable bet amount analysis
print("\n--- 4e. Variable Bet Amount Distribution ---")
var_bets = []
for r in variable_games:
    for h in r["history"]:
        var_bets.append(h["bet"])

if var_bets:
    var_bet_counter = Counter(var_bets)
    total_var = len(var_bets)
    print(f"Total variable bets: {total_var}")
    print(f"Mean bet: ${np.mean(var_bets):.1f}")
    print(f"Median bet: ${np.median(var_bets):.1f}")
    print(f"Std: ${np.std(var_bets):.1f}")
    print()

    # Categorize bets
    small = sum(1 for b in var_bets if b <= 10)
    moderate = sum(1 for b in var_bets if 11 <= b <= 30)
    large = sum(1 for b in var_bets if 31 <= b <= 50)
    extreme = sum(1 for b in var_bets if b > 50)
    print(f"Bet categories:")
    print(f"  Small ($5-$10):     {small:4d} ({small/total_var*100:.1f}%)")
    print(f"  Moderate ($11-$30): {moderate:4d} ({moderate/total_var*100:.1f}%)")
    print(f"  Large ($31-$50):    {large:4d} ({large/total_var*100:.1f}%)")
    print(f"  Extreme ($51+):     {extreme:4d} ({extreme/total_var*100:.1f}%)")

    print(f"\nDetailed distribution:")
    for bet, count in sorted(var_bet_counter.items()):
        bar = "#" * min(40, int(count / 5))
        print(f"  ${bet:3d}: {count:4d} ({count/total_var*100:5.1f}%) {bar}")

# 4f. Betting Aggressiveness Index (I_BA)
print("\n--- 4f. Betting Aggressiveness Index (I_BA) ---")
print("I_BA = bet / balance_before_bet")
iba_values = {"fixed": [], "variable": []}
for r in results:
    balance = 100  # starting balance
    for h in r["history"]:
        if balance > 0:
            iba = h["bet"] / balance
            iba_values[r["bet_type"]].append(iba)
        # Update balance (reconstruct from history)
        balance = h["balance"]

for bt in ["fixed", "variable"]:
    vals = iba_values[bt]
    if vals:
        print(f"  {bt:8s}: mean={np.mean(vals):.4f}, median={np.median(vals):.4f}, "
              f"std={np.std(vals):.4f}")

# 4g. Extreme Choice Index (I_EC)
print("\n--- 4g. Extreme Choice Index (I_EC) ---")
print("I_EC = proportion of bets that are max ($100 or all-in)")
for bt in ["fixed", "variable"]:
    subset = [r for r in results if r["bet_type"] == bt]
    total_bets = 0
    extreme_bets = 0
    for r in subset:
        balance = 100
        for h in r["history"]:
            total_bets += 1
            if h["bet"] >= balance * 0.9 or h["bet"] >= 100:  # 90%+ of balance = extreme
                extreme_bets += 1
            balance = h["balance"]
    if total_bets > 0:
        print(f"  {bt:8s}: {extreme_bets}/{total_bets} = {extreme_bets/total_bets*100:.1f}%")

# ============================================================================
# 5. CONDITION EFFECTS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: CONDITION EFFECTS (Prompt Components)")
print("=" * 80)
print()
print("Prompt components:")
print("  G = Goal setting ('set a target amount')")
print("  M = Maximize ('maximize the reward')")
print("  R = Risk illusion ('hidden patterns')")
print("  W = Win multiplier info ('3.0 times')")
print("  P = Probability info ('Win rate: 30%')")

# Bankruptcy rate by prompt combo
print("\n--- 5a. Bankruptcy Rate by Prompt Combo ---")
combo_stats = {}
for combo in combos:
    subset = [r for r in results if r["prompt_combo"] == combo]
    bankrupt = sum(1 for r in subset if r["outcome"] == "bankruptcy")
    avg_rounds = np.mean([r["total_rounds"] for r in subset])
    avg_balance = np.mean([r["final_balance"] for r in subset])
    combo_stats[combo] = {
        "n": len(subset),
        "bankrupt": bankrupt,
        "bankruptcy_rate": bankrupt / len(subset),
        "avg_rounds": avg_rounds,
        "avg_balance": avg_balance,
    }

# Sort by bankruptcy rate
sorted_combos = sorted(combo_stats.items(), key=lambda x: x[1]["bankruptcy_rate"], reverse=True)
print(f"{'Combo':10s} {'N':>4s} {'Bankrupt':>8s} {'Rate':>7s} {'AvgRnds':>8s} {'AvgBal':>8s}")
print("-" * 50)
for combo, stats in sorted_combos:
    print(f"{combo:10s} {stats['n']:4d} {stats['bankrupt']:8d} {stats['bankruptcy_rate']:7.1%} "
          f"{stats['avg_rounds']:8.1f} ${stats['avg_balance']:7.1f}")

# Component effect analysis
print("\n--- 5b. Individual Component Effects ---")
components = {'G': 'Goal', 'M': 'Maximize', 'R': 'Risk illusion', 'W': 'Win info', 'P': 'Probability'}

for comp, name in components.items():
    with_comp = [r for r in results if comp in r["prompt_combo"]]
    without_comp = [r for r in results if comp not in r["prompt_combo"]]

    with_bankrupt = sum(1 for r in with_comp if r["outcome"] == "bankruptcy") / len(with_comp) * 100
    without_bankrupt = sum(1 for r in without_comp if r["outcome"] == "bankruptcy") / len(without_comp) * 100

    with_rounds = np.mean([r["total_rounds"] for r in with_comp])
    without_rounds = np.mean([r["total_rounds"] for r in without_comp])

    with_balance = np.mean([r["final_balance"] for r in with_comp])
    without_balance = np.mean([r["final_balance"] for r in without_comp])

    print(f"\n  {comp} ({name}):")
    print(f"    Bankruptcy: WITH={with_bankrupt:.1f}% vs WITHOUT={without_bankrupt:.1f}% (diff={with_bankrupt-without_bankrupt:+.1f}pp)")
    print(f"    Avg rounds: WITH={with_rounds:.1f} vs WITHOUT={without_rounds:.1f} (diff={with_rounds-without_rounds:+.1f})")
    print(f"    Avg balance: WITH=${with_balance:.1f} vs WITHOUT=${without_balance:.1f} (diff=${with_balance-without_balance:+.1f})")

# 5c. Bet type × component interaction
print("\n--- 5c. Bet Type x Component Interaction (Bankruptcy Rate) ---")
print(f"{'Component':12s} {'Fixed':>12s} {'Variable':>12s} {'Diff(V-F)':>12s}")
print("-" * 52)

for comp, name in [("BASE", "BASE (none)")] + [(c, f"{c} ({n})") for c, n in components.items()]:
    for bt in ["fixed", "variable"]:
        if comp == "BASE":
            subset = [r for r in results if r["prompt_combo"] == "BASE" and r["bet_type"] == bt]
        else:
            subset = [r for r in results if comp in r["prompt_combo"] and r["bet_type"] == bt]
        bankrupt = sum(1 for r in subset if r["outcome"] == "bankruptcy") / len(subset) * 100 if subset else 0
        if bt == "fixed":
            fixed_rate = bankrupt
        else:
            var_rate = bankrupt
    print(f"  {name:12s} {fixed_rate:10.1f}% {var_rate:10.1f}% {var_rate - fixed_rate:+10.1f}pp")

# ============================================================================
# 6. LOSS CHASING ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: LOSS CHASING ANALYSIS (I_LC)")
print("=" * 80)
print("I_LC: Did the model increase bets after a loss?")
print("(Only meaningful for variable bet games)")

lc_after_loss = 0
lc_after_win = 0
total_after_loss = 0
total_after_win = 0

for r in variable_games:
    for i in range(1, len(r["history"])):
        prev = r["history"][i-1]
        curr = r["history"][i]

        if prev["result"] == "L":
            total_after_loss += 1
            if curr["bet"] > prev["bet"]:
                lc_after_loss += 1
        elif prev["result"] == "W":
            total_after_win += 1
            if curr["bet"] > prev["bet"]:
                lc_after_win += 1

print(f"\nAfter LOSS: increased bet {lc_after_loss}/{total_after_loss} "
      f"({lc_after_loss/total_after_loss*100:.1f}%)" if total_after_loss > 0 else "\nNo sequential loss data")
print(f"After WIN:  increased bet {lc_after_win}/{total_after_win} "
      f"({lc_after_win/total_after_win*100:.1f}%)" if total_after_win > 0 else "No sequential win data")

if total_after_loss > 0 and total_after_win > 0:
    lc_ratio = (lc_after_loss/total_after_loss) / (lc_after_win/total_after_win) if lc_after_win > 0 else float('inf')
    print(f"\nLoss chasing ratio: {lc_ratio:.2f} (>1 = chasing losses, <1 = loss averse)")

# Also track bet size changes
bet_change_after_loss = []
bet_change_after_win = []
for r in variable_games:
    for i in range(1, len(r["history"])):
        prev = r["history"][i-1]
        curr = r["history"][i]
        change = curr["bet"] - prev["bet"]

        if prev["result"] == "L":
            bet_change_after_loss.append(change)
        elif prev["result"] == "W":
            bet_change_after_win.append(change)

if bet_change_after_loss:
    print(f"\nAverage bet change after LOSS: ${np.mean(bet_change_after_loss):+.1f} (std=${np.std(bet_change_after_loss):.1f})")
if bet_change_after_win:
    print(f"Average bet change after WIN:  ${np.mean(bet_change_after_win):+.1f} (std=${np.std(bet_change_after_win):.1f})")

# ============================================================================
# 7. TEMPORAL PATTERNS (Bet progression within games)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7: TEMPORAL PATTERNS")
print("=" * 80)

# Average bet by round number (variable games only)
bets_by_round = defaultdict(list)
for r in variable_games:
    for h in r["history"]:
        bets_by_round[h["round"]].append(h["bet"])

print("\nAverage variable bet by round:")
print(f"{'Round':>6s} {'N':>6s} {'MeanBet':>10s} {'MedianBet':>10s} {'StdBet':>10s}")
for rnd in sorted(bets_by_round.keys())[:20]:
    bets = bets_by_round[rnd]
    print(f"{rnd:6d} {len(bets):6d} ${np.mean(bets):9.1f} ${np.median(bets):9.1f} ${np.std(bets):9.1f}")

# Stop probability by round
print("\nStop probability by round (games still active at round N that stop at round N):")
# For each round, count how many games stopped there vs continued
games_active_at_round = defaultdict(int)
games_stopped_at_round = defaultdict(int)

for r in results:
    if r["total_rounds"] == 0:
        games_stopped_at_round[0] += 1
        games_active_at_round[0] += 1
    else:
        for rnd_num in range(1, r["total_rounds"] + 1):
            games_active_at_round[rnd_num] += 1
        if r["outcome"] == "voluntary_stop":
            games_stopped_at_round[r["total_rounds"]] += 1

print(f"{'Round':>6s} {'Active':>8s} {'Stopped':>8s} {'StopRate':>10s}")
for rnd in sorted(games_active_at_round.keys())[:15]:
    active = games_active_at_round[rnd]
    stopped = games_stopped_at_round.get(rnd, 0)
    rate = stopped / active * 100 if active > 0 else 0
    print(f"{rnd:6d} {active:8d} {stopped:8d} {rate:9.1f}%")

# ============================================================================
# 8. COMPARISON SUMMARY (for later Gemma comparison)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 8: LLAMA BEHAVIORAL PROFILE SUMMARY")
print("=" * 80)

print(f"""
LLaMA-3.1-8B (BASE model) Slot Machine Profile:
================================================
Total games: {total_games}
Parser version: V1 (original, 'stop' in response_lower)
max_new_tokens: 100

OUTCOMES:
  Overall bankruptcy rate: {all_bankrupt/total_games*100:.1f}% ({all_bankrupt}/{total_games})
  Overall voluntary stop rate: {(total_games - all_bankrupt)/total_games*100:.1f}%

BY BET TYPE:
  Fixed bankruptcy:    {sum(1 for r in fixed_games if r['outcome'] == 'bankruptcy')}/{len(fixed_games)} = {sum(1 for r in fixed_games if r['outcome'] == 'bankruptcy')/len(fixed_games)*100:.1f}%
  Variable bankruptcy: {sum(1 for r in variable_games if r['outcome'] == 'bankruptcy')}/{len(variable_games)} = {sum(1 for r in variable_games if r['outcome'] == 'bankruptcy')/len(variable_games)*100:.1f}%

ROUNDS:
  Mean rounds: {np.mean(round_counts):.1f}
  Fixed mean rounds: {np.mean([r['total_rounds'] for r in fixed_games]):.1f}
  Variable mean rounds: {np.mean([r['total_rounds'] for r in variable_games]):.1f}

BALANCE:
  Mean final balance: ${np.mean([r['final_balance'] for r in results]):.1f}
  Fixed mean balance: ${np.mean([r['final_balance'] for r in fixed_games]):.1f}
  Variable mean balance: ${np.mean([r['final_balance'] for r in variable_games]):.1f}

BETTING (Variable):
  Mean bet: ${np.mean(var_bets):.1f}
  Median bet: ${np.median(var_bets):.1f}
  $10 or less: {sum(1 for b in var_bets if b <= 10)/len(var_bets)*100:.1f}%
  $50+: {sum(1 for b in var_bets if b >= 50)/len(var_bets)*100:.1f}%

DATA INTEGRITY:
  Fixed bet anomalies: {anomalous_bets}/{total_fixed_bets} ({anomalous_bets/total_fixed_bets*100:.1f}%) of fixed bets were NOT $10
  Zero-round games: {len(zero_round_games)} ({len(zero_round_games)/total_games*100:.1f}%)

KEY OBSERVATIONS:
  1. Very high voluntary stop rate -- LLaMA base model tends to stop early
  2. Short games (many 0-1 round games) -- consistent with base model prefix-completion
  3. Variable bets cluster around $10 and $50 -- bimodal distribution
  4. Some fixed-bet anomalies exist (V1 parser bug) but affect <10% of bets
  5. Moderate bankruptcy rate despite early stopping tendency
""")

# ============================================================================
# 9. DEEP DIVE: Zero-round games analysis
# ============================================================================
print("=" * 80)
print("SECTION 9: ZERO-ROUND GAMES DEEP DIVE")
print("=" * 80)

# Zero-round by combo
zero_by_combo = Counter(r["prompt_combo"] for r in zero_round_games)
zero_by_bt = Counter(r["bet_type"] for r in zero_round_games)

print(f"\nZero-round games by bet type:")
for bt, count in zero_by_bt.most_common():
    total_bt = sum(1 for r in results if r["bet_type"] == bt)
    print(f"  {bt}: {count}/{total_bt} ({count/total_bt*100:.1f}%)")

print(f"\nZero-round rate by prompt combo (top/bottom 5):")
combo_zero_rates = {}
for combo in combos:
    subset = [r for r in results if r["prompt_combo"] == combo]
    zeros = sum(1 for r in subset if r["total_rounds"] == 0)
    combo_zero_rates[combo] = zeros / len(subset)

sorted_zero = sorted(combo_zero_rates.items(), key=lambda x: x[1], reverse=True)
print("  HIGHEST zero-round rate:")
for combo, rate in sorted_zero[:5]:
    print(f"    {combo:8s}: {rate*100:.1f}%")
print("  LOWEST zero-round rate:")
for combo, rate in sorted_zero[-5:]:
    print(f"    {combo:8s}: {rate*100:.1f}%")

# ============================================================================
# 10. COMPARISON CONTEXT: What to expect from Gemma
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 10: NOTES FOR GEMMA COMPARISON")
print("=" * 80)
print("""
Expected differences between LLaMA (base) and Gemma (instruction-tuned):

1. RESPONSE FORMAT:
   - LLaMA: Short prefix completions (often just "1)" or "$10")
   - Gemma: Full CoT reasoning with "Final Decision:" line

2. V1 PARSER IMPACT:
   - LLaMA: LOW impact - short responses, less text to confuse 'stop' check
   - Gemma: HIGH impact - CoT reasoning like "I should stop..." triggers false stops

3. max_new_tokens=100 IMPACT:
   - LLaMA: LOW impact - base model completions are typically < 50 tokens
   - Gemma: HIGH impact - CoT responses average ~220 tokens, 60% truncation

4. EXPECTED BEHAVIORAL DIFFERENCES:
   - LLaMA may show more genuine base-model tendencies
   - Gemma V1 data is CORRUPTED by parser bugs (need V3 data)
   - Key comparison: Does instruction-tuning increase or decrease gambling?

5. CURRENT DATA USABILITY:
   - LLaMA V1 data: MOSTLY USABLE (moderate parser issues, 90%+ correct)
   - Gemma V1 data: CORRUPTED (parser and truncation bugs)
""")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
