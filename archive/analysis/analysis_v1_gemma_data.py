#!/usr/bin/env python3
"""
Comprehensive analysis of OLD V1 Gemma slot machine data
from HuggingFace dataset, checking for known V1 parser bugs.

V1 bugs:
  1. 'stop' in response_lower parser (matches CoT reasoning text like "I should stop and think")
  2. max_new_tokens=100 (truncates ~60% of Gemma CoT responses)
  3. No valid check in game loop (parse failures silently recorded)
  4. Default bet=$10 on parse failure
  5. bet=$100 (all-in) possible via `'stop' in response_lower` mismatch with bet parsing
"""

import json
import numpy as np
from collections import Counter, defaultdict
import sys

# =============================================================================
# 1. LOAD DATA
# =============================================================================

V1_PATH = "/home/jovyan/beomi/llm-addiction-data/hf-dataset/slot_machine/gemma/final_gemma_20251004_172426.json"
V3_PATH = "/home/jovyan/beomi/llm-addiction-data/slot_machine/experiment_0_gemma_v3/final_gemma_20260225_100215.json"
LLAMA_PATH = "/home/jovyan/beomi/llm-addiction-data/hf-dataset/slot_machine/llama/final_llama_20251004_021106.json"

print("=" * 80)
print("GEMMA V1 SLOT MACHINE DATA ANALYSIS")
print("=" * 80)

with open(V1_PATH) as f:
    v1_data = json.load(f)

with open(V3_PATH) as f:
    v3_data = json.load(f)

with open(LLAMA_PATH) as f:
    llama_data = json.load(f)

v1_results = v1_data["results"]
v3_results = v3_data["results"]
llama_results = llama_data["results"]

# =============================================================================
# 1. SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 1: DATA SUMMARY")
print("=" * 80)

print(f"\nV1 Data: {V1_PATH}")
print(f"  Timestamp: {v1_data['timestamp']}")
print(f"  Total games: {len(v1_results)}")
print(f"  Conditions: {v1_data.get('conditions', 'N/A')}")
print(f"  Reps per condition: {v1_data.get('repetitions_per_condition', 'N/A')}")

# Bet types
bet_types = Counter(r["bet_type"] for r in v1_results)
print(f"\n  Bet types: {dict(bet_types)}")

# Prompt combos
prompt_combos = Counter(r["prompt_combo"] for r in v1_results)
print(f"  Prompt combos: {dict(prompt_combos)}")

# Outcomes
outcomes = Counter(r["outcome"] for r in v1_results)
print(f"\n  Outcomes:")
for k, v in sorted(outcomes.items()):
    print(f"    {k}: {v} ({v/len(v1_results)*100:.1f}%)")

# V1 result keys vs V3 result keys
print(f"\n  V1 result keys: {list(v1_results[0].keys())}")
v3_sample = [r for r in v3_results if r.get('decisions')][0] if any(r.get('decisions') for r in v3_results) else v3_results[0]
print(f"  V3 result keys: {list(v3_sample.keys())}")
print(f"  V1 has 'decisions' field: {'decisions' in v1_results[0]}")
print(f"  V3 has 'decisions' field: {'decisions' in v3_sample}")

# =============================================================================
# 2. RESPONSE LENGTH ANALYSIS (Limited - no raw responses in V1)
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 2: RESPONSE LENGTH ANALYSIS")
print("=" * 80)

print("\n  *** V1 data does NOT contain raw response text ('decisions' field missing) ***")
print("  Cannot directly measure response lengths or check for truncation.")
print("  Using V3 data (which has raw responses) for reference statistics.\n")

# Analyze V3 responses for reference
v3_response_lengths = []
v3_has_final_decision = 0
v3_total_responses = 0
v3_truncation_estimate = 0

for game in v3_results:
    for dec in game.get("decisions", []):
        resp = dec.get("response", "")
        if resp:
            v3_total_responses += 1
            chars = len(resp)
            tokens_est = chars / 3.6  # Gemma ~3.6 chars/token
            v3_response_lengths.append((chars, tokens_est))
            if "final decision" in resp.lower():
                v3_has_final_decision += 1
            if tokens_est < 100:
                v3_truncation_estimate += 1

if v3_response_lengths:
    chars_arr = np.array([x[0] for x in v3_response_lengths])
    tokens_arr = np.array([x[1] for x in v3_response_lengths])
    print(f"  V3 Reference (max_new_tokens=1024):")
    print(f"    Total responses: {v3_total_responses}")
    print(f"    Avg response length: {chars_arr.mean():.0f} chars / {tokens_arr.mean():.0f} est. tokens")
    print(f"    Median: {np.median(chars_arr):.0f} chars / {np.median(tokens_arr):.0f} est. tokens")
    print(f"    Min: {chars_arr.min():.0f} chars / {tokens_arr.min():.0f} est. tokens")
    print(f"    Max: {chars_arr.max():.0f} chars / {tokens_arr.max():.0f} est. tokens")
    print(f"    P10: {np.percentile(chars_arr, 10):.0f} chars / {np.percentile(tokens_arr, 10):.0f} est. tokens")
    print(f"    P25: {np.percentile(chars_arr, 25):.0f} chars / {np.percentile(tokens_arr, 25):.0f} est. tokens")
    print(f"    P75: {np.percentile(chars_arr, 75):.0f} chars / {np.percentile(tokens_arr, 75):.0f} est. tokens")
    print(f"    P90: {np.percentile(chars_arr, 90):.0f} chars / {np.percentile(tokens_arr, 90):.0f} est. tokens")
    print(f"\n    % with 'Final Decision' text: {v3_has_final_decision/v3_total_responses*100:.1f}%")
    print(f"    % < 100 tokens: {v3_truncation_estimate/v3_total_responses*100:.1f}%")

    # Estimate what V1's max_new_tokens=100 would truncate
    would_truncate = sum(1 for t in tokens_arr if t > 100)
    print(f"\n    *** TRUNCATION ESTIMATE for V1 (max_new_tokens=100) ***")
    print(f"    V3 responses > 100 tokens: {would_truncate}/{v3_total_responses} ({would_truncate/v3_total_responses*100:.1f}%)")
    print(f"    These responses would have been TRUNCATED in V1, losing 'Final Decision' line.")

    # What about responses > 250 tokens (V2)?
    would_truncate_v2 = sum(1 for t in tokens_arr if t > 250)
    print(f"    V3 responses > 250 tokens: {would_truncate_v2}/{v3_total_responses} ({would_truncate_v2/v3_total_responses*100:.1f}%)")

# =============================================================================
# 3. PARSER ACCURACY CHECK (indirect - from behavioral anomalies)
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 3: PARSER ACCURACY CHECK (Indirect)")
print("=" * 80)

print("\n  Since V1 has no raw responses, we check for behavioral anomalies")
print("  consistent with V1 parser bugs.\n")

# 3a. Games that stop at round 0 (immediate stop)
r0_stops = [r for r in v1_results if r["total_rounds"] == 0 and r["outcome"] == "voluntary_stop"]
print(f"  3a. Immediate stops (round 0, voluntary_stop):")
print(f"      Count: {len(r0_stops)} / {len(v1_results)} ({len(r0_stops)/len(v1_results)*100:.1f}%)")
print(f"      By bet_type:")
for bt in ["fixed", "variable"]:
    n = sum(1 for r in r0_stops if r["bet_type"] == bt)
    total_bt = sum(1 for r in v1_results if r["bet_type"] == bt)
    print(f"        {bt}: {n}/{total_bt} ({n/total_bt*100:.1f}%)")

# V3 comparison
v3_r0_stops = [r for r in v3_results if r["total_rounds"] == 0 and r["outcome"] == "voluntary_stop"]
print(f"\n      V3 comparison: {len(v3_r0_stops)} / {len(v3_results)} ({len(v3_r0_stops)/len(v3_results)*100:.1f}%)")

# 3b. Stop rate analysis
print(f"\n  3b. Overall voluntary stop rate:")
v1_stops = sum(1 for r in v1_results if r["outcome"] == "voluntary_stop")
v3_stops = sum(1 for r in v3_results if r["outcome"] == "voluntary_stop")
print(f"      V1: {v1_stops}/{len(v1_results)} ({v1_stops/len(v1_results)*100:.1f}%)")
print(f"      V3: {v3_stops}/{len(v3_results)} ({v3_stops/len(v3_results)*100:.1f}%)")
print(f"      Difference: {(v1_stops/len(v1_results) - v3_stops/len(v3_results))*100:+.1f} percentage points")

# 3c. Stop rate by round
print(f"\n  3c. Distribution of stop round (games that stopped voluntarily):")
v1_stop_games = [r for r in v1_results if r["outcome"] == "voluntary_stop"]
v3_stop_games = [r for r in v3_results if r["outcome"] == "voluntary_stop"]

v1_stop_rounds = Counter(r["total_rounds"] for r in v1_stop_games)
v3_stop_rounds = Counter(r["total_rounds"] for r in v3_stop_games)

print(f"\n      {'Round':<8} {'V1 count':>10} {'V1 %':>8} {'V3 count':>10} {'V3 %':>8}")
print(f"      {'-'*44}")
for rd in sorted(set(list(v1_stop_rounds.keys()) + list(v3_stop_rounds.keys()))):
    v1_c = v1_stop_rounds.get(rd, 0)
    v3_c = v3_stop_rounds.get(rd, 0)
    v1_p = v1_c / len(v1_stop_games) * 100 if v1_stop_games else 0
    v3_p = v3_c / len(v3_stop_games) * 100 if v3_stop_games else 0
    print(f"      {rd:<8} {v1_c:>10} {v1_p:>7.1f}% {v3_c:>10} {v3_p:>7.1f}%")

# 3d. Suspicious patterns - bet=$100 all-in in fixed games
print(f"\n  3d. Suspicious bet=$100 all-in (should be $10 in fixed):")
fixed_games = [r for r in v1_results if r["bet_type"] == "fixed"]
allin_fixed = 0
allin_fixed_games = 0
for g in fixed_games:
    has_allin = False
    for h in g["history"]:
        if h["bet"] == 100:
            allin_fixed += 1
            has_allin = True
    if has_allin:
        allin_fixed_games += 1

print(f"      Fixed games with bet=$100 rounds: {allin_fixed_games}/{len(fixed_games)}")
print(f"      Total bet=$100 rounds in fixed games: {allin_fixed}")

# Check bet distribution in fixed games
fixed_bets = []
for g in fixed_games:
    for h in g["history"]:
        fixed_bets.append(h["bet"])

if fixed_bets:
    fixed_bet_dist = Counter(fixed_bets)
    print(f"\n      Bet distribution in fixed games:")
    for b, c in sorted(fixed_bet_dist.items()):
        print(f"        ${b}: {c} ({c/len(fixed_bets)*100:.1f}%)")

# 3e. Variable bet distribution
print(f"\n  3e. Variable bet distribution:")
var_games = [r for r in v1_results if r["bet_type"] == "variable"]
var_bets = []
for g in var_games:
    for h in g["history"]:
        var_bets.append(h["bet"])

if var_bets:
    var_bet_dist = Counter(var_bets)
    print(f"      Total variable bet rounds: {len(var_bets)}")
    for b, c in sorted(var_bet_dist.items()):
        print(f"        ${b}: {c} ({c/len(var_bets)*100:.1f}%)")

# V3 comparison
v3_var_bets = []
for g in v3_results:
    if g["bet_type"] == "variable":
        for h in g["history"]:
            v3_var_bets.append(h["bet"])

if v3_var_bets:
    v3_var_bet_dist = Counter(v3_var_bets)
    print(f"\n      V3 variable bet distribution:")
    print(f"      Total variable bet rounds: {len(v3_var_bets)}")
    for b, c in sorted(v3_var_bet_dist.items()):
        print(f"        ${b}: {c} ({c/len(v3_var_bets)*100:.1f}%)")

# =============================================================================
# 4. BEHAVIORAL METRICS
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 4: BEHAVIORAL METRICS")
print("=" * 80)

def compute_metrics(results, label=""):
    """Compute behavioral metrics for a set of results."""
    total = len(results)
    bankruptcies = sum(1 for r in results if r["outcome"] == "bankruptcy")
    vol_stops = sum(1 for r in results if r["outcome"] == "voluntary_stop")
    max_rounds_reached = sum(1 for r in results if r["outcome"] == "max_rounds")

    rounds = [r["total_rounds"] for r in results]
    balances = [r["final_balance"] for r in results]

    print(f"\n  {label} (n={total}):")
    print(f"    Bankruptcy rate: {bankruptcies}/{total} ({bankruptcies/total*100:.1f}%)")
    print(f"    Voluntary stop rate: {vol_stops}/{total} ({vol_stops/total*100:.1f}%)")
    print(f"    Max rounds reached: {max_rounds_reached}/{total} ({max_rounds_reached/total*100:.1f}%)")
    print(f"    Avg rounds played: {np.mean(rounds):.2f} (std={np.std(rounds):.2f})")
    print(f"    Median rounds: {np.median(rounds):.1f}")
    print(f"    Avg final balance: ${np.mean(balances):.2f} (std=${np.std(balances):.2f})")
    print(f"    Median final balance: ${np.median(balances):.2f}")

    return {
        "total": total,
        "bankruptcy_rate": bankruptcies / total,
        "vol_stop_rate": vol_stops / total,
        "avg_rounds": np.mean(rounds),
        "avg_balance": np.mean(balances),
    }

# Overall
print("\n  --- V1 (OLD) Data ---")
v1_overall = compute_metrics(v1_results, "V1 Overall")

# By bet type
v1_fixed = [r for r in v1_results if r["bet_type"] == "fixed"]
v1_var = [r for r in v1_results if r["bet_type"] == "variable"]
v1_fixed_m = compute_metrics(v1_fixed, "V1 Fixed Bet")
v1_var_m = compute_metrics(v1_var, "V1 Variable Bet")

# V3 comparison
print("\n  --- V3 (FIXED) Data ---")
v3_overall = compute_metrics(v3_results, "V3 Overall")

v3_fixed = [r for r in v3_results if r["bet_type"] == "fixed"]
v3_var = [r for r in v3_results if r["bet_type"] == "variable"]
v3_fixed_m = compute_metrics(v3_fixed, "V3 Fixed Bet")
v3_var_m = compute_metrics(v3_var, "V3 Variable Bet")

# LLaMA comparison (base model, should not be affected by CoT bugs)
print("\n  --- LLaMA V1 Data (base model, no CoT bugs expected) ---")
llama_overall = compute_metrics(llama_results, "LLaMA Overall")
llama_fixed = [r for r in llama_results if r["bet_type"] == "fixed"]
llama_var = [r for r in llama_results if r["bet_type"] == "variable"]
compute_metrics(llama_fixed, "LLaMA Fixed Bet")
compute_metrics(llama_var, "LLaMA Variable Bet")

# 4b. By prompt combo
print("\n  4b. V1 Bankruptcy rate by prompt combo:")
prompt_combos_list = sorted(set(r["prompt_combo"] for r in v1_results))
print(f"\n      {'Combo':<12} {'N':>5} {'Bankrupt':>10} {'Rate':>8} {'AvgRounds':>10} {'AvgBal':>8}")
print(f"      {'-'*58}")
for pc in prompt_combos_list:
    games = [r for r in v1_results if r["prompt_combo"] == pc]
    br = sum(1 for r in games if r["outcome"] == "bankruptcy")
    avg_rd = np.mean([r["total_rounds"] for r in games])
    avg_bal = np.mean([r["final_balance"] for r in games])
    print(f"      {pc:<12} {len(games):>5} {br:>10} {br/len(games)*100:>7.1f}% {avg_rd:>10.2f} ${avg_bal:>7.2f}")

# =============================================================================
# 5. V1 vs V3 CROSS-CHECK
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 5: V1 vs V3 CROSS-CHECK")
print("=" * 80)

print(f"\n  {'Metric':<30} {'V1 (OLD)':>12} {'V3 (FIXED)':>12} {'Delta':>12}")
print(f"  {'-'*66}")

metrics = [
    ("Bankruptcy rate", f"{v1_overall['bankruptcy_rate']*100:.1f}%", f"{v3_overall['bankruptcy_rate']*100:.1f}%",
     f"{(v1_overall['bankruptcy_rate']-v3_overall['bankruptcy_rate'])*100:+.1f}pp"),
    ("Voluntary stop rate", f"{v1_overall['vol_stop_rate']*100:.1f}%", f"{v3_overall['vol_stop_rate']*100:.1f}%",
     f"{(v1_overall['vol_stop_rate']-v3_overall['vol_stop_rate'])*100:+.1f}pp"),
    ("Avg rounds played", f"{v1_overall['avg_rounds']:.2f}", f"{v3_overall['avg_rounds']:.2f}",
     f"{v1_overall['avg_rounds']-v3_overall['avg_rounds']:+.2f}"),
    ("Avg final balance", f"${v1_overall['avg_balance']:.2f}", f"${v3_overall['avg_balance']:.2f}",
     f"${v1_overall['avg_balance']-v3_overall['avg_balance']:+.2f}"),
    ("Fixed bankruptcy", f"{v1_fixed_m['bankruptcy_rate']*100:.1f}%", f"{v3_fixed_m['bankruptcy_rate']*100:.1f}%",
     f"{(v1_fixed_m['bankruptcy_rate']-v3_fixed_m['bankruptcy_rate'])*100:+.1f}pp"),
    ("Variable bankruptcy", f"{v1_var_m['bankruptcy_rate']*100:.1f}%", f"{v3_var_m['bankruptcy_rate']*100:.1f}%",
     f"{(v1_var_m['bankruptcy_rate']-v3_var_m['bankruptcy_rate'])*100:+.1f}pp"),
]

for name, v1_val, v3_val, delta in metrics:
    print(f"  {name:<30} {v1_val:>12} {v3_val:>12} {delta:>12}")

# =============================================================================
# 6. DETAILED GAME INSPECTION
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 6: DETAILED GAME INSPECTION (10-20 sample games)")
print("=" * 80)

print("\n  Since V1 has no raw responses, we sample games with suspicious patterns:\n")

# 6a. Games that stopped at round 0 but have prompt combos that encourage gambling
print("  6a. Round-0 stops with gambling-encouraging prompts:")
encouraging_combos = ["G", "AG", "BG", "ABG", "CG", "ACG", "BCG", "ABCG"]  # G = goal/gamble-encouraging
r0_encouraging = [r for r in v1_results if r["total_rounds"] == 0 and r["outcome"] == "voluntary_stop"
                  and "G" in r["prompt_combo"]]
print(f"      Count: {len(r0_encouraging)}")
for i, g in enumerate(r0_encouraging[:5]):
    print(f"      Game {i}: combo={g['prompt_combo']}, bet_type={g['bet_type']}, rep={g['repetition']}")

# 6b. Games that went bankrupt in round 1 (suspicious all-in)
print(f"\n  6b. Games that went bankrupt in exactly 1 round:")
r1_bankrupt = [r for r in v1_results if r["total_rounds"] == 1 and r["outcome"] == "bankruptcy"]
print(f"      Count: {len(r1_bankrupt)}")
for i, g in enumerate(r1_bankrupt[:10]):
    bet = g["history"][0]["bet"] if g["history"] else "N/A"
    print(f"      Game {i}: combo={g['prompt_combo']}, bet_type={g['bet_type']}, bet=${bet}, balance={g['final_balance']}")

# 6c. Fixed games where bet != $10
print(f"\n  6c. Fixed-bet games with non-$10 bets (should always be $10 in fixed):")
fixed_weird = []
for g in v1_results:
    if g["bet_type"] == "fixed":
        for h in g["history"]:
            if h["bet"] != 10:
                fixed_weird.append((g, h))

print(f"      Count: {len(fixed_weird)} rounds across games")
for i, (g, h) in enumerate(fixed_weird[:10]):
    print(f"      Round {h['round']}: combo={g['prompt_combo']}, bet=${h['bet']}, result={h['result']}, balance={h['balance']}")

# 6d. Games with very high total rounds
print(f"\n  6d. Games with most rounds played (potential non-stopping):")
v1_sorted_by_rounds = sorted(v1_results, key=lambda r: r["total_rounds"], reverse=True)
for i, g in enumerate(v1_sorted_by_rounds[:10]):
    print(f"      Game {i}: combo={g['prompt_combo']}, bet_type={g['bet_type']}, "
          f"rounds={g['total_rounds']}, outcome={g['outcome']}, balance=${g['final_balance']}")

# 6e. Compare specific condition between V1 and V3
print(f"\n  6e. Side-by-side: Same condition (BASE, fixed) in V1 vs V3:")
v1_base_fixed = [r for r in v1_results if r["prompt_combo"] == "BASE" and r["bet_type"] == "fixed"]
v3_base_fixed = [r for r in v3_results if r["prompt_combo"] == "BASE" and r["bet_type"] == "fixed"]

v1_bf_outcomes = Counter(r["outcome"] for r in v1_base_fixed)
v3_bf_outcomes = Counter(r["outcome"] for r in v3_base_fixed)
v1_bf_rounds = [r["total_rounds"] for r in v1_base_fixed]
v3_bf_rounds = [r["total_rounds"] for r in v3_base_fixed]

print(f"      V1 BASE/fixed (n={len(v1_base_fixed)}): outcomes={dict(v1_bf_outcomes)}, avg_rounds={np.mean(v1_bf_rounds):.2f}")
print(f"      V3 BASE/fixed (n={len(v3_base_fixed)}): outcomes={dict(v3_bf_outcomes)}, avg_rounds={np.mean(v3_bf_rounds):.2f}")

# 6f. Detailed look at V3 raw responses to check what V1 parser would have done
print(f"\n  6f. V3 responses: checking 'stop' in response_lower (V1 bug simulation):")
v1_parser_would_misparse = 0
v1_parser_correct_stop = 0
v1_parser_correct_bet = 0
v1_parser_total = 0

# Also check for truncation patterns
responses_with_final_decision = 0
responses_without_final_decision = 0

stop_in_reasoning_examples = []
correctly_parsed_examples = []

for game in v3_results:
    for dec in game.get("decisions", []):
        resp = dec.get("response", "")
        if not resp:
            continue
        v1_parser_total += 1
        resp_lower = resp.lower()
        action = dec.get("action", "")

        has_stop = "stop" in resp_lower
        has_final_decision = "final decision" in resp_lower

        if has_final_decision:
            responses_with_final_decision += 1
        else:
            responses_without_final_decision += 1

        # V1 parser: if 'stop' in response_lower -> action='stop'
        v1_would_say_stop = has_stop
        actual_action = action

        if v1_would_say_stop and actual_action == "bet":
            # V1 parser would have misparsed this as 'stop' when the model actually bet
            v1_parser_would_misparse += 1
            if len(stop_in_reasoning_examples) < 10:
                # Find where 'stop' appears
                idx = resp_lower.find("stop")
                context_start = max(0, idx - 40)
                context_end = min(len(resp), idx + 60)
                stop_in_reasoning_examples.append({
                    "action": actual_action,
                    "bet": dec.get("bet", "?"),
                    "context": resp[context_start:context_end],
                    "full_response_len": len(resp),
                    "has_final_decision": has_final_decision,
                })
        elif v1_would_say_stop and actual_action == "stop":
            v1_parser_correct_stop += 1
        elif not v1_would_say_stop and actual_action == "bet":
            v1_parser_correct_bet += 1

print(f"\n      Total V3 responses analyzed: {v1_parser_total}")
print(f"      'stop' in response AND actual action=bet (MISPARSE): {v1_parser_would_misparse} ({v1_parser_would_misparse/v1_parser_total*100:.1f}%)")
print(f"      'stop' in response AND actual action=stop (correct): {v1_parser_correct_stop} ({v1_parser_correct_stop/v1_parser_total*100:.1f}%)")
print(f"      No 'stop' in response AND actual action=bet (correct): {v1_parser_correct_bet} ({v1_parser_correct_bet/v1_parser_total*100:.1f}%)")
print(f"      With 'Final Decision' text: {responses_with_final_decision} ({responses_with_final_decision/v1_parser_total*100:.1f}%)")
print(f"      Without 'Final Decision' text: {responses_without_final_decision} ({responses_without_final_decision/v1_parser_total*100:.1f}%)")

print(f"\n      === MISPARSED EXAMPLES (model said 'bet' but V1 would parse 'stop') ===")
for i, ex in enumerate(stop_in_reasoning_examples):
    print(f"\n      Example {i+1}: actual_action={ex['action']}, bet=${ex['bet']}, resp_len={ex['full_response_len']}")
    print(f"        has_final_decision: {ex['has_final_decision']}")
    print(f"        Context around 'stop': ...{ex['context']}...")

# =============================================================================
# 7. ADDITIONAL ANALYSES
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 7: ADDITIONAL ANALYSES")
print("=" * 80)

# 7a. Balance trajectory analysis
print("\n  7a. Final balance distribution:")
v1_balances = [r["final_balance"] for r in v1_results]
v3_balances = [r["final_balance"] for r in v3_results]

for label, balances in [("V1", v1_balances), ("V3", v3_balances)]:
    arr = np.array(balances)
    print(f"\n      {label}: mean=${arr.mean():.2f}, median=${np.median(arr):.2f}, "
          f"min=${arr.min()}, max=${arr.max()}")
    # Distribution buckets
    buckets = [0, 50, 100, 150, 200, 300, 500, 1000]
    for i in range(len(buckets)):
        if i == 0:
            count = sum(1 for b in balances if b == 0)
            print(f"        $0 (bankrupt): {count} ({count/len(balances)*100:.1f}%)")
        elif i < len(buckets) - 1:
            count = sum(1 for b in balances if buckets[i-1] < b <= buckets[i])
            print(f"        ${buckets[i-1]+1}-${buckets[i]}: {count} ({count/len(balances)*100:.1f}%)")
        else:
            count = sum(1 for b in balances if b > buckets[i-1])
            print(f"        >${buckets[i-1]}: {count} ({count/len(balances)*100:.1f}%)")

# 7b. Round distribution histogram
print(f"\n  7b. Round distribution:")
v1_rounds = [r["total_rounds"] for r in v1_results]
v3_rounds = [r["total_rounds"] for r in v3_results]

for label, rounds in [("V1", v1_rounds), ("V3", v3_rounds)]:
    arr = np.array(rounds)
    print(f"\n      {label}: mean={arr.mean():.2f}, median={np.median(arr):.1f}, max={arr.max()}")
    round_dist = Counter(rounds)
    for rd in sorted(round_dist.keys())[:20]:
        bar = "#" * (round_dist[rd] // 10)
        print(f"        Round {rd:>3}: {round_dist[rd]:>5} ({round_dist[rd]/len(rounds)*100:>5.1f}%) {bar}")
    if max(round_dist.keys()) > 19:
        remaining = sum(v for k, v in round_dist.items() if k > 19)
        print(f"        Round >19: {remaining:>5} ({remaining/len(rounds)*100:>5.1f}%)")

# 7c. Loss chasing in V1 variable games
print(f"\n  7c. Loss chasing analysis (variable bet games):")
for label, results in [("V1", v1_results), ("V3", v3_results)]:
    chase_after_loss = 0
    reduce_after_loss = 0
    same_after_loss = 0
    total_loss_followed = 0

    for g in results:
        if g["bet_type"] != "variable":
            continue
        hist = g["history"]
        for i in range(1, len(hist)):
            if not hist[i-1]["win"]:  # Previous round was a loss
                total_loss_followed += 1
                if hist[i]["bet"] > hist[i-1]["bet"]:
                    chase_after_loss += 1
                elif hist[i]["bet"] < hist[i-1]["bet"]:
                    reduce_after_loss += 1
                else:
                    same_after_loss += 1

    if total_loss_followed > 0:
        print(f"\n      {label}: After a loss (n={total_loss_followed}):")
        print(f"        Increased bet (chase): {chase_after_loss} ({chase_after_loss/total_loss_followed*100:.1f}%)")
        print(f"        Decreased bet: {reduce_after_loss} ({reduce_after_loss/total_loss_followed*100:.1f}%)")
        print(f"        Same bet: {same_after_loss} ({same_after_loss/total_loss_followed*100:.1f}%)")

# =============================================================================
# 8. VERDICT
# =============================================================================

print("\n" + "=" * 80)
print("SECTION 8: VERDICT AND SUMMARY")
print("=" * 80)

v1_br = v1_overall['bankruptcy_rate'] * 100
v3_br = v3_overall['bankruptcy_rate'] * 100
v1_sr = v1_overall['vol_stop_rate'] * 100
v3_sr = v3_overall['vol_stop_rate'] * 100
v1_ar = v1_overall['avg_rounds']
v3_ar = v3_overall['avg_rounds']

print(f"""
  KEY FINDINGS:

  1. BANKRUPTCY RATE: V1={v1_br:.1f}% vs V3={v3_br:.1f}%
     {'-> LARGE DISCREPANCY: V1 data likely corrupted by parser bugs' if abs(v1_br - v3_br) > 2 else '-> Rates are similar'}

  2. VOLUNTARY STOP RATE: V1={v1_sr:.1f}% vs V3={v3_sr:.1f}%
     {'-> V1 inflated stop rate from "stop" in CoT reasoning' if v1_sr > v3_sr + 5 else '-> Rates are similar' if abs(v1_sr - v3_sr) < 5 else '-> Different stop behavior'}

  3. AVG ROUNDS: V1={v1_ar:.2f} vs V3={v3_ar:.2f}
     {'-> V1 games end much sooner (consistent with premature stop parsing)' if v1_ar < v3_ar - 0.5 else '-> Similar game lengths'}

  4. V1 PARSER SIMULATION (using V3 responses):
     -> {v1_parser_would_misparse}/{v1_parser_total} ({v1_parser_would_misparse/v1_parser_total*100:.1f}%) of bet-decisions would be misparsed as 'stop'

  5. IMMEDIATE STOPS (round 0): V1={len(r0_stops)} ({len(r0_stops)/len(v1_results)*100:.1f}%) vs V3={len(v3_r0_stops)} ({len(v3_r0_stops)/len(v3_results)*100:.1f}%)

  6. ROUND 1 BANKRUPTCIES: {len(r1_bankrupt)} games went all-in and lost immediately
     (Possible cause: truncated response -> parser default bet=$10 or misparsed bet amount)

  CONCLUSION:
""")

if abs(v1_br - v3_br) > 2 or abs(v1_sr - v3_sr) > 5 or v1_ar < v3_ar - 0.5:
    print("  ** V1 data shows SIGNIFICANT behavioral differences from V3 **")
    print("  ** This is CONSISTENT with known V1 parser bugs affecting the data **")
    if v1_parser_would_misparse / max(v1_parser_total, 1) > 0.1:
        print(f"  ** Parser simulation confirms: {v1_parser_would_misparse/v1_parser_total*100:.1f}% of bet decisions")
        print("     would be incorrectly parsed as 'stop' by V1's 'stop' in response_lower logic **")
    print("\n  RECOMMENDATION: V1 Gemma data should NOT be used for analysis.")
    print("  Use V3 data (experiment_0_gemma_v3) instead.")
else:
    print("  V1 and V3 data show SIMILAR behavioral patterns.")
    print("  Parser bugs may have had limited impact on aggregate statistics.")
    print("  However, individual game-level data may still be unreliable.")
    if v1_parser_would_misparse / max(v1_parser_total, 1) > 0.05:
        print(f"\n  CAUTION: Parser simulation shows {v1_parser_would_misparse/v1_parser_total*100:.1f}% misparsing rate.")
        print("  V3 data is still preferred for accurate analysis.")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
