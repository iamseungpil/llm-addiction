#!/bin/bash
# Matched-cap ladder: BASE and the paper's full five-module condition, at every cap.
#
# What this fills in. The paper's matched-cap test aggregates all 32 prompt conditions at each
# cap for one model (GPT-4o-mini). E1 collapsed that to BASE alone, where four of six models sit
# at 0% in both arms and no contrast can be tested; mc32 restored two conditions but only at
# cap $70, so the ladder that carries the paper's claim was never run on another model.
#
# This is NOT a reproduction of the paper's 32-condition aggregate. It samples two conditions —
# the base condition and the richest one — so the result is biased upward relative to that
# aggregate, and the letter must say so.
#
# Layout: one lane per (model, cap), four cells per lane (2 conditions x 2 modes). Sixteen lanes
# run concurrently, four per provider. Cells whose output already exists are skipped, so the
# sixteen mc32 cells at cap $70 are not repeated.
#
# Guard: the runner substitutes a stop response when an API call fails after its retries, and
# records the count in manifest.api_fallback_responses. A cell with a non-zero count is renamed
# out of the way and flagged, because a silent substitution would read as a voluntary stop.

set -u
cd /home/v-seungplee/llm-addiction || exit 1

# Provider credentials live only in ~/.env and are never committed. The runner reads them from
# the environment, so the driver has to export them here; a lane that starts without them dies
# on the first call with "OPENAI_API_KEY not set".
if [ -f "$HOME/.env" ]; then
  set -a; . "$HOME/.env"; set +a
fi
for v in OPENAI_API_KEY ANTHROPIC_API_KEY GOOGLE_API_KEY; do
  [ -n "${!v:-}" ] || { echo "missing $v; aborting before any lane starts" >&2; exit 1; }
done

PY=/home/v-seungplee/miniconda3/envs/metaprobe/bin/python
RUNNER=paper_experiments/track0_w3_replication/src/run_track0_api.py
OUT=/home/v-seungplee/data/llm-addiction/mc32
LOG=/home/v-seungplee/llm-addiction/logs/mc_ladder
mkdir -p "$OUT" "$LOG"

N_GAMES=50
CAPS=(10 30 50 70)
CONDS=("BASE" "GMPRW")   # GMPRW is the code spelling of the paper's GMHWP
MODES=(fixed variable)

declare -A PROVIDER=(
  [gpt-4o-mini]=openai
  [gpt-4.1-mini]=openai
  [gemini-2.5-flash]=google
  [claude-haiku-4-5-20251001]=anthropic
)
declare -A SHORT=(
  [gpt-4o-mini]=gpt-4o-mini
  [gpt-4.1-mini]=gpt-4.1-mini
  [gemini-2.5-flash]=gemini-flash
  [claude-haiku-4-5-20251001]=claude-haiku-4-5-20251001
)

run_lane() {
  local model=$1 cap=$2
  local prov=${PROVIDER[$model]} short=${SHORT[$model]}
  local lanelog="$LOG/lane_${short}_cap${cap}.log"
  : > "$lanelog"
  for cond in "${CONDS[@]}"; do
    for mode in "${MODES[@]}"; do
      local suffix=""
      [ "$cond" != "BASE" ] && suffix="_${cond}"
      local stem="final_${short}_cap${cap}_${mode}${suffix}_persona"
      if compgen -G "$OUT/${stem}_*.json" > /dev/null; then
        echo "SKIP  $stem (already present)" >> "$lanelog"; continue
      fi
      echo "RUN   $stem  $(date +%H:%M:%S)" >> "$lanelog"
      local combo_arg=()
      [ "$cond" != "BASE" ] && combo_arg=(--prompt_combo "$cond")
      "$PY" "$RUNNER" \
        --provider "$prov" --model_id "$model" \
        --cap "$cap" --mode "$mode" "${combo_arg[@]}" --persona \
        --n_games "$N_GAMES" --output_dir "$OUT" >> "$lanelog" 2>&1 \
        || { echo "FAIL  $stem" >> "$lanelog"; continue; }

      # Fallback guard: quarantine any cell whose API calls were silently substituted.
      local produced
      produced=$(ls -t "$OUT/${stem}"_*.json 2>/dev/null | head -1)
      if [ -n "$produced" ]; then
        local nfb
        nfb=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['manifest'].get('api_fallback_responses',-1))" "$produced" 2>/dev/null)
        if [ "$nfb" != "0" ]; then
          mv "$produced" "${produced%.json}.QUARANTINE_fallback${nfb}.json"
          echo "QUARANTINE $stem api_fallback_responses=$nfb" >> "$lanelog"
        else
          echo "OK    $stem  $(date +%H:%M:%S)" >> "$lanelog"
        fi
      fi
    done
  done
  echo "LANE DONE $(date +%H:%M:%S)" >> "$lanelog"
}

for model in "${!PROVIDER[@]}"; do
  for cap in "${CAPS[@]}"; do
    run_lane "$model" "$cap" &
  done
done
wait
echo "all lanes finished: $(ls "$OUT"/final_*_persona_*.json 2>/dev/null | wc -l) cells present"
