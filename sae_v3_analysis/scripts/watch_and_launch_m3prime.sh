#!/usr/bin/env bash
# Watcher: poll for v1 M3 swap process to exit, then chain M3' dose ladder.
# Designed to be the autonomous handoff between v1 (counterfactual swap)
# and v1.5 (indicator-direction steering).
set -uo pipefail

LOG=/home/v-seungplee/llm-addiction/sae_v3_analysis/logs/watch_m3prime.log
mkdir -p "$(dirname "$LOG")"
exec >"$LOG" 2>&1

V1_TARGET="run_m3_counterfactual_swap.py"

echo "=== [watcher start $(date '+%H:%M:%S')] looking for ${V1_TARGET} ==="
SECONDS=0
while pgrep -f "$V1_TARGET" >/dev/null 2>&1; do
    if (( SECONDS % 600 < 60 )); then
        echo "[watcher $(date '+%H:%M:%S')] v1 still running ($(pgrep -f "$V1_TARGET" | wc -l) procs)"
    fi
    sleep 60
done
echo "=== [watcher $(date '+%H:%M:%S')] v1 has exited; starting M3' ladder ==="

# Aggregate v1 results once
/home/v-seungplee/miniconda3/envs/llm-addiction/bin/python \
    -c "
import json
from pathlib import Path
ROOT = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results/v19_multi_patching/M3_swap')
out = {}
for cond in ['baseline_minusG', 'swap_plusG', 'random_swap_ctrl']:
    fp = ROOT / f'gemma_sm_{cond}_n200/trials.jsonl'
    if not fp.exists():
        continue
    rows = [json.loads(l) for l in open(fp)]
    if not rows:
        continue
    bks = [r.get('bankrupt', False) for r in rows]
    vss = [r.get('voluntary_stop', False) for r in rows]
    out[cond] = {'n': len(rows), 'bk_rate': sum(bks)/len(rows),
                 'vstop_rate': sum(vss)/len(rows),
                 'mean_rounds': sum(r.get('n_decisions',0) for r in rows)/len(rows)}
agg = ROOT / 'aggregate_summary.json'
agg.write_text(json.dumps(out, indent=2))
print(f'[v1 aggregate] {agg}')
print(json.dumps(out, indent=2))
"

# Launch M3' dose ladder
exec bash /home/v-seungplee/llm-addiction/sae_v3_analysis/scripts/launch_m3prime_dose_ladder.sh
