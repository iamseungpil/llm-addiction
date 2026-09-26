#!/usr/bin/env bash
# M3' dose-ladder launcher: 6 dose conditions + 3 specificity, N=50 each.
# Total: 450 trials × ~26 sec/trial ≈ 3.3 h on a single A100.
#
# Conditions:
#   alpha-2..alpha+3 : six-point dose ladder using I_BA direction at L22
#   random           : random unit direction, σ matched, α=+2 at L22 (specificity)
#   L8               : I_BA direction projected to L8, α=+2 (layer specificity)
#   ILC              : I_LC direction at L22, α=+2 (indicator specificity)
#
# Resume-safe: each (cond, alpha, layer, dir) writes to its own folder; existing
# trial seeds are skipped on re-launch.
set -euo pipefail

cd /home/v-seungplee/llm-addiction
PY=/home/v-seungplee/miniconda3/envs/llm-addiction/bin/python
RUN=sae_v3_analysis/src/run_m3prime_indicator_steering.py
N=${N:-50}
GPU=${GPU:-0}
MODEL=${MODEL:-gemma}
TASK=${TASK:-sm}

echo "=== [START $(date '+%H:%M:%S')] M3' dose ladder model=${MODEL} task=${TASK} n=${N} gpu=${GPU} ==="

# 1) Six-point dose ladder on I_BA, L22
for spec in "alpha-2:-2.0:i_ba:22" \
            "alpha-1:-1.0:i_ba:22" \
            "alpha+0: 0.0:i_ba:22" \
            "alpha+1:+1.0:i_ba:22" \
            "alpha+2:+2.0:i_ba:22" \
            "alpha+3:+3.0:i_ba:22"; do
    IFS=':' read -r cond alpha dir layer <<< "$spec"
    echo "=== [$(date '+%H:%M:%S')] $cond α=$alpha dir=$dir L=$layer ==="
    $PY $RUN --model $MODEL --task $TASK \
        --condition "$cond" --alpha "$alpha" --direction "$dir" --layer "$layer" \
        --n $N --gpu $GPU
done

# 2) Specificity: random direction, α=+2σ, L22
echo "=== [$(date '+%H:%M:%S')] random direction control ==="
$PY $RUN --model $MODEL --task $TASK \
    --condition random --alpha +2.0 --direction random --layer 22 \
    --n $N --gpu $GPU

# 3) Specificity: I_BA direction, but at L8 instead of L22
echo "=== [$(date '+%H:%M:%S')] L8 layer specificity ==="
$PY $RUN --model $MODEL --task $TASK \
    --condition L8 --alpha +2.0 --direction i_ba --layer 8 \
    --n $N --gpu $GPU

# 4) Specificity: I_LC direction (different indicator) at L22
echo "=== [$(date '+%H:%M:%S')] ILC indicator specificity ==="
$PY $RUN --model $MODEL --task $TASK \
    --condition ILC --alpha +2.0 --direction i_lc --layer 22 \
    --n $N --gpu $GPU

echo "=== [DONE $(date '+%H:%M:%S')] M3' dose ladder complete ==="

# Aggregate
echo "=== [aggregate $(date '+%H:%M:%S')] ==="
$PY sae_v3_analysis/src/aggregate_m3prime_dose_response.py \
    --model $MODEL --task $TASK
