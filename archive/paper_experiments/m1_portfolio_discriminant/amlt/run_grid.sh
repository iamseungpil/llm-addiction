#!/usr/bin/env bash
set -uo pipefail

# M1 portfolio discriminant grid: 2 models x 5 conds x 2 objs x 3 blurbs = 60 cells.

MODE_CHOICE="${1:-open_weight}"
OUTPUT_DIR="/scratch/x3415a02/data/llm-addiction/m1_portfolio"
mkdir -p "${OUTPUT_DIR}" /scratch/logs

if [ "${MODE_CHOICE}" = "open_weight" ]; then
    for model in gemma llama; do
        for cond in BASE +G +M +GM MAX_RISK; do
            for obj in wealth_max capital_preservation; do
                for blurb in salient neutral conservative; do
                    cond_safe=$(echo "${cond}" | sed 's/+/p/g')
                    echo "[m1] running ${model} cond=${cond} obj=${obj} blurb=${blurb}"
                    python paper_experiments/m1_portfolio_discriminant/src/run_m1_open_weight.py \
                        --model "${model}" --gpu 0 --condition "${cond}" \
                        --objective "${obj}" --blurb_variant "${blurb}" \
                        --n_games 200 --output_dir "${OUTPUT_DIR}" \
                        2>&1 | tee -a "/scratch/logs/m1_${model}_${cond_safe}_${obj}_${blurb}.log" || \
                        echo "[m1] cell ${model}/${cond}/${obj}/${blurb} FAILED, continuing"
                done
            done
        done
    done
else
    declare -A IDS=(
        [openai_mini]=gpt-4o-mini
        [anthropic]=claude-3-5-haiku-20241022
        [google]=gemini-2.5-flash
        [openai_full]=gpt-4o
    )
    declare -A PROVIDERS=(
        [openai_mini]=openai
        [anthropic]=anthropic
        [google]=google
        [openai_full]=openai
    )
    for key in openai_mini anthropic google openai_full; do
        for cond in BASE +G +M +GM MAX_RISK; do
            for obj in wealth_max capital_preservation; do
                for blurb in salient neutral conservative; do
                    cond_safe=$(echo "${cond}" | sed 's/+/p/g')
                    echo "[m1_api] running ${key} cond=${cond} obj=${obj} blurb=${blurb}"
                    python paper_experiments/m1_portfolio_discriminant/src/run_m1_api.py \
                        --provider "${PROVIDERS[$key]}" --model_id "${IDS[$key]}" \
                        --condition "${cond}" --objective "${obj}" --blurb_variant "${blurb}" \
                        --n_games 200 --output_dir "${OUTPUT_DIR}" \
                        2>&1 | tee -a "/scratch/logs/m1_${key}_${cond_safe}_${obj}_${blurb}.log" || \
                        echo "[m1_api] cell ${key}/${cond}/${obj}/${blurb} FAILED, continuing"
                done
            done
        done
    done
fi

echo "[m1] grid done"
