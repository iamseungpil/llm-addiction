#!/usr/bin/env bash
set -uo pipefail

# M2 persona-decoupling grid: 2 models x 4 conds x 2 framings = 16 cells (SM only).
# Resumable: skip cells whose .done marker already exists.

MODE_CHOICE="${1:-open_weight}"
OUTPUT_DIR="/scratch/x3415a02/data/llm-addiction/m2_persona"
mkdir -p "${OUTPUT_DIR}/.markers" /scratch/logs

run_cell() {
    local marker="$1"; shift
    local label="$1"; shift
    if [ -f "${OUTPUT_DIR}/.markers/${marker}" ]; then
        echo "[m2] SKIP ${label} (marker exists)"
        return 0
    fi
    echo "[m2] running ${label}"
    "$@" && touch "${OUTPUT_DIR}/.markers/${marker}" || \
        echo "[m2] cell ${label} FAILED, continuing"
}

if [ "${MODE_CHOICE}" = "open_weight" ]; then
    for model in gemma llama; do
        for cond in BASE +G +M +GM; do
            for frame in first_person role_play_gambler; do
                cond_safe=$(echo "${cond}" | sed 's/+/p/g')
                run_cell "${model}_${cond_safe}_${frame}" "${model}/${cond}/${frame}" \
                    bash -c "python paper_experiments/m2_persona_decoupling/src/run_m2_open_weight.py \
                        --model ${model} --gpu 0 --condition ${cond} --framing ${frame} \
                        --task SM --n_games 200 --output_dir ${OUTPUT_DIR} \
                        2>&1 | tee -a /scratch/logs/m2_${model}_${cond_safe}_${frame}.log"
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
        for cond in BASE +G +M +GM; do
            for frame in first_person role_play_gambler; do
                cond_safe=$(echo "${cond}" | sed 's/+/p/g')
                run_cell "${key}_${cond_safe}_${frame}" "${key}/${cond}/${frame}" \
                    bash -c "python paper_experiments/m2_persona_decoupling/src/run_m2_api.py \
                        --provider ${PROVIDERS[$key]} --model_id ${IDS[$key]} \
                        --condition ${cond} --framing ${frame} --task SM \
                        --n_games 200 --output_dir ${OUTPUT_DIR} \
                        2>&1 | tee -a /scratch/logs/m2_${key}_${cond_safe}_${frame}.log"
            done
        done
    done
fi

echo "[m2] grid done"
