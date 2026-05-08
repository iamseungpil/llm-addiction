#!/usr/bin/env bash
set -uo pipefail

# Track 0 W3 grid (Plan v5.2). Three modes:
#   parity     — gpt-4o-mini × 4 caps × 2 modes × 2 sides (v6 + legacy baseline) + parity_check.py
#   open_weight — gemma + llama × 4 caps × 2 modes (Plan v5.2 cross-model arm)
#   api        — 4 API providers × 4 caps × 2 modes (only if user explicitly asks; not Plan v5.2 scope)
#
# Run from /scratch/code/llm-addiction. Resumable via ${OUTPUT_DIR}/.markers/.

MODE_CHOICE="${1:-open_weight}"
OUTPUT_DIR="/scratch/x3415a02/data/llm-addiction/track0_w3"
PARITY_DIR="${OUTPUT_DIR}/parity_legacy_baseline"
mkdir -p "${OUTPUT_DIR}/.markers" "${PARITY_DIR}" /scratch/logs

run_cell() {
    local marker="$1"; shift
    local label="$1"; shift
    if [ -f "${OUTPUT_DIR}/.markers/${marker}" ]; then
        echo "[track0] SKIP ${label} (marker exists)"
        return 0
    fi
    echo "[track0] running ${label}"
    "$@" && touch "${OUTPUT_DIR}/.markers/${marker}" || \
        echo "[track0] cell ${label} FAILED, continuing"
}

if [ "${MODE_CHOICE}" = "parity" ]; then
    # Plan v5.2 §3.5: gpt-4o-mini parity gate (Track 0 v6 vs legacy re-run).
    for cap in 10 30 50 70; do
        for mode in fixed variable; do
            # 1. Track 0 v6 path.
            run_cell "v6_gpt-4o-mini_cap${cap}_${mode}" "v6/gpt-4o-mini/cap${cap}/${mode}" \
                bash -c "python paper_experiments/track0_w3_replication/src/run_track0_api.py \
                    --provider openai --model_id gpt-4o-mini \
                    --cap ${cap} --mode ${mode} \
                    --n_games 200 --output_dir ${OUTPUT_DIR} \
                    2>&1 | tee -a /scratch/logs/track0_v6_gpt-4o-mini_cap${cap}_${mode}.log"
            # 2. Legacy baseline path (untouched legacy classes via __new__ shim).
            run_cell "legacy_gpt-4o-mini_cap${cap}_${mode}" "legacy/gpt-4o-mini/cap${cap}/${mode}" \
                bash -c "python paper_experiments/track0_w3_replication/src/run_legacy_baseline.py \
                    --cap ${cap} --mode ${mode} \
                    --n_games 200 --output_dir ${PARITY_DIR} \
                    2>&1 | tee -a /scratch/logs/track0_legacy_gpt-4o-mini_cap${cap}_${mode}.log"
        done
    done
    # 3. Run parity_check.py to gate cross-model.
    echo "[track0] running parity_check"
    python paper_experiments/track0_w3_replication/src/parity_check.py \
        --v6_dir "${OUTPUT_DIR}" \
        --legacy_dir "${PARITY_DIR}" \
        --output_path "${OUTPUT_DIR}/parity_report.json" \
        2>&1 | tee -a /scratch/logs/track0_parity_check.log

elif [ "${MODE_CHOICE}" = "open_weight" ]; then
    for model in gemma llama; do
        for cap in 10 30 50 70; do
            for mode in fixed variable; do
                run_cell "${model}_cap${cap}_${mode}" "${model}/cap${cap}/${mode}" \
                    bash -c "python paper_experiments/track0_w3_replication/src/run_track0_open_weight.py \
                        --model ${model} --gpu 0 --cap ${cap} --mode ${mode} \
                        --n_games 200 --output_dir ${OUTPUT_DIR} \
                        2>&1 | tee -a /scratch/logs/track0_${model}_cap${cap}_${mode}.log"
            done
        done
    done

elif [ "${MODE_CHOICE}" = "api" ]; then
    # Plan v5.2 §3.1.2 cross-model API arm: paper §3.1 panel APIs minus gpt-4o-mini.
    # gpt-4o-mini handled by parity job; gpt-4.1-mini + claude-haiku + gemini-flash
    # are NEW measurements extending each provider's panel to cap variation.
    # Each provider keeps its own panel system msg + sampling per Plan v5.2 §3.1.2
    # — see OPENAI_PROTOCOL dispatch in run_track0_api.py.
    declare -A IDS=(
        [gpt41_mini]=gpt-4.1-mini
        [claude_haiku]=claude-3-5-haiku-20241022
        [gemini_flash]=gemini-2.5-flash
    )
    declare -A PROVIDERS=(
        [gpt41_mini]=openai
        [claude_haiku]=anthropic
        [gemini_flash]=google
    )
    for key in gpt41_mini claude_haiku gemini_flash; do
        for cap in 10 30 50 70; do
            for mode in fixed variable; do
                run_cell "${key}_cap${cap}_${mode}" "${key}/cap${cap}/${mode}" \
                    bash -c "python paper_experiments/track0_w3_replication/src/run_track0_api.py \
                        --provider ${PROVIDERS[$key]} --model_id ${IDS[$key]} \
                        --cap ${cap} --mode ${mode} \
                        --n_games 200 --output_dir ${OUTPUT_DIR} \
                        2>&1 | tee -a /scratch/logs/track0_${key}_cap${cap}_${mode}.log"
            done
        done
    done

else
    echo "[track0] unknown MODE_CHOICE='${MODE_CHOICE}' — expected one of: parity, open_weight, api"
    exit 1
fi

echo "[track0] grid done (mode=${MODE_CHOICE})"
