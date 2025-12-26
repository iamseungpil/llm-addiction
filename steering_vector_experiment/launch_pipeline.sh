#!/bin/bash
# =============================================================================
# Unified Pipeline Launcher for LLM Gambling Behavior Causal Analysis
# =============================================================================
#
# This script launches the 5-phase causal analysis pipeline.
#
# Direct steering mode:
#   Uses steering vectors for causal validation, SAE only for interpretation.
#   Phases:
#     1: Steering Vector Extraction
#     2: Direct Steering Validation
#     3: SAE Interpretation
#
# Phases:
#   1 (A): Steering Vector Extraction
#   2 (C): SAE Feature Projection
#   3 (B): Soft Interpolation Patching (dose-response validation)
#   4 (D): Head Patching (attention mechanism analysis)
#   5: Gambling-Context Interpretation
#
# Usage:
#   # Full pipeline for LLaMA on GPU 0 (SAE patching pipeline)
#   ./launch_pipeline.sh --model llama --gpu 0
#
#   # Full pipeline for Gemma on GPU 1 (SAE patching pipeline)
#   ./launch_pipeline.sh --model gemma --gpu 1
#
#   # Direct steering pipeline (no SAE patching)
#   ./launch_pipeline.sh --mode direct --model llama --gpu 0
#
#   # Start from Phase 3
#   ./launch_pipeline.sh --model llama --gpu 0 --start-phase 3
#
#   # Run only specific phases
#   ./launch_pipeline.sh --model llama --gpu 0 --phases 1,2,5
#
#   # Individual phase scripts
#   ./launch_pipeline.sh --model llama --gpu 0 --only-phase 1
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL=""
GPU=""
START_PHASE=1
PHASES=""
ONLY_PHASE=""
CONFIG=""
DRY_RUN=false
NO_STRICT=false
MODE="sae"  # sae (default) or direct

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_DIR="${SCRIPT_DIR}/configs"
DEFAULT_CONFIG="${CONFIG_DIR}/experiment_config.yaml"
DIRECT_CONFIG="${CONFIG_DIR}/experiment_config_direct_steering.yaml"
DIRECT_OUTPUT_DIR="/data/llm_addiction/steering_vector_experiment_direct"

# Conda environment
CONDA_ENV="llama_sae_env"

# Print usage
usage() {
    echo "Usage: $0 --model <llama|gemma> --gpu <gpu_id> [options]"
    echo ""
    echo "Required:"
    echo "  --model <name>     Model to analyze (llama or gemma)"
    echo "  --gpu <id>         GPU ID to use"
    echo ""
    echo "Optional:"
    echo "  --mode <type>      Pipeline mode: sae (default) or direct"
    echo "  --start-phase <n>  Phase to start from (1-5, default: 1)"
    echo "  --phases <list>    Comma-separated phases to run (e.g., '1,2,5')"
    echo "  --only-phase <n>   Run only a single phase"
    echo "  --config <path>    Path to config file (default: experiment_config.yaml)"
    echo "  --dry-run          Print commands without executing"
    echo "  --no-strict        Continue on phase failures"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --model llama --gpu 0"
    echo "  $0 --model gemma --gpu 1 --start-phase 3"
    echo "  $0 --model llama --gpu 0 --phases 1,2,5"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --start-phase)
            START_PHASE="$2"
            shift 2
            ;;
        --phases)
            PHASES="$2"
            shift 2
            ;;
        --only-phase)
            ONLY_PHASE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-strict)
            NO_STRICT=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL" ]; then
    echo -e "${RED}Error: --model is required${NC}"
    usage
fi

if [ -z "$GPU" ]; then
    echo -e "${RED}Error: --gpu is required${NC}"
    usage
fi

if [[ "$MODEL" != "llama" && "$MODEL" != "gemma" ]]; then
    echo -e "${RED}Error: --model must be 'llama' or 'gemma'${NC}"
    usage
fi

# Validate mode
if [[ "$MODE" != "sae" && "$MODE" != "direct" ]]; then
    echo -e "${RED}Error: --mode must be 'sae' or 'direct'${NC}"
    usage
fi

# Set config path
if [ -z "$CONFIG" ]; then
    if [ "$MODE" = "direct" ]; then
        CONFIG="$DIRECT_CONFIG"
    else
        CONFIG="$DEFAULT_CONFIG"
    fi
fi

if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG${NC}"
    exit 1
fi

# Print banner
echo -e "${BLUE}"
echo "============================================================================="
echo "  5-PHASE CAUSAL ANALYSIS PIPELINE"
echo "  LLM Gambling Behavior Research"
echo "============================================================================="
echo -e "${NC}"
echo -e "Model:        ${GREEN}${MODEL}${NC}"
echo -e "GPU:          ${GREEN}${GPU}${NC}"
echo -e "Mode:         ${GREEN}${MODE}${NC}"
echo -e "Config:       ${GREEN}${CONFIG}${NC}"
echo -e "Start Phase:  ${GREEN}${START_PHASE}${NC}"
if [ -n "$PHASES" ]; then
    echo -e "Phases:       ${GREEN}${PHASES}${NC}"
fi
if [ -n "$ONLY_PHASE" ]; then
    echo -e "Only Phase:   ${GREEN}${ONLY_PHASE}${NC}"
fi
echo ""

# Activate conda environment
echo -e "${YELLOW}Activating conda environment: ${CONDA_ENV}${NC}"
if [ -f "/home/ubuntu/miniconda3/etc/profile.d/conda.sh" ]; then
    source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
    conda activate ${CONDA_ENV}
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate ${CONDA_ENV}
else
    echo -e "${YELLOW}Warning: Could not find conda.sh, assuming environment is active${NC}"
fi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU

# Function to run individual phase scripts
run_phase() {
    local phase_num=$1
    local script=""
    local extra_args=""

    if [ "$MODE" = "direct" ]; then
        case $phase_num in
            1)
                script="extract_steering_vectors.py"
                ;;
            2)
                script="phase2_direct_steering.py"
                ;;
            3)
                script="phase3_sae_interpretation.py"
                # Find latest Phase 2 results for causal layer selection
                local phase2_file=$(ls -t ${DIRECT_OUTPUT_DIR}/phase2_results/layer_validations_${MODEL}_*.json 2>/dev/null | head -n1)
                if [ -n "$phase2_file" ]; then
                    extra_args="--phase2-results $phase2_file"
                fi
                ;;
            4)
                script="phase4_head_component_analysis.py"
                # Use Phase 2 results to select layers if available
                local phase2_file=$(ls -t ${DIRECT_OUTPUT_DIR}/phase2_results/layer_validations_${MODEL}_*.json 2>/dev/null | head -n1)
                if [ -n "$phase2_file" ]; then
                    extra_args="--phase2-results $phase2_file"
                fi
                ;;
            5)
                script="phase5_activation_patching.py"
                ;;
            *)
                echo -e "${RED}Direct mode supports phases 1-5 only${NC}"
                return 1
                ;;
            *)
        esac
    else
        case $phase_num in
            1)
                script="extract_steering_vectors.py"
                ;;
            2)
                script="sae_feature_projection.py"
                # Find latest steering vectors file
                local vectors_file=$(ls -t /data/llm_addiction/steering_vector_experiment/steering_vectors_${MODEL}_*.npz 2>/dev/null | head -n1)
                if [ -n "$vectors_file" ]; then
                    extra_args="--vectors $vectors_file"
                fi
                ;;
            3)
                script="soft_interpolation_patching.py"
                # Find latest candidate features file
                local candidates_file=$(ls -t /data/llm_addiction/steering_vector_experiment/candidate_features_${MODEL}_*.json 2>/dev/null | head -n1)
                if [ -z "$candidates_file" ]; then
                    # Fall back to sae_analysis file
                    candidates_file=$(ls -t /data/llm_addiction/steering_vector_experiment/sae_analysis_${MODEL}_*.json 2>/dev/null | head -n1)
                fi
                if [ -n "$candidates_file" ]; then
                    extra_args="--candidates $candidates_file"
                fi
                ;;
            4)
                script="head_patching.py"
                # Find latest validated features file
                local validated_file=$(ls -t /data/llm_addiction/steering_vector_experiment/validated_features_${MODEL}_*.json 2>/dev/null | head -n1)
                if [ -n "$validated_file" ]; then
                    extra_args="--validated $validated_file"
                fi
                ;;
            5)
                script="gambling_interpretation.py"
                # Find latest validated features file
                local validated_file=$(ls -t /data/llm_addiction/steering_vector_experiment/validated_features_${MODEL}_*.json 2>/dev/null | head -n1)
                if [ -n "$validated_file" ]; then
                    extra_args="--validated $validated_file"
                fi
                ;;
            *)
                echo -e "${RED}Unknown phase: $phase_num${NC}"
                return 1
                ;;
        esac
    fi

    echo -e "\n${BLUE}=============================================================================${NC}"
    echo -e "${BLUE}  PHASE ${phase_num}: Running ${script}${NC}"
    echo -e "${BLUE}=============================================================================${NC}"

    local cmd="python ${SRC_DIR}/${script} --model ${MODEL} --gpu ${GPU} --config ${CONFIG} ${extra_args}"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN] Would execute: ${cmd}${NC}"
        return 0
    fi

    echo -e "${GREEN}Executing: ${cmd}${NC}"
    eval $cmd
    local exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo -e "${RED}Phase ${phase_num} failed with exit code ${exit_code}${NC}"
        return $exit_code
    fi

    echo -e "${GREEN}Phase ${phase_num} completed successfully${NC}"
    return 0
}

# Main execution
if [ -n "$ONLY_PHASE" ]; then
    # Run only a single phase
    echo -e "${YELLOW}Running only Phase ${ONLY_PHASE}${NC}"
    run_phase $ONLY_PHASE
    exit_code=$?
elif [ -n "$PHASES" ]; then
    # Run specific phases
    echo -e "${YELLOW}Running phases: ${PHASES}${NC}"
    IFS=',' read -ra PHASE_ARRAY <<< "$PHASES"
    for phase in "${PHASE_ARRAY[@]}"; do
        run_phase $phase
        exit_code=$?
        if [ $exit_code -ne 0 ] && [ "$NO_STRICT" = false ]; then
            echo -e "${RED}Stopping due to phase ${phase} failure${NC}"
            exit $exit_code
        fi
    done
else
    if [ "$MODE" = "direct" ]; then
        echo -e "${YELLOW}Running direct steering pipeline (phases 1-5)${NC}"
        for phase in 1 2 3 4 5; do
            run_phase $phase
            exit_code=$?
            if [ $exit_code -ne 0 ] && [ "$NO_STRICT" = false ]; then
                echo -e "${RED}Stopping due to phase ${phase} failure${NC}"
                exit $exit_code
            fi
        done
    else
        # Use the Python orchestrator for full pipeline
        echo -e "${YELLOW}Running full pipeline via orchestrator${NC}"

        cmd="python ${SRC_DIR}/run_full_pipeline.py --model ${MODEL} --gpu ${GPU} --config ${CONFIG} --start-phase ${START_PHASE}"

        if [ "$NO_STRICT" = true ]; then
            cmd="${cmd} --no-strict"
        fi

        if [ "$DRY_RUN" = true ]; then
            cmd="${cmd} --dry-run"
        fi

        echo -e "${GREEN}Executing: ${cmd}${NC}"
        eval $cmd
        exit_code=$?
    fi
fi

# Print final summary
echo ""
echo -e "${BLUE}=============================================================================${NC}"
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}  PIPELINE COMPLETED SUCCESSFULLY${NC}"
else
    echo -e "${RED}  PIPELINE COMPLETED WITH ERRORS (exit code: ${exit_code})${NC}"
fi
echo -e "${BLUE}=============================================================================${NC}"

echo ""
if [ "$MODE" = "direct" ]; then
    echo "Output directory: ${DIRECT_OUTPUT_DIR}"
    echo "Logs directory: ${DIRECT_OUTPUT_DIR}/logs/"
else
    echo "Output directory: /data/llm_addiction/steering_vector_experiment/"
    echo "Logs directory: /data/llm_addiction/steering_vector_experiment/logs/"
fi
echo ""

exit $exit_code
