#!/bin/bash
# SAE Condition Comparison: 중복 파일 정리 스크립트
# 생성일: 2026-02-02

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 디렉토리 설정
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
RESULTS_DIR="$BASE_DIR/results"
LOGS_DIR="$BASE_DIR/logs"
SCRIPTS_DIR="$BASE_DIR/scripts"

cd "$BASE_DIR"

echo -e "${BLUE}=== SAE Condition Comparison Cleanup ===${NC}"
echo "Base directory: $BASE_DIR"
echo ""

# 백업 디렉토리 생성
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo -e "${GREEN}Created backup directory: $BACKUP_DIR${NC}"
echo ""

# 현재 디스크 사용량
echo -e "${YELLOW}[Disk Usage] Before cleanup:${NC}"
du -sh "$RESULTS_DIR" 2>/dev/null || echo "Results directory not found"
du -sh "$LOGS_DIR" 2>/dev/null || echo "Logs directory not found"
echo ""

# 1. 중복 Gemma 결과 삭제
echo -e "${BLUE}[1/5] Removing duplicate Gemma results...${NC}"
GEMMA_FILES=(
    "$RESULTS_DIR/condition_comparison_summary_gemma_20260127_203518.json"
    "$RESULTS_DIR/condition_comparison_summary_gemma_20260127_203626.json"
    "$RESULTS_DIR/four_way_gemma_20260127_203518.json"
    "$RESULTS_DIR/four_way_gemma_20260127_203626.json"
    "$RESULTS_DIR/interaction_gemma_20260127_203518.json"
    "$RESULTS_DIR/variable_vs_fixed_gemma_20260127_203518.json"
    "$RESULTS_DIR/variable_vs_fixed_gemma_20260127_203626.json"
)

GEMMA_COUNT=0
GEMMA_SIZE=0
for file in "${GEMMA_FILES[@]}"; do
    if [ -f "$file" ]; then
        SIZE=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo 0)
        GEMMA_SIZE=$((GEMMA_SIZE + SIZE))
        mv "$file" "$BACKUP_DIR/"
        GEMMA_COUNT=$((GEMMA_COUNT + 1))
        echo "  ✓ Moved: $(basename "$file")"
    fi
done
echo -e "  ${GREEN}→ Moved $GEMMA_COUNT files (~$(numfmt --to=iec-i --suffix=B $GEMMA_SIZE 2>/dev/null || echo "$((GEMMA_SIZE / 1024 / 1024))MB"))${NC}"
echo ""

# 2. 중복 LLaMA 프롬프트 결과
echo -e "${BLUE}[2/5] Removing duplicate LLaMA prompt results...${NC}"
LLAMA_FILES=(
    "$RESULTS_DIR/prompt_component/G_llama_20260201_230709.json"
    "$LOGS_DIR/prompt_component_llama_20260201_230709.log"
)

LLAMA_COUNT=0
for file in "${LLAMA_FILES[@]}"; do
    if [ -f "$file" ]; then
        mv "$file" "$BACKUP_DIR/"
        LLAMA_COUNT=$((LLAMA_COUNT + 1))
        echo "  ✓ Moved: $(basename "$file")"
    fi
done
echo -e "  ${GREEN}→ Moved $LLAMA_COUNT files${NC}"
echo ""

# 3. 구버전 스크립트 이동
echo -e "${BLUE}[3/5] Moving deprecated scripts...${NC}"
mkdir -p "$SCRIPTS_DIR/_deprecated"

DEPRECATED_SCRIPTS=(
    "$SCRIPTS_DIR/visualize_results.py"
    "$SCRIPTS_DIR/visualize_all_results.py"
)

SCRIPT_COUNT=0
for script in "${DEPRECATED_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        mv "$script" "$SCRIPTS_DIR/_deprecated/"
        SCRIPT_COUNT=$((SCRIPT_COUNT + 1))
        echo "  ✓ Moved: $(basename "$script")"
    fi
done
echo -e "  ${GREEN}→ Moved $SCRIPT_COUNT scripts to _deprecated/${NC}"
echo ""

# 4. 로그 압축
echo -e "${BLUE}[4/5] Compressing old logs...${NC}"
LOG_PATTERNS=(
    "condition_comparison_gemma_20260127_*.log"
    "condition_comparison_llama_20260127_*.log"
)

LOG_COUNT=0
for pattern in "${LOG_PATTERNS[@]}"; do
    for log in "$LOGS_DIR"/$pattern; do
        if [ -f "$log" ] && [[ ! "$log" =~ \.gz$ ]]; then
            gzip -f "$log"
            LOG_COUNT=$((LOG_COUNT + 1))
            echo "  ✓ Compressed: $(basename "$log")"
        fi
    done
done
echo -e "  ${GREEN}→ Compressed $LOG_COUNT log files${NC}"
echo ""

# 5. 백업 디렉토리 요약
echo -e "${BLUE}[5/5] Backup summary...${NC}"
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
BACKUP_FILES=$(find "$BACKUP_DIR" -type f | wc -l)
echo "  Location: $BACKUP_DIR"
echo "  Files: $BACKUP_FILES"
echo "  Size: $BACKUP_SIZE"
echo ""

# 정리 후 디스크 사용량
echo -e "${YELLOW}[Disk Usage] After cleanup:${NC}"
du -sh "$RESULTS_DIR" 2>/dev/null || echo "Results directory not found"
du -sh "$LOGS_DIR" 2>/dev/null || echo "Logs directory not found"
echo ""

# 최종 메시지
echo -e "${GREEN}=== Cleanup Complete ===${NC}"
echo ""
echo -e "${YELLOW}IMPORTANT:${NC}"
echo "1. Verify backup integrity before deletion"
echo "2. Backup location: $BACKUP_DIR"
echo "3. Keep backup for 7 days"
echo ""
echo -e "${RED}To permanently delete backup (after verification):${NC}"
echo "  rm -rf $BACKUP_DIR"
echo ""
echo -e "${BLUE}To restore from backup:${NC}"
echo "  cp -r $BACKUP_DIR/* $RESULTS_DIR/"
echo "  cp -r $BACKUP_DIR/* $LOGS_DIR/"
echo ""
