#!/bin/bash

# Phase 4 완료를 기다렸다가 자동으로 분석 실행
# 사용법: ./wait_and_analyze.sh

OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase4_word_feature_FULL"

echo "========================================================================="
echo "Phase 4 완료 대기 및 자동 분석 시작"
echo "========================================================================="
echo ""

# Phase 4 프로세스 PID 확인
pids=$(ps aux | grep "phase4_word_feature_correlation.py" | grep -v grep | awk '{print $2}')

if [ -z "$pids" ]; then
    echo "⚠️  Phase 4 프로세스가 실행 중이 아닙니다!"
    echo "분석을 바로 시작합니다..."
else
    echo "Phase 4 프로세스 확인됨: $pids"
    echo "완료를 기다리는 중..."
    echo ""

    # 모든 Phase 4 프로세스가 종료될 때까지 대기
    while true; do
        running_count=$(ps aux | grep "phase4_word_feature_correlation.py" | grep -v grep | wc -l)

        if [ $running_count -eq 0 ]; then
            echo ""
            echo "✅ Phase 4 완료!"
            break
        fi

        # 진행 상황 출력
        timestamp=$(date '+%H:%M:%S')
        echo "[$timestamp] 아직 실행 중... (프로세스: $running_count개)"

        sleep 30  # 30초마다 체크
    done
fi

echo ""
echo "========================================================================="
echo "Phase 4 결과 검증 중..."
echo "========================================================================="

# 결과 파일 확인
echo ""
echo "=== 결과 파일 크기 ==="
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null

echo ""
echo "=== Coverage 검증 ==="
python3 << 'EOF'
import json
from pathlib import Path

all_phase4_features = set()
total_correlations = 0

for gpu in [4, 5, 6, 7]:
    file = Path("/data/llm_addiction/experiment_pathway_token_analysis/results/phase4_word_feature_FULL/word_feature_correlation_gpu{}.json".format(gpu))

    if not file.exists():
        print(f"❌ GPU {gpu} 파일 없음!")
        continue

    with open(file, 'r') as f:
        data = json.load(f)

    gpu_features = set()
    for corr in data['word_feature_correlations']:
        gpu_features.add(corr['feature'])
        all_phase4_features.add(corr['feature'])
        total_correlations += 1

    print(f"GPU {gpu}: {len(gpu_features):,}개 features, {len(data['word_feature_correlations']):,}개 correlations")

print(f"\n전체: {len(all_phase4_features):,}개 고유 features, {total_correlations:,}개 correlations")

# Phase 1과 비교
all_phase1_features = set()
for gpu in [4, 5, 6, 7]:
    file = Path("/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching_full/phase1_patching_multifeature_gpu{}.jsonl".format(gpu))

    with open(file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                all_phase1_features.add(data['target_feature'])

print(f"\nPhase 1 전체: {len(all_phase1_features):,}개")
print(f"Phase 4 커버: {len(all_phase4_features):,}개 ({100*len(all_phase4_features)/len(all_phase1_features):.1f}%)")

if len(all_phase4_features) == len(all_phase1_features):
    print("\n✅ 100% 커버리지 달성!")
else:
    missing = len(all_phase1_features) - len(all_phase4_features)
    print(f"\n⚠️  {missing:,}개 features 누락 ({100*missing/len(all_phase1_features):.1f}%)")
EOF

echo ""
echo "========================================================================="
echo "단어-Risky/Safe 분석 시작"
echo "========================================================================="
echo ""

# 분석 스크립트 실행
cd /home/ubuntu/llm_addiction/experiment_pathway_token_analysis/visualizations
python3 word_feature_risky_safe_analysis.py

echo ""
echo "========================================================================="
echo "✅ 모든 작업 완료!"
echo "========================================================================="
