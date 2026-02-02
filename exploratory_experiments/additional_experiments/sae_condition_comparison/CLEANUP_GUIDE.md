# SAE Condition Comparison: 정리 가이드

**목적**: 불필요한 파일 삭제, 중복 제거, 디렉토리 최적화

---

## 📊 현재 상황

### 디스크 사용량
```
results/
├── four_way_gemma_*.json: 3.3GB × 3개 = 9.9GB
├── four_way_llama_*.json: 619MB
├── interaction_gemma_*.json: 2.9GB × 2개 = 5.8GB
├── interaction_llama_*.json: 534MB
└── Total: ~17GB

logs/: 13개 로그 파일 (각 수 MB)
문서: 12개 마크다운 파일
```

### 중복 파일
1. **Gemma 결과 3벌**: 동일 분석 3회 실행
   - `_20260127_203518.json`
   - `_20260127_203626.json`
   - `_20260127_203709.json`

2. **LLaMA 프롬프트 분석 2벌**:
   - `G_llama_20260201_230709.json`
   - `G_llama_20260201_231035.json`

3. **중복 시각화 스크립트**:
   - `visualize_results.py` (구버전)
   - `visualize_results_improved.py` (신버전)

---

## 🗑️ 삭제 권장 파일

### 1. 중복된 Gemma 결과 (낮은 우선순위 2개)
```bash
# 가장 최신 실행본 유지: _20260127_203709
rm results/condition_comparison_summary_gemma_20260127_203518.json  # -124KB
rm results/condition_comparison_summary_gemma_20260127_203626.json  # -124KB
rm results/four_way_gemma_20260127_203518.json                      # -3.3GB
rm results/four_way_gemma_20260127_203626.json                      # -3.3GB
rm results/interaction_gemma_20260127_203518.json                   # -2.9GB
rm results/variable_vs_fixed_gemma_20260127_203518.json
rm results/variable_vs_fixed_gemma_20260127_203626.json

# 절약: ~9.5GB
```

### 2. 중복된 LLaMA 프롬프트 결과
```bash
# 가장 최신 실행본 유지: _231035
rm results/prompt_component/G_llama_20260201_230709.json
rm logs/prompt_component_llama_20260201_230709.log

# 절약: ~수십 MB
```

### 3. 구버전 스크립트 (백업 후)
```bash
# improved 버전이 더 나음
mv scripts/visualize_results.py scripts/_deprecated/
mv scripts/visualize_all_results.py scripts/_deprecated/
```

### 4. 오래된 로그 (2주 이상 경과)
```bash
# 2026-01-27 이전 로그는 압축 보관
gzip logs/condition_comparison_gemma_20260127_*.log
gzip logs/condition_comparison_llama_20260127_*.log

# 절약: ~50% 압축률
```

---

## 📦 보관 권장 파일

### 최종 결과 (삭제 금지)
```
results/
├── condition_comparison_summary_gemma_20260127_203709.json  # 최신
├── condition_comparison_summary_llama_20260127_150824.json
├── four_way_gemma_20260127_203709.json                      # 최신
├── four_way_llama_20260127_150824.json
├── interaction_gemma_20260127_203709.json                   # 최신
├── interaction_llama_20260127_150824.json
├── variable_vs_fixed_gemma_20260127_203709.json             # 최신
├── variable_vs_fixed_llama_20260127_150824.json
├── two_way_anova_llama_20260202_162718.json
└── figures/*.png  # 모든 그래프
```

### 문서 (모두 보관)
- `EXPERIMENT_SUMMARY.md` (이 정리의 결과)
- `ANALYSIS_ISSUES_REPORT.md` (중요!)
- `INTERACTION_ETA_PROBLEM_EXPLAINED.md` (중요!)
- 기타 11개 마크다운

---

## 🚀 자동 정리 스크립트

```bash
#!/bin/bash
# cleanup_duplicates.sh

set -e

RESULTS_DIR="results"
LOGS_DIR="logs"
SCRIPTS_DIR="scripts"

echo "=== SAE Condition Comparison Cleanup ==="
echo ""

# 백업 디렉토리 생성
BACKUP_DIR="backup_$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# 1. 중복 Gemma 결과 삭제
echo "[1/4] Removing duplicate Gemma results..."
mv "$RESULTS_DIR"/condition_comparison_summary_gemma_20260127_203518.json "$BACKUP_DIR/" 2>/dev/null || true
mv "$RESULTS_DIR"/condition_comparison_summary_gemma_20260127_203626.json "$BACKUP_DIR/" 2>/dev/null || true
mv "$RESULTS_DIR"/four_way_gemma_20260127_203518.json "$BACKUP_DIR/" 2>/dev/null || true
mv "$RESULTS_DIR"/four_way_gemma_20260127_203626.json "$BACKUP_DIR/" 2>/dev/null || true
mv "$RESULTS_DIR"/interaction_gemma_20260127_203518.json "$BACKUP_DIR/" 2>/dev/null || true
mv "$RESULTS_DIR"/variable_vs_fixed_gemma_20260127_203518.json "$BACKUP_DIR/" 2>/dev/null || true
mv "$RESULTS_DIR"/variable_vs_fixed_gemma_20260127_203626.json "$BACKUP_DIR/" 2>/dev/null || true

SAVED=$(du -sh "$BACKUP_DIR" | cut -f1)
echo "  → Moved to backup: $SAVED"

# 2. 중복 LLaMA 프롬프트 결과
echo "[2/4] Removing duplicate LLaMA prompt results..."
mv "$RESULTS_DIR"/prompt_component/G_llama_20260201_230709.json "$BACKUP_DIR/" 2>/dev/null || true
mv "$LOGS_DIR"/prompt_component_llama_20260201_230709.log "$BACKUP_DIR/" 2>/dev/null || true

# 3. 구버전 스크립트 이동
echo "[3/4] Moving deprecated scripts..."
mkdir -p "$SCRIPTS_DIR/_deprecated"
mv "$SCRIPTS_DIR"/visualize_results.py "$SCRIPTS_DIR/_deprecated/" 2>/dev/null || true
mv "$SCRIPTS_DIR"/visualize_all_results.py "$SCRIPTS_DIR/_deprecated/" 2>/dev/null || true

# 4. 로그 압축
echo "[4/4] Compressing old logs..."
gzip -f "$LOGS_DIR"/condition_comparison_gemma_20260127_*.log 2>/dev/null || true
gzip -f "$LOGS_DIR"/condition_comparison_llama_20260127_*.log 2>/dev/null || true

echo ""
echo "=== Cleanup Complete ==="
echo "Backup location: $BACKUP_DIR"
echo ""
echo "To permanently delete backup (after verification):"
echo "  rm -rf $BACKUP_DIR"
```

---

## 🔍 선택적 추가 정리

### 실험이 완전히 끝난 경우
```bash
# Interaction 결과는 재분석 필요 (sparse filtering)
# → 백업 후 삭제 가능
mv results/interaction_*.json backup/
# 절약: 3.4GB (LLaMA) + 5.8GB (Gemma) = 9.2GB

# Four-way 결과는 요약본(summary)에 top features 포함
# → 전체 파일은 재현성 위해 외부 저장소로 이동
rsync -av results/four_way_*.json /mnt/external/llm-addiction-archive/
rm results/four_way_*.json
# 절약: 619MB (LLaMA) + 3.3GB (Gemma) = 3.9GB
```

### 논문 제출 후
```bash
# 모든 중간 결과를 압축 아카이브
tar -czf sae_condition_comparison_results_$(date +%Y%m%d).tar.gz \
  results/*.json \
  logs/*.log \
  configs/*.yaml

# 원본 삭제 (아카이브만 보관)
# 절약: ~15GB → 1-2GB (압축률 90%)
```

---

## 📁 정리 후 디렉토리 구조

```
sae_condition_comparison/
├── src/                      # 소스 코드 (변경 없음)
├── configs/                  # 설정 (변경 없음)
├── scripts/
│   ├── _deprecated/          # 구버전 스크립트
│   └── [활성 스크립트들]
├── results/
│   ├── condition_comparison_summary_llama_*.json       # 125KB
│   ├── condition_comparison_summary_gemma_20260127_203709.json  # 최신
│   ├── four_way_llama_*.json                          # 619MB
│   ├── four_way_gemma_20260127_203709.json            # 3.3GB (최신만)
│   ├── interaction_llama_*.json                       # 534MB
│   ├── interaction_gemma_20260127_203709.json         # 2.9GB (최신만)
│   ├── variable_vs_fixed_llama_*.json
│   ├── variable_vs_fixed_gemma_20260127_203709.json   # 최신만
│   ├── two_way_anova_llama_*.json
│   ├── figures/              # 18개 PNG
│   ├── prompt_component/     # 9개 JSON (중복 제거)
│   ├── prompt_complexity/    # 2개 JSON
│   └── prompt_combo/         # 2개 JSON
├── logs/
│   ├── [활성 로그들]
│   └── [압축된 오래된 로그들].gz
├── backup_20260202/          # 삭제된 파일 백업 (확인 후 제거)
└── *.md                      # 12개 문서 (모두 보관)

절약: ~9.5GB (중복 제거)
최종 크기: ~7.5GB (압축 전)
```

---

## ✅ 정리 체크리스트

### 즉시 실행
- [ ] `cleanup_duplicates.sh` 실행
- [ ] 백업 디렉토리 확인 (파일 무결성)
- [ ] `results/` 크기 확인 (`du -sh results/`)
- [ ] 최신 결과 파일 동작 테스트

### 검증
- [ ] `condition_comparison_summary_*_최신.json` 존재 확인
- [ ] `figures/` 내 모든 PNG 무손실 확인
- [ ] 스크립트 실행 가능 여부 확인

### 후속 조치
- [ ] 백업 7일 후 삭제 (`rm -rf backup_*`)
- [ ] 외부 저장소에 아카이브 생성 (선택)
- [ ] `.gitignore`에 `backup_*/` 추가

---

## 🎯 예상 효과

### 디스크 절약
- **즉시 정리**: ~9.5GB 절약
- **공격적 정리**: ~13GB 절약 (interaction 백업)
- **최종 아카이브**: ~15GB → 1-2GB (압축)

### 개발 효율성
- 최신 결과만 남아 혼동 방지
- 스크립트 디렉토리 정돈
- 로그 압축으로 검색 속도 향상

### 재현성 유지
- 백업 디렉토리로 롤백 가능
- 외부 아카이브로 장기 보관
- 문서는 모두 보존

---

## ⚠️ 주의사항

1. **백업 먼저**: 삭제 전 반드시 백업 생성
2. **검증 후 삭제**: 백업 디렉토리 1주일 보관 후 제거
3. **Interaction 결과**: 재분석 필요하므로 신중히 삭제
4. **Four-way 결과**: 논문 Figure용 원본 데이터, 외부 저장소 권장

---

**작성일**: 2026-02-02
**실행 권장**: 백업 검증 후 안전하게 진행
**백업 보관 기간**: 7일 (이상 없으면 삭제)
