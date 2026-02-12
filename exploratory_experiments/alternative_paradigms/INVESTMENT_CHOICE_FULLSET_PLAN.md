# Investment Choice 풀세트 실험 계획

## 실험 개요

**Quick mode 결과 검증 완료** (2026-02-04):
- LLaMA-3.1-8B: 160 games, 31.2% bankruptcy rate
- Gemma-2-9B: 160 games, 25.0% bankruptcy rate
- **문제 없음**: 파싱 오류 없음, 모든 게임 정상 작동

## 풀세트 실험 규모

### Quick Mode vs Full Mode 비교

| 항목 | Quick Mode | Full Mode |
|------|-----------|-----------|
| Bet types | 2 (variable, fixed) | 2 (variable, fixed) |
| Prompt conditions | 4 (BASE, G, M, GM) | 4 (BASE, G, M, GM) |
| Repetitions | 20 | **50** |
| **Total games** | **160** | **400** |

### 실험 조건 상세

**Prompt Conditions** (4가지):
1. **BASE**: 기본 투자 선택
2. **G** (Goal): 자기설정 목표 추가
3. **M** (Monitoring): 자기감시 추가
4. **GM** (Goal + Monitoring): 둘 다 적용

**Bet Types** (2가지):
1. **Variable**: 베팅 금액 자유 설정
2. **Fixed**: 고정 베팅 (balance의 10%)

**Bet Constraint**: `unlimited` (잔액 이내 자유)

**Investment Options**:
- Choice 1: Safe Low (100% win $20)
- Choice 2: Safe High (100% win $40)
- Choice 3: Risky Low (50% win $60, 50% lose all)
- Choice 4: Risky High (50% win $120, 50% lose all)

## 실험 실행 계획

### Phase 1: LLaMA-3.1-8B Full Experiment

```bash
# Interactive GPU session
srun -p cas_v100_4 --gres=gpu:1 --time=04:00:00 --pty bash

# Activate environment
conda activate llama_sae_env
cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms

# Run full experiment (400 games)
python src/investment_choice/run_experiment.py \
  --model llama \
  --gpu 0 \
  --constraint unlimited

# Expected output:
# /scratch/x3415a02/data/llm-addiction/investment_choice/llama_investment_unlimited_YYYYMMDD_HHMMSS.json
```

**예상 실행 시간**:
- Quick mode (160 games): ~20분
- Full mode (400 games): **~50분** (2.5배)

### Phase 2: Gemma-2-9B Full Experiment

```bash
# Same session or new session
python src/investment_choice/run_experiment.py \
  --model gemma \
  --gpu 0 \
  --constraint unlimited

# Expected output:
# /scratch/x3415a02/data/llm-addiction/investment_choice/gemma_investment_unlimited_YYYYMMDD_HHMMSS.json
```

**예상 실행 시간**: ~60분 (Gemma가 평균 라운드 수가 더 많음)

### Phase 3: Optional - Constraint Variations

현재는 `unlimited` 조건만 테스트. 추가로 테스트할 수 있는 조건:

```bash
# Constraint variations (optional)
python src/investment_choice/run_experiment.py --model llama --gpu 0 --constraint 50
python src/investment_choice/run_experiment.py --model gemma --gpu 0 --constraint 50

# Or test specific bet types
python src/investment_choice/run_experiment.py --model llama --gpu 0 --bet-type fixed
```

## SLURM 배치 작업 스크립트

장시간 실험을 위한 SLURM 스크립트:

```bash
#!/bin/bash
#SBATCH -J investment_fullset
#SBATCH -p cas_v100_4
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/x3415a02/data/llm-addiction/logs/investment_fullset_%j.out
#SBATCH --error=/scratch/x3415a02/data/llm-addiction/logs/investment_fullset_%j.err

# Activate conda environment
source ~/.bashrc
conda activate llama_sae_env

# Navigate to project directory
cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms

# Run LLaMA full experiment
echo "Starting LLaMA-3.1-8B full experiment..."
python src/investment_choice/run_experiment.py \
  --model llama \
  --gpu 0 \
  --constraint unlimited

# Clear GPU memory
sleep 10

# Run Gemma full experiment
echo "Starting Gemma-2-9B full experiment..."
python src/investment_choice/run_experiment.py \
  --model gemma \
  --gpu 0 \
  --constraint unlimited

echo "All experiments completed!"
```

**사용법**:
```bash
# Save script
nano run_investment_fullset.sh
# (paste above script)

# Submit job
sbatch run_investment_fullset.sh

# Monitor
squeue -u $USER
tail -f /scratch/x3415a02/data/llm-addiction/logs/investment_fullset_*.out
```

## GPU 리소스 요구사항

| 모델 | VRAM 사용량 | 권장 GPU |
|------|-----------|----------|
| LLaMA-3.1-8B | ~19GB | V100 32GB |
| Gemma-2-9B | ~22GB | V100 32GB |

**참고**: 두 모델 모두 bf16으로 로드됨

## 예상 출력 파일

```
/scratch/x3415a02/data/llm-addiction/investment_choice/
├── llama_investment_unlimited_20260204_203537.json  # Quick mode (existing)
├── gemma_investment_unlimited_20260204_202628.json  # Quick mode (existing)
├── llama_investment_unlimited_YYYYMMDD_HHMMSS.json  # Full mode (NEW)
└── gemma_investment_unlimited_YYYYMMDD_HHMMSS.json  # Full mode (NEW)
```

각 파일 크기: ~700KB (160 games) → **~1.8MB (400 games 예상)**

## 분석 계획

풀세트 실험 완료 후 분석할 주요 지표:

### 1. Behavioral Metrics
- **Bankruptcy rate**: 모델별, 조건별 비교
- **Risk preference**: Safe vs Risky 선택 비율
- **Investment patterns**: Choice 분포 (1, 2, 3, 4)
- **Rounds distribution**: 게임 지속 라운드 수

### 2. Condition Effects
- **BASE vs G vs M vs GM**: 각 조건의 효과
- **Variable vs Fixed**: 베팅 타입의 영향
- **Goal escalation**: G 조건에서 목표 변화 패턴

### 3. Model Comparison
- **LLaMA vs Gemma**: 리스크 선호도 차이
- **Strategy differences**: 베팅 패턴, 게임 지속성

### 4. Statistical Tests
- Chi-square test: 선택 분포 비교
- T-test: 평균 final balance, rounds 비교
- Effect size: Cohen's d 계산

## 체크리스트

**실험 전**:
- [ ] Quick mode 결과 확인 완료 (✅ Done)
- [ ] GPU 가용성 확인 (`nvidia-smi`)
- [ ] Conda 환경 활성화 확인
- [ ] 디스크 공간 확인 (`df -h /scratch/x3415a02/`)

**실험 중**:
- [ ] 로그 모니터링 (`tail -f`)
- [ ] GPU 사용률 확인 (`watch -n 1 nvidia-smi`)
- [ ] 중간 체크포인트 확인 (checkpoint JSON 파일)

**실험 후**:
- [ ] 출력 파일 생성 확인
- [ ] 파일 크기 검증 (~1.8MB)
- [ ] JSON 형식 검증
- [ ] 400 games 완료 확인
- [ ] 분석 스크립트 실행

## 예상 일정

| 단계 | 작업 | 예상 시간 |
|------|------|-----------|
| 1 | LLaMA full experiment | 50분 |
| 2 | Gemma full experiment | 60분 |
| 3 | 결과 검증 | 10분 |
| **Total** | | **~2시간** |

**권장**: Interactive session (4시간) 또는 SLURM batch job (6시간 limit)

## 참고 문헌

- Paper experiment: `paper_experiments/investment_choice_experiment/`
- Alternative paradigms: `exploratory_experiments/alternative_paradigms/src/investment_choice/`
- Quick mode 결과: `/scratch/x3415a02/data/llm-addiction/investment_choice/`

## 다음 단계

1. ✅ Quick mode 검증 완료
2. **풀세트 실험 실행** (이 계획서 따라)
3. 결과 분석 및 시각화
4. 다른 paradigm과 비교 (lootbox, blackjack)
5. SAE 분석 (optional - Phase 1 feature extraction)

---

**작성일**: 2026-02-12
**작성자**: Claude Code
**상태**: Ready to Execute
