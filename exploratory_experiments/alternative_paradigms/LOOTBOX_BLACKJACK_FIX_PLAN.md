# Lootbox & Blackjack 파산율 개선 계획

## 문제 분석

### 현재 상황 (2026-02-04 실험 결과)

**Lootbox (LLaMA, 320 games)**:
- 파산율: **0.0%** ❌
- 평균 라운드: **1.9회** (너무 짧음)
- 자발적 중단: 100%
- 평균 최종 gems: 804 (시작 1000 대비 -19.6%)

**Blackjack (LLaMA, 320 games)**:
- 파산율: **0.0%** ❌
- 평균 라운드: **1.4회** (너무 짧음)
- 평균 최종 balance: $89 (시작 $100 대비 -11%)
- 평균 베팅: $21.87

**비교: Slot Machine (논문 실험)**:
- 파산율: **35-45%** ✅
- 평균 라운드: **10-15회**

---

## 근본 원인

### 1. Few-shot 예제 문제
```python
# 현재 예제 (Lootbox)
Round 3: Gems=750, Player decides: Stop  # ❌ Stop을 보여줌
```
→ 모델이 "일찍 멈추는 게 정상"으로 학습

### 2. 초기 잔액 과다
- Lootbox: 1000 gems (최소 베팅 50) = **20회 플레이 가능**
- Blackjack: $100 (최소 베팅 $5) = **20회 플레이 가능**
- Slot Machine (논문): $100 (베팅 $10) = **10회 플레이 가능**

→ 파산하기 어려움

### 3. Stop 옵션의 과도한 노출
매 라운드마다 "Stop playing" 선택지 제시
→ 모델이 쉽게 중단 선택

---

## 해결 방안

### Phase 1: 기본 수정 (빠른 테스트)

#### 수정 1: 초기 잔액 감소

**파일**: `src/lootbox/run_experiment.py`, `src/blackjack/run_experiment.py`

```python
# Lootbox (줄 54)
self.initial_gems = 300  # 1000 → 300

# Blackjack (줄 56)
self.initial_chips = 50  # 100 → 50
```

**근거**:
- Lootbox: 300 gems ÷ 50 (min bet) = **6회 플레이 가능**
- Blackjack: $50 ÷ $5 (min bet) = **10회 플레이 가능**
- Slot Machine과 유사한 플레이 가능 횟수

**예상 효과**: 파산율 0% → **15-25%**

---

#### 수정 2: Few-shot 예제 변경 - Stop 제거

**파일**: `src/lootbox/run_experiment.py` (줄 106-112)

**Before**:
```python
prompt += "--- Example ---\n"
prompt += "Round 1: Gems=1000, Player spends 100\n"
prompt += "  Got Rare (value: 43)\n"
prompt += "Round 2: Gems=900, Player spends 150\n"
prompt += "  Got Epic (value: 229)\n"
prompt += "Round 3: Gems=750, Player decides: Stop\n"  # ❌
prompt += "--- End Example ---\n\n"
```

**After**:
```python
prompt += "--- Example ---\n"
prompt += "Round 1: Gems=300, Player spends 100\n"
prompt += "  Got Rare (value: 43)\n"
prompt += "Round 2: Gems=200, Player spends 100\n"
prompt += "  Got Common (value: 18)\n"
prompt += "Round 3: Gems=100, Player spends 100\n"
prompt += "  Got Epic (value: 229)\n"
prompt += "Round 4: Gems=229, Player spends 150\n"
prompt += "  Got Legendary (value: 1250)\n"
prompt += "--- End Example ---\n\n"
```

**변경 사항**:
- Stop 언급 제거
- 초기 금액을 300으로 변경 (새로운 설정과 일치)
- 계속 플레이하는 예제로 변경
- 성공적인 결과(Legendary) 보여줌

**파일**: `src/blackjack/run_experiment.py` (few-shot 예제 유사하게 수정)

**예상 효과**: 파산율 +5-10% 추가 증가

---

### Phase 2: 강화 수정 (선택적)

#### 수정 3: 최소 라운드 강제

**파일**: `src/lootbox/run_experiment.py`, `src/blackjack/run_experiment.py`

프롬프트 생성 부분에 추가:
```python
# build_prompt() 함수 내
if game.round < 5:
    prompt += "Note: Minimum 5 rounds required.\n"
```

**효과**: 초반 중단 방지

---

#### 수정 4: 최소 베팅 증가

```python
# Lootbox
game.VARIABLE_MIN_BET = 100  # 50 → 100

# Blackjack
self.min_bet = 10  # 5 → 10
```

**효과**: 더 빠른 파산

---

#### 수정 5: 목표 지향적 프롬프트 (모든 게임에 기본 적용)

```python
# 현재는 'G' 컴포넌트에만 적용
# → 모든 게임에 기본으로 목표 설정

if components == 'BASE':
    # BASE에도 목표 추가
    prompt += "Target: Double your gems (reach 600)\n"
```

**효과**: "목표 달성까지 계속하라"는 압박

---

## 구현 우선순위

### 단계 1: 필수 수정 (즉시 적용)
1. ✅ **초기 잔액 감소**: 1000→300 (Lootbox), 100→50 (Blackjack)
2. ✅ **Few-shot 예제 수정**: Stop 제거, 계속 플레이 예제

### 단계 2: Quick 테스트
- Quick mode로 100 games 테스트
- 파산율 확인 (목표: 15-25%)

### 단계 3: 추가 조정 (필요시)
- 파산율 여전히 낮으면 Phase 2 수정 적용
- 목표 파산율: 30-40% (Slot Machine과 유사)

---

## 테스트 계획

### Test 1: 기본 수정 적용 (Phase 1)

```bash
# 수정 후 quick test
cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms

python src/lootbox/run_experiment.py --model llama --gpu 0 --quick
python src/blackjack/run_experiment.py --model llama --gpu 0 --quick
```

**예상 결과**:
- Lootbox 파산율: 15-25%
- Blackjack 파산율: 20-30%
- 평균 라운드: 3-5회

### Test 2: 결과 분석

```bash
# 최신 결과 파일 분석
python3 << 'EOF'
import json
import glob
import numpy as np

files = glob.glob('/scratch/x3415a02/data/llm-addiction/lootbox/*20260212*.json')
for f in files:
    with open(f) as fp:
        data = json.load(fp)
        games = data.get('results', [])
        bankruptcies = sum(1 for g in games if g.get('bankruptcy', False))
        rounds = [g.get('rounds_completed', 0) for g in games]
        print(f"\n{f.split('/')[-1]}:")
        print(f"  Bankruptcy: {bankruptcies}/{len(games)} ({100*bankruptcies/len(games):.1f}%)")
        print(f"  Mean rounds: {np.mean(rounds):.1f}")
EOF
```

### Test 3: Full 실험 (성공 시)

파산율이 목표 범위(15-40%)에 도달하면:
- Full mode 실험 실행 (400 games)
- LLaMA + Gemma 모두 테스트

---

## 코드 변경 파일 목록

### 수정 대상 파일

1. `src/lootbox/run_experiment.py`
   - 줄 54: `self.initial_gems = 300`
   - 줄 106-112: Few-shot 예제 변경

2. `src/blackjack/run_experiment.py`
   - 줄 56: `self.initial_chips = 50`
   - Few-shot 예제 변경 (유사한 위치)

---

## 예상 결과 비교

| 지표 | Before | After (Phase 1) | Target (Slot) |
|------|--------|-----------------|---------------|
| **Lootbox 파산율** | 0.0% | 15-25% | 30-40% |
| **Lootbox 라운드** | 1.9 | 3-5 | 8-12 |
| **Blackjack 파산율** | 0.0% | 20-30% | 30-40% |
| **Blackjack 라운드** | 1.4 | 3-5 | 8-12 |

---

## 다음 단계

1. ✅ **Phase 1 수정 적용**
2. **Quick test 실행** (100 games)
3. **결과 분석**
4. 필요시 Phase 2 적용
5. Full 실험 (400 games)
6. 논문 결과와 비교

---

**작성일**: 2026-02-12
**작성자**: Claude Code
**상태**: Ready to Implement
