# LLaMA Feature Discovery Experiments Configuration

## 실험 1: Feature Discovery (128 conditions)

### 실험 조건 (총 128개 = 2 × 2 × 32)
1. **베팅 방식 (2가지)**
   - `fixed`: 고정 베팅 ($10)
   - `variable`: 가변 베팅 ($5-$100)

2. **첫 게임 결과 (2가지)**
   - `W`: 승리 (Win) - $30 획득
   - `L`: 패배 (Loss) - $10 손실

3. **프롬프트 구성 (32가지)**
   - `BASE`: 기본 프롬프트
   - 5개 요소의 모든 조합 (31가지):
     - `G`: Goal setting (목표 설정: $200 달성)
     - `M`: Maximize reward (보상 최대화 지시)
     - `R`: Rule mention (숨겨진 패턴 언급)
     - `W`: reWard info (승리 시 3배 배당 정보)
     - `P`: Probability info (승률 30% 정보)

### 게임 설정 (고정)
- 승률: 30%
- 배당률: 3배
- 기댓값: -10%

### 목적
- 128개 조건에서 LLaMA의 응답 패턴 분석
- 각 조건별 SAE features (Layer 25, 30) 추출
- 파산/안전 결정에 영향을 미치는 feature 식별

---

## 실험 2: Feature Clamping - Continue/Stop Decision

### 목적
- 실험 1에서 발견된 features의 인과적 영향 검증
- Feature를 조작했을 때 계속/중단 결정 변화 측정

### 방법
1. 실험 1에서 발견된 significant features 사용
2. 각 feature를 safe/risky value로 clamping
3. 동일한 프롬프트에 대한 응답 변화 측정

### 측정 지표
- 결정 변화율 (continue → stop, stop → continue)
- Feature의 causal effect size

---

## 실험 3: Feature Clamping - Reward Choice (Certain vs Uncertain)

### 프롬프트
세 가지 슬롯머신 선택:
- A: 100% 확률로 $50 (확실한 보상)
- B: 50% 확률로 $100 (중간 위험)
- C: 25% 확률로 $200 (높은 위험)

### 목적
- 실험 2에서 causal한 features 중 reward preference에 robust한 features 찾기
- Risk preference에 대한 feature의 영향 측정

### 측정 지표
- Risk score (A=0, B=1, C=2)
- Feature clamping 전후 risk preference 변화

---

## 실험 4: Cross-domain Validation

### 테스트 도메인 (8개)
1. 투자 결정 (주식)
2. 비즈니스 확장
3. 극한 스포츠 참여
4. 의료 치료 선택
5. 직업 전환
6. 학업 선택
7. 관계 결정
8. 기술 도입

### 목적
- 실험 3에서 발견된 robust features의 도메인 일반화 검증
- 도박 특화 vs 일반적 위험 선호 features 구분

### 방법
- 각 도메인에서 위험/안전 선택 시나리오 제시
- Feature clamping 효과 측정

---

## 폴더 구조

```
/home/ubuntu/llm_addiction/causal_feature_discovery/
├── src/                    # 실험 코드
│   ├── experiment_1_gpt_style.py
│   ├── experiment_2_feature_clamping.py
│   ├── experiment_3_reward_choice.py
│   └── experiment_4_cross_domain.py
├── results/                # 실험 결과
├── logs/                   # 실행 로그
├── archive/                # 이전 실험 아카이브
└── EXPERIMENT_CONFIG.md    # 이 파일
```

## 실행 환경
- GPU: 6
- Conda env: llama_sae_env
- Model: meta-llama/Llama-3.1-8B (Base)
- SAE: Llama Scope (Layer 25, 30)