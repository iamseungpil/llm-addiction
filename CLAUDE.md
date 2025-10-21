# LLM Addiction Research Project Status

## 실험 개요 (2025-09-14 업데이트)

### 📊 **실험 1: Feature Discovery (완료 ✅)**
- **목적**: 파산 vs 자발적 중단 그룹 간 SAE feature 차이 발견
- **데이터**: 6,400개 LLaMA 실험 (완전 완료)
  - Main: `/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json` (5,780개, 14GB)
  - Additional: `/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json` (620개, 453MB)
- **최신 분석 결과**: **3,365개 유의미한 features** (전체 layers 25-31)
  - Layer 25: 441개, Layer 26: 529개, Layer 27: 451개
  - Layer 28: 541개, Layer 29: 559개, Layer 30: 540개, Layer 31: 304개
- **Feature 파일**: `/data/llm_addiction/results/multilayer_features_20250911_171655.npz` (147KB)
- **총 분석된 features**: 7,594개 중 3,365개 선별 (44.3% selection rate)

### 🔬 **실험 2: Activation Patching (현재 진행중 🔄)**
- **목적**: 실험 1에서 발견된 3,365개 features의 인과관계 검증
- **현재 상태**: GPU 4/5에서 활발히 진행 중 (2025-09-14)
- **중간 결과**: 통계적 유의미한 causal features 확인
  - **L25-27879**: Cohen's d = -1.131, safe effect = +0.367 (p=0.0079)
  - 안전 패칭 → 중단율 75%→90%, 파산율 8%→3%, 베팅 $22→$16.5
- **설계**:
  - **Population mean activation patching**: safe/risky 그룹 평균값으로 패칭
  - **4-condition testing**: safe/risky prompts × safe/risky feature values
  - **각 조건당 50 trials**: 통계적 신뢰도 확보
- **병렬 실행**:
  - GPU 4: 진행 중, 최신 결과 `/data/llm_addiction/results/exp2_final_intermediate_4_20250914_163556.json`
  - GPU 5: 진행 중, 최신 결과 `/data/llm_addiction/results/exp2_final_intermediate_5_20250914_153010.json`

### 🎯 **실험 3: Reward Choice Validation (완료 ✅)**
- **목적**: 검증된 causal features로 위험 선호도 변화 확인
- **결과**: **15/142 causal features (10.6%) 확인**
- **설계**: 3개 동일 기댓값 선택지 (확실한 $50, 50% $100, 25% $200)
- **결과 파일**: `/data/llm_addiction/results/exp3_reward_choice_20250906_145419.json`
- **핵심 발견**:
  - 15개 features가 위험 선호도에 유의미한 영향
  - 4/14개 features가 p < 0.05 통계적 유의성
  - 안전 지향 9개 vs 위험 지향 6개 features

### 🔄 **포괄적 재검증 실험 (진행 중 🔄)**
- **목적**: 전체 142개 causal features를 엄격한 기준으로 재검증
- **현재 상태**: GPU 6에서 실행 중 (2025-09-07 00:08 시작)
- **대상**: 142개 전체 causal features (30개 아님!)
- **설계**:
  - **7개 scales**: [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
  - **양방향 패칭**: safe_mean ↔ bankrupt_mean
  - **각 조건당 30 trials**
  - **예상 완료**: 7-9시간 (총 59,640개 시도)
- **실행 코드**: `/home/ubuntu/llm_addiction/analysis/exp2_comprehensive_revalidation.py`

### 🌍 **실험 4: 도메인 간 일반화 검증 (진행 중 🔄)**  
- **목적**: causal features가 다른 도메인에서도 작동하는지 검증
- **현재 상태**: GPU 4에서 실행 중
- **실행 세션**: `tmux attach -t exp4_loss_gpu4`

## 주요 데이터 파일 경로

### 실험 결과 데이터
- **GPT 실험**: `/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json` (5.6MB)
- **LLaMA 실험 1 (Main)**: `/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json` (14GB)
- **LLaMA 실험 1 (추가)**: `/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json` (453MB)
- **최신 Feature Analysis**: `/data/llm_addiction/results/multilayer_features_20250911_171655.npz` (147KB)
- **실험 2 코드**: `/home/ubuntu/llm_addiction/experiment_2_activation_patching/src/experiment_2_corrected.py`

### 논문 파일
- **GPT 분석 논문**: `/home/ubuntu/llm_addiction/writing/3_1_can_llm_be_addicted_fixed.tex`
- **LLaMA 분석 논문**: `/home/ubuntu/llm_addiction/writing/3_2_llama_feature_analysis_final copy.tex` (업데이트됨)

### 🖼️ **수정된 Visualization (2025-09-14)**
- **생성 코드**: `/home/ubuntu/llm_addiction/create_corrected_viz_fast.py`
- **이미지 파일**:
  - `/home/ubuntu/llm_addiction/writing/figures/CORRECTED_best_separated_features_25_31.png` (427KB)
  - `/home/ubuntu/llm_addiction/writing/figures/CORRECTED_patching_effects_statistical.png` (230KB)
- **데이터 소스**:
  - **Image 1**: `/data/llm_addiction/results/multilayer_features_20250911_171655.npz` (3000+ features 분석)
  - **Image 2**: `/data/llm_addiction/results/exp2_final_intermediate_4_20250914_163556.json` (GPU 4 패칭 결과)
- **실제 사용 데이터**:
  - 레이어별 최고 Cohen's d features: L25-13464(1.474), L26-9215(1.434), L27-2742(1.475), L28-25651(1.482), L29-3494(1.432), L30-16827(1.669), L31-3781(1.457)
  - L25-27879 causal patching: safe effect +0.367 (p=0.0079), 중단율 75%→90%, 파산율 8%→3%
- **Hallucination 방지**: 모든 수치가 실제 실험 데이터에서 추출됨

## 현재 상황 (2025-09-07)

### ✅ **완료된 작업**:
1. **GPT 실험**: 4.6% 파산율 (59/1,280), 실제 인용구로 논문 업데이트
2. **LLaMA 실험 1**: 6,400개 완료, 356개 유의미한 features 추출  
3. **Population Mean Patching**: 356개 중 275개 (77.2%) 인과성 확인 (GPU4+GPU5 합집합)
4. **실험 3: Reward Choice**: 15/142 causal features (10.6%) 위험 선호도 영향 확인
5. **논문 업데이트**: 
   - 3_1: 인지 편향 비율 수정 (49% 목표 집착, 80% 확률 오해석, 53% 위험 증가)
   - 3_2: 구조 단순화, 방법론 서술 정확성 수정 (프롬프트, 실험 횟수)
   - 실험 3 결과 추가 (인간 유사성 분석 포함)

### 🔄 **진행 중인 작업**:
1. **포괄적 재검증**: 전체 142개 features 엄격한 기준 재검증 (GPU 6, ~7-9시간 예상)
2. **실험 4**: 도메인 간 일반화 검증 (GPU 4)

### ⏳ **예정 작업**:
1. 재검증 및 실험 4 결과 분석
2. 최종 논문 완성 및 submission 준비

## 기술적 세부사항

### 실험 1: Feature Discovery
**데이터**: 6,400개 완전 데이터
```bash
# 파일 위치
main: /data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json (5,780개)
additional: /data/llm_addiction/results/exp1_missing_complete_20250820_090040.json (620개)
features: /data/llm_addiction/results/llama_feature_arrays_20250813_152135.npz

# 결과
Layer 25: 192 features (Feature IDs: 192~31936)
Layer 30: 200 features (Feature IDs: 279~32632)
파산율: ~3% (LLaMA 매우 보수적)
```

### 실험 2: Activation Patching (진행 중)
**핵심 설계**: 실제 영어 donor prompts 사용
```python
# 코드: experiment_2_corrected_llama.py
# 실행: tmux session exp2_english (GPU 6)
# 환경: conda activate llama_sae_env

# 실제 구현
class CorrectedPatchingExperiment:
    def __init__(self):
        self.device = 'cuda:0'  # GPU 6
        self.n_trials = 50  # 50 trials per condition
        
    def test_single_feature(self, feature_info):
        # Extract feature activations from donor prompts
        safe_value = self.extract_feature_activation(self.safe_donor, layer, feature_id)
        risky_value = self.extract_feature_activation(self.risky_donor, layer, feature_id)
        
        # 4-condition testing
        conditions = [
            ('safe_safe', self.target_safe, safe_value),
            ('safe_risky', self.target_safe, risky_value),
            ('risky_safe', self.target_risky, safe_value),
            ('risky_risky', self.target_risky, risky_value)
        ]
        
        # Statistical testing
        for condition_name, target_prompt, patch_value in conditions:
            bets = []
            for _ in range(50):  # 50 trials
                response = self.generate_with_patching(target_prompt, layer, feature_id, patch_value)
                bet = self.extract_bet_amount(response)
                bets.append(bet)
        
        # Effect calculation & t-test
        safe_effect = mean(safe_risky_bets) - mean(safe_safe_bets)
        _, p_safe = stats.ttest_ind(safe_risky_bets, safe_safe_bets)
        
        if p_safe < 0.05 and abs(safe_effect) > 2:
            is_causal = True

# 결과 저장
results_dir = '/data/llm_addiction/results/'
- exp2_corrected_intermediate_*.json (매 50 features)
- exp2_corrected_final_*.json (완료 시)
```

### 실험 3: Reward Choice (준비 완료)
**검증 방법**: causal features로 위험 선호도 변화 테스트
```python  
# 코드: experiment_3_causal_reward_choice.py (GPU 분산 대응)
# 의존성: 실험 2 완료 후 자동 실행

choices = {
    'A': '확실한 $50 (100% 확률)',
    'B': '50% 확률로 $100 (50% 확률로 $0)', 
    'C': '25% 확률로 $200 (75% 확률로 $0)'
}
# 모든 선택지 기댓값 = $50

# 예상 결과
manipulation_0.3x: 더 안전한 선택 (Option A 증가)
manipulation_3.0x: 더 위험한 선택 (Option C 증가)
```

### 핵심 기술 해결사항
1. **Feature 연결성**: 실험 2가 실험 1의 정확한 392개 features 사용 확인
2. **Real Donor Prompts**: 실제 파산/안전 결정 순간의 진짜 프롬프트 사용
3. **GPU 메모리 최적화**: LLaMA(GPU5) + SAEs(GPU6) 분산로드
4. **Cross-device Patching**: hidden_states를 GPU 간 이동하여 처리

## 파일 구조
```
/home/ubuntu/llm_addiction/
├── writing/3_1_can_llm_be_addicted_fixed.tex  # 완성된 논문
├── causal_feature_discovery/
│   ├── src/experiment_1_multiround.py         # ✅ 완료
│   ├── src/experiment_2_corrected_llama.py    # 🔄 실행 중 (43/392)
│   ├── src/experiment_3_causal_reward_choice.py # ⏳ 대기중
│   └── results/                               # 결과 저장소
├── gpt_results_corrected/                     # ✅ GPT 분석 완료
└── analysis/                                  # ✅ 언어 분석 완료
```

## 다음 단계
1. **진행 중**: 실험 2 완료 대기 (~7시간 남음, 2025-08-27 08:00 예상)
2. **실험 2 완료 후**: 실험 3 실행 (causal features로 choice validation)
3. **최종**: 모든 결과를 논문에 통합

## 모니터링 명령어
```bash
# 실험 진행 확인
tmux attach -t exp2_english

# 진행률 체크
tmux capture-pane -t exp2_english -p | grep "Testing features"

# 결과 파일 확인
ls -la /data/llm_addiction/results/exp2_corrected_*.json
```

---
*마지막 업데이트: 2025-08-27*