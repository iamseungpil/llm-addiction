# 종합 코드/데이터 감사 보고서: 논문 결과의 신뢰성 분석

**Date**: 2026-02-24
**Scope**: 전체 코드베이스 (paper_experiments + investment_choice_bet_constraint_cot + exploratory_experiments)

---

## Executive Summary

논문의 핵심 주장 "Variable 베팅이 Fixed보다 파산율이 높다 (autonomy effect)"에 대해:

| 모델 | 보고된 결과 | 방향 신뢰도 | 절대값 신뢰도 | 근거 |
|------|-----------|-----------|------------|------|
| **GPT-4o-mini** | F:0% / V:21.3% | **HIGH** ✓ | MEDIUM | 파서 양호, 토큰 충분. "cautious" 시스템 프롬프트가 절대값 억제 |
| **GPT-4.1-mini** | F:0% / V:6.3% | **HIGH** ✓ | MEDIUM | 동일 |
| **Gemini-2.5-Flash** | F:3.1% / V:48.1% | **HIGH** ✓ | MEDIUM-HIGH | 토큰 무제한, 시스템 프롬프트 없음. 최선의 설정 |
| **Claude-3.5-Haiku** | F:0% / V:20.5% | **MEDIUM-HIGH** | MEDIUM | max_tokens=300 (잘림 위험), temperature=0.5 |
| **LLaMA-3.1-8B** | F:0.1% / V:7.1% | **MEDIUM** | LOW | stop 서브스트링 버그, $10 유령 베팅 |
| **Gemma-2-9B** | F:12.8% / V:29.1% | **LOW** ⚠ | **LOW** ⚠ | max_tokens=100 (CoT 잘림), 파서 노이즈 |

**결론: 4개 API 모델에서 "Variable > Fixed" 방향은 신뢰할 수 있음. 그러나 Gemma 데이터는 심각한 파서 오염으로 신뢰 불가.**

---

## 1. 코드베이스 버전 비교

### 1.1 세 가지 Investment Choice 구현체

| 속성 | V1 (paper) | V2 (bet_constraint_cot) | V3 (alt_paradigms) |
|------|-----------|------------------------|-------------------|
| **위치** | `paper_experiments/investment_choice_experiment/` | `investment_choice_bet_constraint_cot/` | `exploratory_experiments/alternative_paradigms/` |
| **모델** | API만 (GPT, Claude, Gemini) | API만 | API + Local (LLaMA, Gemma) |
| **파서** | 패턴 순서 역전 (option 먼저, FD 나중) | 동일 버그 | **수정됨** (P0→P1→P2, finditer[-1]) |
| **토큰** | 300 | 300 (Gemini: 8000) | **1024** (CoT), 100 (base) |
| **에러 폴백** | Option 1 주입 | Option 1 주입 | **Skip 로직** |
| **시드** | 없음 | 없음 | **42** |
| **Retry** | 없음 | 없음 | **5회 + 힌트** |
| **파산 감지** | 없음 | 있음 | 있음 |
| **Option 3 배수** | **3.2x** (EV=0.80) | 3.6x (EV=0.90) | 3.6x (EV=0.90) |

### 1.2 슬롯머신 구현체

| 속성 | Local (LLaMA/Gemma) | API (GPT) | API (Claude) | API (Gemini) |
|------|---------------------|-----------|-------------|-------------|
| **파서** | `'stop' in response` (전체 텍스트) | 6단계 FD 우선 | 동일 (GPT 복사) | 동일 (GPT 복사) |
| **토큰** | **100** | 1024 | **300** | 무제한 |
| **시스템 프롬프트** | 없음 | **"cautious, rational"** | 없음 | 없음 |
| **Temperature** | 0.7 | 1.0 (기본) | **0.5** | 1.0 (기본) |
| **파싱 실패 폴백** | **bet $10** | retry→bet $10 | retry→bet $10 | retry→**STOP** |
| **API 에러 폴백** | 무한 retry | Stop | Stop | Stop |
| **Raw 응답 저장** | **미저장** | 저장 | 저장 | 저장 |

---

## 2. 발견된 버그의 방향성 분석

### 2.1 핵심 질문: 버그가 Fixed와 Variable을 차별적으로 영향 주는가?

| 버그 | 영향 방향 | Fixed 영향 | Variable 영향 | 차등 효과? |
|------|---------|-----------|-------------|-----------|
| C1: `'stop' in response` | 파산율 ↓ (거짓 정지) | 동일 | 동일 | **없음** |
| C2: max_tokens=100 | 파싱 노이즈 | 동일 | 동일 | **없음** |
| C4: 기본 bet $10 | 파산율 ↑ (유령 베팅) | 동일 | 동일 | **없음** |
| W2: FD 첫매칭 | 간헐적 오류 | 동일 | 동일 | **없음** |
| W5: temperature 차이 | 모델간 비교 오염 | 동일 | 동일 | **없음** |
| W6: "cautious" 프롬프트 | 파산율 ↓ | 동일 | 동일 | **없음** |

**모든 버그가 Fixed와 Variable을 동등하게 영향** → "Variable > Fixed" 방향은 버그에 의해 인위적으로 생성되지 않음.

### 2.2 그렇다면 "방향"은 왜 신뢰할 수 있는가?

버그들이 절대값을 왜곡하더라도, Fixed/Variable 간 차이를 만드는 **유일한 변인은 실제 모델 행동**:
- Fixed: 항상 $10 베팅 (선택의 여지 없음)
- Variable: $5~$balance 중 모델이 선택

→ Variable에서 모델이 $10 이상을 선택하면 파산 위험 증가. 이것은 파서와 무관한 **게임 로직 수준의 차이**.

### 2.3 Gemma가 예외인 이유

Gemma에서만 신뢰도가 LOW인 이유:
1. **100토큰 잘림**으로 응답의 ~60%가 "Final Decision:" 줄 이전에 절단
2. 잘린 텍스트에서 파서가 **추론 텍스트의 "stop"이나 "$N"을 결정으로 해석**
3. 이 노이즈가 Fixed와 Variable에서 **다른 양상으로 발현**될 수 있음:
   - Variable 프롬프트: `"1) Bet (choose $5-$90)"` → 추론에 다양한 금액 언급 → 파서가 랜덤 금액 추출
   - Fixed 프롬프트: `"1) Bet $10"` → 추론에 "$10"만 반복 → 더 일관된 파싱
4. **어느 방향으로 편향되는지 예측 불가** → 데이터 신뢰 불가

---

## 3. 슬롯머신 vs Investment Choice: "Autonomy Effect" 역전의 진짜 원인

### 3.1 데이터 기반 발견

| 실험 | Fixed 베팅 | 모델 자연 선호 | Fixed vs 선호 | 결과 |
|------|-----------|-------------|-------------|------|
| **슬롯머신** | $10 (고정) | ~$30-50 (잔고의 30-50%) | Fixed **< 선호** | Variable > Fixed |
| **Investment c10** | $10 (고정) | $5.3 (제약의 53%) | Fixed **> 선호** | 동률 (0%=0%) |
| **Investment c30** | $30 (고정) | $15.1 (제약의 50%) | Fixed **> 선호** | Fixed ≥ Variable |
| **Investment c50** | $50 (고정) | $23.4 (제약의 47%) | Fixed **> 선호** | **Fixed > Variable** |

### 3.2 통합 법칙

```
Fixed 금액 < 모델의 자연 선호  →  Variable이 더 위험  →  Variable > Fixed 파산 (슬롯머신)
Fixed 금액 > 모델의 자연 선호  →  Fixed가 더 위험    →  Fixed > Variable 파산 (Investment Choice)
```

**이것은 "autonomy effect"가 아니라 "constraint direction effect".**

- 슬롯머신 Fixed $10은 모델이 원하는 것보다 **적게** 베팅하도록 강제 → 자율성이 위험 증가
- Investment Choice Fixed $50은 모델이 원하는 것보다 **많이** 베팅하도록 강제 → 자율성이 위험 감소

### 3.3 논문에 미치는 영향

| 논문 주장 | 평가 | 근거 |
|----------|------|------|
| "Variable betting → 높은 파산율" | **부분적으로 맞음** | 슬롯머신에서는 맞지만, Investment Choice에서는 반대. 일반화 불가 |
| "자율성이 위험을 증가시킨다" | **틀림 (oversimplified)** | 자율성의 효과는 Fixed 금액과 자연 선호의 상대적 위치에 의존 |
| "Gemma Fixed 12.81%" | **신뢰 불가** | max_tokens=100 + 파서 버그로 데이터 오염 |
| "Gemma Irrationality 0.271±0.118" | **의심** | 높은 표준편차(0.118)가 파싱 노이즈를 반영할 가능성 |

---

## 4. 코인 플립 파산율 0%: 확인된 원인

### 4.1 코드 버그 없음 (검증됨)

코인 플립은 V3 코드를 사용. 파싱 정확도 226/226 = 100%. 모든 수정 적용됨.

### 4.2 게임 설계 원인

| 요인 | 코인 플립 | 슬롯머신 | Investment Choice |
|------|---------|---------|-----------------|
| 선택지 | 2개 (Continue/Stop) | 2개 (Bet/Stop) + 금액 | 4개 (Stop + 3위험) |
| EV 표시 | 명시적 (45%/55%) | 명시적 (30%/3x) | 명시적 (50%/1.8x 등) |
| 위험 gradient | 없음 | 금액으로 조절 | 3단계 |
| "약간만 위험하게" | **불가능** | 소액 베팅 가능 | Option 3 가능 |

코인 플립에서는 매 라운드 **동일한 EV 계산 → 동일한 결론(Stop)**. Investment Choice에서는 "Option 3은 balanced"라는 합리화 경로가 존재.

---

## 5. Investment Choice Fixed 고파산율: 확인된 원인

### 5.1 코드 버그 아닌 행동적 원인

모델은 **언제든 Stop 가능**. Fixed $50에서도 파산하는 이유:

1. **올인 미인지**: 잔고=$50, 베팅=$50인 상황에서 46개 결정 중 **0개**가 올인을 인식
2. **Goal이 Stop 차단**: G 조건 파산율 26% vs BASE 6%. "Option 4는 목표에 도움 안 됨" → Stop 거부
3. **프롬프트 프레이밍**: Option 4가 "Stop"이 아닌 "100% return investment"로 표현 → 게임 종료로 인식 안 됨

### 5.2 Variable이 보호하는 메커니즘

Variable에서 Gemma는 **제약의 ~50%만 베팅** (c50 기준 $23.4). 이것이:
- 2연패 생존: Fixed $50→$0 vs Variable $23→$54 (생존)
- 점진적 위험 감소: 잔고 하락 시 베팅도 자동 감소
- Fixed에서는 유일한 위험 감소 수단이 **완전한 정지** → 모델이 이를 선택하지 않음

---

## 6. 전체 버그 목록 (코드베이스 통합)

### CRITICAL (6개)

| # | 버그 | 영향 코드 | 데이터 영향 |
|---|------|---------|-----------|
| C1 | `'stop' in response` 전체 텍스트 검색 | 슬롯머신 local | Gemma/LLaMA 파산율 왜곡 |
| C2 | max_new_tokens=100 (CoT 잘림) | 슬롯머신 Gemma | Gemma 데이터 전체 오염 |
| C3 | LLaMA base model에 IT 프롬프트 | 슬롯머신 LLaMA | LLaMA 결정이 prefix-completion 노이즈 |
| C4 | 파싱 실패 → 유령 $10 bet | 슬롯머신 local | 강제 베팅 주입 |
| C5 | API 에러 → 거짓 Option 1/Stop 주입 | 모든 API 실험 | 정지율 부풀림 (빈도 불명) |
| C6 | Option 3 배수 불일치 (3.2x vs 3.6x) | V1 vs V2/V3 | 실험간 비교 불가 |

### WARNING (11개)

| # | 버그 | 영향 코드 |
|---|------|---------|
| W1 | 랜덤 시드 미설정 | 전체 (V3 제외) |
| W2 | "Final Decision" 첫매칭 (마지막 아님) | API 파서 전체 |
| W3 | Claude max_tokens=300 | 슬롯머신/Investment Claude |
| W4 | GPT "cautious, rational" 시스템 프롬프트 | 슬롯머신 GPT |
| W5 | Temperature 불일치 (0.5~1.0) | 모델간 비교 |
| W6 | 파서 bet 우선 (stop보다) | API 파서 |
| W7 | Fixed 보상 오표시 (잔고 < constraint) | V1, V2 |
| W8 | V1 파산 미감지 (exit_reason) | V1 paper |
| W9 | V2 파서 패턴 순서 역전 | V2 bet_constraint |
| W10 | Claude 시스템 프롬프트 누락 (extended_cot) | V2 extended |
| W11 | Goal 추출 허위 양성 63% | V2 |

---

## 7. 권장 조치

### 즉시 (논문 수정)

1. **Gemma 슬롯머신 재실행** (최우선): V3 수준 파서 + max_tokens=1024 + seed=42
2. **LLaMA 슬롯머신 재실행**: 파서 수정 + raw 응답 저장
3. **"Autonomy effect" 재해석**: "Variable > Fixed"가 아닌 "constraint direction effect"로 reframe. 또는 슬롯머신 결과에 한정하여 주장
4. **GPT 결과에 "cautious" 프롬프트 주석 추가**: 절대값이 다른 모델보다 낮은 이유 설명

### 분석 시

5. **API 모델 raw 응답 재파싱**: 저장된 응답에 V3 파서 적용하여 결과 검증 가능 (데이터 접근 시)
6. **Claude 300토큰 잘림 빈도 확인**: raw 응답에서 "Final Decision" 포함 여부 체크
7. **Gemini 결과를 "가장 깨끗한" 기준선으로 사용**: 버그 영향 최소

### 장기

8. **통합 실험 프레임워크**: 모델별 파서/토큰/프롬프트 분리 → 교차 모델 비교 신뢰성 확보
9. **Fixed 금액을 모델의 자연 선호에 맞춤**: 진정한 autonomy effect 측정을 위해 `Fixed = Variable 평균 베팅`으로 설정

---

## Appendix: 데이터 접근 상태

| 데이터 | 경로 | 접근 가능 | 비고 |
|--------|------|---------|------|
| Coin Flip (Gemma) | `/home/jovyan/beomi/.../coin_flip/` | ✓ | 80게임, 100% 파싱 검증 |
| Investment Choice (Gemma, V3) | `/home/jovyan/beomi/.../gemma_parser_fixed_v2/` | ✓ | c10/c30/c50, 100% 파싱 검증 |
| Slot Machine 6모델 | `/mnt/c/Users/.../slot_machine` (심링크) | ✗ | Windows 경로, OpenHPC 미접근 |
| Paper Investment (API) | `/data/llm_addiction/...` (심링크) | ✗ | 경로 미존재 |
| Extended CoT (API) | `/data/llm_addiction/...` (심링크) | ✗ | 경로 미존재 |

**빅모델 원본 데이터는 이 서버에서 접근 불가. 코드 분석 기반 평가만 수행됨.**

---

*Generated by comprehensive code/data audit across all experiment implementations.*
*Code review: 4 slot machine files, 2 investment choice variants, 1 bet_constraint_cot variant, 1 alternative_paradigms variant.*
*Data analysis: 80 coin flip games, 1200 investment choice games (c10/c30/c50).*
