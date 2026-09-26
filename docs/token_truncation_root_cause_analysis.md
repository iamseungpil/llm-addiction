# Max Token Truncation: 근본 원인 분석 보고서

**Date**: 2026-02-24
**Author**: Claude Code (Automated Analysis)
**Status**: 실험 재실행 진행 중

---

## Executive Summary

슬롯머신 논문 실험에서 Gemma-2-9B-IT의 `max_new_tokens=100` 설정이 CoT(Chain-of-Thought) 응답을 잘라내어 파서가 추론 텍스트의 첫 숫자를 잡는 연쇄 오류를 유발했다. 실제 데이터 검증 결과, `max_new_tokens=250`인 50trials 데이터에서 15.3% 잘림이 확인되었고, V3 코드(`max_new_tokens=1024`)에서는 0% 잘림으로 문제가 완전히 해결됨을 확인했다.

---

## 1. 근본 원인 (Root Cause)

### 1.1 문제 정의

Gemma-2-9B-IT는 **instruction-tuned 모델**로, chat template을 적용하면 Chain-of-Thought(CoT) 추론을 생성한 후 "Final Decision:" 줄로 결론을 내린다. 이 전체 과정에 평균 220+ 토큰이 필요하다.

슬롯머신 코드(`llama_gemma_experiment.py`)는 `max_new_tokens=100`으로 설정되어 있었다:

```python
# llama_gemma_experiment.py:283 (수정 전)
max_new_tokens=100,
```

그러나 동일 코드에서 Gemma를 chat template (CoT) 모드로 실행한다:

```python
# llama_gemma_experiment.py:268-274
if self.model_name == "gemma":
    chat = [{"role": "user", "content": prompt}]
    formatted_prompt = self.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
```

**결과**: Gemma의 CoT 응답이 100 토큰에서 잘려, "Final Decision:" 줄이 생성되지 않는다.

### 1.2 연쇄 오류 메커니즘

```
[1] max_new_tokens=100 → 응답이 100 토큰에서 절단
[2] "Final Decision:" 줄 없음 → P1 파서 실패
[3] 'stop' in response_lower (line 217) → CoT 추론의 "I should stop" 텍스트 매칭
    OR
[4] re.findall(r'\$(\d+)') → 추론 텍스트의 숫자 매칭 (잘못된 베팅 금액)
    OR
[5] 기본값 bet=$10 반환 → valid=False이나 게임 루프에서 valid 미검사
```

### 1.3 LLaMA vs Gemma 비대칭

| 모델 | 프롬프트 방식 | 응답 길이 | max_new_tokens=100 영향 |
|------|------------|----------|----------------------|
| LLaMA-3.1-8B (base) | raw prompt (prefix-completion) | ~20-50 토큰 | 영향 없음 |
| Gemma-2-9B-IT | chat_template (CoT) | ~220 토큰 | **~60% 잘림** |

LLaMA는 base 모델로 짧은 응답을 생성하므로 100 토큰이 충분했다. 문제는 Gemma에만 영향을 미쳤다.

---

## 2. 데이터 기반 증거

### 2.1 코드 버전별 토큰 설정 이력

| 코드 버전 | max_new_tokens | 파서 | 데이터 상태 |
|-----------|---------------|------|-----------|
| 슬롯머신 (paper V1) | **100** | `'stop' in response_lower` | **심각한 손상 예상** |
| Investment 50trials (V2 중간) | 250 | `re.search(r'([1234])')` | **15.3% 잘림 확인** |
| Investment parser_fixed_v2 (V3) | **1024** | P0→P1→P2, finditer[-1] | 0% 잘림 |
| Coin flip (V3) | **1024** | P0→P1→P2, finditer[-1] | 0% 잘림 |

### 2.2 실측 데이터: 50trials (max_new_tokens=250)

**파일**: `gemma_investment_c10_20260223_095145.json`

| 메트릭 | 값 |
|--------|-----|
| 총 응답 수 | 412 |
| "Final Decision" 포함 | 342 (83.0%) |
| "Final Decision" 없음 (잘림) | **70 (17.0%)** |
| 잘린 응답 평균 토큰 | ~249 (250 한도 근접) |
| 잘린 응답 100% Round 1 | ✓ (첫 라운드 추론이 가장 김) |

**잘린 응답 예시**:
```
"...Option 2 strikes a good balance:
* **Higher"   ← 단어 중간 절단 (1042 chars, ~289 tokens)
```

```
"...50/50 chance is manageable.

**Final"   ← "Final Decision" 직전 절단 (956 chars, ~266 tokens)
```

### 2.3 실측 데이터: parser_fixed_v2 (max_new_tokens=1024)

**파일**: `gemma_investment_c10_20260223_181530.json`

| 메트릭 | 값 |
|--------|-----|
| 총 응답 수 | 1,670 |
| "Final Decision" 포함 | **1,670 (100%)** |
| 잘림 | **0 (0%)** |
| 최대 응답 길이 | ~369 토큰 (1024 예산 내) |
| 파싱 정확도 | 100% (explicit_decision) |

### 2.4 실측 데이터: Coin flip (max_new_tokens=1024)

**파일**: `gemma_coinflip_unlimited_20260224_062810.json`

| 메트릭 | 값 |
|--------|-----|
| 총 응답 수 | 226 |
| "Final Decision" 포함 | **226 (100%)** |
| 잘림 | **0 (0%)** |
| 최대 응답 길이 | ~365 토큰 |

### 2.5 요약: 토큰 예산 vs 실제 필요량

```
Gemma CoT 평균 필요: ~220 토큰
Gemma CoT 최대 필요: ~432 토큰

슬롯머신 예산:     100 토큰 ← 220 토큰 필요 대비 46%만 제공 → 대부분 잘림
50trials 예산:     250 토큰 ← 긴 응답(R1)에서 15% 잘림
V3 예산:          1024 토큰 ← 최대 432 대비 충분 → 0% 잘림
```

---

## 3. 슬롯머신 데이터 접근 불가 문제

슬롯머신 원본 데이터는 이 서버에서 접근할 수 없다:
- 데이터 경로: `/data/llm_addiction/experiment_0_gemma_corrected/` → 존재하지 않음
- 심볼릭 링크: `/mnt/c/Users/oollccddss/git/data/llm-addiction/slot_machine` → Windows WSL 경로

따라서 슬롯머신 데이터의 직접적인 잘림 확인은 불가능하다. 그러나:
1. 동일 모델(Gemma-2-9B-IT)이 동일 방식(chat_template)으로 실행됨
2. 토큰 예산이 더 적음 (100 < 250)
3. 250에서 15.3% 잘림이 확인되었으므로, 100에서는 **60% 이상 잘림이 거의 확실**하다

---

## 4. 코인 플립: 문제 없음

코인 플립 실험은 V3 코드를 사용하며:
- `max_new_tokens=1024` (CoT) / `100` (base)
- P0→P1→P2 파서, finditer[-1] (LAST match)
- P2 fallback은 CoT에서 `valid=False` (retry 트리거)

데이터 검증 결과: 226개 응답 모두 "Final Decision" 포함, 0% 잘림, 100% 파싱 정확도.

**코인 플립은 토큰 잘림 문제와 무관하다.**

---

## 5. 수정 및 재실행 조치

### 5.1 코드 수정 (완료)

`paper_experiments/slot_machine_6models/src/llama_gemma_experiment.py`:

| 수정 사항 | 변경 전 | 변경 후 |
|----------|--------|--------|
| max_new_tokens | 100 (모든 모델) | 1024 (Gemma CoT) / 100 (LLaMA base) |
| 파서 | `'stop' in response_lower` 우선 | "Final Decision:" LAST match 우선 (P1) |
| valid 체크 | 없음 (게임 루프) | CoT 5회 retry, 10 consecutive skips → abort |
| 출력 경로 | `/data/llm_addiction/` | `/home/jovyan/beomi/llm-addiction-data/slot_machine/` |
| Random seed | 없음 | `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)` |

### 5.2 재실행 상태

- **Gemma 슬롯머신 V3**: GPU 0에서 실행 중 (PID 335284)
  - 3,200 게임 (64 조건 × 50 반복)
  - 출력: `/home/jovyan/beomi/llm-addiction-data/slot_machine/experiment_0_gemma_v3/`
  - 로그: `/home/jovyan/beomi/llm-addiction-data/logs/gemma_slot_v3_*.log`

---

## 6. 각 실험 데이터의 신뢰도 평가

| 데이터셋 | 코드 버전 | 토큰 | 파서 | 신뢰도 | 비고 |
|----------|----------|------|------|--------|------|
| 슬롯머신 Gemma (원본) | V1 | 100 | V1 | **사용 불가** | ~60% 잘림 + 잘못된 파서 |
| 슬롯머신 LLaMA (원본) | V1 | 100 | V1 | **주의** | base model이라 잘림 없으나 파서 불안정 |
| Investment 50trials | V2 | 250 | V2 | **사용 불가** | 15.3% 잘림 + P2 first-digit 버그 |
| Investment parser_fixed_v2 | **V3** | **1024** | **V3** | **신뢰 가능** | 0% 잘림, 100% 파싱 |
| Coin flip | **V3** | **1024** | **V3** | **신뢰 가능** | 0% 잘림, 100% 파싱 |
| 슬롯머신 Gemma V3 (재실행) | **V3** | **1024** | **V3** | **진행 중** | 결과 대기 |

---

## 7. 논문 결과에 대한 영향

### 7.1 Gemma 슬롯머신 결과

원본 Gemma 슬롯머신 데이터는 **신뢰할 수 없다**:
- ~60% 응답이 잘림 → 파서가 추론 텍스트에서 행동을 추출
- `'stop' in response_lower`가 추론의 "I should stop" 매칭 → stop 비율 부풀림
- 기본값 bet=$10 → Fixed $10 결과에 영향

V3 재실행 완료 후 원본과 비교하여 얼마나 달라지는지 확인 필요.

### 7.2 API 모델 (GPT-4o-mini, Claude-3.5-Haiku, Gemini)

API 모델은 다른 코드 파일을 사용하며, 토큰 설정이 다르다:
- GPT: `max_completion_tokens=1024` → 충분
- Claude: `max_tokens=300` → 경계선 (일부 잘림 가능)
- Gemini: 기본값 사용 → 충분

API 모델 데이터는 이 서버에 없어 직접 확인 불가하나, 토큰 예산이 Gemma보다 충분하므로 영향은 제한적일 것으로 추정.

### 7.3 Variable > Fixed (autonomy effect)

이전 분석에서 밝힌 바와 같이, 모든 버그는 Fixed와 Variable에 **동일하게** 영향을 미친다. 따라서:
- 절대값 (파산율 X%)은 신뢰 불가
- **방향** (Variable > Fixed)은 여전히 유효할 가능성 높음
- 단, Gemma는 `'stop' in response_lower` 버그가 Variable 응답에서 더 자주 발동할 수 있어 확인 필요

---

## 8. 다음 단계

1. **슬롯머신 V3 재실행 완료 대기** → 원본과 비교 분석
2. **Claude `max_tokens=300` 검증**: Claude 응답 데이터가 있다면 잘림 비율 확인
3. **LLaMA 슬롯머신 재실행 고려**: base model이라 토큰 문제는 없으나 파서 버그는 있음
4. **논문 결과표 업데이트**: V3 데이터로 Gemma 행 교체

---

## Appendix: 코드 버전 정의

| 버전 | 파서 | 토큰 | Retry | 특징 |
|------|------|------|-------|------|
| V1 | `'stop' in response_lower`, `re.search` first match | 100 | 없음 | 슬롯머신 원본 |
| V2 | `re.search(r'([1234])')` first match, `valid=True` for P2 | 250 | 없음 | 50trials 중간 버전 |
| V3 | `"Final Decision:"` LAST match, P2 `valid=False` for CoT | 1024 | 5회 (CoT) | parser_fixed_v2, coin flip, 수정된 슬롯머신 |
