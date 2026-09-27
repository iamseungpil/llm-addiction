# L22_PLAN_v5 ↔ multilayer-causal 트랙 관계 정리 (2026-06-11)

두 사전등록 계획이 같은 repo군에 공존하므로 관계를 공식화한다. **모순되는 사전등록이 아니다** —
프로토콜과 질문이 다르다.

| | L22_PLAN_v5 (이 repo) | multilayer-causal W1–W3 (llm-addiction/multilayer_causal/) |
|---|---|---|
| 개입 시점 | **full-game**: 전 결정에 걸쳐 누적 steering | **single-decision**: −G 상태 1결정, prefill 개입 |
| 개입 대상 | ĥ_BK (과제별 파산-정지 방향), L22 | 다층 윈도 상태 패치 / 행동축·디코더축 steering, L16–21 |
| 1차 지표 | 게임 종말점 bk_rate | 라운드-수준 bet_ratio·3대 비합리성 지표 |
| 주 모델 | LLaMA (H1/H2/H4/H5) + Gemma H3 | Gemma (W3에서 LLaMA 확장) |
| 질문 | 누적 BK-방향 개입이 종말점을 옮기나 | 단일 결정의 인과 기질이 어디·무엇인가 |

정리:
1. multilayer 트랙의 "L22 쓰기 불능" 결과(단일-결정 prefill 패치, W1 w1e_2223 회복 1.2%)는
   v5의 "+α(L22, 전게임 누적)가 bk_rate를 올린다" 기대와 **논리적으로 양립** — 누적 steering은
   매 forward에 작용하므로 단일-prefill 불능과 모순 없음.
2. 두 결과를 본문 같은 절에 쓸 때는 프로토콜 차이(누적 vs 단일-결정)를 반드시 병기한다.
3. W3의 BK 관련 arm(w3bk)은 v5와 **다른 객체**(LOTO rank-1 공유축, 표적과제 제외)와
   다른 창(L16–21)을 다루며, v5의 셀과 중복되지 않는다.
4. v5는 별도 트랙으로 유지(실행 여부·결과는 v5 문서 계보를 따름). 본 노트가 v5를 폐기하지 않는다.

관련: llm-addiction/multilayer_causal/RUN_PLAN_W3.md §"L22_PLAN_v5 관계".
