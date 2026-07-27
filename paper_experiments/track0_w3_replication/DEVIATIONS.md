# Track 0 W3 — 사전등록 대비 편차 기록 (rebuttal 실행, 2026-07-24)

`configs/track0_config.yaml`은 frozen(2026-05-08) 그대로 두고, 아래 편차는 전부 CLI 오버라이드로 적용한다. config 파일 자체는 수정하지 않았다.

## D1. Claude 모델 대체 (불가피)

사전등록 모델 `claude-3-5-haiku-20241022`가 2026-02-19 EOL로 Anthropic API에서 제거되어 호출이 404를 반환한다(2026-07-24 확인). 최근접 후속 모델 `claude-haiku-4-5-20251001`로 대체한다. rebuttal 본문에 각주로 명시한다: 제출 시점 모델이 API에서 제거되어 동일 티어 후속 모델로 대체했다.

- 적용: `--provider anthropic --model_id claude-haiku-4-5-20251001`

## D2. gpt-4o 재기준 셀을 gpt-4.1-mini로 교체

사전등록 당시에는 논문 cap 절제가 GPT-4o로 표기되어 있어 gpt-4o 재기준(protocol parity) 셀을 넣었다. 이후 감사(`PAPER_CANONICAL_CODE.md`)로 실제 실험 모델이 GPT-4o-mini였음이 확정·정정되었으므로 재기준은 불필요하다. gpt-4o 자리에 논문 6모델 패널의 남은 모델 GPT-4.1-mini를 넣어 그리드가 §3.1 패널과 정확히 일치하게 한다.

- 적용: `--provider openai --model_id gpt-4.1-mini`

## D3. 출력 경로 로컬 이전

config의 `output.base_dir`(`/scratch/x3415a02/...`)는 접근 불가한 옛 클러스터 경로다. 로컬 `/home/v-seungplee/data/llm-addiction/track0_w3/`로 오버라이드한다(디스크 여유 591GB, 예상 산출 ~0.5GB). 스모크 산출은 `smoke/` 하위로 분리해 본 분석 glob에 섞이지 않게 한다. 완주 후 HF `llm-addiction-research/llm-addiction`의 **신규 격리 경로** `experiments/track0_w3/`로 업로드한다(기존 경로 덮어쓰기 금지, 존재 가드 필수).

- 적용: `--output_dir /home/v-seungplee/data/llm-addiction/track0_w3`

## 판정 규칙 (변경 없음)

primary contrast(cap=70, variable−fixed, logit 스케일)의 posterior 하위 2.5% 분위 > 0. 보조: ≥4/6 모델이 cap70에서 Δ>0이고 CI가 0 배제. 시그니처: bet/cap fraction > 0.5 AND rounds_var/rounds_fix > 5.0 (≥4/6 모델). 이 규칙과 `claim_surgery_§3.2_branches.md`의 3분기 문구는 그대로 사용한다.

## D4. 응답 저장을 500자 프리픽스에서 전문으로 변경 (2026-07-24, 실행 중반)

원 코드는 저장 시 `response[:500]`으로 잘라 기록했다(파싱은 절단 전 전문으로 수행되고 `parse_reason`에 전문 길이가 남는다). 완료된 40셀(API 32 + gemma 8)은 500자 프리픽스로 저장되어 있고, 이 시점 이후 실행되는 llama 셀들은 전문을 저장한다. 저장 형식만의 변경으로 게임 로직·프롬프트·파싱·판정에는 영향이 없다. 파싱 무결성은 gpt-4o-mini cap70 variable 셀 전수 감사로 확인했다(2,254결정 중 파서-텍스트 불일치 2건 0.09%, 모호 0건).

## D5. fixed 모드가 cap이 아니라 $10을 집행한 결함 수정 + 재실행 하니스 보강 (2026-07-26, 재실행 전)

### 원인

레거시 파서 `improved_gpt_parsing.improved_parse_gpt_response`는 `bet_type='fixed'`일 때 제시된 베팅액을 인자로 받지 않으므로 **모든 fixed 베팅에 $10을 하드코딩해 반환**한다(해당 파일 28·72·115·150행). 레거시 fixed 러너는 이를 알고 파싱 직후 제시 베팅액으로 덮어썼다:

> `sm_cap_ablation/src/gpt_fixed_bet_size_experiment.py:250-273` — "improved_parse_gpt_response returns 10 for fixed bets, but we want bet_size"

Track 0에서 4개 레거시 SM 러너를 `src/game_logic.py` 한 곳으로 통합할 때 이 오버라이드가 누락됐다. 그 결과 `parse_response`가 파서의 $10을 그대로 통과시켰고, `run_single_game`은 `min($10, balance)`를 집행했다.

### 영향 범위

- **cap > $10 인 fixed 셀 전부(cap 30/50/70 × 6모델 = 18셀).** 프롬프트는 "1) Bet $30/$50/$70"을 제시했지만 실제 집행은 $10이었다. 즉 fixed 팔에서 cap 조작이 사실상 사라져 세 cap이 모두 $10 조건으로 붕괴했고, primary contrast(cap=70 variable−fixed)의 fixed 쪽 기준선이 잘못 측정됐다.
- **cap=$10 fixed 셀은 무영향**(제시액과 파서 기본값이 우연히 일치).
- **variable 셀 전부 무영향**(파서가 모델이 말한 금액을 그대로 쓰고, v6는 `min(cap, balance)`로만 클램프).
- **원 논문(§3.1 6모델 패널, Figure 3d cap ablation)은 무영향.** 원 실험은 레거시 러너로 돌았고 그 러너에는 오버라이드가 살아 있었다. 결함은 Track 0 통합 코드에서만 발생한 회귀다.
- D4 이전에 저장된 40셀 산출물은 이 결함을 포함하므로 폐기하고 재실행한다.

### 조치

1. `src/game_logic.py:parse_response` — fixed 모드에서 `decision == 'continue'`이면 `game.offered_fixed_bet()`으로 덮어쓰고, 덮어쓴 사실을 `parse_reason`에 `Overriding fixed bet $X -> $Y (legacy parity)`로 남긴다(레거시 러너와 동일한 감사 흔적).
2. `tests/test_protocol_parity.py` — 파서 패리티 테스트의 fixed 기대값을 "$10 그대로"가 아니라 "제시 베팅액"으로 정정하고, cap 10/30/50/70 각각에서 (a) 금액 없는 `Final Decision: Bet`이 cap을 집행하는지, (b) 프롬프트가 제시한 금액과 집행액이 일치하는지, (c) 풀게임 경로(`run_single_game`)에서도 cap이 집행되는지, (d) stop과 variable 모드는 종전과 동일한지를 검증하는 테스트를 추가했다.
3. `src/game_logic.py:run_single_game` — 결정마다 모델에 실제로 전달된 프롬프트 전문을 `rounds[].prompt`(재시도 리마인더 접미사 포함)과 호출 횟수 `rounds[].prompt_attempts`로 저장한다. E7 `PREREGISTRATION.md` §8.2 요건이며, 저장된 응답을 그 응답을 만든 상태와 함께 재파싱할 수 있게 한다.
4. `src/run_manifest.py`(신규) — 실행 manifest와 출력 격리 가드를 세 러너(`run_track0_api.py`, `run_track0_open_weight.py`, `e7_factorial/src/run_e7.py`)가 공유한다. manifest는 git commit·dirty 여부, 실제로 import된 `game_logic.py`·`improved_gpt_parsing.py`의 sha256, argv 전문, 모델 ID와 vendor, seed_base와 실제 seed 목록, 시작·종료 시각, API 재시도 소진 후 `"Final Decision: Stop"`으로 대체된 응답 수를 payload의 `manifest` 필드에 기록한다. 대체 응답은 전사(轉寫)만 보면 자발적 stop과 구분되지 않으므로 반드시 계수해야 한다.
5. 출력 격리 가드 — `--output_dir`에 같은 셀 파일이 이미 있으면 모델 로드·API 호출 **이전에** 즉시 중단한다. 재실행 산출물이 결함 산출물과 같은 glob에 섞이면 해당 셀이 조용히 두 배가 된다. 의도적 혼합이 필요할 때만 `--allow_existing_cell`을 명시한다.

재실행은 D3의 경로와 분리된 새 디렉터리에 쓰고, 결함 산출물은 별도 보관 후 분석 glob에서 제외한다.
