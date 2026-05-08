# Plan — 4-node H100 keep-alive + SAE multi-feature extension (Discovery)

> **Status**: Discovery doc (2026-05-07). 실행 X. 사용자 승인 후 별도 실행 plan 작성.
> **Project**: `llm-addiction` (LLM gambling addiction NeurIPS submission)
> **Reference**: `/home/v-seungplee/metacognition/h200_meta_opd_R7_0506.yaml` 패턴

---

## 0. 목적

Reviewer 피드백 + 사용자 의문에 답하기 위해 **4 H100 노드를 keep-alive holder로 신청**하고, SSH로 들어가 필요할 때 SAE 인과성 patching 확장 실험을 돌리는 인프라를 갖춘다.

**핵심 의문 (사용자)**:
1. 인과성 patching을 단일 feature가 아닌 multi-feature 동시 조작으로 확장할 수 있는가
2. 지금 §4 실험들이 모두 complete한가
3. Correlation 분석이 상위 feature만 본 것은 위험하지 않은가
4. "올바른 feature"를 본 것이 맞는가 (selection 검증)

---

## 1. AMLT 노드 신청 방식 (metacognition 폴더에서 추출)

### 1.1 yaml 패턴 — `meta_opd_r7` 사례

```yaml
description: <experiment description>

target:
  service: sing
  name: msrresrchbasicvc          # H200/H100 모두 가능
  workspace_name: msra-sh-aml-ws

environment:
  image: amlt-sing/acpt-torch2.7.1-py3.10-cuda12.6-ubuntu22.04

code:
  local_dir: $CONFIG_DIR/

jobs:
  - name: <holder_name>
    sku: 80G4-H100                # 1 node × 4 H100 80GB
    sla_tier: Standard            # H100은 Standard, H200은 Basic만 가능
    priority: high
    identity: managed
    submit_args:
      max_run_duration_seconds: 604800   # 7일
      env:
        _AZUREML_SINGULARITY_JOB_UAI: <UAI ARM ID>
        HF_TOKEN: <token>
        HUGGING_FACE_HUB_TOKEN: <token>
        WANDB_API_KEY: <token>
        WANDB_PROJECT: skilldiscovery2
    command:
      - nvidia-smi
      - |
        bash -c '
        # 1. Code from HF tarball (no SSH base64 — too big)
        # 2. Bootstrap conda env (idempotent fast-path)
        # 3. Stage data + checkpoints from HF
        # 4. Launch background daemons (gpu_keeper + push_to_hf)
        # 5. Foreground task (or sleep 86400 for keep-alive)
        '
```

### 1.2 SKU 선택 (메모리 기반)

| SKU | GPU | 사용처 | sla_tier | preempt? |
|---|---|---|---|---|
| `80G4-H100` | 4× H100 80GB | OPD/RLSD training, multi-GPU SAE | **Standard** | no |
| `80G4-H200` | 4× H200 141GB | 큰 모델 inference, long-context | Basic only | yes (~17 min idle) |
| `80G4-A100` | 4× A100 80GB | NCv4 retired | — | — |

→ **본 실험에는 80G4-H100 / Standard tier 권장** (preempt risk 없음, 7일 max).

### 1.3 Keep-alive 패턴

```bash
# Background daemons (이미 metacognition에 있는 검증된 도구)
nohup python /scratch/<repo>/scripts/gpu_keeper.py > /scratch/logs/gpu_keeper.log 2>&1 &
nohup python /scratch/<repo>/scripts/push_ckpts_to_hf.py \
    --ckpt_dir /scratch/checkpoints/<exp> \
    --repo_id iamseungpil/<repo_id> \
    --token $HF_TOKEN \
    --interval 90 > /scratch/logs/push.log 2>&1 &

# Foreground sleep 또는 실험
sleep 86400   # 1일 keep-alive (반복 가능). BSC H200은 idle suspend 17분이라 gpu_keeper 필수.
```

**중요 (memory 참조)**:
- BSC H200 idle-suspend ~17분 → `gpu_keeper.py`로 GPU touch 유지
- AMLT outer foreground exit = job pass → keep-alive command를 outer에서 끊기지 않게 sleep
- Big code uploads via HF `code_snapshots/<repo>.tar.gz` (SSH base64 X)

### 1.4 NODE_POLICY 패턴 (frozen contract)

```markdown
N. <holder_name>
   - AMLT experiment: <experiment>
   - Hardware: 4× H100 80GB (msrresrchbasicvc, Standard)
   - Project owner: <project>
   - Allowed work: <specific tasks>
   - Disallowed work: <other projects' jobs>
```

→ **각 노드별 owner project 명시 + cross-project 사용 금지**. AMLT 점유 충돌 방지.

### 1.5 Code distribution

`metacognition` 패턴: `iamseungpil/metacot/code_snapshots/metacognition.tar.gz` HF에 push → 노드에서 pull.

→ **`llm-addiction`도 동일 패턴**: 코드를 tarball로 묶어 HF dataset (e.g., `iamseungpil/llm-addiction-code-snapshot`) push → AMLT yaml에서 download.

---

## 2. LLM_Addiction §4 SAE 분석 — 현재 상태

### 2.1 Paper claim (현재 본문)

| Cell | Paper에 보고 | Code 위치 |
|---|---|---|
| Gemma/LLaMA × SM/IC/MW × {I_BA, I_LC, I_EC} | ✅ Table 1 (R²) | `sae_v3_analysis/src/run_rq1_group_selectivity_l2.py` |
| Cross-task BK PCA (LOTO) | ✅ Table 2 | `sae_v3_analysis/src/build_neural_readout_panel.py` |
| Autonomy modulation (±G/±M) | ✅ Table 3 | `sae_v3_analysis/src/condition_analysis.py` |
| Causal patching (M3) → null | ✅ §4.3 + Appendix E.4 | `paper_experiments/llama_sae_analysis/src/phase4_causal_pilot_v2.py` |
| **Multi-feature patching** | ❌ **본문 없음** | `paper_experiments/pathway_token_analysis/src/phase1_patching_multifeature.py` (코드 존재, 결과 미보고) |
| **Top-K sensitivity** (k=100/200/500) | ❌ **본문 없음** | `sae_v3_analysis/src/run_correlation_analysis.py` (k 변경 가능) |

### 2.2 사용자 의문 매핑

#### Q1. "Multi-feature 동시 조작 가능?"
- **답**: ✅ 코드 이미 존재 (`phase1_patching_multifeature.py` 358 lines).
- 현재 `top_n=2787` (모든 causal feature) 지원. 단, paper §4.3 M3 결과는 **단일 feature 단위** patching → null 보고.
- **새 실험 후보**: Top-K subset (예: top-10, top-50) 동시 patching → behavioural 변화 측정.

#### Q2. "지금 실험들 모두 complete?"
- §4 본문 cells 모두 보고됨 (Table 1).
- **빠진 것**:
  - Multi-feature simultaneous patching (위)
  - Top-K sensitivity (아래)
  - Open-weight cap ablation (reviewer Q3, rebuttal 핵심)
  - ILC balance-windowed + 절대 베팅 delta (reviewer Q1)
  - LOTO PCA random-init stability (reviewer Q7)

#### Q3. "상위 feature만 본 게 위험?"
- **현 상태**: top-200 features (Spearman rank correlation 기준)에서 Ridge readout fit.
- **위험 요소**:
  1. k=200 cherry-picked (sensitivity 미검증)
  2. Spearman vs Pearson sensitivity 미검증
  3. FDR threshold sensitivity 미검증
- **새 실험 후보**: k ∈ {50, 100, 200, 500, 1000} × 6 cells × 3 indicators = 90 R² 조합 stability check.

#### Q4. "올바른 feature를 본 것이 맞나?"
- **현 상태**: Per-fold rank correlation으로 selection (training fold-only, no leakage).
- **검증 방향**:
  1. Selection 기준 다양화 (rank corr / mutual info / random forest importance)
  2. Top-K vs random-K baseline 비교 (현 쇼핑 결과와 noise floor 차이)
  3. Selected features의 semantic interpretation (이미 phase3_semantic_analysis.py 있음)

---

## 3. 4-node H100 keep-alive 활용 plan (high-level)

### 3.1 4 노드 분배 (가설)

| Node | AMLT name (가설) | 주 실험 | 우선순위 |
|---|---|---|---|
| 1 | `addiction_cap_ablation` | Open-weight cap ablation (LLaMA, Gemma) | **Critical** (rebuttal Q3) |
| 2 | `addiction_multifeature_patching` | Multi-feature simultaneous patching (LLaMA L22) | **Important** (사용자 Q1) |
| 3 | `addiction_topk_sensitivity` | k ∈ {50,100,200,500,1000} sensitivity | **Important** (rebuttal Q2 + 사용자 Q3) |
| 4 | `addiction_robustness` | ILC balance-windowed, decoding params, LOTO stability | **Useful** (rebuttal Q1, Q4, Q7) |

### 3.2 NODE_POLICY 추가 항목 (가설)

```markdown
7. addiction_cap_ablation
   - AMLT experiment: <new>
   - Hardware: 4× H100 80GB (msrresrchbasicvc, Standard)
   - Project owner: llm-addiction
   - Allowed work: cap ablation, ILC stratification
   - Disallowed: metacognition Meta-CoT, RSP GRPO

8. addiction_multifeature_patching
   - 동일 spec, owner: llm-addiction
   - Allowed: multi-feature patching, semantic analysis
   - Disallowed: 다른 프로젝트

9. addiction_topk_sensitivity
   - 동일 spec, owner: llm-addiction
   - Allowed: top-K sweep, correlation alternatives
   - Disallowed: 다른 프로젝트

10. addiction_robustness
    - 동일 spec, owner: llm-addiction
    - Allowed: decoding param sweep, LOTO stability, BK split
    - Disallowed: 다른 프로젝트
```

→ NODE_POLICY 본 패턴 그대로 6 → 10 노드로 확장. metacognition holder는 그대로 유지 (cross-project 충돌 없음).

### 3.3 Code distribution 준비

1. `llm-addiction` 코드를 tarball로 압축 → HF dataset push
2. AMLT yaml에서 HF download → `/scratch/llm-addiction/` extract
3. Bootstrap script (`scripts/bootstrap_addiction_node.sh` 신규 작성 필요):
   - conda env (`llm-addiction` from memory `env_conda.md`)
   - PyTorch + sae_lens 설치
   - LlamaScope/GemmaScope HF download
   - Existing raw data (SM/IC/MW JSONs) HF download

---

## 4. 핵심 결정 포인트 (사용자 승인 필요)

### 4.1 노드 owner 명명
- 옵션 A: `addiction_*` 4개 (project-specific, 위 안)
- 옵션 B: `sae_audit_node_1~4` (analysis-specific)
- 옵션 C: 단일 holder + 4-job (1 yaml로 4 jobs)

### 4.2 첫 번째 실행 노드
- 옵션 A: 가장 critical인 cap ablation부터
- 옵션 B: 가장 빠른 Top-K sensitivity (raw data 이미 있어 inference 불필요)부터
- 옵션 C: 4 노드 동시 신청 → 모든 실험 병렬

### 4.3 데이터 제공 경로
- LLaMA-3.1-8B-Instruct/Gemma-2-9B-IT는 HF에서 직접 download
- SM raw data (3,200 games × 6 models): HF dataset push 필요? 또는 현재 host에서 access?
- IC raw data: 동일

### 4.4 Multi-feature patching 실험 설계
- Top-K subset 정의 (k=10, 50, 100?)
- Patching value (zero-out / mean-replace / negate-direction?)
- Outcome metric (bankruptcy rate / I_EC change?)
- Trial count per condition

### 4.5 Top-K sensitivity 실험 설계
- k 값 grid (50/100/200/500/1000)
- 6 cells × 3 indicators × 5 k = 90 fits
- Selection 기준 (rank corr / mutual info / RF importance) 추가?

### 4.6 7일 max 후 처리
- 노드 자동 expire 후 재신청
- 또는 7일 안에 모든 실험 완료 후 cancel

---

## 5. 다음 단계 (사용자 승인 후)

승인되면 다음 4가지 산출물 작성:

1. **`PLAN_4NODE_EXECUTION_2026_05_07.md`** — 구체적 실행 plan (어떤 노드에 무슨 실험, 어떤 순서)
2. **AMLT yamls (4개)** — `addiction_node_1.yaml` … `addiction_node_4.yaml`
3. **`bootstrap_addiction_node.sh`** — env setup
4. **`NODE_POLICY.md` 갱신안** — `llm-addiction` 4개 holder 추가

추가 결정 필요:
- Tarball에 포함할 코드 범위 (`paper_experiments/` + `sae_v3_analysis/` + `exploratory_experiments/`?)
- HF dataset 이름 (`iamseungpil/llm-addiction-code-snapshot`?)
- Raw data 경로 (현재 `/home/v-seungplee/data/llm-addiction/...`, 노드에서 access 어떻게?)
- LLaMA/Gemma SAE dictionary cache (LlamaScope ~수GB)

---

## 6. Memory references

- `reference_amlt_submit.md` — AMLT 노드 신청 검증 패턴
- `feedback_amlt_project.md` — 절대 checkout 금지, .amltconfig
- `feedback_amlt_tier_h200_h100.md` — H200=Basic only(preempt), H100 Standard
- `feedback_amlt_queue.md` — Never cancel queued siblings
- `feedback_amlt_job_preservation.md` — outer foreground exit = job pass
- `feedback_hf_bootstrap.md` — HF tarball, BSC idle-suspend 17min
- `feedback_bsc_idle_suspend.md` — gpu_keeper.py 필수
- `feedback_hf_sync.md` — 모든 모델/데이터/결과 HF 동기화

---

## Status

**현재 단계**: Discovery only. 사용자 승인 시 → §5 산출물 작성.
**실험 진행 X** — 명령 명시: "실험을 진행하진 말고, 노드 방식 같은 걸 살펴본 뒤 어떻게 동작시킬지 고민해줘".

승인 받을 항목:
- §3.1 노드 분배 (4 노드 × 4 실험)
- §3.2 NODE_POLICY 추가 (`addiction_*` 4개)
- §4 결정 포인트 6개 (명명/순서/데이터/multi-feature 설계/top-K 설계/만료 처리)
