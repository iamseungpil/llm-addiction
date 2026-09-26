# Multi-Layer Causal Intervention (E1→E2→E3) — Design Spec

**Date**: 2026-06-10
**Status**: Approved (user, 2026-06-10) — full ladder E1→E2→E3; all metrics recorded exploratorily
**Repo home**: `multilayer_causal/` (new top-level folder; zero edits to existing experiment code)
**Paper context**: NeurIPS/EMNLP 2026 "Can LLMs Develop Gambling Addiction?" — follow-up to the
M3/M3′/M3″ single-layer causal nulls (Appendix F.4) and the open question of a distributed
multi-layer controller locus.

---

## 1. Scientific framing

§4.1–4.3 established that gambling-risk state is *readable* (Ridge readout R² at L22).
M3 (prompt swap), M3′ (L22 additive dose ladder, r=+0.013), and M3″ (L22 full-prompt paired
patch, all-position, Welch p>0.15 vs natural −G) established that it is *not writable at any
single layer*. This matches the hydra-effect / self-repair literature (McGrath et al. 2023)
and the 2025 steering consensus that distributed behaviours require simultaneous multi-layer
intervention.

Three questions, three experiments:

| | Question | Manipulation | Protocol |
|---|---|---|---|
| **E1** | WHERE is the −G→+G autonomy effect consolidated? | Replace hidden states of a layer **set** with paired +G values | single-decision (M3″ ext.) |
| **E2** | WHAT RANK of subspace carries it? | Replace only top-r PCA components within S\* | single-decision |
| **E3** | Is the axis a CONTROLLER (dose-response on irrationality indices)? | Add α·v̂ at all layers of S\* | single-decision ladder **+ full-game** |

Positive control exists by construction: patching ALL layers ≈ running the +G prompt, so the
endpoint of E1's sweep is pinned. The deliverable is the *minimal* layer set / rank, or a
systematic full-sweep null (both publishable).

Prior-work anchors: hydra effect (McGrath 2023); multi-layer simultaneous steering necessity
(LayerNavigator, OpenReview 2025); whole-network rank-1 intervention precedent (Arditi et al.
2024 refusal direction); subspace-patching illusion + random-subspace controls (Makelov et al.
2023, arXiv:2311.17030); SAE-targeted steering (Chalnev et al. 2024, already cited in paper).

## 2. Protocol details

### 2.0 Shared scaffolding (inherited from M3″, frozen)

- Model: `google/gemma-2-9b-it`, bf16, single GPU per arm. 42 decoder layers, d=3584.
- Task: slot machine (SM), variable betting, −G state pool from §3 corpus
  (`behavioral/slot_machine/gemma_v4_role/*.json`, rounds 2–6 per game, shuffled with
  `random.Random(42)`, same as M3″ `load_minusG_states`).
- Generation: temperature 0.7, do_sample, max_new_tokens 200, per-trial seed `42 + i*997`
  (M3″ convention). Trial i uses state `states[i % len(states)]`.
- Prompt builder + parser: byte-identical copies of M3″ `build_prompt` / `parse_response`
  (copied into `multilayer_causal/src/prompts.py` with provenance header; a CPU test asserts
  equality with the originals to prevent drift).
- Paired construction per trial: −G prompt and +G twin (= same state, combo + 'G'),
  cache +G forward once at ALL target layers, then patch hooks fire once on the −G
  generation prefill (`seq_len > 1`, first forward). Patched prompt activations persist
  through the KV cache.
- Anchors per phase: `natural_minusG` and `natural_plusG` (no hook), n matched to arms.

### 2.1 E1 — multi-layer paired patch sweep (17 arms + 2 anchors, n=50 each)

Patch mode: all-position (`patched_all` semantics, suffix-aligned like M3″).

| Family | Arms |
|---|---|
| full (positive control) | `L0-41` |
| cumulative bottom-up | `L0-8`, `L0-16`, `L0-22`, `L0-30`, `L0-36` |
| cumulative top-down | `L8-41`, `L16-41`, `L22-41`, `L30-41` |
| sliding width-6 | `L0-5`, `L6-11`, `L12-17`, `L18-23`, `L24-29`, `L30-35`, `L36-41` |

Decision-level metrics (recorded for every trial, every phase): action (bet/stop), bet amount,
bet_ratio (decision-level I_BA), extreme-bet flag (bet == max allowed; I_EC proxy),
parse-validity, response text (300 chars), per-layer last-token hidden vectors of BOTH the
−G run and the +G twin (fp16; powers E2/E3 directions for free).

**Gate G1 (harness)**: full-layer arm must move behaviour to the +G anchor
(Welch p>0.05 vs natural_plusG on bet_ratio AND |Δstop-rate| < 0.15). Fail → stop, debug.

**S\* selection (pre-registered)**: among arms statistically indistinguishable from
natural_plusG (same criterion), pick the smallest layer count; ties → the deeper window.
If only the full arm passes → S\* := all layers, interpretation "distributed", E2/E3 proceed
on S\* = L0–41.

### 2.2 E2 — rank-r subspace patch at S\* (6 arms + random controls, n=50 each)

For layer ℓ ∈ S\*: `h ← h_minus + P_r^ℓ (h_plus − h_minus)` applied at the same positions as E1,
where `P_r^ℓ` projects onto the top-r PCA components of natural decision-point states at ℓ
(basis fixed offline from HF `sae_features_v3/slot_machine/gemma/checkpoint/phase_a_hidden_states.npz`,
21,421 rounds; mean-centered, components frozen and shipped with the job).

- r ∈ {1, 2, 8, 32, 128, 3584(=full, consistency check vs E1)}.
- **Random-subspace controls** (Makelov-illusion defence): r ∈ {1, 8, 32} × 10 random
  orthonormal bases each (`numpy PCG64(2026061000+k)`), reported as null distributions.
- Report cos similarity of the r=1 component vs (a) Table 2 LOTO-PCA shared axis,
  (b) the paired mean-delta direction from E1 logs.

### 2.3 E3 — multi-layer additive steering (dose-response)

Direction per layer: `v̂_ℓ = normalize(mean_i(h_plus^{ℓ,i} − h_minus^{ℓ,i}))` from E1's logged
pairs (last-token). **Per-layer magnitude calibration** (v16 lesson — Gemma hidden norms vary
3–81 across depth): unit dose α=1 adds `0.03 · median_token_norm(ℓ) · v̂_ℓ`, i.e. 3% of the
layer's natural hidden-state norm. Steering hook adds at ALL positions on EVERY forward
(prefill + decode), no firing guard.

- **E3a single-decision ladder** (direct M3′ comparison): α ∈ {−2, −1, −0.5, 0, +0.5, +1, +2},
  7 arms × n=50, at S\*. Random-direction control at α=+2 (10 dirs). Spearman ρ(α, bet_ratio),
  10,000-permutation randomization test.
- **E3b full-game mode** (irrationality indices live here): via read-only import of
  `sae_v3_analysis/src/exact_behavioral_replay.py` (`play_exact_behavioral_game`), steering
  active for the whole game. α ∈ {−1, 0, +1} × n=30 games. Metrics: per-round I_BA, I_LC
  (post-loss bet escalation), I_EC, bankruptcy rate, stop rate, rounds played, final balance.
- **Generalization arm (gated on E3a positive)**: Gemma IC single-decision ladder, same axis
  transferred (tests the Table-2 shared-axis claim causally).

### 2.4 Statistics policy

All metrics recorded for all arms; all tests computed per metric and labelled EXPLORATORY.
Any result promoted to the paper body triggers a separate confirmatory run: n=200, held-out
state pool (g≥1000 convention from L22_PLAN_v5), the single pre-named metric, 10,000-sign-flip
randomization test. This two-stage design avoids multiplicity games.

## 3. Code design (clean-room, isolated)

```
multilayer_causal/
├── README.md                  # what/why/how, arm registry table, HF paths
├── configs/
│   └── arms.yaml              # every arm: id, phase, mode, layers, r, alpha, n
├── src/
│   ├── prompts.py             # build_prompt/parse_response (frozen copies + provenance)
│   ├── states.py              # −G state pool loader (frozen shuffle), HF download helper
│   ├── hooks.py               # MultiLayerPatcher / SubspacePatcher / MultiLayerSteerer
│   ├── pca_basis.py           # offline: phase_a npz → per-layer PCA bases (.npz)
│   ├── runner.py              # trial loop (single-decision) + game loop (E3b)
│   ├── checkpoint.py          # JSONL append + HF latest-state sync + resume
│   └── analyze.py             # arm summaries, anchors, gates, stats
├── run_experiment.py          # CLI: --arm <id> [--n N] [--gpu K] [--smoke]
├── run_arms.sh                # in-job scheduler: shard arm list across visible GPUs
├── amlt/
│   ├── smoke.yaml             # 1 GPU, full-layer arm, n=3
│   └── e1_main.yaml           # 4×GPU, 19 arms sharded
└── tests/                     # CPU-only: hooks on a tiny mock transformer, prompt parity,
                               # checkpoint resume round-trip, arm registry sanity
```

Principles (karpathy-guidelines): no framework, one runner; hooks are 3 small classes with
`install(model)/remove()`; every assumption is an assert (layer count, d_model, suffix
alignment); arms fully declarative in `arms.yaml`; no edits to any existing experiment file.
`sae_v3_analysis` is imported read-only in exactly one place (E3b replay).

## 4. Operations: resume, HF sync, monitoring

### 4.1 Checkpoint + resume (preemption-safe)

- Local: `out/{arm}.jsonl`, one line per completed trial/game, `flush+fsync` after each
  (M3″ pattern). Resume = read done seeds, skip.
- HF sync: every 10 completed trials (and at exit), upload the arm's JSONL to
  `experiments/multilayer_causal/checkpoints/{phase}/{arm}.jsonl` in
  `llm-addiction-research/llm-addiction` — **same path, overwritten** (latest state, not an
  archive; git history preserves the trail automatically).
- Job start: download the arm's checkpoint from HF if present → resume. Therefore a
  preempted job is resumed by simple resubmission; frozen seeds make the result identical
  to an uninterrupted run.
- Final results + summaries → `experiments/multilayer_causal/results/`.

### 4.2 amlt jobs (metacognition-math pattern)

- Target `sing/msrresrchbasicvc`, image `amlt-sing/acpt-torch2.7.1-py3.10-cuda12.6-ubuntu22.04`,
  managed identity, HF_TOKEN env (verify token has write access to the org dataset before
  baking in), `max_run_duration_seconds` set.
- Code delivery: tarball `experiments/multilayer_causal/code/multilayer_causal.tar.gz` on the
  HF dataset (overwritten on each push); job pulls + extracts + `pip install -q` the few deps.
- Data delivery: job downloads behavioural catalog + (E2 only) PCA bases from the HF dataset.
- Iterative submission: ① smoke (1 GPU: env check, model load, full-layer arm n=3, parse
  validity ≥0.8) → ② E1 main (4 GPUs, 19 arms via `run_arms.sh`) → ③ E2 → ④ E3a/E3b.
  Each phase submitted only after the previous one's gate is read.

### 4.3 Monitoring loop

Periodic (ScheduleWakeup, ~20–30 min): `amlt status` + tail logs (env-setup success on first
check), HF checkpoint sizes (arm progress), parse-validity rates. On preempt/failure:
diagnose from logs, resubmit (resume makes this cheap). Report anomalies to user; report
phase results + gate decisions as each phase completes.

## 5. Paper placement (decided up front)

| Outcome | Action |
|---|---|
| E1 minimal window found | Body §4.4 +2–4 sentences ("writable at L_a–L_b"); full table → Appendix F.4 as M3‴ |
| Only full-layer works | Appendix F.4 ext.; discussion gains "distributed, not locally writable" |
| E2 small r\* | Table 2 LOTO axis promoted predictive→causal; body paragraph "Localizing and controlling the risk axis" |
| E3 dose-response on I_BA/I_LC | The paper's first positive causal result on irrationality indices; direct contrast with M3′ single-layer null |
| Any | EMNLP `6.limitations.tex` open-question sentence replaced with the answer; NeurIPS checklist limitations answer checked |

## 6. Compute budget

| Phase | Arms × n | GPU-hours (A100/H100 est.) |
|---|---|---|
| smoke | 1 × 3 | <0.5 |
| E1 | 19 × 50 | ~24–40 (4 GPUs ≤ 1 day) |
| E2 | 6 + 30 ctrl × 50 | ~40 |
| E3a | 7 + 10 ctrl × 50 | ~20 |
| E3b | 3 × 30 games (~30 decisions/game) | ~30 |

Single-decision trial ≈ 1 generation of ≤200 tokens + 1 cache forward ≈ 10–20 s.

## 7. Out of scope

- LLaMA SM/MW arms (need phase_a re-extraction; revisit only if Gemma results are positive).
- SAE-feature-targeted steering (Chalnev-style) — separate follow-up.
- NMI manuscript updates.
