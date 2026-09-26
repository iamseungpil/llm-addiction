# §4 Causal-Strengthening — Phase 1 Implementation Plan

> **For agentic workers:** Implement task-by-task. Steps use `- [ ]` checkboxes. Clean-code (karpathy) discipline: no dead code, each file one responsibility, cleanly separated from prior waves (spine/xtask/xtaskd) by the `sec4_` prefix and the `experiments/sec4_causal/` HF namespace.

**Goal:** Establish ONE bulletproof causal result — *the −G→+G autonomy effect is writable into Gemma-2-9B slot-machine behaviour at the L16–21 multi-layer band, and the direction that WRITES behaviour is distinct from the direction that READS it (monitor ≠ controller)* — with every prior-failure bug pre-empted, then expand only if it passes.

**Architecture:** A direction-library builder (`indicator_axes.py`) produces three deconfounded steering axes per (model, task, indicator) — readout (SAE-decoder-projected), behavioural (residualised top/bottom split), confound (balance+round). The existing generic steering runner replays exact addiction prompts, injects `α·scale·dir` at L16–21, and logs per-arm outcomes to HF (resume-safe) and W&B. An analysis module (`sec4_stats.py`) scores dose-response, sign-reversal, coherence, null band, and read-vs-write cosine. An autoresearch loop polls, analyses, checks pre-registered criteria, and gates expansion.

**Tech Stack:** Python, PyTorch, HF transformers, Gemma-Scope SAE via `sae_lens`, amlt (sing/msrresrchbasicvc 80G4-H100), Weights & Biases, HF datasets for checkpoints.

---

## PART I — INTENT & HYPOTHESES (pre-registration)

### 0. North star
The paper §4 shows internal risk is *readable* (correlational): "monitoring, not control." Five prior causal attempts (V12/V14 steering, M3/M3′/M3″ patching, aligned-axis A/B/C) all produced **null, invalidated, or sign-inverted** results. We do NOT overturn the paper's scoping. We *strengthen* it: prove causally that the read direction is inert and a distinct, narrow-band write direction exists — turning the paper's null into a characterised, mechanistic positive.

### 1. The two axes (the whole intellectual core)
For an indicator `k` (I_BA bet-aggression, I_LC loss-chasing, I_EC extreme-choice) on task `t`, two candidate "risk directions" in the residual stream:
- **Readout axis** `r_{t,k}` = SAE-decoder projection of the §4.1 Ridge readout — the direction that best *predicts* `k`. The "thermometer."
- **Behavioural axis** `b_{t,k}` = mean(top-residual-quantile states) − mean(bottom) — the direction along which `k` *behaviour* varies. The candidate "heater."
Both are **deconfounded identically** (indicator residualised against balance+round within-fold) so any read-vs-write difference is NOT a deconfounding artefact.
- **Confound axis** `c_t` = the balance+round direction — a control proving the write is not merely "tell the model it has more money."

### 2. Hypotheses (pre-registered; sign fixed in advance)
- **H1 (writable window).** Injecting `b_{SM,BA}` at L16–21 (multi-layer) shifts betting **monotonically** with α and **reverses sign** (−α ⇒ less betting, +α ⇒ more), staying coherent (parse ≥ 0.8), above the random-null band. *Prediction:* recovery of the ±G anchor gap (0.141) clearly > null (~0.11), Spearman(α, bet)≥0.7, correctly signed.
- **H2 (monitor ≠ controller).** At matched injection magnitude (σ-units) and matched n, the **readout axis `r_{SM,BA}` is inert** (no monotone dose, indistinguishable from null) while `b_{SM,BA}` writes. Report decoding AUC of both — the finding is *decoding strength does not confer control* (the better decoder is inert).
- **H3 (not a confound).** `b_{SM,BA}` writes **beyond** the confound axis `c_SM`; steering `c_SM` reproduces at most a fraction of the effect.
- **H4 (locality).** Single-layer injection is insufficient; recovery concentrates in the L16–21 band and over-widening derails (parse↓). (Confirms Stage D with fixed bugs.)

### 3. Success / failure semantics (every outcome has a claim)
- **H1+H2 hold** → CORE positive: "a narrow-band write axis controls betting; the read axis is causally inert." GO to Phase 2.
- **H1 holds, H2 fails (readout also writes)** → bigger story (read direction is also causal); re-frame, still positive.
- **H1 fails (behavioural inert even at L16–21)** → NO-GO for the write claim; pre-registered pivot: report as "no writable locus survives exact-prompt replay" (extends the M3 null honestly), inspect α-range / manifold projection once (stopping rule ≤1 refinement), else terminal null.
- **H3 fails (confound explains it)** → the effect is a balance-signalling artefact, not risk control; report as such.

### 4. Past-failure fixes baked in (weakness elimination)
| Prior failure | Root cause | Fix in this plan |
|---|---|---|
| V12/V14 headline invalid | direction built on addiction prompts, tested on bare prompts | **exact addiction-prompt replay** (ROLE+G+M); assert prompt-set hash matches build set |
| paired dose broken | α-dependent seeding | **α-independent seeds** `42+997·i` |
| M3/M3′/L22 null | single-layer | **L16–21 multi-layer** + cumulative window arm |
| aligned A/C "significant but negative" | sign ignored | **pre-registered sign**; require monotone + sign-reversal; no auto-"significant" label |
| off-diag parse→0.02 | α>2–3 off-manifold | **α ∈ [−3,+3]**, per-arm parse gate, never read behaviour off <0.8 parse |
| readout non-causal but confounded compare | raw axis carries balance/round | **both axes deconfounded**; explicit confound axis |
| underpowered null | n50, single null dir | **n≥200, ≥5 random-null seeds** |
| manip check dead | `manip_proj` NaN (key path) | **read `vector_log.proj`**; assert non-NaN |
| provenance drift | 10-s smoke JSONs as real | **run-metadata assertion** (n, runtime, replay hash) in every result |
| IC/MW no headroom | floor 0.13 / ceiling 0.78 | Phase 1 is **SM-only**; IC/MW deferred to Phase 2 with re-baseline |

### 5. Guardrails
Pre-registration (this doc) + discovery/held-out game split + full-ladder reporting (every arm, incl. nulls) + stopping rule (≤3 refinements/phase) + all controls carried every arm (random null, sign-reversal, parse gate, confound axis).

---

## PART II — FILE STRUCTURE (clean separation)

New code lives under the `sec4_` prefix; HF/results under `experiments/sec4_causal/`. Existing generic infra (`runner.py`, `checkpoint.py`, `hooks.py`, `paper_axes.py`) is **reused unmodified except two additive hooks** (wandb call, manip_proj key fix). Prior waves (spine/xtask/xtaskd) are untouched.

- Create `src/indicator_axes.py` — build readout/behavioural/confound axes (deconfounded), save to `assets/sec4/`.
- Create `src/wandb_logger.py` — thin per-arm W&B logger (no-op if `WANDB_DISABLED`).
- Create `src/sec4_stats.py` — dose-response / sign / parse / null / cosine / decoding-AUC analysis + figure.
- Create `configs/arms_sec4_p0.yaml` — Phase-1 CORE arms (Gemma SM).
- Create `amlt/sec4_p0.yaml.template` — node job (mirror `xtaskd.yaml.template` + `sae_lens` pip + `experiments/sec4_causal` env).
- Create `scripts/watch_sec4.sh` — poll amlt + HF cell count.
- Create `scripts/sec4_autoresearch.py` — poll→analyse→check criteria→emit GO/NO-GO + INDEX append.
- Create `experiments/sec4_causal/INDEX.md` (committed stub; HF mirror) — rung ledger.
- Modify `src/runner.py` — (a) fix `manip_proj` to read `vector_log.proj`; (b) call `wandb_logger.log_arm(...)` at each arm DONE; (c) assert `arm['prompt_set_hash']` matches direction build hash (replay guard).
- Modify `src/checkpoint.py` — record run-metadata (n, start/end, replay hash) in the arm summary.
- Test `tests/test_indicator_axes.py`, `tests/test_sec4_stats.py`, `tests/test_replay_guard.py`.

---

## PART III — TASKS

### Task 1: Direction-library builder — deconfounded readout axis
**Files:** Create `src/indicator_axes.py`; Test `tests/test_indicator_axes.py`

- [ ] **Step 1: failing test — readout axis shape & unit norm & AUC gate**
```python
# tests/test_indicator_axes.py
import numpy as np
from multilayer_causal.src import indicator_axes as ia

def test_readout_axis_smoke(tmp_path):
    # synthetic: 200 rows, 64-dim SAE feats, 8-dim hidden, 2 layers
    d = ia.build_readout_axis_from_arrays(
        feats=np.random.RandomState(0).randn(200, 64),
        indicator=np.random.RandomState(1).randn(200),
        balance=np.random.RandomState(2).rand(200),
        rounds=np.random.RandomState(3).randint(1, 20, 200),
        groups=np.arange(200) // 4,
        decoder=np.random.RandomState(4).randn(64, 8),
        layers=[0, 1],
    )
    assert d["directions"].shape == (2, 8)
    assert np.allclose(np.linalg.norm(d["directions"], axis=1), 1.0, atol=1e-5)
    assert "auc" in d and "provenance" in d
```
- [ ] **Step 2: run → FAIL** `pytest tests/test_indicator_axes.py::test_readout_axis_smoke -v`
- [ ] **Step 3: implement `build_readout_axis_from_arrays`** — within-fold RF deconfound (reuse `paper_axes.rf_deconfound_split`), top-200 Spearman select, ridge fit (`paper_axes.fit_full_ridge`), `paper_axes.decoder_map` per layer, unit-norm; compute held-out decoding AUC; return dict with `directions,scales,auc,provenance`.
- [ ] **Step 4: run → PASS**
- [ ] **Step 5: commit** `git add -A && git commit -m "feat(sec4): deconfounded readout axis builder"`

### Task 2: Behavioural + confound axes (deconfounded, matched)
**Files:** Modify `src/indicator_axes.py`; Test `tests/test_indicator_axes.py`

- [ ] **Step 1: failing test** — behavioural axis from residual top/bottom split returns (L,d) unit; confound axis returns (L,d) unit; behavioural is deconfounded (residualise indicator vs balance+round before split).
```python
def test_behavioural_and_confound_axes():
    rs = np.random.RandomState(0)
    hidden = rs.randn(300, 2, 8); ind = rs.randn(300)
    bal = rs.rand(300); rnd = rs.randint(1, 20, 300); grp = np.arange(300)//3
    b = ia.build_behavioural_axis_from_arrays(hidden, ind, bal, rnd, grp, layers=[0,1], q=0.25)
    c = ia.build_confound_axis_from_arrays(hidden, bal, rnd, layers=[0,1])
    assert b["directions"].shape == (2,8) and c["directions"].shape == (2,8)
    assert np.allclose(np.linalg.norm(b["directions"],axis=1),1,atol=1e-5)
```
- [ ] **Step 2: run → FAIL**
- [ ] **Step 3: implement** both functions; behavioural = mean(top-q residual) − mean(bottom-q residual) per layer, unit-norm, `scales_from_phase_a`; confound = unit direction of OLS(hidden ~ balance+round) coefficient norm per layer. Record `cos(readout,behavioural)` when both available.
- [ ] **Step 4: run → PASS**
- [ ] **Step 5: commit** `git commit -am "feat(sec4): deconfounded behavioural + confound axes"`

### Task 3: CLI build from cached HF SAE/hidden → assets/sec4/
**Files:** Modify `src/indicator_axes.py` (add `main()`); Test: manual smoke on Gemma SM I_BA

- [ ] **Step 1** add `load_task_arrays(model, task, layers)` pulling `sae_features_v3/{task}/{model}/sae_features_L*.npz` + `hidden_states_dp.npz` + indicators (compute I_BA/I_LC/I_EC + balance/round from meta), with a **discovery/held-out game split** (hash game_id).
- [ ] **Step 2** add `main()`: `--model gemma --task slot_machine --indicators i_ba,i_lc,i_ec --axes readout,behavioural,confound --layers 16 21 --dest assets/sec4/`. Saves `assets/sec4/{model}_{task}_{indicator}_{axis}.npz` with provenance, AUC, and `cos_read_write`.
- [ ] **Step 3** smoke: `python -m multilayer_causal.src.indicator_axes --model gemma --task slot_machine --indicators i_ba --axes readout,behavioural,confound --layers 16 21 --dest assets/sec4/`. Expected: 3 npz, readout AUC ≥ 0.6, cos_read_write printed.
- [ ] **Step 4: commit** `git commit -am "feat(sec4): CLI axis library build from cached HF SAE/hidden"`

### Task 4: Runner fixes — manip_proj + replay guard + wandb
**Files:** Modify `src/runner.py`, `src/checkpoint.py`; Create `src/wandb_logger.py`; Test `tests/test_replay_guard.py`

- [ ] **Step 1: failing test** — `wandb_logger.log_arm` is a no-op under `WANDB_DISABLED=1`; replay guard raises if `arm['prompt_set_hash'] != direction_hash`.
```python
def test_replay_guard_mismatch():
    import os; os.environ["WANDB_DISABLED"]="1"
    from multilayer_causal.src import runner
    with pytest.raises(ValueError):
        runner.assert_replay_match(arm_hash="a", dir_hash="b")
```
- [ ] **Step 2: run → FAIL**
- [ ] **Step 3: implement** `wandb_logger.py` (`init_run`, `log_arm(arm_id, dose, metrics)`, guarded by env); `runner.assert_replay_match`; fix `manip_proj` extraction to `rec.get("vector_log",{}).get("proj")`; call `log_arm` at each arm DONE; `checkpoint` records `{n, t_start, t_end, replay_hash}`.
- [ ] **Step 4: run → PASS**; also re-run an existing xtaskd arm locally-dry to confirm no regression on sm/ic paths.
- [ ] **Step 5: commit** `git commit -am "fix(sec4): manip_proj key + replay guard + wandb per-arm logging"`

### Task 5: Phase-1 CORE config (Gemma SM)
**Files:** Create `configs/arms_sec4_p0.yaml`

- [ ] **Step 1** author arms: for `axis ∈ {readout, behavioural, confound}` and `α ∈ {−3,−2,−1,0,+1,+2,+3}` → 21 arms; `+ random_null × 5` (α=+3, distinct seeds); `+ baseline (no-steer)`; `+ cum_window {L18-19, L16-21, L14-23}` behavioural at α=+3 for H4. All: model gemma, task slot_machine, layers [16,21] (except cum arms), n 200, seed_base 2000042, state_offset 300 (held-out), `prompt_set: addiction_role_gm`, `log_vectors: true`. ~30 arms.
- [ ] **Step 2** validate: `python -c "from multilayer_causal.src.registry import load_arms; a=load_arms('configs/arms_sec4_p0.yaml'); print(len(a))"` → ~30, all `prompt_set_hash` set.
- [ ] **Step 3: commit** `git commit -am "feat(sec4): Phase-1 CORE arms (Gemma SM, 2-axis + confound + nulls + cum)"`

### Task 6: Analysis — dose/sign/parse/null/cos/AUC
**Files:** Create `src/sec4_stats.py`; Test `tests/test_sec4_stats.py`

- [ ] **Step 1: failing test** — `analyze` returns per-axis `{recovery_by_dose, spearman, sign_ok, parse_ok_by_dose, above_null, monotone}` and a `verdict` in {WRITE_CONFIRMED, READOUT_INERT, NULL, CONFOUNDED}. Synthetic monotone behavioural + flat readout → `verdict` reflects H1+H2.
- [ ] **Step 2: run → FAIL**
- [ ] **Step 3: implement** recovery vs ±G anchors (reuse anchor pull), Spearman(α,bet) + sign check, parse gate (drop dose cells <0.8), null band (mean±2sd of random arms), `cos_read_write` from assets, decoding-AUC table from build meta; `make_figure` (dose curves: behavioural vs readout vs confound vs null band). Assert `manip_proj` non-NaN.
- [ ] **Step 4: run → PASS**
- [ ] **Step 5: commit** `git commit -am "feat(sec4): Phase-1 analysis (dose/sign/parse/null/cos/AUC + figure)"`

### Task 7: amlt job + watch + autoresearch loop
**Files:** Create `amlt/sec4_p0.yaml.template`, `scripts/watch_sec4.sh`, `scripts/sec4_autoresearch.py`, `experiments/sec4_causal/INDEX.md`

- [ ] **Step 1** `sec4_p0.yaml.template`: mirror `xtaskd.yaml.template`; add `pip install sae_lens` to the pip line; env `MLC_ARMS_YAML=multilayer_causal/configs/arms_sec4_p0.yaml`, `MLC_HF_BASE=experiments/sec4_causal/checkpoints`, `MLC_OUT=multilayer_causal/results/sec4_p0`, `WANDB_PROJECT=llm-addiction-sec4`, `WANDB_API_KEY` from placeholder; description uses " - " not ": " (YAML guard).
- [ ] **Step 2** `watch_sec4.sh`: poll `amlt status` + HF cell count under `experiments/sec4_causal/checkpoints/p0`, exit on terminal/paused/N-cells (mirror `watch_xtask.sh`).
- [ ] **Step 3** `sec4_autoresearch.py`: loop — pull completed arms from HF, run `sec4_stats.analyze`, log summary to W&B, append a rung row to `INDEX.md` (hypothesis, config, numbers, verdict), print GO/NO-GO against §3 criteria. No auto-launch of Phase 2 (human gate).
- [ ] **Step 4** `INDEX.md` stub: table header (rung, phase, hypothesis, config, key numbers, status).
- [ ] **Step 5: commit** `git commit -am "feat(sec4): amlt job + watch + autoresearch ledger loop"`

### Task 8: Render, submit, monitor
- [ ] **Step 1** build axis library (Task 3 CLI) for Gemma SM I_BA (all 3 axes), verify AUC + cos, upload `assets/sec4/` to HF.
- [ ] **Step 2** `bash amlt/render.sh` (amlt env python), confirm `sec4_p0.rendered.yaml` gitignored.
- [ ] **Step 3** `amlt run amlt/sec4_p0.rendered.yaml mlc-sec4-p0-0706 -y`.
- [ ] **Step 4** `bash scripts/watch_sec4.sh` until cells land; on pause, resubmit (resume-safe).
- [ ] **Step 5** `python scripts/sec4_autoresearch.py` → verdict; append INDEX; report GO/NO-GO.

---

## PART IV — PHASE 2/3 (outline, gated on Phase-1 GO)
- **Phase 2 (3 parallel, contingent):** §4.1 grid (I_LC/I_EC + specificity, Gemma SM), §4.3 condition dose-slope (±G/±M), §4.2 cross-task **after IC/MW re-baseline to mid-range**; LLaMA replication on its own write window (absolute-effect metric; watch for prior sign-inversion).
- **Phase 3:** necessity ablation (project-out ⇒ betting↓).
- Integration: appendix **M4** confirming "monitoring not control"; body untouched until camera-ready.

---

## Self-review
- Spec coverage: H1–H4 → Tasks 1–3 (axes), 4–5 (replay/config), 6 (analysis), 7–8 (run/loop). ✓
- Past-failure table → each row maps to a fix in Tasks 4/5/6. ✓
- No placeholders: each task has concrete code/commands. ✓
- Clean separation: `sec4_` prefix + `experiments/sec4_causal/` + reuse-not-fork of runner. ✓
