# Patching Plan v4 — Master plan after HF audit + reviewer feedback

**Updated**: 2026-05-04
**Supersedes**: v3 (causal supplement structure)
**Scope**: HF audit + reviewer feedback mapping + complete code/data plan + autonomous execution

---

## §0 What this plan delivers

A single autonomous execution path that:
1. Builds all remaining code (smoke-critic-improve loop) until clean
2. Runs experiments (autoresearch-style autonomous loop) on local A100
3. Pushes results to HF every 10 min
4. Stops when paper §4 is strengthened (reviewer concern coverage ≥ 8/10)

Two tracks running in parallel:
- **Track A (causal)**: M3' indicator-direction steering — 1/4 files done, 3 more
- **Track B (confound)**: E2/E3/E7 re-analyses — 0/3 files done, all needed
- (Track C deferred unless Tier-1 results need additional support)

---

## §1 HF dataset audit (2026-05-04 snapshot)

| Category | Files | Status | Notes |
|----------|-------|--------|-------|
| §3 SM behavioral | 222 | ✅ | source data for E1, E5, E7 re-analyses |
| §3 IC behavioral | 2,858 | ✅ | source data for E1, E5, E7 |
| §3 MW behavioral | 98 | ✅ | source data |
| §4 SAE feature index | 0 (filtered) | — | 341 files in `sae_features_v3/` (not flagged by audit but exists) |
| §4 GroupKFold canonical | 28 | ✅ | table1_groupkfold_L{8,12,22,25,30}, condition_modulation_L22, etc. |
| v1 M3_swap (running) | 8 | 🟡 | baseline 200/200 ✅, swap_plusG ~120/200, ctrl 0/200 |
| **M3' direction_metadata** | **2** | **🚧** | **gemma_sm_i_ba + i_lc Ridge weights pushed (this commit)** |
| Confound removal (Track B) | 0 | ❌ | E2/E3/E7 not built yet |
| Legacy V12–V17 | 81 | quarantined | legacy/v12_steering_invalidated/, etc. |

**What's missing for paper completion**:
1. M3' steering trial outputs (450 trials × 3-4 cells)
2. Confound removal results (E2/E3/E7 JSON outputs)
3. Aggregate analysis JSONs

**Compute estimate for missing work**:
- M3' (3 files left + 450 trials): ~5h on local A100
- Track B (3 files, CPU only): ~2h
- Total: ~7h to full Tier 1 completion

---

## §2 Reviewer feedback → experiment mapping

Reviewer scored "accept after revisions" with 7 specific questions. Mapping each to experiments:

| Reviewer concern | Experiment | Tier | Status |
|------------------|-----------|------|--------|
| Q1 — variance inflation in §4.3 R² | E2 variance-normalized §4.3 | 1 | ❌ to build |
| Q2 — autonomy vs output-format complexity | E1 discrete bet grid | 2 | deferred |
| Q3 — BK direction balance confound | E3 balance-stratified §4.2 | 1 | ❌ to build |
| Q4 — distortion language validation | E5 LLM-judge | 2 | deferred |
| Q5 — decoding parameter sensitivity | E8 decoding sweep | 2 | deferred |
| Q6 — DeLLMa mitigation baseline | E6 mitigation prompt | 2 | deferred |
| Q7 — per-layer curves, SAE sparsity | already in appendix C.2 | — | ✅ (existing) |
| general — mixed-effects modelling | E7 mixedlm §3 | 1 | ❌ to build |
| (user) — causal supplement at §4.1 | M3' indicator steering | 1 | 🚧 building |

**Tier 1 completion covers reviewer Q1, Q3, Q7, general stat + user's causal.**
**That's 5/7 reviewer concerns + user request → expected to substantially raise paper acceptance probability.**

---

## §3 Track A — M3' Indicator-direction Steering (causal supplement at §4.1)

### 3.1 Intent (paper claim being tested)

§4.1 finds: **"L22 SAE features predict $I_\text{BA}$, $I_\text{LC}$, $I_\text{EC}$
at $R^2 \in [0.06, 0.30]$ via Top-K=200 + Ridge"** — this is *correlational
recoverability* of behavioural indicators from internal state.

**Open question**: Is the same Ridge weight vector $w$ that predicts $I_\text{BA}$
*also* a *controller* for $I_\text{BA}$? I.e., if we project $w$ back to L22
hidden state and ADD it to a forward pass, does the model bet more aggressively?

If yes → §4.1 readout has **causal validity** (predictor = controller).
If no → §4.1 readout is a **decoder of upstream-determined behaviour**.

Either result is a paper contribution. Positive → strengthens §4 with causal evidence.
Negative → strengthens §6 with explicit causal scope statement.

### 3.2 Hypothesis tree (v3.3 carry-over, refined)

**H_M3'-A** (continuous): Pearson($\alpha$, mean bet ratio across 6 levels) $\geq 0.6$
**H_M3'-B** (binary): bk_rate($\alpha=+3\sigma$) - bk_rate($\alpha=-2\sigma$) $\geq 0.10$, Fisher $p<0.05$, Cohen $h \geq 0.20$
**H_M3'-C** (random control specificity): random direction same norm at $\alpha=+2\sigma$ → Pearson < 0.2 AND Cohen $h < 0.10$
**H_M3'-D** (layer specificity): same direction at L8 → effect $\leq$ 0.5 × L22 effect
**H_M3'-E** (indicator specificity): I_LC direction at $\alpha=+2\sigma$ → does NOT increase bet ratio

H_M3' overall = (A AND B AND C). D, E are robustness.

### 3.3 Specific method

**Model**: Gemma-2-9B-IT (primary; LLaMA Phase 2 deferred)
**Task**: SM (slot machine)
**Prompt**: BASE + variable betting (no G/M/H/W/P modules — clean baseline)
**Layer**: L22 (matches §4 entire chain)
**Position**: Last input token (decision token)

**Direction extraction** (one-time):
```python
# §4.1 Ridge w (from extract_section4_ridge_weights.py — DONE)
w_200 = ridge.coef_                              # (200,)
top_K_idx = active_subset[selected_indices]      # full SAE 131k space
scaler_scale = StandardScaler().scale_           # (200,)

# Load Gemma-Scope L22 SAE
sae = SAE.from_pretrained('gemma-scope-9b-it-res',
                          'layer_22/width_131k/canonical')
W_dec = sae.W_dec  # (131072, 4096)

# Build direction (correctly accounts for standardization)
selected_dec = W_dec[top_K_idx]                  # (200, 4096)
d_pre_norm = (w_200 / scaler_scale) @ selected_dec   # (4096,)
d_unit = d_pre_norm / np.linalg.norm(d_pre_norm)

# σ from baseline distribution
projection_baseline = h_baseline @ d_unit         # 100 baseline rounds
sigma = projection_baseline.std()
```

**Steering hook**:
```python
def steer_hook(module, _input, output):
    out = output[0] if isinstance(output, tuple) else output
    if out.shape[1] > 1:  # only first forward pass (full prompt)
        out[:, -1, :] += alpha * sigma * d_unit
    return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out
```

**Conditions** (9 cells × n=50 = **450 trials**):

| Cell | Direction | α (σ units) | n | Purpose |
|------|-----------|-------------|---|---------|
| α=-2 | $\vec{d}_{I_\text{BA}}$ | -2 | 50 | dose-response left tail |
| α=-1 | $\vec{d}_{I_\text{BA}}$ | -1 | 50 | dose-response |
| **α=0** | none | 0 | 50 | **baseline** |
| α=+1 | $\vec{d}_{I_\text{BA}}$ | +1 | 50 | dose-response |
| α=+2 | $\vec{d}_{I_\text{BA}}$ | +2 | 50 | dose-response |
| α=+3 | $\vec{d}_{I_\text{BA}}$ | +3 | 50 | dose-response right tail |
| **random_dir** | $\mathcal{N}(0,I)$, norm-matched | +2 | 50 | **direction specificity** |
| **L8_dir** | $\vec{d}_{I_\text{BA}}$ at L8 | +2 | 50 | **layer specificity** |
| **ILC_dir** | $\vec{d}_{I_\text{LC}}$ at L22 | +2 | 50 | **indicator specificity** |

**Compute**: ~3.5h on local A100 (~28s/trial avg; 450 × 28 = 12,600s).

### 3.4 Data

| What | Source |
|------|--------|
| §4.1 Ridge weights (Gemma SM I_BA, I_LC) | `direction_metadata/{cell}.json` ✅ on HF |
| Gemma-Scope L22 SAE decoder | HuggingFace `google/gemma-scope-9b-it-res` |
| Hidden states for σ baseline | `sae_features_v3/slot_machine/gemma/hidden_states_dp.npz` ✅ |
| Decoded behaviour (in-the-loop) | model.generate() at runtime |
| Per-trial JSONL output | `M3prime_indicator_steering/{cell}/trials.jsonl` (auto-pushed every 10min) |

### 3.5 Aggregate analysis (after 450 trials done)

```python
# Primary continuous outcome
mean_bet_ratio_per_alpha = [mean(t.observed_I_BA for t in trials_at_alpha)]
pearson_r, pearson_p = pearsonr([-2,-1,0,1,2,3], mean_bet_ratio_per_alpha)
# Bootstrap 95% CI on Pearson via 1000 resamples

# Secondary binary outcome
bk_rate_high = bk_rate(α=+3 trials)
bk_rate_low  = bk_rate(α=-2 trials)
fisher_p     = fisher_exact([[bk_high*50, (1-bk_high)*50],
                             [bk_low*50,  (1-bk_low)*50]])
cohen_h      = 2*(arcsin(sqrt(bk_high)) - arcsin(sqrt(bk_low)))

# Specificity
pearson_random_dir = ...  # baseline+random_dir → should approach 0
pearson_L8         = ...  # baseline+L8        → should be < 0.5 × L22 r
ILC_BA_effect      = ...  # ILC dir's effect ON bet ratio (should be ~0)
```

### 3.6 Distinct from prior

- Templeton 2024 (Anthropic): single-feature SAE clamping → ours is **multi-feature
  Ridge-weighted direction**
- Marks 2024 SFC: attribution-patching feature selection → ours is **predictive
  correlation** selection
- Geiger 2024 DAS: learnable subspace optimization → ours is **fixed §4.1 Ridge direction**
- Nostalgebraist 2023 (linear-probe steering): raw hidden-state probe → ours uses
  **SAE-basis projection** for interpretability

**M3' novelty**: Direct test of "predictor → controller" symmetry using the SAME
§4.1 Ridge weight, projected via SAE decoder. Connects §4.1 correlational claim
to §4.5 causal supplement with no logical gap.

### 3.7 Falsification

If at $\alpha \in \{-2, +3\}$:
- |median bet_ratio diff| < 0.05 AND
- Pearson < 0.3 AND
- bk_rate diff < 0.05

→ **§4.1 readout is decoder, not controller**. §6 strengthens with explicit
"the predictor cannot be inverted into a controller at single-pass L22 modification."

---

## §4 Track B — Confound-removal re-analyses

### 4.1 E2 — Variance-normalized §4.3 readout (reviewer Q1)

**Intent**: §4.3 reports +143% Gemma SM I_BA $\Delta R^2$ between -G and +G. Reviewer
concern: this could partly reflect higher I_BA *target variance* under +G
(mechanical inflation), not internal sharpening.

**Hypothesis H_E2**: After variance-normalization, the +G effect persists with
positive standardized effect size:
- Pearson($\hat{I}_\text{BA}, I_\text{BA}$): +G ≥ -G (should remain positive)
- Partial $\eta^2$: +G ≥ -G
- Δ% post-normalization: ≥ +50% (vs raw +143%)

**Method**:
```python
for cond in ['minus_G', 'plus_G', 'minus_M', 'plus_M']:
    target = I_BA[cond]
    mu, std = target.mean(), target.std()
    target_z = (target - mu) / std
    pred = ridge.predict(X[cond])

    metrics[cond] = {
        'r2_raw': r2_score(target, pred),
        'r2_z':   r2_score(target_z, (pred - mu) / std),
        'pearson': pearsonr(target, pred),
        'partial_eta2': partial_eta_squared(target, pred,
                                            controls=[balance, round]),
    }
```

**Data**: Existing `condition_modulation_groupkfold_L22.json` (already on HF).
No new compute. Pure CPU re-analysis. ~30 min.

**Output**: `confound_removal/E2_variance_normalized_section43.json`

**Falsification**: Δ% post-norm < 20% AND Pearson +G ≈ -G → §4.3 +143% is variance
inflation. §6 hedge required.

### 4.2 E3 — Balance-stratified §4.2 BK direction (reviewer Q3)

**Intent**: §4.2 BK direction = mean(h | bankrupt) − mean(h | voluntary stop).
Reviewer concern: bankruptcy correlates with low-balance rounds; the BK direction
may just be a "low balance" detector.

**Hypothesis H_E3**: Within-balance-stratum BK separation persists. AUC of BK
projection within mid-balance stratum ($30 ≤ balance < $150$) ≥ 0.6.

**Method**:
```python
for bucket in [(0,30), (30,100), (100,200), (200,inf)]:
    h_bk_in    = h[bk    & (balance in bucket)]
    h_stop_in  = h[stop  & (balance in bucket)]

    bk_dir_within   = h_bk_in.mean(0) - h_stop_in.mean(0)
    auc_within      = roc_auc(np.concatenate([proj_bk, proj_stop]),
                              np.concatenate([1s, 0s]))
    auc_pooled_in_bucket = ...  # use pooled BK direction within this bucket

    report[bucket] = {within: auc_within, pooled: auc_pooled_in_bucket, n_bk, n_stop}
```

**Data**: Existing hidden states `sae_features_v3/.../hidden_states_dp.npz` + game outcomes.
No new compute. Pure CPU. ~1h.

**Output**: `confound_removal/E3_balance_stratified_section42.json`

**Falsification**: Within-bucket AUC drops to ~0.5 in mid/high balance →
BK direction is just balance detector. §4.2 claim weakens.

### 4.3 E7 — Mixed-effects §3 modelling (reviewer general stat)

**Intent**: §3 reports per-condition aggregate effects. Reviewer wants proper
hierarchical modelling, CIs, per-model random effects.

**Hypothesis H_E7**: Hierarchical logistic model (per-game bk outcome ~ condition
+ (1 | model)) gives +G fixed effect significantly positive (95% CI excludes 0)
across all 6 models.

**Method**:
```python
import statsmodels.formula.api as smf

mlm = smf.mixedlm("bk ~ variable + plus_G + plus_M + variable:plus_G",
                   df, groups=df['model'], re_formula="~1").fit()

# Report fixed effect coef + 95% CI for each module
# Random intercept variance per model
# LRT for interactions
```

**Data**: §3 game outcomes from `paper_experiments/slot_machine_6models/data/...`.
Pure CPU. ~30 min.

**Output**: `confound_removal/E7_mixed_effects_section3.json`

**Falsification**: +G fixed effect 95% CI includes 0 → §3 +G claim weakens.
(Highly unlikely given §3 effect size.)

---

## §5 Implementation file structure

```
sae_v3_analysis/src/
├── extract_section4_ridge_weights.py            ✅ DONE (S1-S3 PASS)
├── compute_section4_steering_directions.py     ❌ TO BUILD (Step 2)
├── run_m3prime_indicator_steering.py            ❌ TO BUILD (Step 3)
├── aggregate_m3prime_dose_response.py           ❌ TO BUILD (Step 4)
├── e2_variance_normalized_section43.py          ❌ TO BUILD (Step 5)
├── e3_balance_stratified_section42.py           ❌ TO BUILD (Step 6)
└── e7_mixed_effects_section3.py                 ❌ TO BUILD (Step 7)

sae_v3_analysis/scripts/
└── m3_push_scheduler.py                         ✅ running (10-min push)

sae_v3_analysis/results/v19_multi_patching/
├── M3_swap/                                     🟡 v1 running
├── M3prime_indicator_steering/
│   ├── direction_metadata/
│   │   ├── gemma_sm_i_ba_L22.json              ✅
│   │   ├── gemma_sm_i_lc_L22.json              ✅
│   │   └── {model}_{task}_{ind}_L22_steering.json    ❌ Step 2 produces
│   ├── gemma_sm_IBA_alpha-2_n50/                ❌ Step 3 produces
│   ├── gemma_sm_IBA_alpha-1_n50/                ❌
│   ├── ... (9 cells total)                      ❌
│   └── aggregate_dose_response.json             ❌ Step 4 produces
└── confound_removal/
    ├── E2_variance_normalized_section43.json    ❌ Step 5
    ├── E3_balance_stratified_section42.json     ❌ Step 6
    └── E7_mixed_effects_section3.json           ❌ Step 7
```

---

## §6 Smoke-critic-improve protocol per file

For each new file, run:

**Smoke S1-S4** (in order):
- S1: syntax + import check (1 min)
- S2: 1-input dummy run (verify shape/types) (1-5 min)
- S3: small real run (1 trial / 1 cell) (5-15 min)
- S4: Resume safety check (kill mid-run, restart, verify continuation)

**Critic ITER 1**: hooks/handles/memory leaks/numerical stability
**Critic ITER 2**: edge cases (small samples, single rounds)
**Critic ITER 3**: cross-platform/file-system/logging clarity

When all clean → autoresearch launch.

---

## §7 Autonomous run protocol

```
[NOW]                v1 swap_plusG running on local A100, push scheduler 10min
[+30min]             Step 2 build + smoke-critic complete (compute_directions)
                     → produces 6+3 = 9 steering JSONs (gemma I_BA + I_LC + L8)
[+1h]                Step 5/6/7 build (CPU-only Track B) — parallel to GPU work
                     → produce E2/E3/E7 result JSONs
[+1.5h]              Step 3 build + smoke (run_m3prime_indicator_steering)
[+2h]                v1 random_swap_ctrl finishes
[+2.5h]              Step 4 build + smoke (aggregate)
[+3h]                M3' Phase 1 launch (after v1 GPU free)
                     → 450 trials × ~28s = ~3.5h
[+6.5h]              M3' Phase 1 complete → aggregate analysis
[+7h]                Push final results, draft §4.5 or §6 update
```

10-min push throughout, resume-safe per cell. Total wall time ~7h.

---

## §8 Plan iteration log (intent/hypothesis/verification refinement)

### v3 → v3.3 (carry from prior)
[As documented in v3 plan]

### v4 self-critique (this version)

#### Critic 1: HF audit revealed M3' direction_metadata is 2/9+ files

→ Plan v4 explicitly tracks build status (✅ DONE / 🚧 building / ❌ TO BUILD)
per file. Run protocol §7 specifies build sequence with smoke gates.

#### Critic 2: Track B was vague in v3

→ Plan v4 §4 specifies exact metrics (Pearson, partial η², AUC strata, mixedlm
coefficients) per E2/E3/E7. Pseudocode in each subsection.

#### Critic 3: Reviewer Q mapping was implicit

→ Plan v4 §2 has explicit reviewer-Q→experiment table. Tier 1 covers Q1, Q3,
Q7, general stat + user causal = 5/7 reviewer + user.

#### Critic 4: Falsification per experiment was inconsistent

→ Plan v4 each experiment has explicit falsification statement. Negative
result → §6 limitation strengthening (publishable in either direction).

#### Critic 5: Compute estimate was rough

→ Plan v4 §7 has timeline with concrete +N min markers. Total ~7h to Tier 1
completion. Track B parallel-able with v1/M3' on GPU.

### v4 final state

- All 4 critic concerns addressed
- File-level build sequence specified
- Reviewer mapping explicit
- Falsification stated per experiment

Plan v4 ready for autonomous execution.

---

## §9 Open decisions (no longer blocking)

1. AMLT 4 paused nodes: **abandon** (5 resume attempts failed; backend not responding).
   Local A100 sufficient.
2. Phase 2 LLaMA replication: defer until Gemma M3' result.
3. Tier 2 (E1, E5, E6, E8): defer until Tier 1 lands.
4. v1 (M3 swap): keep running as supplementary §4.3 evidence.

---

## §10 Paper integration plan (after Tier 1 complete)

### M3' positive case
- Add §4.5 (1 paragraph + dose-response figure):
  > "We provide preliminary causal evidence for §4.1's readout direction. Steering along
  > the Ridge weight $w$ projected via the Gemma-Scope SAE decoder produces a monotonic
  > dose-response in observed bet ratio (Pearson $r=X.XX$, $p<0.001$ over 6 α levels;
  > random-direction control shows $|r|<0.2$). The §4.1 readout is therefore a
  > controller, not just a decoder, of bet aggressiveness at L22."
- §6 limitation hedge for steering can be removed/softened

### M3' negative case
- §6 strengthens:
  > "We tested whether the §4.1 readout direction can be inverted into a controller via
  > additive steering at L22 (n=50 per α level, 6 levels). The dose-response is null
  > (Pearson $r=X.XX$, $p>0.05$). The §4.1 readout decodes behaviour but does not
  > causally control it under single-pass L22 modification; future causal validation
  > requires multi-layer or learnable-subspace interventions."

### E2/E3/E7 results
- E2: §4.3 prose adds "After variance-normalization, the +G effect persists at..."
- E3: §4.2 prose adds "Within-balance-stratum AUC remains..."
- E7: Appendix B adds mixed-effects table; §3 prose adds "(95% CI..., mixed-effects p<...)"

---

## §11 v4 status as of writing

- ✅ extract_section4_ridge_weights.py (Code 1/7)
- 🚧 v1 M3 swap on local A100 (~60-70% done)
- ❌ compute_section4_steering_directions.py (Code 2/7)
- ❌ run_m3prime_indicator_steering.py (Code 3/7)
- ❌ aggregate_m3prime_dose_response.py (Code 4/7)
- ❌ e2_variance_normalized_section43.py (Code 5/7)
- ❌ e3_balance_stratified_section42.py (Code 6/7)
- ❌ e7_mixed_effects_section3.py (Code 7/7)

Next: Build Code 2 (smoke-critic-improve), then 5/6/7 in parallel (CPU), then 3, 4.
