# Audit Report: §4 Paper Pipeline & HF Data Integrity
**Date**: 2026-05-03
**Scope**: Paper §4 + appendix + supporting data on HuggingFace

---

## TL;DR for first-time readers

We are submitting a NeurIPS 2026 / EMNLP 2026 paper claiming that LLMs show
gambling-like risk behaviour, that this behaviour intensifies under autonomy
conditions, and that the same contrast leaves a recoverable trace inside the
model's decision-time hidden state. The §4 claim chain is:

```
behavioural metric (§3)  →  internal readout R²  →  task-shared geometry  →  autonomy modulation
                                       ↑                  ↑                       ↑
                            Table 1 (5 layers)     LOTO PCA (Fig 5)         Table 2 (±G/±M)
                                       ↑
                            paper-canonical pipeline
                            (GroupKFold by game_id)
```

This document audits whether **every number in the body** traces to a single,
internally consistent file lineage on HuggingFace; whether the HF dataset
contains stale outputs that contradict body claims; and whether the paper's
prose still conforms to the user's "flowing prose, no bullets, two-paragraph
voice" principle (commit `1cd4a40`).

---

## §1. Intent of this audit

**The question**: Is the §4 evidence chain — *prose claim → table → JSON file →
script that produced it* — fully traceable, leak-free, and voice-aligned?

**Why this matters**:
- A reviewer who follows any number in §4 back to HF should land on a single
  authoritative file with no ambiguity.
- Stale or contradicting files that look authoritative (`paper_neural_audit.json`,
  `v17_nonlinear_deconfound.txt`, V12 steering JSONs) silently undermine
  reproducibility.
- Voice drift (bold paragraph leaders, bullet-style audits) breaks the
  iterative-writing principle the user has explicitly enforced.

**How this audit is structured** (five sub-questions, each posed as
intent / hypothesis / verification):

| Sub-Q | Topic | Section |
|-------|-------|---------|
| Q1 | EMNLP format compliance | §2 |
| Q2 | Voice / style alignment with user's principles | §3 |
| Q3 | Number traceability (paper → file → script) | §4 |
| Q4 | HF data hygiene (canonical vs stale) | §5 |
| Q5 | Paper–data–code labeling completeness | §6 |

---

## §2. Q1 — EMNLP format compliance

**Intent**: The paper compiles cleanly under the EMNLP 2026 ACL template and
respects the 8-page body limit + free Limitations section.

**Hypothesis**: The current build (`emnlp_en.tex` + `emnlp.tex`) sits within
EMNLP submission constraints and does not require manual format surgery.

**Verification**:
- `emnlp_en.tex` builds → 25 pages total. Body = 8p (lines up to §6 Limitations
  start). Limitations + References + Appendix = 17p free.
- ACL style loaded via `\usepackage[review]{acl}` (line 88 of `paper_core.tex`).
- `\bibliographystyle{acl_natbib}` is auto-set by `acl.sty` (no explicit call,
  per the bibtex error fix recorded in earlier commits).
- KO build (`emnlp.tex`) uses `kotex` + ACL two-column — verified working at
  24 pages.
- Title block, author block (`Anonymous EMNLP submission`), and `\maketitle`
  all guarded by `\ifx\paperversion\emnlpversion`.

**Status**: ✓ PASS. EMNLP build is format-compliant.

**Caveat**: `Overfull \hbox` warnings exist at lines 467–488 (pre-existing,
~10pt over) and one Unicode hyperref warning. Neither blocks submission.

---

## §3. Q2 — Voice and style alignment

**Intent**: The §4 prose maintains the "flowing prose, no bullets, no bold
paragraph leaders, two-sentence implicit Intent/Hypothesis/Verification flow"
voice that the user codified in commit `1cd4a40` (2026-05-02).

**Hypothesis**: Recent edits introduced by the iterative-writing pass (commits
`a3fff04`, `eca2e3d`, `5d2b457`) preserve this voice. Where they diverge, the
divergence is either (a) a strict improvement that the user would accept, or
(b) a violation that needs to be reverted.

**Verification (per edit)**:

| Edit (file:section) | Type | Voice impact | Status |
|---------------------|------|--------------|--------|
| `paper_core.tex:142,146,150,170,220` — Title reframe | Structural | User has never touched title; this is a unilateral change | ⚠ NEEDS USER APPROVAL |
| `0.abstract.tex` — added `$+143\%$` + `$p<0.0001$` | Content | User has only ever *compressed* the abstract; new numbers never added | ⚠ NEEDS FACT-BASE CONFIRMATION |
| `4.neural.tex` (EN) prologue — em-dash run-on → 3 sentences | Style | Flowing prose preserved, lead-claim-first | ✓ |
| `4.neural.tex` (EN+KO) §4.1 method — regress→Ridge reorder | Content reorder | Logical flow improved, no voice change | ✓ |
| `4.neural.tex` (EN+KO) §4.2 — `(i)/(ii)/(iii)` → declarative sentences | Style | Removes bullet-disguised-prose; matches user principle | ✓ |
| `4.neural.tex` (EN) §4.4 — 130w semicolon run-on → 4 sentences | Style | One-idea-per-sentence, no markers | ✓ |
| `5.discussion.tex` (EN+KO) — `+126%` → `+143%` | Number | Aligns §5 with §4.4 and Table 2 — pure consistency fix | ✓ |
| `5.discussion.tex` (EN) — 1-paragraph → 2-paragraph mirror of KO | Structure | User had already written KO 2-paragraph; EN now matches | ✓ |
| `appendix.tex` (EN+KO) — new C.2 GroupKFold sweep subsection + table | New content | Defends body claim with new data; flowing prose | ✓ |

**Status**: 7 of 9 edits ✓. Two need user decision before next phase:
- **Title reframe**: revert or keep?
- **Abstract numbers**: keep `+143%`/`p<0.0001` (Fact Base says these are in
  Table 2 + §4.1 prose, so factual), or revert to user's original which had no
  abstract numbers?

---

## §4. Q3 — Number traceability

**Intent**: Every quantitative claim in §4 (body) traces through a single
documented chain: prose → table cell → JSON file → script.

**Hypothesis**: After the GroupKFold transition (user commit `35bfad4`,
2026-05-03), all body §4 numbers come from `table1_groupkfold_L22.json` (Table 1)
or `condition_modulation_groupkfold_L22.json` (Table 2). The 5-layer appendix
sweep (`table1_groupkfold_L{8,12,22,25,30}.json`) is consistent at L22.

**Verification (canonical chain)**:

```
§4.1 Body Table 1
  ↓ source
table1_groupkfold_L22.json  (Top-200 SAE, Ridge α=100, 5-fold GroupKFold)
  ↓ produced by
src/run_groupkfold_recompute.py

§4.1 prose "p<0.0001"
  ↓ source
table1_perm_null.json (50-iter game-block permutation)
  ↓ produced by
src/run_table1_perm_null.py

§4.3 Body Table 2 + prose
  ↓ source
condition_modulation_groupkfold_L22.json
  ↓ produced by
src/run_groupkfold_recompute.py (§4.3 part)

§4.2 prose AUC + cosine + sparse-feature transfer
  ↓ source
iba_cross_task_transfer.json + rq2_audit_consistent_layer.json
  ↓ produced by
src/run_rq2_aligned_hidden_transfer_sweep.py

§4 figure (Fig 5)
  ↓ source
gen_fig5b_pca.py reads hidden_states_dp.npz directly

Appendix C.2 Table tab:appendix-groupkfold-sweep
  ↓ source
table1_groupkfold_L{8,12,22,25,30}.json
  ↓ produced by
src/run_groupkfold_layer_sweep.py
```

**Spot-check (at random)**:
- §4.1 prose: "Gemma SM reaches $R^2{=}0.167$" → `table1_groupkfold_L22.json["gemma_sm_i_ba_L22"]["r2_mean"]` = 0.16657615... ✓
- §4.4 summary: "$+143\%$ on Gemma" → Table 2 ΔG% Gemma SM I_BA = (0.153 − 0.063)/0.063 × 100 = +142.86% → +143% ✓
- §4.3 prose: "$0.081 \to 0.138$" (Gemma MW I_LC) → MW condition modulation file. NOT verified in this audit — need to confirm.

**Status**: ✓ PASS for spot-checked cells; **§4.3 prose MW numbers not
spot-verified** (deferred to Phase B).

---

## §5. Q4 — HuggingFace data hygiene

**Intent**: HF dataset (`llm-addiction-research/llm-addiction`) contains the
canonical files cleanly tagged, and stale files are either removed or marked
do-not-cite.

**Hypothesis**: There exist files on HF that look authoritative but contradict
body numbers. Specifically: `paper_neural_audit.json` (legacy V17 audit),
`v17_nonlinear_deconfound.txt` (leaky RF), V12/V14/V16 steering JSONs, and the
`sweep_3metrics/` random-KFold sweep.

**Verification (audit by category)**:

| Category | Count | Status | Action |
|----------|-------|--------|--------|
| GroupKFold canonical (paper-cited) | 8 | ✓ Authoritative | Keep + label |
| Pre-GroupKFold `sweep_3metrics/*` | 18 | Random KFold; LLaMA cells diverge from body | Tag DO-NOT-CITE; keep for traceability |
| `paper_neural_audit.json` | 1 | Legacy V17 binary I_LC pipeline | Tag DEPRECATED; keep for archive |
| `v17_nonlinear_deconfound.txt` (+REFERENCE) | 2 | Leaky RF deconfound; partly inflated | Tag LEAKY; keep with disclaimer |
| V12 steering | 25 | Invalidated 2026-04-14 (prompts mismatch) | Move to `legacy/` or quarantine |
| V14 steering | 14 | Status unclear, likely stale | Audit needed |
| V16 steering | 7 | Multilayer follow-up, paper does not cite | Audit needed |
| `json/*` raw outputs | 40 | Authoritative (raw experiment) | Keep |
| `reports/*` planning docs | 13 | Mostly stale planning | Tag historical |
| `logs/*` | 2,002 | Mostly just runtime logs | Move to `logs/legacy/` or compress |
| `figures/*` | 91 | Old figure versions | Audit + prune |

**Status**: ⚠ Mixed. Canonical files are present but stale files are
unlabeled. A reviewer cloning the dataset cannot easily distinguish.

**Required action** (proposed):
1. Add a `MANIFEST.md` at HF root mapping every paper claim → file → script.
2. Move all V12/V14/V16 steering JSONs to `legacy/steering_invalidated/`
   under HF dir tree.
3. Move `paper_neural_audit.json` and `v17_nonlinear_deconfound*.txt` to
   `legacy/v17_pipeline/` with a README explaining the leak.
4. Move 2,002 raw `logs/` files to `logs/raw/` with index.
5. Add per-paper-claim labels to canonical files via README links.

---

## §6. Q5 — Paper–data–code labeling

**Intent**: A reader of the paper can find the exact file behind every claim
by following a labeled lookup table.

**Hypothesis**: We currently have an updated `sae_v3_analysis/results/README.md`
on HF that lists the paper-canonical pipeline, but no claim-by-claim mapping.

**Verification**:
- `sae_v3_analysis/results/README.md` — has the §4 paper-canonical section
  (added in this iteration). Lists files by section. ✓
- Body §4 prose / table / figure does NOT cite HF paths or filenames.
- Appendix §C.2 references `run_groupkfold_layer_sweep.py` by name but no
  HF link.
- No `MANIFEST.md` at HF root.

**Status**: ✗ FAIL. Reproducibility currently relies on the reader knowing the
pipeline. We need a single canonical mapping table that links paper claims to
file paths.

---

## §7. Findings summary

| Finding | Severity | Action class |
|---------|----------|--------------|
| Title reframe in `5d2b457` lacks user precedent | High | User decision |
| Abstract +143% added without abstract-revision precedent | Medium | User decision |
| 4 minor §4 edits OK (audit list, summary split, etc.) | Low | Keep |
| §4.3 MW prose numbers not spot-verified | Medium | Phase B verify |
| 25 V12 steering files unlabeled on HF (invalidated) | High | Phase D move to legacy/ |
| 4 V17 leaky pipeline files unlabeled on HF | Medium | Phase D move to legacy/ |
| 18 pre-GroupKFold sweep files unlabeled on HF | Medium | Phase D add do-not-cite README |
| 2,002 `logs/*` files clutter HF | Low | Phase D move to `logs/raw/` |
| No HF root `MANIFEST.md` | High | Phase E create |
| No paper-claim → HF-path table | High | Phase E create |

---

## §8. Phased plan (proposed)

Each phase has explicit intent / hypothesis / verification, scoped so it can
be approved or vetoed independently.

### Phase A — Voice reconciliation (≈30 min)
- **Intent**: Decide whether to revert title reframe and abstract numbers.
- **Hypothesis**: User wants to keep the new voice-improvement edits but may
  veto title/abstract structural changes.
- **Verification**: User decision.

### Phase B — Number spot-verification (≈45 min)
- **Intent**: Every number in §4 body and §4.4 summary traces to a single
  HF file with matching value (within rounding).
- **Hypothesis**: All currently survive verification post-GroupKFold.
- **Verification**: Run a Python script that walks §4 prose, extracts every
  numeric mention, and matches against canonical JSONs. List mismatches.

### Phase C — Plan iteration loop (≈1 h)
- **Intent**: Produce a plan v1 → critic → v2 → ... until intent / hypothesis
  / verification triplets are robust for every remaining task.
- **Hypothesis**: 3 iterations converge.
- **Verification**: Self-critic + user critic per iteration.

### Phase D — HF cleanup (≈1.5 h)
- **Intent**: Quarantine stale data so a reader cannot accidentally cite it.
- **Hypothesis**: Moving V12/V14/V16/V17 to `legacy/` + adding READMEs is
  reversible and does not break paper claims.
- **Verification**: After cleanup, every file under `sae_v3_analysis/results/`
  is either canonical (cited in body/appendix) or under `legacy/` with a
  README disclaimer.

### Phase E — Paper-claim ↔ HF-path manifest (≈45 min)
- **Intent**: Single `MANIFEST.md` at HF root that maps every paper claim
  (section + table + claim) to file path.
- **Hypothesis**: One manifest is enough — body need not cite HF paths
  inline.
- **Verification**: Random spot-check 5 paper claims → manifest → file.

### Phase F — Code smoke-critic-improve (≈1 h)
- **Intent**: GroupKFold scripts run end-to-end on a 1-cell smoke test, fail
  loudly on bad inputs, document hyperparameters in module docstring.
- **Hypothesis**: 2 iterations converge (smoke, then critic feedback).
- **Verification**: Smoke run reproduces L22 cells within ±0.005 R²; critic
  flags every assumption.

### Phase G — Final verification (≈30 min)
- **Intent**: Rebuild all 4 PDFs, full body+appendix integrity check, push
  to HF with new MANIFEST + cleaned legacy/.
- **Hypothesis**: All builds pass; HF state matches paper claims.
- **Verification**: Build exits 0; HF audit shows 100% labeled.

**Total estimated effort**: ~5.5 h interactive + waiting time.

---

## §9. What we need from you

Three decisions before Phase A:

1. **Title reframe** — keep `When Do LLMs Take Gambling-Like Risk? An
   Autonomy-Conditioned Behavioural and Representational Audit`, or revert to
   `Can Large Language Models Develop Gambling Addiction?`?

2. **Abstract numbers** — keep `+143%` and `p<0.0001` in abstract, or revert
   to the no-numbers abstract you compressed previously?

3. **HF cleanup destructiveness** — OK to move 25 V12 + 14 V14 + 7 V16 + 4 V17
   files to `legacy/` subdirectories on HF? They remain accessible, just
   re-routed and tagged.
