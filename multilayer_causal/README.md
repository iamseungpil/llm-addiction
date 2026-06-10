# multilayer_causal — Multi-Layer Causal Intervention (E1→E2→E3)

Follow-up to the M3/M3′/M3″ single-layer causal nulls: where (E1), at what rank
(E2), and with what dose-response (E3) is the −G→+G autonomy effect writable
into Gemma-2-9B-IT on the slot-machine task?

- **Spec**: `docs/superpowers/specs/2026-06-10-multilayer-causal-intervention-design.md`
- **Plan**: `docs/superpowers/plans/2026-06-10-multilayer-causal-harness.md`

## Protocol in one paragraph

Single-decision trials inherited byte-identically from
`sae_v3_analysis/src/run_m3pp_strong_patching.py` (M3″): a −G slot-machine
state from the §3 corpus, its +G twin prompt, generation at T=0.7 with frozen
seeds (`42 + 997·i`). Interventions: replace hidden states of a **layer set**
with the +G twin's (E1, `MultiLayerPatcher`); replace only the top-r PCA
subspace component (E2, `SubspacePatcher`); add `α · 0.03·‖h‖_median · v̂_ℓ` at
all positions of the layer set (E3, `MultiLayerSteerer`). Anchors:
`natural_minusG` / `natural_plusG`.

## Layout

| Path | What |
|---|---|
| `configs/arms.yaml` | Declarative arm registry (E1 now; E2/E3 appended after the E1 gate fixes S\*) |
| `src/prompts.py`, `src/states.py` | **FROZEN** M3″ copies — do not edit; parity-tested |
| `src/hooks.py` | The three intervention hook sets + activation cache |
| `src/checkpoint.py` | JSONL checkpoint + HF latest-state sync + resume |
| `src/runner.py`, `run_experiment.py` | Trial loop + CLI |
| `src/analyze.py` | Summaries, G1 gate, S\* selection, steering-direction builder |
| `src/pca_basis.py` | E2 prep: phase_a npz → per-layer rank-128 PCA bases |
| `run_arms.sh` | Shard arms across GPUs inside one job |
| `scripts/push_code_to_hf.py` | Tarball → HF (latest, overwritten) |
| `amlt/*.yaml.template` + `amlt/render.sh` | Job specs (tokens substituted locally; rendered files are gitignored) |

## HF dataset paths (`llm-addiction-research/llm-addiction`)

```
experiments/multilayer_causal/
├── code/multilayer_causal.tar.gz          # latest code (overwritten)
├── checkpoints/{phase}/{arm}.jsonl        # LATEST resumable state (overwritten)
├── checkpoints/e1/e1_full_vectors.npz     # −G/+G paired vectors → E3 directions
├── assets/pca_bases_gemma_sm.npz          # E2 bases
└── results/                               # final summaries
```

## Resume semantics (preemption-safe)

Every trial is appended to `{arm}.jsonl` (fsync'd) and synced to HF every 10
trials at the SAME path — the dataset always holds the latest state, not an
archive. On start, each arm downloads its checkpoint and skips done seeds.
A preempted amlt job is recovered by plain resubmission; frozen seeds make the
result identical to an uninterrupted run.

## Run

```bash
# local tests (CPU)
pytest multilayer_causal/tests -q

# package + submit (iterative: smoke → e1 → gate → e2/e3)
HF_TOKEN=$(python -c "from huggingface_hub import get_token; print(get_token())") \
  python multilayer_causal/scripts/push_code_to_hf.py
bash multilayer_causal/amlt/render.sh
amlt run multilayer_causal/amlt/smoke.rendered.yaml mlc-smoke
amlt run multilayer_causal/amlt/e1_main.rendered.yaml mlc-e1

# after E1: gates + S*
python -m multilayer_causal.src.analyze summary --out multilayer_causal/out
```

## Gates (pre-registered)

- **G1**: `e1_full` indistinguishable from `e1_anchor_plus` (Welch p>0.05 on
  bet_ratio AND stop-rate gap < 0.15) — else stop, debug harness.
- **S\***: smallest passing layer set; ties → deeper window. None passing →
  S\* = all layers ("distributed" interpretation).
- All per-metric tests are EXPLORATORY; paper-body promotion requires an n=200
  confirmatory rerun on a held-out state pool with one pre-named metric.
