# Blackjack Experiment & SAE Analysis Plan

## 📋 Current Status

### ✅ Completed
- **Gemma Fixed Betting**: 160 games (8 components × 20 reps)
  - File: `blackjack_gemma_20260220_023838.json`
  - Bankruptcy: 15.0%, Loss: $41.9 avg
- **LLaMA Variable Betting**: 160 games (8 components × 20 reps)
  - File: `blackjack_llama_20260219_005625.json`
  - Bankruptcy: 11.2%, Loss: $28.3 avg

### ⏳ To Do
1. **Gemma Variable Betting**: Complete the experiment
2. **SAE Feature Extraction**: Extract features from both models
3. **SAE Correlation Analysis**: Find features correlated with gambling behaviors

---

## 🚀 Execution Plan

### Phase 1: Gemma Variable Betting Experiment

**Purpose**: Complete Gemma variable betting to enable fair model comparison

```bash
cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/src
sbatch blackjack/slurm_gemma_variable.sbatch
```

**Expected output**:
- File: `/scratch/x3415a02/data/llm-addiction/blackjack/blackjack_gemma_YYYYMMDD_HHMMSS.json`
- Size: ~3-5 MB (160 games)
- Time: ~2-3 hours

**Monitor progress**:
```bash
# Check job status
squeue -u $USER

# Watch logs
tail -f /scratch/x3415a02/data/llm-addiction/logs/blackjack_gemma_var_*.out
```

---

### Phase 2: SAE Feature Extraction

**Purpose**: Extract SAE features from decision-making prompts to identify neural mechanisms

#### 2A. LLaMA SAE Extraction

```bash
sbatch blackjack/slurm_sae_llama.sbatch
```

**Configuration**:
- Model: LLaMA-3.1-8B
- SAE: LlamaScope (`fnlp/Llama3_1-8B-Base-LXR-8x`)
- Layers: 25-31 (7 layers, 32K features each)
- Input: Most recent `blackjack_llama_*.json`
- Output: `/scratch/x3415a02/data/llm-addiction/blackjack/sae_features/llama/`

**Expected output files**:
```
sae_features/llama/
├── blackjack_llama_YYYYMMDD_HHMMSS_L25.npz
├── blackjack_llama_YYYYMMDD_HHMMSS_L26.npz
├── ...
└── blackjack_llama_YYYYMMDD_HHMMSS_L31.npz
```

**Time**: ~4-5 hours (7 layers × 160 games)

#### 2B. Gemma SAE Extraction

⚠️ **Wait for Gemma variable experiment to complete first!**

```bash
sbatch blackjack/slurm_sae_gemma.sbatch
```

**Configuration**:
- Model: Gemma-2-9B
- SAE: GemmaScope (`gemma-scope-9b-pt-res-canonical`)
- Layers: 26, 28, 30, 32, 35, 38, 40 (7 layers, 131K features each)
- Input: Most recent `blackjack_gemma_*.json` (will use FIXED or VARIABLE depending on timestamp)
- Output: `/scratch/x3415a02/data/llm-addiction/blackjack/sae_features/gemma/`

**Time**: ~6-8 hours (7 layers × 160 games, larger SAE)

---

### Phase 3: SAE Correlation Analysis

**Purpose**: Find features that differentiate gambling behaviors

#### 3A. LLaMA Analysis

```bash
python blackjack/phase2_correlation_analysis.py \
  --model llama \
  --feature-dir /scratch/x3415a02/data/llm-addiction/blackjack/sae_features/llama/ \
  --fdr 0.05
```

#### 3B. Gemma Analysis

```bash
python blackjack/phase2_correlation_analysis.py \
  --model gemma \
  --feature-dir /scratch/x3415a02/data/llm-addiction/blackjack/sae_features/gemma/ \
  --fdr 0.05
```

**Output**:
- `{model}_correlation_results.json`: Significant features by layer
- Console output: Top features with effect sizes

**Time**: ~10-20 minutes per model

---

## 📊 Expected Insights from SAE Analysis

### Research Questions

1. **Component Encoding**:
   - Which features activate for GMHWP vs BASE?
   - Do features encode specific components (G, M, H, W, P)?

2. **Behavioral Prediction**:
   - Which features predict bankruptcy?
   - Are there "risk-taking" features vs "conservative" features?

3. **Model Differences**:
   - Why does GMHWP have opposite effects (Gemma 0% vs LLaMA 20% bankruptcy)?
   - Do models encode information differently?

4. **Near-Miss Pattern (P component)**:
   - P component reduces bankruptcy to 0-5% in both models
   - What features drive this protective effect?

### Analysis Targets

**Component Comparisons** (FDR < 0.05):
- BASE vs G (Goal setting effect)
- BASE vs M (Maximize effect)
- BASE vs GMHWP (Full information effect)
- BASE vs P (Pattern awareness effect)

**Expected Feature Types**:
- Decision-making features (risk assessment)
- Goal-tracking features (target monitoring)
- Pattern recognition features (near-miss detection)
- Outcome evaluation features (win/loss processing)

---

## 🎯 Quick Start (All-in-One)

```bash
# Step 1: Submit Gemma variable experiment
cd /scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/src
sbatch blackjack/slurm_gemma_variable.sbatch

# Step 2: Check when Gemma completes
squeue -u $USER

# Step 3: When Gemma is done, submit both SAE extractions
sbatch blackjack/slurm_sae_llama.sbatch
sbatch blackjack/slurm_sae_gemma.sbatch

# Step 4: When SAE extractions complete, run correlation analysis
python blackjack/phase2_correlation_analysis.py --model llama \
  --feature-dir /scratch/x3415a02/data/llm-addiction/blackjack/sae_features/llama/

python blackjack/phase2_correlation_analysis.py --model gemma \
  --feature-dir /scratch/x3415a02/data/llm-addiction/blackjack/sae_features/gemma/
```

---

## 📈 Timeline Estimate

| Phase | Task | Time | Dependency |
|-------|------|------|------------|
| 1 | Gemma variable experiment | 2-3 hrs | None |
| 2A | LLaMA SAE extraction | 4-5 hrs | Gemma completion (parallel) |
| 2B | Gemma SAE extraction | 6-8 hrs | Gemma completion |
| 3 | Correlation analysis | 30 min | Phase 2 completion |

**Total time**: ~8-12 hours (with parallelization)

---

## 🔍 Monitoring & Debugging

### Check job status
```bash
squeue -u $USER
```

### View logs
```bash
# Experiment logs
tail -f /scratch/x3415a02/data/llm-addiction/logs/blackjack_gemma_var_*.out

# SAE extraction logs
tail -f /scratch/x3415a02/data/llm-addiction/logs/blackjack_sae_llama_*.out
tail -f /scratch/x3415a02/data/llm-addiction/logs/blackjack_sae_gemma_*.out
```

### Check outputs
```bash
# List experiment results
ls -lht /scratch/x3415a02/data/llm-addiction/blackjack/*.json | head

# List SAE features
ls -lh /scratch/x3415a02/data/llm-addiction/blackjack/sae_features/llama/
ls -lh /scratch/x3415a02/data/llm-addiction/blackjack/sae_features/gemma/
```

### GPU usage
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

---

## 📝 Notes

- **Gemma SAE** is large (131K features) → requires A100 GPU (80GB VRAM)
- **LLaMA SAE** is smaller (32K features) → V100 GPU (32GB) is sufficient
- Phase 2 analysis is CPU-intensive but can run on login node (quick)
- Results are automatically checkpointed during extraction
