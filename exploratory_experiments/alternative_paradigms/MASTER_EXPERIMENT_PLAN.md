# Master Experiment Execution Plan

## 📋 Current Status Summary

### ✅ Completed & Valid
- **Gemma Investment Choice**: c10, c30, c50 (400 games each, parsing OK)
- **Gemma Blackjack**: Fixed $10, $30 (160 games each)
- **LLaMA Blackjack**: Variable (160 games)

### ⚠️ Completed But Invalid
- **LLaMA Investment Choice**: c10, c30, c50 (data exists but **100% parsing failure**)
  - Hallucination issue: bullet-point format confusion
  - **MUST RE-RUN with new prompt format**

### ❌ Not Started
- **Gemma Blackjack**: Variable betting
- **Investment Choice**: New prompt format test (LLaMA)

---

## 🚀 Execution Order (Priority-Based)

### 🔥 Priority 1: Critical Missing Experiments

#### 1A. Investment Choice - New Prompt Test (LLaMA)
**Purpose**: Validate that new prompt format fixes LLaMA hallucination

```bash
cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/src
sbatch investment_choice/slurm_prompt_test.sbatch
```

**Details**:
- Script: `test_new_prompt.py`
- Format: OLD vs NEW comparison
- Games: 10 reps × 4 conditions × 2 formats = 80 games
- Time: ~1-2 hours
- Output: `llama_prompt_format_test_YYYYMMDD.json`

**Success criteria**:
- NEW format hallucination rate < 10%
- NEW format valid parsing > 80%

---

#### 1B. Blackjack - Gemma Variable Betting
**Purpose**: Complete Gemma experiments for fair model comparison

```bash
cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/src
sbatch blackjack/slurm_gemma_variable.sbatch
```

**Details**:
- Games: 8 components × 20 reps = 160 games
- Time: ~2-3 hours
- Output: `blackjack_gemma_YYYYMMDD.json`
- Constraint: None (variable betting, like LLaMA)

---

### 🔄 Priority 2: Re-run with New Prompt (After 1A Success)

⚠️ **WAIT FOR PRIORITY 1A TO COMPLETE AND VERIFY SUCCESS**

#### 2A. Investment Choice - LLaMA c30 (New Prompt)
```bash
cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/src
sbatch investment_choice/slurm_llama_c30_new.sbatch
```

#### 2B. Investment Choice - LLaMA c50 (New Prompt)
```bash
sbatch investment_choice/slurm_llama_c50_new.sbatch
```

**Details** (each):
- Games: 2 bet types × 4 conditions × 50 reps = 400 games
- Time: ~3-4 hours each
- New prompt format applied
- Output: `llama_investment_c{30,50}_YYYYMMDD.json`

---

### 📊 Priority 3: SAE Analysis (After Experiments Complete)

#### 3A. Blackjack SAE Feature Extraction

**LLaMA** (can start now - data already available):
```bash
sbatch blackjack/slurm_sae_llama.sbatch
```

**Gemma** (wait for Priority 1B):
```bash
sbatch blackjack/slurm_sae_gemma.sbatch
```

**Time**: 4-8 hours each

---

#### 3B. Investment Choice SAE Feature Extraction

**Gemma** (can start now):
```bash
# Create SAE extraction script for investment choice
sbatch investment_choice/slurm_sae_gemma.sbatch
```

**LLaMA** (wait for Priority 2A/2B):
```bash
sbatch investment_choice/slurm_sae_llama.sbatch
```

**Time**: 4-8 hours each

---

#### 3C. SAE Correlation Analysis

**After feature extraction completes**:
```bash
# Blackjack
python blackjack/phase2_correlation_analysis.py --model llama --feature-dir ...
python blackjack/phase2_correlation_analysis.py --model gemma --feature-dir ...

# Investment Choice
python investment_choice/phase2_correlation_analysis.py --model llama --feature-dir ...
python investment_choice/phase2_correlation_analysis.py --model gemma --feature-dir ...
```

**Time**: ~30 minutes total

---

## 🎯 Quick Start - Execute All Priority 1 Now

```bash
cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms/src

# Priority 1A: New prompt test (CRITICAL)
sbatch investment_choice/slurm_prompt_test.sbatch

# Priority 1B: Gemma variable blackjack
sbatch blackjack/slurm_gemma_variable.sbatch

# Check status
squeue -u $USER
```

**Total time for Priority 1**: ~3 hours
**Total time for all experiments**: ~20-30 hours (with parallelization)

---

## 📈 Timeline Estimate

| Phase | Tasks | Time | Can Start |
|-------|-------|------|-----------|
| **Priority 1** | New prompt test + Gemma blackjack | 3 hrs | **NOW** |
| **Priority 2** | LLaMA investment re-run (c30, c50) | 8 hrs | After 1A validates |
| **Priority 3A** | SAE feature extraction | 16 hrs | After experiments complete |
| **Priority 3B** | SAE correlation analysis | 1 hr | After 3A completes |

**Total**: ~28 hours wall time (many tasks run in parallel)

---

## 🔍 Monitoring Commands

```bash
# Job status
squeue -u $USER

# Logs
tail -f /home/jovyan/beomi/llm-addiction-data/logs/inv_prompt_test_*.out
tail -f /home/jovyan/beomi/llm-addiction-data/logs/blackjack_gemma_var_*.out

# Results
ls -lht /home/jovyan/beomi/llm-addiction-data/investment_choice/*.json | head
ls -lht /home/jovyan/beomi/llm-addiction-data/blackjack/*.json | head
```

---

## ✅ Success Criteria

### Priority 1A (New Prompt Test)
- [ ] Hallucination rate < 10% (NEW format)
- [ ] Valid parsing > 80% (NEW format)
- [ ] Demonstrates improvement over OLD format

### Priority 1B (Gemma Variable)
- [ ] 160 games completed
- [ ] Data structure matches LLaMA variable
- [ ] Enables fair model comparison

### Priority 2 (LLaMA Re-run)
- [ ] Valid parsing > 80%
- [ ] No hallucination issues
- [ ] 400 games per constraint (c30, c50)

### Priority 3 (SAE Analysis)
- [ ] Features extracted for all layers
- [ ] Significant features identified (FDR < 0.05)
- [ ] Component effects explained
