# Revised Launch Plan: Claude + Gemini Parallel Execution

## Date: 2025-11-25

## Ultrathink Analysis Complete

### API Key Status (Verified)

| Model | API Key | Status | Test Result |
|-------|---------|--------|-------------|
| Claude Haiku | ✅ Valid | READY | Pilot test: 30% ref rate |
| Gemini Flash | ✅ Valid | READY | Test successful |
| GPT-4o-mini | ❌ Invalid (401) | BLOCKED | Cannot proceed |
| GPT-4.1-mini | ❌ Invalid (401) | BLOCKED | Cannot proceed |

**Decision**: Proceed with Claude + Gemini, skip GPT for now

---

## Revised Experiment Scope

### Reduced Scale (Claude + Gemini Only)
- **Models**: 2 (Claude-3.5-Haiku, Gemini-2.0-Flash)
- **Constraints**: 4 ($10, $30, $50, $70)
- **Bet types**: 2 (fixed, variable)
- **Prompts**: 4 (BASE, G, M, GM)
- **Trials**: 50 per condition
- **Total games**: 2 × 4 × 2 × 4 × 50 = **3,200 games**
- **Estimated rounds**: ~10 per game = **32,000 API calls**

### Cost & Time (Revised)
- **Claude Haiku**: 1,600 games × ~3 sec/round × 10 rounds = ~13 hours
- **Gemini Flash**: 1,600 games × ~3 sec/round × 10 rounds = ~13 hours
- **Parallel execution**: **~13-14 hours wall time**
- **Cost**: ~$100-150 (mostly Claude, Gemini is rate-limited free tier)

---

## Execution Plan

### Phase 1: API Key Setup ✅ COMPLETE
- ✅ Claude API key validated (pilot test success)
- ✅ Gemini API key validated (test successful)

### Phase 2: Run Experiment (Two Models in Parallel)

#### Terminal 1: Claude Haiku
```bash
cd /home/ubuntu/llm_addiction/investment_choice_bet_constraint_cot

export ANTHROPIC_API_KEY="your-api-key-here"

# Run all conditions for Claude
python3 src/run_all_experiments.py --model claude_haiku
```

#### Terminal 2: Gemini Flash
```bash
cd /home/ubuntu/llm_addiction/investment_choice_bet_constraint_cot

export GEMINI_API_KEY="AIzaSyC5noPhw0WTM051ifvAoL7fsYTQkVMhvUs"

# Run all conditions for Gemini
python3 src/run_all_experiments.py --model gemini_flash
```

**Expected completion**: 13-14 hours (if run in parallel)

---

## Implementation Check

### Need to verify: `run_all_experiments.py` structure

```python
# Expected structure:
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                       choices=['claude_haiku', 'gemini_flash', 'gpt4o_mini', 'gpt41_mini'])
    parser.add_argument('--bet_constraint', type=int, default=None)
    parser.add_argument('--bet_type', default=None)

    args = parser.parse_args()

    # Run all conditions for specified model
    for constraint in [10, 30, 50, 70]:
        for bet_type in ['fixed', 'variable']:
            runner = get_runner(args.model, constraint, bet_type)
            runner.run_experiment(trials_per_condition=50)
```

---

## Risk Assessment (Updated)

### ✅ Resolved Risks
1. ~~GPT API key invalid~~ → Skip GPT, use only Claude + Gemini
2. ~~Gemini API key untested~~ → Tested and working
3. ~~Regex error~~ → Fixed in lines 247-250

### ⚠️ Remaining Risks

**Medium Risk**:
1. **Gemini rate limits**
   - Free tier has rate limits
   - Mitigation: Built-in exponential backoff (lines 61-65)
   - Impact: Gemini might take longer (~16-18 hours vs 13)

2. **Checkpoint/resume not tested at scale**
   - Saves every 10 games (line 406)
   - Resume logic exists (lines 325-355)
   - Mitigation: Test with 100 games first
   - Impact: If crash, need manual restart

3. **Disk space**
   - Need: ~500 MB (down from 1 GB)
   - Should check: `df -h /data/llm_addiction/`

**Low Risk**:
1. **API failures** - Good retry logic
2. **Code crashes** - Checkpoints every 10 games
3. **Data corruption** - JSON format is robust

---

## Pre-Launch Checklist (Final)

### Critical ✅
- [x] Claude API key validated
- [x] Gemini API key validated
- [x] Regex pattern fixed
- [x] Pilot test shows 30% reference rate
- [ ] Check disk space (>2 GB free)
- [ ] Verify `run_all_experiments.py` exists and works

### Important
- [ ] Test checkpoint/resume (run 20 games → kill → resume)
- [ ] Check if we need separate script or can use existing
- [ ] Backup pilot results (already done)

### Optional
- [ ] Setup monitoring dashboard
- [ ] Email alerts on completion

---

## Modified Launch Sequence

### Step 1: Verify Infrastructure (5 minutes)
```bash
# Check disk space
df -h /data/llm_addiction/
# Should show >2 GB free

# Check if run script exists
ls -la src/run_all_experiments.py
```

### Step 2: Quick Test (30 minutes)
```bash
# Run 10 games with Claude to test full pipeline
export ANTHROPIC_API_KEY="..."
python3 -c "
from src.models.claude_runner import ClaudeRunner
runner = ClaudeRunner(bet_constraint=10, bet_type='variable')
runner.run_experiment(trials_per_condition=10)
"
# Verify:
# - Data saves correctly
# - No errors
# - Goal references present
```

### Step 3: Launch Parallel Execution (13-14 hours)

**Terminal 1 (tmux session: claude_exp)**:
```bash
tmux new -s claude_exp
cd /home/ubuntu/llm_addiction/investment_choice_bet_constraint_cot
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# Method A: If run_all_experiments.py exists
python3 src/run_all_experiments.py --model claude_haiku

# Method B: If need to run manually
for constraint in 10 30 50 70; do
  for bet_type in fixed variable; do
    python3 -c "
from src.models.claude_runner import ClaudeRunner
runner = ClaudeRunner(bet_constraint=$constraint, bet_type='$bet_type')
runner.run_experiment(trials_per_condition=50)
"
  done
done
```

**Terminal 2 (tmux session: gemini_exp)**:
```bash
tmux new -s gemini_exp
cd /home/ubuntu/llm_addiction/investment_choice_bet_constraint_cot
export GEMINI_API_KEY="AIzaSyC5noPhw0WTM051..."

# Same structure as Claude
python3 src/run_all_experiments.py --model gemini_flash
```

**Monitor progress**:
```bash
# Terminal 3
watch -n 60 'tail -10 /data/llm_addiction/investment_choice_bet_constraint_cot/logs/*.log'
```

### Step 4: Validation (1 hour)
```bash
# Check all result files
ls -lh /data/llm_addiction/investment_choice_bet_constraint_cot/results/

# Verify data integrity
python3 -c "
import json
import glob

files = glob.glob('/data/llm_addiction/investment_choice_bet_constraint_cot/results/*.json')
print(f'Total result files: {len(files)}')

for f in files:
    with open(f) as fh:
        data = json.load(fh)
        print(f'{f}: {len(data.get(\"results\", []))} games')
"
```

---

## Success Criteria (Revised)

**Experiment successful if**:
- ✅ All 3,200 games completed (1,600 per model)
- ✅ All result files saved
- ✅ Goal reference rate ≥25% in G/GM conditions
- ✅ Claude: <5% API errors
- ✅ Gemini: <10% API errors (rate limits expected)

**Needs investigation if**:
- ⚠️ Goal reference rate <20%
- ⚠️ >10% API errors for Claude
- ⚠️ >20% API errors for Gemini
- ⚠️ Systematic bias (>80% Option 1)

---

## Timeline

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| Infrastructure check | 5 min | T+0 | T+0:05 |
| Quick test (10 games) | 30 min | T+0:05 | T+0:35 |
| Launch parallel | 5 min | T+0:35 | T+0:40 |
| **Experiment running** | **13-14 hrs** | **T+0:40** | **T+14:40** |
| Validation | 1 hr | T+14:40 | T+15:40 |
| **Total** | **~16 hrs** | | |

**If start now (03:30)**: Complete by ~19:30 tomorrow

---

## Contingency Plans

### If Gemini hits rate limits
- **Option A**: Add sleep(2) between calls
- **Option B**: Run Gemini solo after Claude completes
- **Option C**: Reduce Gemini to 25 trials/condition

### If experiment crashes
- **Option A**: Resume from last checkpoint (every 10 games)
- **Option B**: Check logs for error pattern
- **Option C**: Skip problematic condition, continue

### If disk space runs out
- **Option A**: Compress old checkpoints
- **Option B**: Move logs to different location
- **Option C**: Stream results incrementally

---

## Go/No-Go Decision

### Current Status: ✅ **READY TO LAUNCH**

**All blockers resolved**:
- ✅ Claude API working (pilot validated)
- ✅ Gemini API working (test passed)
- ✅ Regex fixed
- ✅ Goal tracking validated (30% rate)

**Remaining checks** (5-35 minutes):
1. Disk space check (1 min)
2. Verify run script exists (1 min)
3. Quick test 10 games (30 min)

**Recommendation**:
1. Do quick checks (5 min)
2. Run quick test (30 min)
3. If all good → **LAUNCH FULL EXPERIMENT**

---

## Next Immediate Actions

1. **Check disk space**: `df -h /data/llm_addiction/`
2. **Check run script**: `ls src/run_all_experiments.py`
3. **Quick test**: 10 games with Claude
4. **If all pass → LAUNCH**

**Estimated time to launch**: 35-40 minutes from now

---

**Created**: 2025-11-25 03:30
**Status**: Ready for pre-launch checks
**Next**: Infrastructure verification → Quick test → Launch
