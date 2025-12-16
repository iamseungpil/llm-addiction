# Full Experiment Launch Plan: Comprehensive Review

## Date: 2025-11-25

## Current Status

**✅ Pilot Test V2 Completed Successfully**
- Reference rate: 30.0% (V1: 0.0%)
- Design validated: LLMs track and reference previous goals
- Ready for full experiment

---

## Experiment Scope

### Total Scale
- **Models**: 4 (GPT-4o-mini, GPT-4.1-mini, Gemini-2.5-Flash, Claude-3.5-Haiku)
- **Constraints**: 4 ($10, $30, $50, $70)
- **Bet types**: 2 (fixed, variable)
- **Prompts**: 4 (BASE, G, M, GM)
- **Trials per condition**: 50
- **Total games**: 4 × 4 × 2 × 4 × 50 = **6,400 games**
- **Estimated rounds**: ~10 per game = **64,000 API calls**

### Data Generation
- **~128 MB per model** (based on pilot: 5 games = ~500 KB)
- **Total data**: ~512 MB (4 models)
- **With analysis files**: ~1 GB total

---

## Cost & Time Estimation

### API Costs (Conservative)
| Model | Cost per 1M tokens | Tokens per game | Total cost |
|-------|-------------------|-----------------|------------|
| Claude Haiku | $0.80 / $4.00 | ~2K | ~$150-200 |
| GPT-4.1-mini | $0.15 / $0.60 | ~1.5K | ~$20-30 |
| GPT-4o-mini | $0.15 / $0.60 | ~1.5K | ~$20-30 |
| Gemini Flash | Free (rate limited) | ~2K | $0 |

**Total estimated cost**: ~$200-260 per full run

### Time Estimation
- **API call latency**: ~3 seconds per round
- **64,000 calls sequential**: ~53 hours
- **With 4 models parallel**: ~13 hours per model
- **Total wall time**: 13-15 hours (if all parallel)

---

## Critical Issues to Check

### 1. API Keys Validation ⚠️

**Issue**: Only tested Claude Haiku, GPT key failed in earlier test

**Required**:
```bash
# Test all API keys before launch
export OPENAI_API_KEY="sk-proj-5SaQ..."  # ⚠️ Failed earlier
export ANTHROPIC_API_KEY="sk-ant..."     # ✅ Tested, works
export GOOGLE_API_KEY="AIzaSy..."        # ❓ Not tested
```

**Action needed**:
- Validate GPT API key (failed with 401 earlier)
- Test Gemini API key
- Confirm all keys have sufficient quota

### 2. Code Issues ✅ FIXED

**Issue**: Regex pattern had double backslashes
**Status**: ✅ Fixed in lines 247-250
**Verified**: Pilot test V2 ran successfully

### 3. Disk Space ✅

**Current usage**: Check available space
**Required**: ~2 GB (conservative: data + checkpoints + logs)
**Action**: Verify `/data/llm_addiction/` has >5 GB free

### 4. Checkpoint System ✅

**Implementation**: Lines 420-445 in `base_experiment.py`
- Saves every 10 games
- Includes resume functionality
- Tested: Not yet tested at scale

**Risk**: If experiment crashes, need to ensure resume works

### 5. Error Handling ⚠️

**Current implementation**:
```python
try:
    response = self.get_model_response(prompt)
except Exception as e:
    self.log(f"❌ API error: {e}")
    response = "I choose Option 1 (Safe Exit)"
```

**Issues**:
- Infinite retry with exponential backoff (good)
- But fallback to Option 1 might bias results
- No alerting mechanism if errors accumulate

**Risk**: Silent failures could bias data

### 6. Parallel Execution ⚠️

**Current design**: Sequential, one model at a time

**Problem**: 53 hours sequential is too long

**Solution needed**:
- Run 4 models in parallel (4 separate processes)
- Reduces wall time to ~13-15 hours
- Need to coordinate/monitor

**Implementation**:
```bash
# Terminal 1
python3 src/run_experiment.py --model gpt4o_mini

# Terminal 2
python3 src/run_experiment.py --model gpt41_mini

# Terminal 3
python3 src/run_experiment.py --model gemini_flash

# Terminal 4
python3 src/run_experiment.py --model claude_haiku
```

---

## Pre-Launch Checklist

### Critical (Must Do)
- [ ] **Validate ALL API keys** (especially GPT, which failed earlier)
- [ ] **Test resume functionality** (run 20 games, kill, resume)
- [ ] **Check disk space** (need >5 GB free)
- [ ] **Review error handling** (ensure no silent failures)
- [ ] **Test one full condition** (50 games) before full launch

### Important (Should Do)
- [ ] **Setup monitoring** (tail logs in separate terminals)
- [ ] **Plan parallel execution** (which models on which terminals)
- [ ] **Backup pilot results** (already done: pilot_results_v1_backup)
- [ ] **Document expected completion time** (13-15 hours if parallel)

### Nice to Have (Optional)
- [ ] Setup email/Slack alerts for completion
- [ ] Create progress dashboard
- [ ] Pre-compute expected storage needs

---

## Execution Strategy

### Option A: Sequential (Safe but Slow)
**Pros**: Simple, less error-prone
**Cons**: 50+ hours total
**Verdict**: ❌ Not recommended (too slow)

### Option B: Parallel Models (Recommended)
**Pros**: 13-15 hours, manageable
**Cons**: Need to monitor 4 processes
**Verdict**: ✅ **Recommended**

**Implementation**:
1. Open 4 terminal sessions (or tmux windows)
2. Run one model per terminal
3. Monitor logs for errors
4. Checkpoints save every 10 games

### Option C: Parallel Everything (Risky)
**Pros**: Fastest (2-3 hours)
**Cons**: Complex, rate limits, hard to debug
**Verdict**: ⚠️ Not recommended for first run

---

## Risk Assessment

### High Risk ⚠️
1. **GPT API key invalid** (401 error in earlier test)
   - Mitigation: Validate before launch
   - Impact: 50% of data lost if not caught early

2. **Rate limits** (especially Gemini)
   - Mitigation: Add rate limiting logic
   - Impact: Experiment could stall

3. **Disk space exhaustion**
   - Mitigation: Check before launch
   - Impact: Data loss, corruption

### Medium Risk ⚠️
1. **Checkpoint resume not working**
   - Mitigation: Test with 20 games first
   - Impact: Need to restart from beginning

2. **Silent API failures**
   - Mitigation: Review error logs frequently
   - Impact: Biased results (too many Option 1)

### Low Risk ✅
1. **Regex parsing errors**
   - Status: Fixed and tested
   - Impact: Minimal

2. **Code crashes**
   - Mitigation: Good error handling
   - Impact: Resume from checkpoint

---

## Launch Sequence (Recommended)

### Phase 1: Final Validation (30 minutes)
1. ✅ Validate all API keys
   ```bash
   # Test GPT
   export OPENAI_API_KEY="..." && python3 -c "from openai import OpenAI; OpenAI().chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':'test'}], max_tokens=5)"

   # Test Gemini
   export GOOGLE_API_KEY="..." && python3 -c "import google.generativeai as genai; genai.configure(api_key='...'); genai.GenerativeModel('gemini-2.0-flash-exp').generate_content('test')"
   ```

2. ✅ Check disk space
   ```bash
   df -h /data/llm_addiction/
   ```

3. ✅ Test checkpoint resume
   ```bash
   # Run 20 games
   python3 pilot_test_claude.py  # Modified to run 20 games
   # Kill it
   # Resume from checkpoint
   ```

### Phase 2: Single Condition Test (2 hours)
1. Run ONE full condition (50 games, one model)
2. Verify:
   - All data saved correctly
   - Goal references present
   - Checkpoints working
   - No errors

### Phase 3: Parallel Launch (13-15 hours)
1. Open 4 terminals/tmux sessions
2. Start all 4 models simultaneously
3. Monitor logs for first hour
4. Check every 3-4 hours
5. Expected completion: 13-15 hours

### Phase 4: Validation (1 hour)
1. Check all result files created
2. Verify data integrity
3. Run preliminary analysis
4. Compare to pilot results

---

## Contingency Plans

### If GPT API key fails
- **Option A**: Use only Claude + Gemini (3,200 games)
- **Option B**: Get new GPT API key
- **Option C**: Run GPT separately later

### If rate limits hit
- **Option A**: Add sleep delays (1-2 sec between calls)
- **Option B**: Run models sequentially instead of parallel
- **Option C**: Spread over multiple days

### If disk space runs out
- **Option A**: Compress old checkpoints
- **Option B**: Move data to different drive
- **Option C**: Stream results to remote storage

### If experiment crashes
- **Option A**: Resume from last checkpoint
- **Option B**: Check logs for error pattern
- **Option C**: Run remaining games manually

---

## Success Criteria

**Experiment is successful if**:
- ✅ All 6,400 games completed
- ✅ All result files saved
- ✅ Goal reference rate ≥25% in G/GM conditions
- ✅ Data quality matches pilot (no systematic errors)
- ✅ Checkpoint/resume worked correctly

**Experiment needs revision if**:
- ❌ >10% of games have API errors
- ❌ Goal reference rate drops <10%
- ❌ Systematic bias detected (e.g., all Option 1)
- ❌ Data corruption or loss

---

## Go/No-Go Decision

### GO if ✅:
1. All API keys validated
2. Disk space sufficient (>5 GB)
3. Single condition test passes
4. Checkpoint resume works
5. Team agrees on 13-15 hour timeline

### NO-GO if ❌:
1. GPT API key still invalid
2. Checkpoint resume fails
3. Disk space <2 GB
4. Single condition test shows issues
5. Need more design iterations

---

## Current Recommendation

**Status**: ⚠️ **NOT READY - Need Pre-Launch Validation**

**Next Steps**:
1. **Validate GPT API key** (critical blocker)
2. **Test checkpoint resume** (20 games → kill → resume)
3. **Run single condition test** (50 games, verify quality)
4. **Then decide on full launch**

**Do NOT launch full experiment until**:
- GPT API key works (currently fails with 401)
- Checkpoint tested
- Single condition verified

**Estimated time to ready**: 3-4 hours (validation + single condition)

---

**Created**: 2025-11-25
**Status**: Ready for validation phase
**Next action**: API key validation + checkpoint testing
