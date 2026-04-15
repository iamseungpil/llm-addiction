# Archived: Old-Prompt Steering Results

**Archived**: 2026-04-14
**Reason**: These steering experiments used OLD prompts (no ROLE_INSTRUCTION, no G+M components).

The game mechanics (100 rounds, MW 4-zone, correct win tracking) were correct
in `run_v12_all_steering.py`, but the prompts did not match the original
behavioral experiments (v4_role for SM, v2_role for IC/MW).

## Paper impact

- `v12_n200` and `v12_crossdomain_steering` are still referenced in the paper
  for RQ3 (ρ=0.919, cross-domain sign reversal). These are kept in `json/`
  but should be replaced once corrected steering experiments complete.
- All other V12/V13 files are superseded.

## Files

- V12 per-model/task steering runs
- V12 sensitivity scans (s1a, s1b, s1c)
- V12 random direction controls
- V12 cross-analysis and sign reversal analysis
- V13 RQ3 analysis summary
