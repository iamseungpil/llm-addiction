# Archived: Non-Paired-Design Steering Results

**Archived**: 2026-04-14
**Reason**: These experiments used α-dependent seeding, violating paired design.

## Details

`run_v16_multilayer_steering.py:117` computes:
```python
seed = g + seed_offset + abs(alpha) * 100000
```

This means different α values get different random sequences for the same game
index, preventing within-game paired comparison of dose-response. The causal
experiment plan (v3/v4) requires α-independent seeding:
```python
seed = g + TASK_SEED_OFFSET[task]  # same seed regardless of α
```

## Game mechanics

These used `play_game` imported from `run_v12_all_steering.py`, so game
mechanics (100 rounds, MW 4-zone, correct win tracking, 512 tokens) are correct.
Only the seeding design is flawed for confirmatory causal claims.

## Files

- V14 experiments (exp1-4): SM/IC/MW steering on LLaMA and Gemma
- V15 pilots: Exp B on LLaMA SM, Gemma pilot
- V16 multilayer scan: LLaMA layer sweep
