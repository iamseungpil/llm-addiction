// Every number the page draws. Each block names where it is printed in the
// NeurIPS 2026 camera-ready, so the page can be checked against the paper.
window.PAPER_DATA = {
  // Appendix Table "Comprehensive slot-machine results by model"
  // (tab:appendix-slot-comprehensive); Figure 2a. 1,600 games per model and arm.
  slotBankruptcy: [
    { model: "LLaMA-3.1-8B", fixed: 0.44, variable: 72.31 },
    { model: "Gemini-2.5-Flash", fixed: 3.12, variable: 48.06 },
    { model: "GPT-4o-mini", fixed: 0.0, variable: 21.31 },
    { model: "Claude-3.5-Haiku", fixed: 0.0, variable: 20.5 },
    { model: "GPT-4.1-mini", fixed: 0.0, variable: 6.31 },
    { model: "Gemma-2-9B", fixed: 0.0, variable: 5.44 },
  ],

  // Finding 4, first run: GPT-4o-mini, matched caps, all 32 prompt conditions.
  matchedCap: [
    { cap: 30, fixed: 0, variable: 14 },
    { cap: 50, fixed: 5, variable: 16 },
    { cap: 70, fixed: 0.4, variable: 17 },
  ],

  // Finding 5 / Appendix Table tab:choice-ladder. LLaMA-3.1-8B, $70 cap.
  choiceLadder: [
    { arm: "The game sets the wager at $70", short: "Game sets it", bankrupt: 2.0, rebet: 18, rounds: 1.0, games: 100 },
    { arm: "The model names its wager once", short: "Names it once", bankrupt: 5.0, rebet: 45, rounds: 1.9, games: 200 },
    { arm: "The model may revise it every round", short: "Revises every round", bankrupt: 85.0, rebet: 100, rounds: 15.9, games: 100 },
    { arm: "Revises every round, cap raised to $100", short: "…and a $100 cap", bankrupt: 80.0, rebet: 100, rounds: 15.1, games: 100 },
  ],

  // Finding 3 / Figure 3. Pooled over six models, 2,400 games per condition.
  goal: {
    bankruptcy: { base: 19, goal: 36 },
    highVariance: { base: 26, goalLo: 38, goalHi: 42 },
    movingTarget: { baseLo: 11, baseHi: 17, goalLo: 48, goalHi: 50 },
  },

  // Figure 4 sidecar (paper_data/fig04_causal_battery.json): mean bet ratio,
  // 200 trials per dose. Random band = mean ± 2 SD of norm-matched random
  // directions (Gemma: 20 directions at ±3; LLaMA: 5 directions at +3).
  steering: {
    doses: [-3, -2, -1, 0, 1, 2, 3],
    gemma: {
      layers: "L16–21",
      behaviour: [0.0137, 0.0199, 0.0192, 0.0608, 0.1499, 0.229, 0.2559],
      readout: [0.0701, 0.0641, 0.0527, 0.0608, 0.0881, 0.0935, 0.1166],
      band: { "-3": [0.012, 0.146], "3": [0.025, 0.142] },
      removal: -0.037,
    },
    llama: {
      layers: "L14–19",
      behaviour: [0.1616, 0.1642, 0.177, 0.2132, 0.2182, 0.2561, 0.2743],
      readout: [0.223, 0.2254, 0.1989, 0.2132, 0.1924, 0.1933, 0.1885],
      band: { "3": [0.177, 0.242] },
      removal: -0.052,
    },
  },
};
