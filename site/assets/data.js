// Every number the page draws. Each block names where it is printed in the
// NeurIPS 2026 camera-ready, so the page can be checked against the paper.
window.PAPER_DATA = {
  // Appendix Table "Comprehensive slot-machine results by model"
  // (tab:appendix-slot-comprehensive). LLaMA-3.1-8B, 1,600 games per arm;
  // bankruptcy %. Also recomputable from the released run (tools/extract_games.py
  // prints 7 and 1,157 bankruptcies of 1,600).
  slotLlama: { fixed: 0.44, variable: 72.31, gamesPerArm: 1600 },

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
      band: { "-3": [0.013, 0.146], "3": [0.023, 0.143] },
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
