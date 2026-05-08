# M1 Outcome Claim Surgery Branches (Plan v4 §4 / §9bis)

Per codex Round 9: "Do not wait for results to decide prose." Three §3.2 paragraph
variants are committed in advance and selected at the M1 readout by the pre-registered
decision rule (lower 95% CI on logit-scale interaction `β_{+G × gambling} − β_{+G × portfolio}`).

The decision rule is single-test on logit scale; probability-scale marginal differences
and per-model interaction CIs are reported alongside as descriptive secondary output but
never substitute for the primary gate.

## M1-passes branch (gambling-specific; lower 95% CI on logit interaction > 0)

> "The `+G` autonomy framing operates as a gambling-specific propensity rather than a
> generic prompt-conditioned risk amplifier. Under a matched non-gambling-but-still-risky
> portfolio allocation task — temptation-rich asset menu including a 3× leveraged ETF, a
> single volatile stock, and an OTM call / crypto-like binary asset, with salient
> performance blurbs — `+G` produces no comparable shift in the bankruptcy-equivalent
> drawdown event rate; the cross-domain `domain × condition` mixed-logit interaction is
> `[β_interaction] [CI]` (logit scale, lower 95% CI > 0). The same six models that show
> the gambling-domain `+G` effect show only a `[Δ_portfolio]` shift on the portfolio
> task, while a `MAX_RISK` positive control verifies the portfolio task can register
> risk-seeking ([base_rate] → [max_risk_rate], +[uplift]). We therefore retain the §3.2
> 'gambling-addiction-like' framing: `+G` cues a domain-specific propensity, not generic
> risk amplification."

## M1-mixed branch (interaction CI excludes 0 but partial generalisation; |Δ_portfolio| > ε)

> "The `+G` autonomy framing is partly gambling-specific but also generalises in
> attenuated form to other prompt-conditioned risky decisions. The cross-domain
> `domain × condition` mixed-logit interaction is `[β_interaction] [CI]` (logit scale,
> lower 95% CI > 0), but `+G` also raises portfolio drawdown-equivalent risk by
> `[Δ_portfolio]` over BASE — non-zero though smaller than the gambling-domain
> `[Δ_gambling]` shift. The mechanism therefore amplifies risk-seeking generally with
> additional gambling-specific potentiation. We narrow the §3.2 framing from
> 'gambling-addiction-like' to 'autonomy-induced risk increase with domain-conditional
> amplification' and discuss the cross-domain readout in §6 as evidence that gambling
> tasks selectively recruit components of `+G` that other risky tasks do not."

## M1-fails branch (interaction CI includes 0 OR Δ_portfolio comparable to Δ_gambling)

> "The `+G` autonomy framing generalises to any prompt-conditioned risky decision
> rather than being gambling-specific. Replicating the prompt manipulation on a
> matched non-gambling portfolio allocation task with the same six models, `+G` raises
> the bankruptcy-equivalent drawdown event rate by `[Δ_portfolio]` — comparable in
> magnitude to the gambling-domain `[Δ_gambling]` shift — and the cross-domain
> `domain × condition` mixed-logit interaction is `[β_interaction] [CI]` (logit scale,
> lower 95% CI includes 0). We therefore narrow the §3.2 claim from 'gambling-
> addiction-like' to *autonomy-induced risk increase across decision domains*: `+G`
> reliably amplifies risk-taking but the gambling-domain framing of the effect is not
> licensed by the cross-domain evidence. Section 6 takes up the consequence — that
> gambling-specificity must be defended at the mechanistic level (§3.2 SAE features /
> §4 readout) rather than assumed from behaviour."

These three branches are committed in
`paper_experiments/m1_portfolio_discriminant/claim_surgery_M1_outcome_branches.md`.
The branch is selected at the M1 readout by the pre-registered decision rule and
inserted verbatim into the rebuttal (with the bracketed quantities filled from
`summary.json` in `analyze_m1.py` output).
