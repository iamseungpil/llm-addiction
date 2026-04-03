#!/usr/bin/env python3
"""Generate a V14 causal validation report from completed experiment JSON files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis")
RESULTS_DIR = BASE_DIR / "results"
JSON_DIR = RESULTS_DIR / "json"
FIG_DIR = RESULTS_DIR / "figures"
MD_PATH = RESULTS_DIR / "sae_v14_causal_validation_study.md"
SUMMARY_JSON_PATH = RESULTS_DIR / "v14_latest_summary.json"
READY_MD_PATH = RESULTS_DIR / "v14_READY_FOR_APPROVAL.md"
NEXT_PROMPT_PATH = RESULTS_DIR / "v14_NEXT_CODEX_PROMPT.txt"

FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
EXPERIMENT_SPECS = {
    "exp1": {
        "glob": "v14_exp1_llama_sm_perm20_*.json",
        "label": "LLaMA SM",
        "family": "within",
        "source": "SM",
        "target": "SM",
    },
    "exp2a": {
        "glob": "v14_exp2a_llama_ic_n100_*.json",
        "label": "LLaMA IC",
        "family": "within",
        "source": "IC",
        "target": "IC",
    },
    "exp2b": {
        "glob": "v14_exp2b_llama_mw_n100_*.json",
        "label": "LLaMA MW",
        "family": "within",
        "source": "MW",
        "target": "MW",
    },
    "exp4": {
        "glob": "v14_exp4_gemma_mw_n100_*.json",
        "label": "Gemma MW",
        "family": "within",
        "source": "MW",
        "target": "MW",
    },
    "exp5": {
        "glob": "v14_exp5_gemma_sm_n100_*.json",
        "label": "Gemma SM",
        "family": "within",
        "source": "SM",
        "target": "SM",
    },
    "exp6": {
        "glob": "v14_exp6_gemma_ic_n100_*.json",
        "label": "Gemma IC",
        "family": "within",
        "source": "IC",
        "target": "IC",
    },
    "exp3_mw2ic": {
        "glob": "v14_exp3_mw2ic_*.json",
        "label": "MW -> IC",
        "family": "cross",
        "source": "MW",
        "target": "IC",
    },
    "exp3_mw2sm": {
        "glob": "v14_exp3_mw2sm_*.json",
        "label": "MW -> SM",
        "family": "cross",
        "source": "MW",
        "target": "SM",
    },
    "exp3_ic2sm": {
        "glob": "v14_exp3_ic2sm_*.json",
        "label": "IC -> SM",
        "family": "cross",
        "source": "IC",
        "target": "SM",
    },
}

VERDICT_COLORS = {
    "BK_SPECIFIC_CONFIRMED": "#1B9E77",
    "BK_SIGNIFICANT_NOT_SPECIFIC": "#D95F02",
    "NOT_SIGNIFICANT": "#7570B3",
}


def _load_json(path: Path) -> dict[str, Any]:
    text = path.read_text()
    text = text.replace("NaN", "null").replace("Infinity", "null")
    return json.loads(text)


def _latest(pattern: str) -> Path | None:
    matches = sorted(JSON_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def load_latest_results() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    missing = []
    for key, spec in EXPERIMENT_SPECS.items():
        path = _latest(spec["glob"])
        if path is None:
            missing.append(key)
            continue
        data = _load_json(path)
        data["_path"] = str(path)
        data["_label"] = spec["label"]
        data["_family"] = spec["family"]
        data["_source"] = spec["source"]
        data["_target"] = spec["target"]
        results[key] = data
    if missing:
        raise FileNotFoundError(f"Missing V14 result files for: {', '.join(missing)}")
    return results


def assess_outcome(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    confirmed = [k for k, r in results.items() if r["verdict"] == "BK_SPECIFIC_CONFIRMED"]
    borderline = [k for k, r in results.items() if r["verdict"] == "BK_SIGNIFICANT_NOT_SPECIFIC"]
    cross_confirmed = [k for k in confirmed if results[k]["_family"] == "cross"]
    within_confirmed = [k for k in confirmed if results[k]["_family"] == "within"]
    gemma_within_confirmed = [k for k in confirmed if k in {"exp4", "exp5", "exp6"}]
    positive = "exp1" in confirmed and (len(confirmed) >= 2 or len(cross_confirmed) >= 1)
    if positive:
        recommendation = "positive"
        recommendation_text = (
            "V14 supports a narrowed causal claim. The strongest evidence remains "
            "LLaMA SM, and at least one additional experiment also satisfies "
            "direction specificity against random controls."
        )
    elif "exp1" in confirmed and gemma_within_confirmed:
        recommendation = "borderline"
        recommendation_text = (
            "V14 supports a conservative cross-model claim. LLaMA SM remains the "
            "anchor result, and at least one Gemma within-domain replication also "
            "passes random-direction controls, but broader generalization should "
            "still be framed cautiously."
        )
    elif "exp1" in confirmed or borderline:
        recommendation = "borderline"
        recommendation_text = (
            "V14 supports a conservative claim. LLaMA SM may retain strong causal "
            "evidence, but broader within-domain or cross-domain generalization "
            "should be presented as exploratory."
        )
    else:
        recommendation = "negative"
        recommendation_text = (
            "V14 does not support a direction-specific causal claim beyond chance "
            "controls. Any steering result should be reported as exploratory."
        )
    return {
        "recommendation": recommendation,
        "recommendation_text": recommendation_text,
        "confirmed": confirmed,
        "borderline": borderline,
        "within_confirmed": within_confirmed,
        "cross_confirmed": cross_confirmed,
        "gemma_within_confirmed": gemma_within_confirmed,
    }


def _extract_curve(result: dict[str, Any]) -> tuple[list[float], list[float], list[list[float]]]:
    bk = [result["bk_direction"]["bk_rates"][str(a)] for a in ALPHAS]
    randoms = [
        [rc["bk_rates"][str(a)] for a in ALPHAS]
        for rc in result["random_controls"]
    ]
    return ALPHAS, bk, randoms


def generate_within_domain_figure(results: dict[str, dict[str, Any]]) -> str:
    keys = ["exp1", "exp2a", "exp2b", "exp4", "exp5", "exp6"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, key in zip(axes, keys):
        r = results[key]
        xs, bk, randoms = _extract_curve(r)
        for rr in randoms:
            ax.plot(xs, rr, color="#CCCCCC", linewidth=1.0, alpha=0.6, zorder=1)
        if randoms:
            arr = np.array(randoms)
            ax.plot(xs, arr.mean(axis=0), color="#999999", linestyle="--", linewidth=1.4, label="Random mean", zorder=2)
        ax.plot(xs, bk, color="#0072B2", linewidth=2.6, marker="o", label="BK direction", zorder=3)
        ax.axhline(r["baseline_bk"], color="#444444", linestyle=":", linewidth=1.2)
        ax.set_title(
            f"{r['_label']} | {r['verdict']}",
            fontsize=10,
            color=VERDICT_COLORS.get(r["verdict"], "#222222"),
        )
        ax.text(
            0.03,
            0.05,
            f"rho={r['bk_direction']['rho']}, perm_p={r['permutation_p']}\n"
            f"rand sig={r['n_random_significant']}/{r['n_random']}",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BBBBBB", alpha=0.9),
        )
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)
    for ax in axes[len(keys):]:
        ax.axis("off")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Figure 1: Within-Domain Direction Steering with Random Controls", fontsize=14, fontweight="bold")
    fig.supxlabel("Steering strength (alpha)")
    fig.supylabel("Bankruptcy rate")
    out = FIG_DIR / "v14_fig1_within_domain_dose_response.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"figures/{out.name}"


def generate_crossdomain_figure(results: dict[str, dict[str, Any]]) -> str:
    keys = ["exp3_mw2ic", "exp3_mw2sm", "exp3_ic2sm"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, key in zip(axes, keys):
        r = results[key]
        xs, bk, randoms = _extract_curve(r)
        for rr in randoms:
            ax.plot(xs, rr, color="#CCCCCC", linewidth=1.0, alpha=0.7, zorder=1)
        if randoms:
            arr = np.array(randoms)
            ax.plot(xs, arr.mean(axis=0), color="#999999", linestyle="--", linewidth=1.4, zorder=2)
        ax.plot(xs, bk, color="#D55E00", linewidth=2.6, marker="o", zorder=3)
        ax.axhline(r["baseline_bk"], color="#444444", linestyle=":", linewidth=1.2)
        ax.set_title(
            f"{r['_label']} | {r['verdict']}",
            fontsize=10,
            color=VERDICT_COLORS.get(r["verdict"], "#222222"),
        )
        ax.text(
            0.04,
            0.05,
            f"rho={r['bk_direction']['rho']}, perm_p={r['permutation_p']}\n"
            f"rand sig={r['n_random_significant']}/{r['n_random']}",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BBBBBB", alpha=0.9),
        )
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.2)
    fig.suptitle("Figure 2: Cross-Domain Steering Validation", fontsize=14, fontweight="bold")
    fig.supxlabel("Steering strength (alpha)")
    fig.supylabel("Bankruptcy rate")
    out = FIG_DIR / "v14_fig2_crossdomain_dose_response.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"figures/{out.name}"


def generate_summary_figure(results: dict[str, dict[str, Any]]) -> str:
    ordered = ["exp1", "exp2a", "exp2b", "exp4", "exp5", "exp6", "exp3_mw2ic", "exp3_mw2sm", "exp3_ic2sm"]
    labels = [results[k]["_label"] for k in ordered]
    values = [abs(results[k]["bk_direction"]["rho"] or 0.0) for k in ordered]
    colors = [VERDICT_COLORS.get(results[k]["verdict"], "#666666") for k in ordered]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    y = np.arange(len(ordered))
    ax.barh(y, values, color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("|Spearman rho| for BK direction")
    ax.set_title("Figure 3: Summary of V14 Steering Evidence", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    for yi, key in zip(y, ordered):
        r = results[key]
        ax.text(
            values[yi] + 0.01,
            yi,
            f"{r['verdict']}, perm_p={r['permutation_p']}",
            va="center",
            fontsize=8,
        )
    out = FIG_DIR / "v14_fig3_verdict_summary.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"figures/{out.name}"


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_markdown(results: dict[str, dict[str, Any]], assessment: dict[str, Any], figs: dict[str, str]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    specific_count = len(assessment["confirmed"])
    borderline_count = len(assessment["borderline"])
    lines: list[str] = [
        "# V14: Rigorous Causal Validation of Bankruptcy Steering in LLMs",
        "",
        "**Authors**: Seungpil Lee, Donghyeon Shin, Yunjeong Lee, Sundong Kim (GIST)",
        f"**Date**: {now}",
        "**Scope**: Follow-up validation of V12/V13 steering claims with expanded random controls",
        "",
        "---",
        "",
        "## 1. Introduction",
        "",
        "V14 was designed to resolve the main interpretive weakness of the previous steering analysis: a monotonic BK dose-response is not sufficient to claim direction specificity if random directions also produce significant monotonic effects. The new experiments therefore retain the same activation-level steering protocol while substantially increasing the number of random controls and evaluating permutation-style specificity directly.",
        "",
        f"The completed V14 run yields {specific_count} direction-specific confirmations and {borderline_count} BK-significant but non-specific results across seven experiments. The overall recommendation is **{assessment['recommendation']}**, meaning that the causal claim should be framed as: {assessment['recommendation_text']}",
        "",
        "## 2. Experimental Setup",
        "",
        "All experiments apply a BK direction vector at layer 22 of the residual stream and sweep alpha across {-2, -1, -0.5, 0, +0.5, +1, +2}. Each experiment compares the BK direction against matched-norm random control directions and evaluates both the BK-direction Spearman correlation and a random-control permutation score.",
        "",
        "| Experiment | Source | Target | n_games | n_random | Verdict |",
        "|-----------|--------|--------|--------:|---------:|---------|",
    ]

    ordered = ["exp1", "exp2a", "exp2b", "exp4", "exp3_mw2ic", "exp3_mw2sm", "exp3_ic2sm"]
    for key in ordered:
        r = results[key]
        lines.append(
            f"| {r['_label']} | {r['_source']} | {r['_target']} | {r['n_games']} | {r['n_random']} | {r['verdict']} |"
        )

    lines.extend([
        "",
        "## 3. Within-Domain Validation",
        "",
        "The within-domain experiments test whether each task-specific BK direction remains distinguishable from a broad random-direction null. This is the strongest setting for a causal claim because the steering direction and the evaluation task match.",
        "",
        "| Experiment | BK rho | BK p | Permutation p | Random significant | Baseline BK | Verdict |",
        "|-----------|-------:|-----:|--------------:|-------------------:|------------:|---------|",
    ])
    for key in ["exp1", "exp2a", "exp2b", "exp4"]:
        r = results[key]
        lines.append(
            f"| {r['_label']} | {fmt_float(r['bk_direction']['rho'])} | {fmt_float(r['bk_direction']['p'], 6)} | {fmt_float(r['permutation_p'])} | {r['n_random_significant']}/{r['n_random']} | {fmt_float(r['baseline_bk'])} | {r['verdict']} |"
        )

    exp1 = results["exp1"]
    lines.extend([
        "",
        f"LLaMA SM remains the anchor case. Its BK direction produced rho = {fmt_float(exp1['bk_direction']['rho'])} with permutation p = {fmt_float(exp1['permutation_p'])}, making it the primary benchmark for whether V12's causal claim survives the stricter random-control regime.",
        "",
        "Other within-domain results should be interpreted strictly by verdict. `BK_SPECIFIC_CONFIRMED` supports a direction-specific causal statement. `BK_SIGNIFICANT_NOT_SPECIFIC` supports only an observed BK-sensitive steering effect, not a claim that the BK direction is unique. `NOT_SIGNIFICANT` means the dose-response itself does not survive significance filtering.",
        "",
        f"![Fig. 1: Within-domain V14 dose-response curves. Blue lines show the BK direction, gray lines show matched random controls, and the verdict text indicates whether the BK curve is specific rather than merely monotonic.]({figs['within']})",
        "",
        "## 4. Cross-Domain Validation",
        "",
        "The cross-domain experiments test whether a direction extracted from one task changes behavior in another task beyond what random directions can achieve. This is the strongest test of domain generalization, but it is also the most demanding because both task mismatch and baseline geometry can attenuate the effect.",
        "",
        "| Transfer | BK rho | BK p | Permutation p | Random significant | Verdict |",
        "|---------|-------:|-----:|--------------:|-------------------:|---------|",
    ])
    for key in ["exp3_mw2ic", "exp3_mw2sm", "exp3_ic2sm"]:
        r = results[key]
        lines.append(
            f"| {r['_label']} | {fmt_float(r['bk_direction']['rho'])} | {fmt_float(r['bk_direction']['p'], 6)} | {fmt_float(r['permutation_p'])} | {r['n_random_significant']}/{r['n_random']} | {r['verdict']} |"
        )
    lines.extend([
        "",
        "A cross-domain `BK_SPECIFIC_CONFIRMED` result would justify a stronger claim that at least part of the BK representation is causally shared across tasks. A cross-domain `BK_SIGNIFICANT_NOT_SPECIFIC` result is still informative, but only as exploratory evidence because the same monotonic pattern can be reproduced by generic directions.",
        "",
        f"![Fig. 2: Cross-domain V14 dose-response curves. Each panel compares the task-transferred BK direction against matched random controls under the same alpha sweep.]({figs['cross']})",
        "",
        "## 5. Overall Assessment",
        "",
        "| Experiment | |rho| | Permutation p | Random significant | Verdict |",
        "|-----------|-----:|--------------:|-------------------:|---------|",
    ])
    for key in ordered:
        r = results[key]
        rho = abs(r["bk_direction"]["rho"] or 0.0)
        lines.append(
            f"| {r['_label']} | {fmt_float(rho)} | {fmt_float(r['permutation_p'])} | {r['n_random_significant']}/{r['n_random']} | {r['verdict']} |"
        )
    lines.extend([
        "",
        f"V14's final recommendation is **{assessment['recommendation']}**. {assessment['recommendation_text']}",
        "",
        "The practical writing implication is straightforward. If the recommendation is positive, the paper can keep a causal steering subsection but must explicitly limit strong claims to the confirmed cases. If the recommendation is borderline, the paper should retain only the strongest case as causal and demote the rest to exploratory evidence. If the recommendation is negative, the steering section should be reframed as an interesting but non-specific perturbation result.",
        "",
        f"![Fig. 3: Summary of V14 evidence strength. Bars show the magnitude of the BK-direction dose-response, while color encodes whether that response remains specific against random controls.]({figs['summary']})",
        "",
        "## 6. Conclusion",
        "",
        "V14 answers a narrower and more defensible question than V12: not simply whether BK steering can move behavior, but whether the BK direction is more informative than random directions of the same norm. That distinction is the key boundary between a genuine causal interpretation and an overclaim. This report is therefore intended as the decision document for updating the paper's causal language.",
        "",
        "## Appendix: Source Files",
        "",
        "| Experiment | JSON source |",
        "|-----------|-------------|",
    ])
    for key in ordered:
        lines.append(f"| {results[key]['_label']} | `{results[key]['_path']}` |")
    return "\n".join(lines) + "\n"


def main() -> None:
    results = load_latest_results()
    assessment = assess_outcome(results)
    figs = {
        "within": generate_within_domain_figure(results),
        "cross": generate_crossdomain_figure(results),
        "summary": generate_summary_figure(results),
    }
    md = build_markdown(results, assessment, figs)
    MD_PATH.write_text(md)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assessment": assessment,
        "results": {
            key: {
                "label": value["_label"],
                "path": value["_path"],
                "verdict": value["verdict"],
                "bk_rho": value["bk_direction"]["rho"],
                "bk_p": value["bk_direction"]["p"],
                "permutation_p": value["permutation_p"],
                "n_random_significant": value["n_random_significant"],
                "n_random": value["n_random"],
            }
            for key, value in results.items()
        },
        "report_md": str(MD_PATH),
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2))
    ready_lines = [
        "# V14 Ready For Approval",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Recommendation: `{assessment['recommendation']}`",
        f"- Summary: {assessment['recommendation_text']}",
        f"- Report markdown: `{MD_PATH}`",
    ]
    if assessment["recommendation"] == "positive":
        ready_lines.append("- Next action: review the report, then update the Korean paper's causal steering section with narrowed claims.")
    elif assessment["recommendation"] == "borderline":
        ready_lines.append("- Next action: review the report, then revise the paper conservatively and keep only the strongest causal case.")
    else:
        ready_lines.append("- Next action: do not strengthen the paper's causal claims; rewrite steering as exploratory evidence.")
    ready_lines.extend([
        "",
        "## Key Verdicts",
        "",
    ])
    for key, value in summary["results"].items():
        ready_lines.append(
            f"- `{value['label']}`: `{value['verdict']}` "
            f"(rho={value['bk_rho']}, perm_p={value['permutation_p']}, "
            f"rand_sig={value['n_random_significant']}/{value['n_random']})"
        )
    READY_MD_PATH.write_text("\n".join(ready_lines) + "\n")

    next_prompt = (
        "V14 monitoring is complete. Read "
        f"{READY_MD_PATH} and {MD_PATH}, explain the result quality, and if the "
        "recommendation is positive then update "
        "/home/v-seungplee/LLM_Addiction_NMT_KOR/content/3.results.tex, "
        "/home/v-seungplee/LLM_Addiction_NMT_KOR/content/1.introduction.tex, and "
        "/home/v-seungplee/LLM_Addiction_NMT_KOR/content/4.discussion.tex using "
        "iterative-academic-writing and academic-latex-pipeline principles. Ask "
        "for approval before editing the paper."
    )
    NEXT_PROMPT_PATH.write_text(next_prompt + "\n")
    print(f"Wrote {MD_PATH}")
    print(f"Wrote {SUMMARY_JSON_PATH}")
    print(f"Wrote {READY_MD_PATH}")
    print(f"Wrote {NEXT_PROMPT_PATH}")
    print(f"Recommendation: {assessment['recommendation']}")


if __name__ == "__main__":
    main()
