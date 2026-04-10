#!/usr/bin/env python3
"""
Build a paper-facing neural metrics manifest with explicit provenance.

The direct-rerun values below were audited on 2026-04-09 against the current
canonical evaluation helpers:
  - run_comprehensive_robustness.py
  - run_perm_null_ilc.py
  - run_probe_selectivity_controls.py

They are stored here as a locked manifest because the full re-evaluation pass is
expensive and the paper needs one stable source-of-truth file for text and
figure generation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis")
RESULTS = ROOT / "results"
OUT = RESULTS / "paper_neural_audit.json"
V17_PATH = RESULTS / "v17_nonlinear_deconfound.txt"
STEERING_MAIN = RESULTS / "json" / "v14_exp1_llama_sm_perm20_20260331_153127.json"
STEERING_CROSS = RESULTS / "json" / "v12_crossdomain_steering.json"
DIRECT_RERUN = {
    "rq1_direct": {
        "gemma_sm_i_ba": {
            "model": "gemma", "paradigm": "sm", "layer": 24, "metric": "i_ba",
            "n": 12246, "target_mean": 0.2638894711348347, "r2": 0.16123525659057136, "source": "direct_rerun_2026_04_09",
        },
        "llama_sm_i_ba": {
            "model": "llama", "paradigm": "sm", "layer": 16, "metric": "i_ba",
            "n": 45551, "target_mean": 0.23801507541219966, "r2": 0.12110926944222482, "source": "direct_rerun_2026_04_09",
        },
        "gemma_mw_i_ba": {
            "model": "gemma", "paradigm": "mw", "layer": 24, "metric": "i_ba",
            "n": 8948, "target_mean": 0.12890749498643178, "r2": 0.05628875990042441, "source": "direct_rerun_2026_04_09",
        },
        "llama_mw_i_ba": {
            "model": "llama", "paradigm": "mw", "layer": 16, "metric": "i_ba",
            "n": 57220, "target_mean": 0.20903361078476404, "r2": 0.06812122492433696, "source": "direct_rerun_2026_04_09",
        },
        "gemma_sm_i_ec": {
            "model": "gemma", "paradigm": "sm", "layer": 24, "metric": "i_ec",
            "n": 12246, "target_mean": 0.08639555773313735, "r2": 0.05344758255586415, "source": "direct_rerun_2026_04_09",
        },
        "llama_sm_i_ec": {
            "model": "llama", "paradigm": "sm", "layer": 16, "metric": "i_ec",
            "n": 45551, "target_mean": 0.14258742947465478, "r2": 0.04176792168039154, "source": "direct_rerun_2026_04_09",
        },
    },
    "rq3_condition_i_ba": {
        "gemma_sm_i_ba": {
            "model": "gemma",
            "paradigm": "sm",
            "layer": 24,
            "metric": "i_ba",
            "source": "direct_rerun_2026_04_09",
            "subsets": {
                "all_variable": {"n": 12246, "mean": 0.2638894711348347, "r2": 0.1612352565905713},
                "plus_G": {"n": 8040, "mean": 0.28969401582074034, "r2": 0.15185268777377667},
                "minus_G": {"n": 4206, "mean": 0.21456266674237595, "r2": 0.0779104222087033},
                "plus_M": {"n": 6556, "mean": 0.28737532683926964, "r2": 0.1487925364798381},
                "minus_M": {"n": 5690, "mean": 0.23682914248838907, "r2": 0.12704159908616527},
                "fixed_all": {"n": 6062, "mean": 0.10200013351821144, "r2": -0.016410718165550155},
            },
        },
        "llama_sm_i_ba": {
            "model": "llama",
            "paradigm": "sm",
            "layer": 16,
            "metric": "i_ba",
            "source": "direct_rerun_2026_04_09",
            "subsets": {
                "all_variable": {"n": 45551, "mean": 0.23801507541219966, "r2": 0.12110926944222485},
                "plus_G": {"n": 22288, "mean": 0.2591108691147531, "r2": 0.12161671259280993},
                "minus_G": {"n": 23263, "mean": 0.2178034496527313, "r2": 0.0874469612147235},
                "plus_M": {"n": 19753, "mean": 0.2693028807790319, "r2": 0.09866285891295504},
                "minus_M": {"n": 25798, "mean": 0.21405864392871102, "r2": 0.07349497080022119},
                "fixed_all": {"n": 14418, "mean": 0.10409443352435441, "r2": 0.0016447648244561908},
            },
        },
    },
}


def _parse_v17_ilc() -> dict:
    text = V17_PATH.read_text()
    pattern = re.compile(
        r"--- (?P<model>GEMMA|LLAMA)/(?P<task>SM|IC|MW) L(?P<layer>\d+) \(n=\s*(?P<n>\d+)\) ---\n"
        r"(?:.*\n)*?\s+I_LC: R²=(?P<r2>-?\d+\.\d+), .*? p=(?P<p>\d+\.\d+)",
        re.MULTILINE,
    )
    out = {}
    for m in pattern.finditer(text):
        model = m.group("model").lower()
        task = m.group("task").lower()
        out[f"{model}_{task}"] = {
            "model": model,
            "paradigm": task,
            "layer": int(m.group("layer")),
            "metric": "i_lc",
            "n": int(m.group("n")),
            "r2": float(m.group("r2")),
            "p": float(m.group("p")),
            "source": "archived_v17",
        }
    return out


def _load_steering() -> dict:
    same = json.loads(STEERING_MAIN.read_text())
    cross = json.loads(STEERING_CROSS.read_text())
    significant_cross = []
    for row in cross["cross_domain_results"]:
        if row["p_value"] < 0.05:
            significant_cross.append(
                {
                    "combo": row["combo"],
                    "rho": row["rho"],
                    "p_value": row["p_value"],
                }
            )
    return {
        "same_domain": {
            "model": same["model"],
            "task": same["task"],
            "layer": same["layer"],
            "rho": same["bk_direction"]["rho"],
            "p": same["bk_direction"]["p"],
            "permutation_p": same["permutation_p"],
            "baseline_bk": same["baseline_bk"],
            "source": "v14_json",
        },
        "cross_domain_significant": significant_cross,
    }


def main() -> None:
    ilc_archived = _parse_v17_ilc()
    audit = {
        "rq1_ilc": ilc_archived,
        "rq1_direct": DIRECT_RERUN["rq1_direct"],
        "rq3_condition_i_ba": DIRECT_RERUN["rq3_condition_i_ba"],
        "steering": _load_steering(),
    }
    OUT.write_text(json.dumps(audit, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
