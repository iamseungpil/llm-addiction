"""
Extract ALL numerical results from Gemma SAE/activation analysis JSON files.
Organize by 3 Research Questions:
  RQ1: BK prediction (AUCs, correlation, feature importance)
  RQ2: Domain-invariant representations (cross-domain, feature overlap)
  RQ3: Condition differences (condition encoding, per-condition AUCs)
"""

import json
import os
import sys
from pathlib import Path

JSON_DIR = Path("/home/jovyan/llm-addiction/sae_v3_analysis/results/json")

def load(name):
    with open(JSON_DIR / name) as f:
        return json.load(f)


def extract_best_layer(layer_results, metric="auc"):
    """From a list of per-layer dicts, find best by metric."""
    if not layer_results:
        return None
    best = max(layer_results, key=lambda x: x.get(metric, 0))
    return best


def extract_top_n_layers(layer_results, metric="auc", n=5):
    """Top N layers by metric."""
    sorted_layers = sorted(layer_results, key=lambda x: x.get(metric, 0), reverse=True)
    return [{"layer": r["layer"], metric: round(r[metric], 4)} for r in sorted_layers[:n]]


def safe_round(v, d=4):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return round(v, d)
    return v


###############################################################################
# FILE 1: all_analyses_20260306_091055.json
###############################################################################
def extract_file1():
    data = load("all_analyses_20260306_091055.json")
    results = {}

    # Goal A: SAE BK Decision Point classification
    goal_a = data.get("goal_a_classification", {})
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm not in goal_a:
            continue
        sae_layers = goal_a[paradigm].get("sae", [])
        best = extract_best_layer(sae_layers)
        all_layers = []
        for r in sae_layers:
            all_layers.append({
                "layer": r["layer"],
                "auc": round(r["auc"], 4),
                "auc_std": round(r["auc_std"], 4),
                "f1": round(r["f1"], 4),
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "n_pos": r["n_pos"],
                "n_neg": r["n_neg"],
                "n_features": r.get("n_features"),
            })
        results[f"goal_a_sae_bk_dp_{paradigm}"] = {
            "best_layer": best["layer"] if best else None,
            "best_auc": round(best["auc"], 4) if best else None,
            "best_auc_std": round(best["auc_std"], 4) if best else None,
            "best_f1": round(best["f1"], 4) if best else None,
            "n_pos": best["n_pos"] if best else None,
            "n_neg": best["n_neg"] if best else None,
            "top5_layers": extract_top_n_layers(sae_layers),
            "all_layers": all_layers,
        }

    # Goal B: SAE R1 classification
    goal_b = data.get("goal_b_r1", {})
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm not in goal_b:
            continue
        sae_layers = goal_b[paradigm].get("sae", [])
        best = extract_best_layer(sae_layers)
        all_layers = []
        for r in sae_layers:
            all_layers.append({
                "layer": r["layer"],
                "auc": round(r["auc"], 4),
                "auc_std": round(r.get("auc_std", 0), 4),
                "n_pos": r.get("n_pos"),
                "n_neg": r.get("n_neg"),
                "n_features": r.get("n_features"),
            })
        results[f"goal_b_sae_r1_{paradigm}"] = {
            "best_layer": best["layer"] if best else None,
            "best_auc": round(best["auc"], 4) if best else None,
            "best_auc_std": round(best.get("auc_std", 0), 4) if best else None,
            "n_pos": best.get("n_pos") if best else None,
            "n_neg": best.get("n_neg") if best else None,
            "top5_layers": extract_top_n_layers(sae_layers),
            "all_layers": all_layers,
        }

    # Goal C: Cross-domain SAE transfer
    goal_c = data.get("goal_c_cross_domain", {})
    cross_results = []
    if isinstance(goal_c, dict):
        for direction, vals in goal_c.items():
            if isinstance(vals, dict):
                cross_results.append({
                    "direction": direction,
                    "auc": safe_round(vals.get("auc")),
                    "auc_std": safe_round(vals.get("auc_std")),
                    "n_features": vals.get("n_features"),
                    "train_paradigm": vals.get("train_paradigm"),
                    "test_paradigm": vals.get("test_paradigm"),
                })
            elif isinstance(vals, list):
                for v in vals:
                    cross_results.append({
                        "direction": direction,
                        "auc": safe_round(v.get("auc")),
                        "layer": v.get("layer"),
                    })
    results["goal_c_cross_domain"] = cross_results

    return results


###############################################################################
# FILE 2: extended_analyses_20260306_211214.json
###############################################################################
def extract_file2():
    data = load("extended_analyses_20260306_211214.json")
    results = {}

    # Exp 2a: Balance-matched SAE BK
    exp2a = data.get("exp2a_balance_matched", {})
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm not in exp2a:
            continue
        layers = exp2a[paradigm]
        if isinstance(layers, list):
            best = extract_best_layer(layers)
            all_layers = []
            for r in layers:
                all_layers.append({
                    "layer": r.get("layer"),
                    "auc": safe_round(r.get("auc")),
                    "auc_std": safe_round(r.get("auc_std")),
                    "n_pos": r.get("n_pos"),
                    "n_neg": r.get("n_neg"),
                    "n_features": r.get("n_features"),
                })
            results[f"exp2a_balance_matched_{paradigm}"] = {
                "best_layer": best["layer"] if best else None,
                "best_auc": safe_round(best.get("auc")) if best else None,
                "best_auc_std": safe_round(best.get("auc_std")) if best else None,
                "n_pos": best.get("n_pos") if best else None,
                "n_neg": best.get("n_neg") if best else None,
                "top5_layers": extract_top_n_layers(layers),
                "all_layers": all_layers,
            }
        elif isinstance(layers, dict):
            results[f"exp2a_balance_matched_{paradigm}"] = layers

    # Exp 3: Risk preference classification
    exp3 = data.get("exp3_risk", {})
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm not in exp3:
            continue
        layers = exp3[paradigm]
        if isinstance(layers, list):
            best = extract_best_layer(layers)
            all_layers = []
            for r in layers:
                all_layers.append({
                    "layer": r.get("layer"),
                    "auc": safe_round(r.get("auc")),
                    "auc_std": safe_round(r.get("auc_std")),
                    "n_pos": r.get("n_pos"),
                    "n_neg": r.get("n_neg"),
                    "n_features": r.get("n_features"),
                })
            results[f"exp3_risk_{paradigm}"] = {
                "best_layer": best["layer"] if best else None,
                "best_auc": safe_round(best.get("auc")) if best else None,
                "top5_layers": extract_top_n_layers(layers),
                "all_layers": all_layers,
            }

    # Exp 4: Condition encoding
    exp4 = data.get("exp4_condition_encoding", {})
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm not in exp4:
            continue
        cond_data = exp4[paradigm]
        if isinstance(cond_data, list):
            results[f"exp4_condition_{paradigm}"] = []
            for item in cond_data:
                results[f"exp4_condition_{paradigm}"].append({
                    "layer": item.get("layer"),
                    "accuracy": safe_round(item.get("accuracy")),
                    "accuracy_std": safe_round(item.get("accuracy_std")),
                    "n_classes": item.get("n_classes"),
                    "chance": safe_round(item.get("chance")),
                    "auc": safe_round(item.get("auc")),
                    "auc_std": safe_round(item.get("auc_std")),
                })
        elif isinstance(cond_data, dict):
            for sub_key, sub_val in cond_data.items():
                if isinstance(sub_val, list):
                    best = max(sub_val, key=lambda x: x.get("auc", x.get("accuracy", 0)))
                    results[f"exp4_condition_{paradigm}_{sub_key}"] = {
                        "best_layer": best.get("layer"),
                        "best_auc": safe_round(best.get("auc")),
                        "best_accuracy": safe_round(best.get("accuracy")),
                        "n_classes": best.get("n_classes"),
                        "all_layers": [{
                            "layer": r.get("layer"),
                            "auc": safe_round(r.get("auc")),
                            "accuracy": safe_round(r.get("accuracy")),
                        } for r in sub_val],
                    }

    return results


###############################################################################
# FILE 3: improved_v4_20260308_032435.json
###############################################################################
def extract_file3():
    data = load("improved_v4_20260308_032435.json")
    results = {}

    # Cross-domain bootstrap
    cdb = data.get("cross_domain_bootstrap", {})
    for direction, vals in cdb.items():
        if isinstance(vals, dict):
            results[f"cross_domain_bootstrap_{direction}"] = {
                "mean_auc": safe_round(vals.get("mean_auc")),
                "ci_lower": safe_round(vals.get("ci_lower")),
                "ci_upper": safe_round(vals.get("ci_upper")),
                "std_auc": safe_round(vals.get("std_auc")),
                "n_bootstrap": vals.get("n_bootstrap"),
                "layer": vals.get("layer"),
                "train_paradigm": vals.get("train_paradigm"),
                "test_paradigm": vals.get("test_paradigm"),
            }
        elif isinstance(vals, list):
            for item in vals:
                key = f"cross_domain_bootstrap_{direction}_L{item.get('layer', '?')}"
                results[key] = {
                    "mean_auc": safe_round(item.get("mean_auc")),
                    "ci_lower": safe_round(item.get("ci_lower")),
                    "ci_upper": safe_round(item.get("ci_upper")),
                    "std_auc": safe_round(item.get("std_auc")),
                    "n_bootstrap": item.get("n_bootstrap"),
                    "layer": item.get("layer"),
                }

    # Same-layer features
    slf = data.get("same_layer_features", {})
    for layer_key, vals in slf.items():
        if isinstance(vals, dict):
            results[f"same_layer_features_{layer_key}"] = {
                "ic_top100": vals.get("ic_top100"),
                "sm_top100": vals.get("sm_top100"),
                "mw_top100": vals.get("mw_top100"),
                "ic_sm_overlap": vals.get("ic_sm_overlap"),
                "ic_mw_overlap": vals.get("ic_mw_overlap"),
                "sm_mw_overlap": vals.get("sm_mw_overlap"),
                "ic_sm_jaccard": safe_round(vals.get("ic_sm_jaccard")),
                "ic_mw_jaccard": safe_round(vals.get("ic_mw_jaccard")),
                "sm_mw_jaccard": safe_round(vals.get("sm_mw_jaccard")),
                "shared_features_ic_sm": vals.get("shared_features_ic_sm"),
                "shared_features_ic_mw": vals.get("shared_features_ic_mw"),
                "shared_features_sm_mw": vals.get("shared_features_sm_mw"),
            }

    # R1 permutation test
    r1p = data.get("r1_permutation_test", {})
    for paradigm, vals in r1p.items():
        if isinstance(vals, dict):
            results[f"r1_permutation_{paradigm}"] = {
                "observed_auc": safe_round(vals.get("observed_auc")),
                "null_mean": safe_round(vals.get("null_mean")),
                "null_std": safe_round(vals.get("null_std")),
                "p_value": vals.get("p_value"),
                "n_permutations": vals.get("n_permutations"),
                "layer": vals.get("layer"),
            }
        elif isinstance(vals, list):
            for item in vals:
                results[f"r1_permutation_{paradigm}_L{item.get('layer', '?')}"] = {
                    "observed_auc": safe_round(item.get("observed_auc")),
                    "null_mean": safe_round(item.get("null_mean")),
                    "null_std": safe_round(item.get("null_std")),
                    "p_value": item.get("p_value"),
                    "n_permutations": item.get("n_permutations"),
                    "layer": item.get("layer"),
                }

    # Gemma balance-controlled R1
    gbc = data.get("gemma_balance_controlled", {})
    for paradigm, vals in gbc.items():
        if isinstance(vals, dict):
            results[f"gemma_balance_controlled_{paradigm}"] = {
                "auc": safe_round(vals.get("auc")),
                "auc_std": safe_round(vals.get("auc_std")),
                "layer": vals.get("layer"),
                "n_pos": vals.get("n_pos"),
                "n_neg": vals.get("n_neg"),
            }
        elif isinstance(vals, list):
            best = extract_best_layer(vals)
            all_l = [{"layer": r.get("layer"), "auc": safe_round(r.get("auc"))} for r in vals]
            results[f"gemma_balance_controlled_{paradigm}"] = {
                "best_layer": best.get("layer") if best else None,
                "best_auc": safe_round(best.get("auc")) if best else None,
                "all_layers": all_l,
            }

    return results


###############################################################################
# FILE 4: comprehensive_gemma_20260309_063511.json
###############################################################################
def extract_file4():
    data = load("comprehensive_gemma_20260309_063511.json")
    results = {}

    # Hidden BK classification
    hidden_bk = data.get("hidden_bk", {})
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm not in hidden_bk:
            continue
        layers = hidden_bk[paradigm]
        if isinstance(layers, list):
            best = extract_best_layer(layers)
            all_layers = [{"layer": r.get("layer"), "auc": safe_round(r.get("auc")),
                           "auc_std": safe_round(r.get("auc_std"))} for r in layers]
            results[f"hidden_bk_dp_{paradigm}"] = {
                "best_layer": best["layer"] if best else None,
                "best_auc": safe_round(best.get("auc")) if best else None,
                "best_auc_std": safe_round(best.get("auc_std")) if best else None,
                "top5_layers": extract_top_n_layers(layers),
                "all_layers": all_layers,
            }

    # Hidden risk classification
    hidden_risk = data.get("hidden_risk", {})
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm not in hidden_risk:
            continue
        layers = hidden_risk[paradigm]
        if isinstance(layers, list):
            best = extract_best_layer(layers)
            all_layers = [{"layer": r.get("layer"), "auc": safe_round(r.get("auc")),
                           "auc_std": safe_round(r.get("auc_std"))} for r in layers]
            results[f"hidden_risk_{paradigm}"] = {
                "best_layer": best["layer"] if best else None,
                "best_auc": safe_round(best.get("auc")) if best else None,
                "top5_layers": extract_top_n_layers(layers),
                "all_layers": all_layers,
            }

    # Per-condition SAE BK
    pc_sae_bk = data.get("percondition_sae_bk", {})
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm not in pc_sae_bk:
            continue
        cond_data = pc_sae_bk[paradigm]
        if isinstance(cond_data, dict):
            for cond_key, layers in cond_data.items():
                if isinstance(layers, list):
                    best = extract_best_layer(layers)
                    results[f"percondition_sae_bk_{paradigm}_{cond_key}"] = {
                        "best_layer": best["layer"] if best else None,
                        "best_auc": safe_round(best.get("auc")) if best else None,
                        "n_pos": best.get("n_pos") if best else None,
                        "n_neg": best.get("n_neg") if best else None,
                    }

    # Per-condition SAE risk
    pc_sae_risk = data.get("percondition_sae_risk", {})
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm not in pc_sae_risk:
            continue
        cond_data = pc_sae_risk[paradigm]
        if isinstance(cond_data, dict):
            for cond_key, layers in cond_data.items():
                if isinstance(layers, list):
                    best = extract_best_layer(layers)
                    results[f"percondition_sae_risk_{paradigm}_{cond_key}"] = {
                        "best_layer": best["layer"] if best else None,
                        "best_auc": safe_round(best.get("auc")) if best else None,
                        "n_pos": best.get("n_pos") if best else None,
                        "n_neg": best.get("n_neg") if best else None,
                    }

    # Per-condition hidden BK
    pc_hidden_bk = data.get("percondition_hidden_bk", {})
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm not in pc_hidden_bk:
            continue
        cond_data = pc_hidden_bk[paradigm]
        if isinstance(cond_data, dict):
            for cond_key, layers in cond_data.items():
                if isinstance(layers, list):
                    best = extract_best_layer(layers)
                    results[f"percondition_hidden_bk_{paradigm}_{cond_key}"] = {
                        "best_layer": best["layer"] if best else None,
                        "best_auc": safe_round(best.get("auc")) if best else None,
                        "n_pos": best.get("n_pos") if best else None,
                        "n_neg": best.get("n_neg") if best else None,
                    }

    return results


###############################################################################
# FILE 5: comprehensive_gemma_20260309_095339.json (percondition_hidden_bk detailed)
###############################################################################
def extract_file5():
    data = load("comprehensive_gemma_20260309_095339.json")
    results = {}

    # This file has detailed per-condition hidden BK with all layers
    pc_hidden_bk = data.get("percondition_hidden_bk", {})
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm not in pc_hidden_bk:
            continue
        cond_data = pc_hidden_bk[paradigm]
        if isinstance(cond_data, dict):
            for cond_key, layers in cond_data.items():
                if isinstance(layers, list) and len(layers) > 0:
                    best = extract_best_layer(layers)
                    all_layers = [{"layer": r.get("layer"), "auc": safe_round(r.get("auc")),
                                   "n_pos": r.get("n_pos"), "n_neg": r.get("n_neg")} for r in layers]
                    results[f"percondition_hidden_bk_detailed_{paradigm}_{cond_key}"] = {
                        "best_layer": best["layer"] if best else None,
                        "best_auc": safe_round(best.get("auc")) if best else None,
                        "n_pos": best.get("n_pos") if best else None,
                        "n_neg": best.get("n_neg") if best else None,
                        "top5_layers": extract_top_n_layers(layers),
                        "all_layers": all_layers,
                    }

    # Check for other keys
    for key in data:
        if key != "percondition_hidden_bk":
            val = data[key]
            if isinstance(val, dict):
                results[f"file5_extra_{key}"] = {k: safe_round(v) if isinstance(v, (int, float)) else str(type(v)) for k, v in list(val.items())[:10]}

    return results


###############################################################################
# FILE 6: comprehensive_gemma_20260309_182926.json
###############################################################################
def extract_file6():
    data = load("comprehensive_gemma_20260309_182926.json")
    results = {}

    # Hidden risk corrected
    hidden_risk = data.get("hidden_risk", {})
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm not in hidden_risk:
            continue
        layers = hidden_risk[paradigm]
        if isinstance(layers, list):
            best = extract_best_layer(layers)
            all_layers = [{"layer": r.get("layer"), "auc": safe_round(r.get("auc")),
                           "auc_std": safe_round(r.get("auc_std"))} for r in layers]
            results[f"hidden_risk_corrected_{paradigm}"] = {
                "best_layer": best["layer"] if best else None,
                "best_auc": safe_round(best.get("auc")) if best else None,
                "top5_layers": extract_top_n_layers(layers),
                "all_layers": all_layers,
            }

    # Per-condition SAE risk
    pc_sae_risk = data.get("percondition_sae_risk", {})
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm not in pc_sae_risk:
            continue
        cond_data = pc_sae_risk[paradigm]
        if isinstance(cond_data, dict):
            for cond_key, layers in cond_data.items():
                if isinstance(layers, list):
                    best = extract_best_layer(layers)
                    results[f"percondition_sae_risk_corrected_{paradigm}_{cond_key}"] = {
                        "best_layer": best["layer"] if best else None,
                        "best_auc": safe_round(best.get("auc")) if best else None,
                        "n_pos": best.get("n_pos") if best else None,
                        "n_neg": best.get("n_neg") if best else None,
                    }

    # Check for other keys
    for key in data:
        if key not in ("hidden_risk", "percondition_sae_risk"):
            results[f"file6_extra_{key}"] = str(type(data[key]))

    return results


###############################################################################
# FILE 7: hidden_gaps_20260309_181059.json
###############################################################################
def extract_file7():
    data = load("hidden_gaps_20260309_181059.json")
    results = {}

    # Balance-matched hidden
    bm_hidden = data.get("balance_matched_hidden", {})
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm not in bm_hidden:
            continue
        layers = bm_hidden[paradigm]
        if isinstance(layers, list):
            best = extract_best_layer(layers)
            all_layers = [{"layer": r.get("layer"), "auc": safe_round(r.get("auc")),
                           "n_pos": r.get("n_pos"), "n_neg": r.get("n_neg")} for r in layers]
            results[f"balance_matched_hidden_{paradigm}"] = {
                "best_layer": best["layer"] if best else None,
                "best_auc": safe_round(best.get("auc")) if best else None,
                "n_pos": best.get("n_pos") if best else None,
                "n_neg": best.get("n_neg") if best else None,
                "top5_layers": extract_top_n_layers(layers),
                "all_layers": all_layers,
            }

    # Per-condition hidden BK R1
    pc_hbk_r1 = data.get("percondition_hidden_bk_r1", {})
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm not in pc_hbk_r1:
            continue
        cond_data = pc_hbk_r1[paradigm]
        if isinstance(cond_data, dict):
            for cond_key, layers in cond_data.items():
                if isinstance(layers, list):
                    best = extract_best_layer(layers)
                    results[f"percondition_hidden_bk_r1_{paradigm}_{cond_key}"] = {
                        "best_layer": best["layer"] if best else None,
                        "best_auc": safe_round(best.get("auc")) if best else None,
                        "n_pos": best.get("n_pos") if best else None,
                        "n_neg": best.get("n_neg") if best else None,
                    }

    # Per-condition hidden risk
    pc_h_risk = data.get("percondition_hidden_risk", {})
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm not in pc_h_risk:
            continue
        cond_data = pc_h_risk[paradigm]
        if isinstance(cond_data, dict):
            for cond_key, layers in cond_data.items():
                if isinstance(layers, list):
                    best = extract_best_layer(layers)
                    results[f"percondition_hidden_risk_{paradigm}_{cond_key}"] = {
                        "best_layer": best["layer"] if best else None,
                        "best_auc": safe_round(best.get("auc")) if best else None,
                        "n_pos": best.get("n_pos") if best else None,
                        "n_neg": best.get("n_neg") if best else None,
                    }

    # Hidden cross-domain
    hcd = data.get("hidden_cross_domain", {})
    for direction, vals in hcd.items():
        if isinstance(vals, dict):
            results[f"hidden_cross_domain_{direction}"] = {
                "mean_auc": safe_round(vals.get("mean_auc")),
                "ci_lower": safe_round(vals.get("ci_lower")),
                "ci_upper": safe_round(vals.get("ci_upper")),
                "std_auc": safe_round(vals.get("std_auc")),
                "layer": vals.get("layer"),
            }
        elif isinstance(vals, list):
            for item in vals:
                results[f"hidden_cross_domain_{direction}_L{item.get('layer', '?')}"] = {
                    "mean_auc": safe_round(item.get("mean_auc")),
                    "ci_lower": safe_round(item.get("ci_lower")),
                    "ci_upper": safe_round(item.get("ci_upper")),
                    "layer": item.get("layer"),
                }

    return results


###############################################################################
# FILE 8: correlation_20260309_092256.json
###############################################################################
def extract_file8():
    data = load("correlation_20260309_092256.json")
    results = {}

    # DP correlation
    dp_corr = data.get("dp_correlation", {})
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm not in dp_corr:
            continue
        p_data = dp_corr[paradigm]
        if isinstance(p_data, dict):
            # summary stats
            results[f"dp_correlation_{paradigm}"] = {}
            for key in ["n_significant", "total_features", "top_features", "best_layer",
                         "fdr_alpha", "n_pos", "n_neg"]:
                if key in p_data:
                    results[f"dp_correlation_{paradigm}"][key] = p_data[key]
            # per-layer summaries
            if "per_layer" in p_data:
                layer_summaries = []
                for l_data in p_data["per_layer"]:
                    if isinstance(l_data, dict):
                        layer_summaries.append({
                            "layer": l_data.get("layer"),
                            "n_significant": l_data.get("n_significant"),
                            "total_tested": l_data.get("total_tested"),
                            "top_feature_corr": safe_round(l_data.get("top_feature_corr")),
                        })
                results[f"dp_correlation_{paradigm}"]["per_layer_summary"] = layer_summaries
            # top features (first 10)
            if "top_features" in p_data and isinstance(p_data["top_features"], list):
                results[f"dp_correlation_{paradigm}"]["top_10_features"] = []
                for feat in p_data["top_features"][:10]:
                    results[f"dp_correlation_{paradigm}"]["top_10_features"].append({
                        "layer": feat.get("layer"),
                        "feature_idx": feat.get("feature_idx"),
                        "correlation": safe_round(feat.get("correlation")),
                        "p_value": feat.get("p_value"),
                        "cohens_d": safe_round(feat.get("cohens_d")),
                    })
        elif isinstance(p_data, list):
            # flat list of per-layer results
            results[f"dp_correlation_{paradigm}"] = [{
                "layer": r.get("layer"),
                "n_significant": r.get("n_significant"),
            } for r in p_data]

    # R1 correlation
    r1_corr = data.get("r1_correlation", {})
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm not in r1_corr:
            continue
        p_data = r1_corr[paradigm]
        if isinstance(p_data, dict):
            results[f"r1_correlation_{paradigm}"] = {}
            for key in ["n_significant", "total_features", "best_layer",
                         "fdr_alpha", "n_pos", "n_neg"]:
                if key in p_data:
                    results[f"r1_correlation_{paradigm}"][key] = p_data[key]
            if "top_features" in p_data and isinstance(p_data["top_features"], list):
                results[f"r1_correlation_{paradigm}"]["top_10_features"] = []
                for feat in p_data["top_features"][:10]:
                    results[f"r1_correlation_{paradigm}"]["top_10_features"].append({
                        "layer": feat.get("layer"),
                        "feature_idx": feat.get("feature_idx"),
                        "correlation": safe_round(feat.get("correlation")),
                        "p_value": feat.get("p_value"),
                        "cohens_d": safe_round(feat.get("cohens_d")),
                    })

    # Cross-paradigm overlap
    cp_overlap = data.get("cross_paradigm_overlap", {})
    results["cross_paradigm_feature_overlap"] = {}
    if isinstance(cp_overlap, dict):
        for key, val in cp_overlap.items():
            if isinstance(val, (int, float)):
                results["cross_paradigm_feature_overlap"][key] = safe_round(val)
            elif isinstance(val, dict):
                results["cross_paradigm_feature_overlap"][key] = {
                    k: safe_round(v) if isinstance(v, (int, float)) else v
                    for k, v in val.items()
                }
            elif isinstance(val, list):
                results["cross_paradigm_feature_overlap"][key] = val[:20]  # first 20

    # Behavioral-SAE linkage
    bsl = data.get("behavioral_sae_linkage", {})
    results["behavioral_sae_linkage"] = {}
    if isinstance(bsl, dict):
        for key, val in bsl.items():
            if isinstance(val, (int, float)):
                results["behavioral_sae_linkage"][key] = safe_round(val)
            elif isinstance(val, dict):
                results["behavioral_sae_linkage"][key] = {
                    k: safe_round(v) if isinstance(v, (int, float)) else v
                    for k, v in list(val.items())[:20]
                }
            elif isinstance(val, list):
                results["behavioral_sae_linkage"][key] = [{
                    k: safe_round(v) if isinstance(v, (int, float)) else v
                    for k, v in item.items()
                } for item in val[:10]] if all(isinstance(x, dict) for x in val[:1]) else val[:10]

    return results


###############################################################################
# FILE 9: condition_v2_20260308_151953.json
###############################################################################
def extract_file9():
    data = load("condition_v2_20260308_151953.json")
    results = {}

    # Per-condition SAE BK with component marginal effects
    for key, val in data.items():
        if isinstance(val, dict):
            extracted = {}
            for k, v in val.items():
                if isinstance(v, (int, float)):
                    extracted[k] = safe_round(v)
                elif isinstance(v, list) and len(v) > 0:
                    if isinstance(v[0], dict):
                        extracted[k] = [{kk: safe_round(vv) if isinstance(vv, (int, float)) else vv
                                         for kk, vv in item.items()} for item in v[:10]]
                    else:
                        extracted[k] = v[:10]
                elif isinstance(v, dict):
                    extracted[k] = {kk: safe_round(vv) if isinstance(vv, (int, float)) else vv
                                    for kk, vv in list(v.items())[:10]}
            results[f"condition_v2_{key}"] = extracted
        elif isinstance(val, list):
            if len(val) > 0 and isinstance(val[0], dict):
                results[f"condition_v2_{key}"] = [{k: safe_round(v) if isinstance(v, (int, float)) else v
                                                    for k, v in item.items()} for item in val[:20]]
            else:
                results[f"condition_v2_{key}"] = val[:20]

    return results


###############################################################################
# MAIN
###############################################################################
def main():
    import pprint

    all_results = {}

    print("=" * 80)
    print("FILE 1: all_analyses_20260306_091055.json")
    print("  (Goal A: SAE BK DP, Goal B: SAE R1, Goal C: Cross-domain)")
    print("=" * 80)
    f1 = extract_file1()
    all_results["file1_all_analyses"] = f1
    pprint.pprint(f1, width=120)

    print("\n" + "=" * 80)
    print("FILE 2: extended_analyses_20260306_211214.json")
    print("  (Exp2a: balance-matched, Exp3: risk, Exp4: condition encoding)")
    print("=" * 80)
    f2 = extract_file2()
    all_results["file2_extended"] = f2
    pprint.pprint(f2, width=120)

    print("\n" + "=" * 80)
    print("FILE 3: improved_v4_20260308_032435.json")
    print("  (cross_domain_bootstrap, same_layer_features, r1_permutation, balance_controlled)")
    print("=" * 80)
    f3 = extract_file3()
    all_results["file3_improved_v4"] = f3
    pprint.pprint(f3, width=120)

    print("\n" + "=" * 80)
    print("FILE 4: comprehensive_gemma_20260309_063511.json")
    print("  (hidden_bk, hidden_risk, percondition_sae_bk/risk, percondition_hidden_bk)")
    print("=" * 80)
    f4 = extract_file4()
    all_results["file4_comprehensive"] = f4
    pprint.pprint(f4, width=120)

    print("\n" + "=" * 80)
    print("FILE 5: comprehensive_gemma_20260309_095339.json")
    print("  (percondition_hidden_bk detailed)")
    print("=" * 80)
    f5 = extract_file5()
    all_results["file5_percondition_detailed"] = f5
    pprint.pprint(f5, width=120)

    print("\n" + "=" * 80)
    print("FILE 6: comprehensive_gemma_20260309_182926.json")
    print("  (hidden_risk corrected, percondition_sae_risk)")
    print("=" * 80)
    f6 = extract_file6()
    all_results["file6_corrected"] = f6
    pprint.pprint(f6, width=120)

    print("\n" + "=" * 80)
    print("FILE 7: hidden_gaps_20260309_181059.json")
    print("  (balance_matched_hidden, percondition_hidden_bk_r1, percondition_hidden_risk, hidden_cross_domain)")
    print("=" * 80)
    f7 = extract_file7()
    all_results["file7_hidden_gaps"] = f7
    pprint.pprint(f7, width=120)

    print("\n" + "=" * 80)
    print("FILE 8: correlation_20260309_092256.json")
    print("  (dp_correlation, r1_correlation, cross_paradigm_overlap, behavioral_sae_linkage)")
    print("=" * 80)
    f8 = extract_file8()
    all_results["file8_correlation"] = f8
    pprint.pprint(f8, width=120)

    print("\n" + "=" * 80)
    print("FILE 9: condition_v2_20260308_151953.json")
    print("  (per-condition SAE BK with component marginal effects)")
    print("=" * 80)
    f9 = extract_file9()
    all_results["file9_condition_v2"] = f9
    pprint.pprint(f9, width=120)

    # Save combined output
    out_path = "/home/jovyan/llm-addiction/sae_v3_analysis/results/extracted_all_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nSaved combined results to: {out_path}")


if __name__ == "__main__":
    main()
