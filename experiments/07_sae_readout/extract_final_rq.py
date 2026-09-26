"""
FINAL comprehensive extraction of ALL numerical results from Gemma SAE/activation analysis.
Organized by 3 Research Questions. Outputs Python dict to stdout.
"""
import json
from pathlib import Path

J = Path("/home/jovyan/llm-addiction/sae_v3_analysis/results/json")

def load(name):
    with open(J / name) as f:
        return json.load(f)

def r(v, d=4):
    if v is None: return None
    if isinstance(v, (int, float)): return round(v, d)
    return v

def best(layers, m="auc"):
    if not layers: return {}
    return max(layers, key=lambda x: x.get(m, 0))

def top5(layers, m="auc"):
    s = sorted(layers, key=lambda x: x.get(m, 0), reverse=True)
    return [(x["layer"], r(x[m])) for x in s[:5]]

# Load all files
f1 = load("all_analyses_20260306_091055.json")
f2 = load("extended_analyses_20260306_211214.json")
f3 = load("improved_v4_20260308_032435.json")
f4 = load("comprehensive_gemma_20260309_063511.json")
f5 = load("comprehensive_gemma_20260309_095339.json")
f6 = load("comprehensive_gemma_20260309_182926.json")
f7 = load("hidden_gaps_20260309_181059.json")
f8 = load("correlation_20260309_092256.json")
f9 = load("condition_v2_20260308_151953.json")

results = {
    "RQ1_bankruptcy_prediction": {},
    "RQ2_domain_invariant": {},
    "RQ3_condition_differences": {},
}

###############################################################################
# RQ1: BK PREDICTION
###############################################################################
rq1 = results["RQ1_bankruptcy_prediction"]

# 1A. SAE BK Decision-Point (all rounds)
rq1["sae_bk_dp"] = {}
for p in ["ic", "sm", "mw"]:
    layers = f1["goal_a_classification"][p]["sae"]
    b = best(layers)
    rq1["sae_bk_dp"][p] = {
        "best_layer": b["layer"], "best_auc": r(b["auc"]), "best_auc_std": r(b["auc_std"]),
        "best_f1": r(b["f1"]), "best_precision": r(b["precision"]), "best_recall": r(b["recall"]),
        "best_n_features": b.get("n_features"),
        "n_pos": b["n_pos"], "n_neg": b["n_neg"],
        "top5": top5(layers),
        "auc_range": (r(min(x["auc"] for x in layers)), r(max(x["auc"] for x in layers))),
        "all_42_layers": [(x["layer"], r(x["auc"]), r(x["auc_std"]), x.get("n_features")) for x in sorted(layers, key=lambda x: x["layer"])],
    }

# 1B. SAE R1 (early prediction) - Goal B is actually round-by-round
rq1["sae_r1_early_prediction"] = {}
for p in ["ic", "sm", "mw"]:
    ep = f1["goal_b_early_prediction"][p]
    rq1["sae_r1_early_prediction"][p] = {
        "layer": ep["layer"],
        "rounds": [(rd["round"], r(rd["auc"]), rd["n_games"], rd["n_bk"]) for rd in ep["rounds"]],
    }

# 1C. SAE R1 from exp2a (file2) - separate R1 classification
rq1["sae_r1_classification"] = {}
for p in ["ic", "sm", "mw"]:
    layers = f2["exp2a_balance_controlled"][p]["r1"]
    b = best(layers)
    rq1["sae_r1_classification"][p] = {
        "best_layer": b["layer"], "best_auc": r(b["auc"]), "best_auc_std": r(b["auc_std"]),
        "n_pos": b["n_pos"], "n_neg": b["n_neg"],
        "n_features": b.get("n_features"),
        "top5": top5(layers),
        "all_layers": [(x["layer"], r(x["auc"]), r(x["auc_std"])) for x in sorted(layers, key=lambda x: x["layer"])],
    }

# 1D. Balance-matched SAE BK
rq1["sae_bk_balance_matched"] = {}
for p in ["ic", "sm", "mw"]:
    layers = f2["exp2a_balance_controlled"][p]["balanced_matched"]
    b = best(layers)
    rq1["sae_bk_balance_matched"][p] = {
        "best_layer": b["layer"], "best_auc": r(b["auc"]), "best_auc_std": r(b["auc_std"]),
        "n_pos": b["n_pos"], "n_neg": b["n_neg"],
        "n_matched_pairs": b.get("n_matched_pairs"),
        "bk_bal_mean": r(b.get("bk_bal_mean")), "safe_bal_mean": r(b.get("safe_bal_mean")),
        "top5": top5(layers),
        "all_layers": [(x["layer"], r(x["auc"])) for x in sorted(layers, key=lambda x: x["layer"])],
    }

# 1E. R1 Permutation Test (file3)
rq1["r1_permutation_test"] = {}
for p in ["ic", "sm", "mw"]:
    v = f3["r1_permutation_test"][p]
    rq1["r1_permutation_test"][p] = {
        "layer": v["layer"],
        "observed_auc": r(v["observed_auc"]), "observed_auc_std": r(v.get("observed_auc_std")),
        "null_mean": r(v["null_mean"]), "null_std": r(v["null_std"]),
        "p_value": v["p_value"],
        "n_permutations": v["n_permutations"],
        "n_pos": v["n_pos"], "n_neg": v["n_neg"],
    }

# 1F. Balance-Controlled R1 SAE (file3)
rq1["sae_r1_balance_controlled"] = {}
for p in ["ic", "sm", "mw"]:
    pdata = f3["gemma_balance_controlled"][p]
    for mode in ["r1", "decision_point", "balance_matched"]:
        if mode in pdata:
            layers = pdata[mode]
            if isinstance(layers, list) and len(layers) > 0 and isinstance(layers[0], dict):
                b = best(layers)
                rq1["sae_r1_balance_controlled"][f"{p}_{mode}"] = {
                    "best_layer": b["layer"], "best_auc": r(b["auc"]),
                    "best_auc_std": r(b.get("auc_std", 0)),
                    "n_pos": b.get("n_pos"), "n_neg": b.get("n_neg"),
                    "top5": top5(layers),
                    "all_layers": [(x["layer"], r(x["auc"])) for x in sorted(layers, key=lambda x: x["layer"])],
                }

# 1G. Hidden State BK DP (file4)
rq1["hidden_bk_dp"] = {}
for p in ["ic", "sm", "mw"]:
    layers = f4["hidden_bk"][p]["dp"]
    b = best(layers)
    rq1["hidden_bk_dp"][p] = {
        "best_layer": b["layer"], "best_auc": r(b["auc"]), "best_auc_std": r(b.get("auc_std",0)),
        "best_f1": r(b.get("f1")),
        "n_pos": b["n_pos"], "n_neg": b["n_neg"],
        "top5": top5(layers),
        "all_layers": [(x["layer"], r(x["auc"]), r(x["auc_std"])) for x in sorted(layers, key=lambda x: x["layer"])],
    }

# 1H. Hidden State BK R1 (file4)
rq1["hidden_bk_r1"] = {}
for p in ["ic", "sm", "mw"]:
    layers = f4["hidden_bk"][p]["r1"]
    b = best(layers)
    rq1["hidden_bk_r1"][p] = {
        "best_layer": b["layer"], "best_auc": r(b["auc"]),
        "n_pos": b["n_pos"], "n_neg": b["n_neg"],
        "all_layers": [(x["layer"], r(x["auc"]), r(x["auc_std"])) for x in sorted(layers, key=lambda x: x["layer"])],
    }

# 1I. Balance-Matched Hidden BK (file7) - DP and R1
rq1["hidden_bk_balance_matched"] = {}
for p in ["ic", "sm", "mw"]:
    for mode in ["dp", "r1"]:
        layers = f7["balance_matched_hidden"][p][mode]
        b = best(layers)
        rq1["hidden_bk_balance_matched"][f"{p}_{mode}"] = {
            "best_layer": b["layer"], "best_auc": r(b["auc"]),
            "n_pos": b.get("n_pos"), "n_neg": b.get("n_neg"),
            "all_layers": [(x["layer"], r(x["auc"])) for x in sorted(layers, key=lambda x: x["layer"])],
        }

# 1J. SAE Risk (file2 exp3)
rq1["sae_risk"] = {}
for key in ["ic_risk_choice", "sm_bet_magnitude", "mw_bet_magnitude"]:
    layers = f2["exp3_round_level_risk"][key]
    b = best(layers)
    rq1["sae_risk"][key] = {
        "best_layer": b["layer"], "best_auc": r(b["auc"]), "best_auc_std": r(b.get("auc_std",0)),
        "n_pos": b["n_pos"], "n_neg": b["n_neg"],
        "top5": top5(layers),
    }
# IC risk by outcome
ic_rbo = f2["exp3_round_level_risk"].get("ic_risk_by_outcome", {})
rq1["sae_risk"]["ic_risk_by_outcome"] = {}
for outcome, data in ic_rbo.items():
    if isinstance(data, dict):
        rq1["sae_risk"]["ic_risk_by_outcome"][outcome] = {
            "auc": r(data.get("auc")), "auc_std": r(data.get("auc_std")),
            "f1": r(data.get("f1")),
        }
    elif isinstance(data, list):
        b = best(data)
        rq1["sae_risk"]["ic_risk_by_outcome"][outcome] = {
            "best_layer": b.get("layer"), "best_auc": r(b.get("auc")),
        }

# 1K. Hidden Risk (file6 corrected)
rq1["hidden_risk_corrected"] = {}
for p in ["ic", "sm", "mw"]:
    layers = f6["hidden_risk"][p]
    b = best(layers)
    rq1["hidden_risk_corrected"][p] = {
        "best_layer": b["layer"], "best_auc": r(b["auc"]), "best_auc_std": r(b.get("auc_std",0)),
        "all_layers": [(x["layer"], r(x["auc"])) for x in sorted(layers, key=lambda x: x["layer"])],
    }

# 1L. Feature Importance (file2 exp2b)
rq1["feature_importance"] = {}
exp2b = f2["exp2b_feature_importance"]
for p in ["ic", "sm", "mw"]:
    fi = exp2b[p]
    rq1["feature_importance"][p] = {
        "layer": fi["layer"], "auc": r(fi["auc"]),
        "n_active_features": fi["n_active_features"],
        "top_10_features": fi.get("top_features", [])[:10],
        "top_feature_indices": fi.get("top_feature_indices", [])[:20],
    }
# Cross-paradigm from feature importance
rq1["feature_importance"]["cross_paradigm_overlap"] = exp2b.get("cross_paradigm_overlap", {})

# 1M. DP Correlation (file8)
rq1["dp_correlation"] = {}
for p in ["ic", "sm", "mw"]:
    rq1["dp_correlation"][p] = {}
    for lk in ["L0", "L6", "L8", "L10", "L12", "L16", "L18", "L22"]:
        ld = f8["dp_correlation"][p][lk]
        top = ld.get("top_features", [])
        t1 = top[0] if top else {}
        rq1["dp_correlation"][p][lk] = {
            "n_significant": ld.get("n_significant"),
            "n_features_tested": ld.get("n_features_tested"),
            "top1_feature": t1.get("feature_idx"),
            "top1_cohen_d": r(t1.get("cohen_d")),
            "top1_p_fdr": t1.get("p_fdr"),
            "top1_direction": t1.get("direction"),
            "top1_rate_bk": r(t1.get("rate_bk")),
            "top1_rate_safe": r(t1.get("rate_safe")),
            "top1_mean_bk": r(t1.get("mean_bk")),
            "top1_mean_safe": r(t1.get("mean_safe")),
        }

# 1N. R1 Correlation (file8)
rq1["r1_correlation"] = {}
for p in ["ic", "sm", "mw"]:
    rq1["r1_correlation"][p] = {}
    for lk in ["L0", "L6", "L8", "L10", "L12", "L16", "L18", "L22"]:
        ld = f8["r1_correlation"][p][lk]
        top = ld.get("top_features", [])
        t1 = top[0] if top else {}
        rq1["r1_correlation"][p][lk] = {
            "n_significant": ld.get("n_significant"),
            "n_features_tested": ld.get("n_features_tested"),
            "top1_feature": t1.get("feature_idx"),
            "top1_cohen_d": r(t1.get("cohen_d")),
            "top1_direction": t1.get("direction"),
        }

# 1O. Behavioral-SAE Linkage (file8) - SM only has data
rq1["behavioral_sae_linkage_sm"] = {}
sm_link = f8["behavioral_sae_linkage"]["sm"]
for cond_key in list(sm_link.keys())[:12]:
    features = sm_link[cond_key]
    if isinstance(features, list) and len(features) > 0:
        rq1["behavioral_sae_linkage_sm"][cond_key] = [{
            "feature_idx": f.get("feature_idx"),
            "cohen_d": r(f.get("cohen_d")),
            "spearman_i_ba": r(f.get("spearman_i_ba")),
            "p_i_ba": f.get("p_i_ba"),
            "spearman_i_lc": r(f.get("spearman_i_lc")),
            "p_i_lc": f.get("p_i_lc"),
        } for f in features[:3]]

###############################################################################
# RQ2: DOMAIN-INVARIANT REPRESENTATIONS
###############################################################################
rq2 = results["RQ2_domain_invariant"]

# 2A. Cross-domain SAE Transfer (file1 goal_c)
rq2["sae_cross_domain_transfer"] = {}
for direction in ["ic_to_sm", "ic_to_mw", "sm_to_ic", "sm_to_mw", "mw_to_ic", "mw_to_sm"]:
    layers = f1["goal_c_cross_domain"][direction]
    b = max(layers, key=lambda x: x.get("transfer_auc", 0))
    rq2["sae_cross_domain_transfer"][direction] = {
        "best_layer": b["layer"], "best_transfer_auc": r(b["transfer_auc"]),
        "n_shared_features": b["n_shared_features"],
        "src_n_bk": b["src_n_bk"], "tgt_n_bk": b["tgt_n_bk"],
        "all_layers": [(x["layer"], r(x["transfer_auc"]), x["n_shared_features"]) for x in sorted(layers, key=lambda x: x["layer"])],
    }

# 2B. Cross-domain Bootstrap (file3)
rq2["sae_cross_domain_bootstrap"] = {}
for direction in ["ic_to_sm", "ic_to_mw", "sm_to_ic", "sm_to_mw", "mw_to_ic", "mw_to_sm"]:
    v = f3["cross_domain_bootstrap"][direction]
    rq2["sae_cross_domain_bootstrap"][direction] = {
        "layer": v["layer"],
        "transfer_auc": r(v["transfer_auc"]),
        "ci_lower": r(v["ci_lower"]), "ci_upper": r(v["ci_upper"]),
        "n_shared_features": v["n_shared_features"],
        "n_bootstrap": v["n_bootstrap"],
        "src_n_bk": v["src_n_bk"], "tgt_n_bk": v["tgt_n_bk"],
    }

# 2C. Same-Layer Feature Overlap (file3)
overlap = f3["same_layer_features"]["overlap"]
rq2["same_layer_feature_overlap_L22"] = {
    "ic_vs_sm": {
        "n_shared": overlap["ic_vs_sm"]["n_shared"],
        "jaccard": r(overlap["ic_vs_sm"]["jaccard"]),
        "shared_features": overlap["ic_vs_sm"]["shared_features"],
    },
    "ic_vs_mw": {
        "n_shared": overlap["ic_vs_mw"]["n_shared"],
        "jaccard": r(overlap["ic_vs_mw"]["jaccard"]),
        "shared_features": overlap["ic_vs_mw"]["shared_features"],
    },
    "sm_vs_mw": {
        "n_shared": overlap["sm_vs_mw"]["n_shared"],
        "jaccard": r(overlap["sm_vs_mw"]["jaccard"]),
        "shared_features": overlap["sm_vs_mw"]["shared_features"],
    },
}
# Per-paradigm L22 info
for p in ["ic", "sm", "mw"]:
    info = f3["same_layer_features"][p]
    rq2["same_layer_feature_overlap_L22"][f"{p}_info"] = {
        "layer": info["layer"], "auc": r(info["auc"]), "n_active": info["n_active"],
    }

# 2D. Cross-Paradigm Correlation Feature Overlap (file8)
rq2["correlation_feature_overlap"] = {}
for lk in sorted(f8["cross_paradigm_overlap"].keys(), key=lambda x: int(x[1:])):
    lo = f8["cross_paradigm_overlap"][lk]
    rq2["correlation_feature_overlap"][lk] = {}
    for pair in ["ic_vs_sm", "ic_vs_mw", "sm_vs_mw"]:
        po = lo[pair]
        rq2["correlation_feature_overlap"][lk][pair] = {
            "shared_active_pool": po["shared_active_pool"],
            "K": po["K"],
            "overlap_count": po["overlap_count"],
            "jaccard": r(po["jaccard"]),
            "hypergeom_p": po["hypergeom_p"],
            "significant": po["significant"],
            "overlap_features": po.get("overlap_features", []),
        }

# 2E. Hidden State Cross-Domain Transfer (file7)
rq2["hidden_cross_domain_transfer"] = {}
for direction in ["ic_to_sm", "ic_to_mw", "sm_to_ic", "sm_to_mw", "mw_to_ic", "mw_to_sm"]:
    layers = f7["hidden_cross_domain"][direction]
    b = best(layers)
    rq2["hidden_cross_domain_transfer"][direction] = {
        "best_layer": b["layer"], "best_auc": r(b["auc"]),
        "all_layers": [(x["layer"], r(x["auc"])) for x in sorted(layers, key=lambda x: x["layer"])],
    }

# 2F. Cross-model comparison LLaMA vs Gemma (file3)
rq2["cross_model_comparison"] = f3.get("cross_model_comparison", {})

# LLaMA BK classification (file3)
llama_bk = f3.get("llama_bk_classification", {})
if isinstance(llama_bk, list) and len(llama_bk) > 0:
    b = best(llama_bk)
    rq2["llama_bk_classification"] = {
        "best_layer": b["layer"], "best_auc": r(b["auc"]),
        "best_auc_std": r(b.get("auc_std", 0)),
        "n_pos": b["n_pos"], "n_neg": b["n_neg"],
        "top5": top5(llama_bk),
        "all_layers": [(x["layer"], r(x["auc"]), x.get("n_features")) for x in sorted(llama_bk, key=lambda x: x["layer"])],
    }
elif isinstance(llama_bk, dict):
    rq2["llama_bk_classification"] = {}
    for key, layers in llama_bk.items():
        if isinstance(layers, list) and len(layers) > 0:
            b = best(layers)
            rq2["llama_bk_classification"][key] = {
                "best_layer": b["layer"], "best_auc": r(b["auc"]),
                "top5": top5(layers),
            }

###############################################################################
# RQ3: CONDITION DIFFERENCES
###############################################################################
rq3 = results["RQ3_condition_differences"]

# 3A. Condition Encoding (file2 exp4)
rq3["condition_encoding"] = {}
exp4 = f2["exp4_condition_level"]

# IC bet constraint (4-class)
layers = exp4["ic_bet_constraint"]
b = best(layers)
rq3["condition_encoding"]["ic_bet_constraint"] = {
    "best_layer": b["layer"], "best_auc": r(b["auc"]), "n_classes": b["n_classes"],
    "class_counts": b.get("class_counts"),
    "top5": top5(layers),
    "all_layers": [(x["layer"], r(x["auc"]), r(x.get("auc_std",0))) for x in sorted(layers, key=lambda x: x["layer"])],
}

# IC prompt condition (4-class)
layers = exp4["ic_prompt_condition"]
b = best(layers)
rq3["condition_encoding"]["ic_prompt_condition"] = {
    "best_layer": b["layer"], "best_auc": r(b["auc"]), "n_classes": b["n_classes"],
    "top5": top5(layers),
}

# Bet type per paradigm
for p in ["ic", "sm", "mw"]:
    layers = exp4["bet_type"][p]
    b = best(layers)
    rq3["condition_encoding"][f"bet_type_{p}"] = {
        "best_layer": b["layer"], "best_auc": r(b["auc"]),
        "top5": top5(layers),
    }

# Prompt components
if "prompt_components" in exp4:
    rq3["condition_encoding"]["prompt_components"] = {}
    for comp_key, layers in exp4["prompt_components"].items():
        if isinstance(layers, list):
            b = best(layers)
            rq3["condition_encoding"]["prompt_components"][comp_key] = {
                "best_layer": b["layer"], "best_auc": r(b["auc"]),
            }

# 3B. Per-Condition SAE BK (file4) - best per condition at each key layer
rq3["percondition_sae_bk"] = {}
for p in ["ic", "sm", "mw"]:
    conds = f4["percondition_sae_bk"][p]
    rq3["percondition_sae_bk"][p] = {}
    for ck, v in conds.items():
        if isinstance(v, dict):
            rq3["percondition_sae_bk"][p][ck] = {
                "auc": r(v["auc"]), "auc_std": r(v.get("auc_std")),
                "n_pos": v["n_pos"], "n_neg": v["n_neg"],
                "layer": v["layer"],
                "condition": v.get("condition"),
            }

# 3C. Per-Condition SAE Risk corrected (file6)
rq3["percondition_sae_risk"] = {}
for p in ["ic", "sm", "mw"]:
    conds = f6["percondition_sae_risk"][p]
    rq3["percondition_sae_risk"][p] = {}
    for ck, v in conds.items():
        if isinstance(v, dict):
            rq3["percondition_sae_risk"][p][ck] = {
                "auc": r(v.get("auc")), "n_pos": v.get("n_pos"), "n_neg": v.get("n_neg"),
                "layer": v.get("layer"),
            }
        elif isinstance(v, list) and len(v) > 0:
            # It's a flat dict, not a list. Skip non-dict entries.
            pass

# 3D. Per-Condition Hidden BK R1 (file7) - best per condition
rq3["percondition_hidden_bk_r1"] = {}
for p in ["ic", "sm", "mw"]:
    conds = f7["percondition_hidden_bk_r1"][p]
    rq3["percondition_hidden_bk_r1"][p] = {}
    for ck, v in conds.items():
        if isinstance(v, dict):
            rq3["percondition_hidden_bk_r1"][p][ck] = {
                "auc": r(v["auc"]), "auc_std": r(v.get("auc_std")),
                "n_pos": v["n_pos"], "n_neg": v["n_neg"],
                "layer": v["layer"],
                "condition": v.get("condition"),
            }

# 3E. Per-Condition Hidden Risk (file7)
rq3["percondition_hidden_risk"] = {}
for p in ["ic", "sm", "mw"]:
    conds = f7["percondition_hidden_risk"][p]
    rq3["percondition_hidden_risk"][p] = {}
    for ck, v in conds.items():
        if isinstance(v, dict):
            rq3["percondition_hidden_risk"][p][ck] = {
                "auc": r(v.get("auc")), "n_pos": v.get("n_pos"), "n_neg": v.get("n_neg"),
                "layer": v.get("layer"),
            }

# 3F. Condition V2 - Component Marginal Effects (file9)
rq3["condition_marginal_effects"] = {}
for p in ["ic", "sm", "mw"]:
    d = f9[p]
    rq3["condition_marginal_effects"][p] = {
        "paradigm": d["paradigm"],
        "layer": d["layer"],
        "n": d["n"], "n_bk": d["n_bk"],
        "n_features": d["n_features"],
        "bet_fixed": {k: r(v) if isinstance(v, float) else v for k, v in d["bet_fixed"].items()},
        "bet_variable": {k: r(v) if isinstance(v, float) else v for k, v in d["bet_variable"].items()},
        "bet_type_fisher_p": d["bet_type_fisher_p"],
        "component_marginal": {},
    }
    for comp, cm in d["component_marginal"].items():
        rq3["condition_marginal_effects"][p]["component_marginal"][comp] = {
            "n_with": cm["n_with"], "n_without": cm["n_without"],
            "bk_with": cm["bk_with"], "bk_without": cm["bk_without"],
            "bk_rate_with": r(cm["bk_rate_with"]), "bk_rate_without": r(cm["bk_rate_without"]),
            "bk_rate_diff": r(cm["bk_rate_diff"]),
            "fisher_p": cm["fisher_p"],
            "perm_p": r(cm["perm_p"]),
            "auc_with": r(cm["auc_with"]), "auc_without": r(cm["auc_without"]),
            "auc_diff": r(cm["auc_diff"]),
        }
    # IC-specific: per-condition AUC
    if p == "ic":
        for cond in ["cond_BASE", "cond_G", "cond_M", "cond_GM"]:
            if cond in d:
                rq3["condition_marginal_effects"][p][cond] = {
                    "n": d[cond]["n"], "n_bk": d[cond]["n_bk"],
                    "bk_rate": r(d[cond]["bk_rate"]), "auc": r(d[cond]["auc"]),
                }
        rq3["condition_marginal_effects"][p]["ic_4cond_chi2_p"] = d.get("ic_4cond_chi2_p")

# 3G. R1 within-condition correlation (file8)
rq3["r1_within_condition_correlation"] = {}
for p in ["ic", "sm", "mw"]:
    r1wc = f8["r1_within_condition"][p]
    rq3["r1_within_condition_correlation"][p] = {}
    for ck, v in r1wc.items():
        if isinstance(v, dict):
            rq3["r1_within_condition_correlation"][p][ck] = {
                "n_significant": v.get("n_significant"),
                "n_features_tested": v.get("n_features_tested"),
                "n_total": v.get("n_total"),
                "n_bk": v.get("n_bk"),
            }
            top = v.get("top_features", [])
            if top:
                t1 = top[0]
                rq3["r1_within_condition_correlation"][p][ck]["top1_feature"] = t1.get("feature_idx")
                rq3["r1_within_condition_correlation"][p][ck]["top1_cohen_d"] = r(t1.get("cohen_d"))

# 3H. Per-condition IC fixed (file8)
rq3["percondition_ic_fixed_correlation"] = {}
for ck, v in f8["percondition_ic_fixed"].items():
    rq3["percondition_ic_fixed_correlation"][ck] = {
        "constraint": v["constraint"],
        "n_total": v["n_total"], "n_bk": v["n_bk"],
        "n_features_tested": v["n_features_tested"],
        "n_significant": v["n_significant"],
    }
    if v.get("top_features"):
        rq3["percondition_ic_fixed_correlation"][ck]["top_features"] = [{
            "feature_idx": f.get("feature_idx"),
            "cohen_d": r(f.get("cohen_d")),
        } for f in v["top_features"][:5]]

###############################################################################
# OUTPUT
###############################################################################
import pprint
print("=" * 100)
print("COMPREHENSIVE RESULTS ORGANIZED BY RESEARCH QUESTION")
print("=" * 100)

for rq_name, rq_data in results.items():
    print(f"\n{'=' * 100}")
    print(f"  {rq_name}")
    print(f"{'=' * 100}")
    pprint.pprint(rq_data, width=140, depth=5)

# Save to JSON
out = "/home/jovyan/llm-addiction/sae_v3_analysis/results/rq_comprehensive_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n\nSaved to: {out}")
