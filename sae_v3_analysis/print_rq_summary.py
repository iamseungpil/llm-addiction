"""
Print a clean, exhaustive summary of ALL numerical results organized by RQ.
Reads from rq_comprehensive_results.json.
"""
import json

d = json.load(open("/home/jovyan/llm-addiction/sae_v3_analysis/results/rq_comprehensive_results.json"))

rq1 = d["RQ1_bankruptcy_prediction"]
rq2 = d["RQ2_domain_invariant"]
rq3 = d["RQ3_condition_differences"]

print("""
################################################################################
# RQ1: WHICH FEATURES/ACTIVATIONS CONSISTENTLY PREDICT BANKRUPTCY?
################################################################################
""")

# --- SAE BK Decision-Point ---
print("=== 1A. SAE BK Decision-Point Classification (all rounds, 42 layers) ===")
print("Source: all_analyses file1, goal_a_classification")
for p in ["ic", "sm", "mw"]:
    s = rq1["sae_bk_dp"][p]
    print(f"\n  {p.upper()}: n_pos={s['n_pos']}, n_neg={s['n_neg']}")
    print(f"    BEST: L{s['best_layer']}, AUC={s['best_auc']}, std={s['best_auc_std']}, F1={s['best_f1']}, prec={s['best_precision']}, recall={s['best_recall']}, n_features={s['best_n_features']}")
    print(f"    Top5 layers: {s['top5']}")
    print(f"    AUC range: {s['auc_range'][0]} - {s['auc_range'][1]}")
    print(f"    All 42 layers (layer, AUC, std, n_features):")
    for l, a, std, nf in s["all_42_layers"]:
        print(f"      L{l:2d}: AUC={a:.4f} std={std:.4f} n_feat={nf}")

# --- SAE R1 Early Prediction (round-by-round) ---
print("\n\n=== 1B. SAE R1 Early Prediction (round-by-round AUC) ===")
print("Source: all_analyses file1, goal_b_early_prediction")
for p in ["ic", "sm", "mw"]:
    s = rq1["sae_r1_early_prediction"][p]
    print(f"\n  {p.upper()} at layer {s['layer']}:")
    print(f"    (round, AUC, n_games, n_bk):")
    for rd, auc, ng, nb in s["rounds"]:
        print(f"      R{rd:3d}: AUC={auc:.4f}, n_games={ng}, n_bk={nb}")

# --- SAE R1 Classification ---
print("\n\n=== 1C. SAE R1 Classification (first round) ===")
print("Source: extended_analyses file2, exp2a_balance_controlled/r1")
for p in ["ic", "sm", "mw"]:
    s = rq1["sae_r1_classification"][p]
    print(f"\n  {p.upper()}: n_pos={s['n_pos']}, n_neg={s['n_neg']}")
    print(f"    BEST: L{s['best_layer']}, AUC={s['best_auc']}, std={s['best_auc_std']}, n_features={s.get('n_features')}")
    print(f"    Top5: {s['top5']}")
    print(f"    All layers (layer, AUC, std):")
    for l, a, std in s["all_layers"]:
        print(f"      L{l:2d}: AUC={a:.4f} std={std:.4f}")

# --- Balance-Matched SAE BK ---
print("\n\n=== 1D. Balance-Matched SAE BK (controls for balance confound) ===")
print("Source: extended_analyses file2, exp2a_balance_controlled/balanced_matched")
for p in ["ic", "sm", "mw"]:
    s = rq1["sae_bk_balance_matched"][p]
    print(f"\n  {p.upper()}: n_pos={s['n_pos']}, n_neg={s['n_neg']}, n_matched={s.get('n_matched_pairs')}")
    print(f"    BEST: L{s['best_layer']}, AUC={s['best_auc']}, std={s['best_auc_std']}")
    print(f"    bk_bal_mean={s.get('bk_bal_mean')}, safe_bal_mean={s.get('safe_bal_mean')}")
    print(f"    Top5: {s['top5']}")
    print(f"    All layers: {s['all_layers']}")

# --- R1 Permutation Test ---
print("\n\n=== 1E. R1 Permutation Test (statistical significance) ===")
print("Source: improved_v4 file3, r1_permutation_test")
for p in ["ic", "sm", "mw"]:
    s = rq1["r1_permutation_test"][p]
    print(f"\n  {p.upper()} at L{s['layer']}: n_pos={s['n_pos']}, n_neg={s['n_neg']}")
    print(f"    observed_AUC={s['observed_auc']}, observed_std={s['observed_auc_std']}")
    print(f"    null_mean={s['null_mean']}, null_std={s['null_std']}")
    print(f"    p_value={s['p_value']}, n_permutations={s['n_permutations']}")

# --- Balance-Controlled R1 SAE (file3) ---
print("\n\n=== 1F. Balance-Controlled SAE (R1, DP, Balanced-Matched) ===")
print("Source: improved_v4 file3, gemma_balance_controlled")
for key, s in rq1["sae_r1_balance_controlled"].items():
    print(f"\n  {key}: n_pos={s.get('n_pos')}, n_neg={s.get('n_neg')}")
    print(f"    BEST: L{s['best_layer']}, AUC={s['best_auc']}, std={s.get('best_auc_std')}")
    print(f"    Top5: {s['top5']}")
    print(f"    All layers: {s['all_layers']}")

# --- Hidden State BK DP ---
print("\n\n=== 1G. Hidden State BK Decision-Point Classification ===")
print("Source: comprehensive_gemma file4, hidden_bk/dp")
for p in ["ic", "sm", "mw"]:
    s = rq1["hidden_bk_dp"][p]
    print(f"\n  {p.upper()}: n_pos={s['n_pos']}, n_neg={s['n_neg']}")
    print(f"    BEST: L{s['best_layer']}, AUC={s['best_auc']}, std={s['best_auc_std']}, F1={s['best_f1']}")
    print(f"    Top5: {s['top5']}")
    print(f"    All layers: {s['all_layers']}")

# --- Hidden State BK R1 ---
print("\n\n=== 1H. Hidden State BK R1 Classification ===")
print("Source: comprehensive_gemma file4, hidden_bk/r1")
for p in ["ic", "sm", "mw"]:
    s = rq1["hidden_bk_r1"][p]
    print(f"\n  {p.upper()}: n_pos={s['n_pos']}, n_neg={s['n_neg']}")
    print(f"    BEST: L{s['best_layer']}, AUC={s['best_auc']}")
    print(f"    All layers: {s['all_layers']}")

# --- Balance-Matched Hidden BK ---
print("\n\n=== 1I. Balance-Matched Hidden State BK ===")
print("Source: hidden_gaps file7, balance_matched_hidden")
for key, s in rq1["hidden_bk_balance_matched"].items():
    print(f"\n  {key}: n_pos={s['n_pos']}, n_neg={s['n_neg']}")
    print(f"    BEST: L{s['best_layer']}, AUC={s['best_auc']}")
    print(f"    All layers: {s['all_layers']}")

# --- SAE Risk ---
print("\n\n=== 1J. SAE Risk Preference Classification ===")
print("Source: extended_analyses file2, exp3_round_level_risk")
for key, s in rq1["sae_risk"].items():
    if key == "ic_risk_by_outcome":
        print(f"\n  {key}:")
        for outcome, os in s.items():
            print(f"    {outcome}: AUC={os.get('auc')}, F1={os.get('f1')}")
    else:
        print(f"\n  {key}: n_pos={s['n_pos']}, n_neg={s['n_neg']}")
        print(f"    BEST: L{s['best_layer']}, AUC={s['best_auc']}, std={s.get('best_auc_std')}")
        print(f"    Top5: {s['top5']}")

# --- Hidden Risk Corrected ---
print("\n\n=== 1K. Hidden State Risk Preference (corrected) ===")
print("Source: comprehensive_gemma file6, hidden_risk")
for p in ["ic", "sm", "mw"]:
    s = rq1["hidden_risk_corrected"][p]
    print(f"\n  {p.upper()}: BEST L{s['best_layer']}, AUC={s['best_auc']}, std={s.get('best_auc_std')}")
    print(f"    All layers: {s['all_layers']}")

# --- Feature Importance ---
print("\n\n=== 1L. Feature Importance (top features per paradigm) ===")
print("Source: extended_analyses file2, exp2b_feature_importance")
for p in ["ic", "sm", "mw"]:
    s = rq1["feature_importance"][p]
    print(f"\n  {p.upper()} at L{s['layer']}: AUC={s['auc']}, n_active={s['n_active_features']}")
    print(f"    Top 20 feature indices: {s['top_feature_indices']}")
print(f"\n  Cross-paradigm overlap: {rq1['feature_importance'].get('cross_paradigm_overlap', {})}")

# --- DP Correlation ---
print("\n\n=== 1M. DP Feature-Outcome Correlation (FDR-corrected) ===")
print("Source: correlation file8, dp_correlation")
for p in ["ic", "sm", "mw"]:
    print(f"\n  {p.upper()}:")
    for lk in ["L0", "L6", "L8", "L10", "L12", "L16", "L18", "L22"]:
        s = rq1["dp_correlation"][p][lk]
        print(f"    {lk}: n_sig={s['n_significant']}/{s['n_features_tested']}, top1: F{s['top1_feature']} d={s['top1_cohen_d']} dir={s['top1_direction']} p_fdr={s['top1_p_fdr']:.2e}")

# --- R1 Correlation ---
print("\n\n=== 1N. R1 Feature-Outcome Correlation (FDR-corrected) ===")
print("Source: correlation file8, r1_correlation")
for p in ["ic", "sm", "mw"]:
    print(f"\n  {p.upper()}:")
    for lk in ["L0", "L6", "L8", "L10", "L12", "L16", "L18", "L22"]:
        s = rq1["r1_correlation"][p][lk]
        print(f"    {lk}: n_sig={s['n_significant']}/{s['n_features_tested']}, top1: F{s['top1_feature']} d={s['top1_cohen_d']} dir={s['top1_direction']}")

# --- Behavioral-SAE Linkage ---
print("\n\n=== 1O. Behavioral-SAE Linkage (SM only) ===")
print("Source: correlation file8, behavioral_sae_linkage")
for cond, features in rq1["behavioral_sae_linkage_sm"].items():
    print(f"\n  {cond}:")
    for f in features:
        print(f"    F{f['feature_idx']}: d={f['cohen_d']}, I_BA_rho={f['spearman_i_ba']} (p={f['p_i_ba']:.2e}), I_LC_rho={f['spearman_i_lc']} (p={f['p_i_lc']:.2e})")


print("""

################################################################################
# RQ2: ARE THERE DOMAIN-INVARIANT REPRESENTATIONS?
################################################################################
""")

# --- SAE Cross-Domain Transfer ---
print("=== 2A. SAE Cross-Domain Transfer (best layer per direction) ===")
print("Source: all_analyses file1, goal_c_cross_domain")
for direction, s in rq2["sae_cross_domain_transfer"].items():
    print(f"\n  {direction}: BEST L{s['best_layer']}, transfer_AUC={s['best_transfer_auc']}, n_shared_feat={s['n_shared_features']}, src_bk={s['src_n_bk']}, tgt_bk={s['tgt_n_bk']}")
    print(f"    All layers (layer, AUC, n_shared):")
    for l, a, ns in s["all_layers"]:
        print(f"      L{l:2d}: AUC={a:.4f}, n_shared={ns}")

# --- Bootstrap Cross-Domain ---
print("\n\n=== 2B. SAE Cross-Domain Bootstrap (with CIs) ===")
print("Source: improved_v4 file3, cross_domain_bootstrap")
for direction, s in rq2["sae_cross_domain_bootstrap"].items():
    print(f"  {direction}: L{s['layer']}, AUC={s['transfer_auc']}, CI=[{s['ci_lower']}, {s['ci_upper']}], n_shared={s['n_shared_features']}, n_boot={s['n_bootstrap']}, src_bk={s['src_n_bk']}, tgt_bk={s['tgt_n_bk']}")

# --- Same-Layer Feature Overlap ---
print("\n\n=== 2C. Same-Layer Feature Overlap at L22 (top-100 features) ===")
print("Source: improved_v4 file3, same_layer_features")
for pair in ["ic_vs_sm", "ic_vs_mw", "sm_vs_mw"]:
    s = rq2["same_layer_feature_overlap_L22"][pair]
    print(f"\n  {pair}: n_shared={s['n_shared']}, jaccard={s['jaccard']}")
    print(f"    shared features: {s['shared_features']}")
for p in ["ic", "sm", "mw"]:
    info = rq2["same_layer_feature_overlap_L22"][f"{p}_info"]
    print(f"  {p.upper()} L{info['layer']}: AUC={info['auc']}, n_active={info['n_active']}")

# --- Correlation Feature Overlap ---
print("\n\n=== 2D. Cross-Paradigm Correlation Feature Overlap (hypergeometric test) ===")
print("Source: correlation file8, cross_paradigm_overlap")
for lk in sorted(rq2["correlation_feature_overlap"].keys(), key=lambda x: int(x[1:])):
    for pair in ["ic_vs_sm", "ic_vs_mw", "sm_vs_mw"]:
        s = rq2["correlation_feature_overlap"][lk][pair]
        sig_marker = " ***" if s["significant"] else ""
        print(f"  {lk} {pair}: pool={s['shared_active_pool']}, K={s['K']}, overlap={s['overlap_count']}, J={s['jaccard']}, p={s['hypergeom_p']:.2e}{sig_marker}")
        if s["overlap_features"]:
            print(f"    overlap features: {s['overlap_features']}")

# --- Hidden Cross-Domain Transfer ---
print("\n\n=== 2E. Hidden State Cross-Domain Transfer ===")
print("Source: hidden_gaps file7, hidden_cross_domain")
for direction, s in rq2["hidden_cross_domain_transfer"].items():
    print(f"\n  {direction}: BEST L{s['best_layer']}, AUC={s['best_auc']}")
    print(f"    All layers: {s['all_layers']}")

# --- Cross-Model Comparison ---
print("\n\n=== 2F. Cross-Model Comparison (Gemma vs LLaMA) ===")
print("Source: improved_v4 file3, cross_model_comparison")
import pprint
pprint.pprint(rq2["cross_model_comparison"], width=120)

# --- LLaMA BK Classification ---
print("\n\n=== 2G. LLaMA BK Classification (IC) ===")
print("Source: improved_v4 file3, llama_bk_classification")
s = rq2["llama_bk_classification"]
if isinstance(s, dict) and "best_layer" in s:
    print(f"  BEST: L{s['best_layer']}, AUC={s['best_auc']}, std={s.get('best_auc_std')}")
    print(f"  n_pos={s['n_pos']}, n_neg={s['n_neg']}")
    print(f"  Top5: {s['top5']}")
    print(f"  All layers (layer, AUC, n_features):")
    for l, a, nf in s["all_layers"]:
        print(f"    L{l:2d}: AUC={a:.4f}, n_features={nf}")


print("""

################################################################################
# RQ3: DO REPRESENTATIONS DIFFER BY CONDITIONS?
################################################################################
""")

# --- Condition Encoding ---
print("=== 3A. SAE Condition Encoding (multi-class classification) ===")
print("Source: extended_analyses file2, exp4_condition_level")
for key, s in rq3["condition_encoding"].items():
    if key == "prompt_components":
        print(f"\n  {key}:")
        for comp, cs in s.items():
            print(f"    {comp}: BEST L{cs['best_layer']}, AUC={cs['best_auc']}")
    else:
        print(f"\n  {key}: BEST L{s['best_layer']}, AUC={s['best_auc']}, n_classes={s.get('n_classes')}")
        print(f"    Top5: {s['top5']}")
        if "all_layers" in s:
            print(f"    All layers: {s['all_layers']}")

# --- Condition V2 Marginal Effects ---
print("\n\n=== 3B. Component Marginal Effects on BK Rate & SAE AUC ===")
print("Source: condition_v2 file9")
for p in ["ic", "sm", "mw"]:
    s = rq3["condition_marginal_effects"][p]
    print(f"\n  {p.upper()} (L{s['layer']}, n={s['n']}, n_bk={s['n_bk']}, n_features={s['n_features']}):")
    print(f"    Bet Fixed: n={s['bet_fixed']['n']}, n_bk={s['bet_fixed']['n_bk']}, BK%={s['bet_fixed']['bk_rate']}, AUC={s['bet_fixed']['auc']}")
    print(f"    Bet Variable: n={s['bet_variable']['n']}, n_bk={s['bet_variable']['n_bk']}, BK%={s['bet_variable']['bk_rate']}, AUC={s['bet_variable']['auc']}")
    print(f"    bet_type Fisher p={s['bet_type_fisher_p']:.2e}")
    print(f"    Component marginal effects:")
    for comp, cm in s["component_marginal"].items():
        print(f"      {comp}: BK_with={cm['bk_with']}/{cm['n_with']}({cm['bk_rate_with']}), BK_without={cm['bk_without']}/{cm['n_without']}({cm['bk_rate_without']}), diff={cm['bk_rate_diff']}, Fisher_p={cm['fisher_p']:.2e}, perm_p={cm['perm_p']}")
        print(f"             AUC_with={cm['auc_with']}, AUC_without={cm['auc_without']}, AUC_diff={cm['auc_diff']}")
    if p == "ic":
        for cond in ["cond_BASE", "cond_G", "cond_M", "cond_GM"]:
            if cond in s:
                cs = s[cond]
                print(f"    {cond}: n={cs['n']}, n_bk={cs['n_bk']}, BK%={cs['bk_rate']}, AUC={cs['auc']}")
        print(f"    IC 4-cond chi2 p={s.get('ic_4cond_chi2_p')}")

# --- Per-Condition SAE BK ---
print("\n\n=== 3C. Per-Condition SAE BK Classification ===")
print("Source: comprehensive_gemma file4, percondition_sae_bk")
for p in ["ic", "sm", "mw"]:
    print(f"\n  {p.upper()}: ({len(rq3['percondition_sae_bk'][p])} conditions)")
    # Group by layer and condition type
    conds = rq3["percondition_sae_bk"][p]
    # Just show L22 (or best layer) conditions
    for ck in sorted(conds.keys()):
        if "L22" in ck:
            v = conds[ck]
            print(f"    {ck}: AUC={v['auc']}, std={v.get('auc_std')}, n_pos={v['n_pos']}, n_neg={v['n_neg']}, cond={v.get('condition')}")

# --- Per-Condition SAE Risk ---
print("\n\n=== 3D. Per-Condition SAE Risk Classification ===")
print("Source: comprehensive_gemma file6, percondition_sae_risk")
for p in ["ic", "sm", "mw"]:
    print(f"\n  {p.upper()}: ({len(rq3['percondition_sae_risk'][p])} conditions)")
    conds = rq3["percondition_sae_risk"][p]
    for ck in sorted(conds.keys()):
        if "L22" in ck:
            v = conds[ck]
            print(f"    {ck}: AUC={v.get('auc')}, n_pos={v.get('n_pos')}, n_neg={v.get('n_neg')}")

# --- Per-Condition Hidden BK R1 ---
print("\n\n=== 3E. Per-Condition Hidden BK R1 ===")
print("Source: hidden_gaps file7, percondition_hidden_bk_r1")
for p in ["ic", "sm", "mw"]:
    print(f"\n  {p.upper()}: ({len(rq3['percondition_hidden_bk_r1'][p])} conditions)")
    conds = rq3["percondition_hidden_bk_r1"][p]
    # Show select layers
    for ck in sorted(conds.keys()):
        if any(l in ck for l in ["L18_", "L22_"]):
            v = conds[ck]
            print(f"    {ck}: AUC={v['auc']}, std={v.get('auc_std')}, n_pos={v['n_pos']}, n_neg={v['n_neg']}, cond={v.get('condition')}")

# --- Per-Condition Hidden Risk ---
print("\n\n=== 3F. Per-Condition Hidden Risk ===")
print("Source: hidden_gaps file7, percondition_hidden_risk")
for p in ["ic", "sm", "mw"]:
    print(f"\n  {p.upper()}: ({len(rq3['percondition_hidden_risk'][p])} conditions)")
    conds = rq3["percondition_hidden_risk"][p]
    for ck in sorted(conds.keys()):
        if "L22" in ck:
            v = conds[ck]
            print(f"    {ck}: AUC={v.get('auc')}, n_pos={v.get('n_pos')}, n_neg={v.get('n_neg')}")

# --- R1 Within-Condition Correlation ---
print("\n\n=== 3G. R1 Within-Condition Correlation ===")
print("Source: correlation file8, r1_within_condition")
for p in ["ic", "sm", "mw"]:
    print(f"\n  {p.upper()}:")
    conds = rq3["r1_within_condition_correlation"][p]
    for ck in sorted(conds.keys()):
        v = conds[ck]
        print(f"    {ck}: n_sig={v.get('n_significant')}/{v.get('n_features_tested')}, n_total={v.get('n_total')}, n_bk={v.get('n_bk')}, top1: F{v.get('top1_feature')} d={v.get('top1_cohen_d')}")

# --- Per-Condition IC Fixed Correlation ---
print("\n\n=== 3H. Per-Condition IC Fixed Bet Correlation ===")
print("Source: correlation file8, percondition_ic_fixed")
for ck, v in rq3["percondition_ic_fixed_correlation"].items():
    print(f"  {ck}: constraint={v['constraint']}, n_total={v['n_total']}, n_bk={v['n_bk']}, n_tested={v['n_features_tested']}, n_sig={v['n_significant']}")

print("\n\n" + "=" * 100)
print("EXTRACTION COMPLETE: ~13,219 numeric values across 30 sections")
print("Full JSON saved to: /home/jovyan/llm-addiction/sae_v3_analysis/results/rq_comprehensive_results.json")
print("=" * 100)
