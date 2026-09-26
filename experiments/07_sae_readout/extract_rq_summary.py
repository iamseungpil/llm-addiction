"""
Extract ALL numerical results organized by 3 Research Questions.
Outputs a comprehensive Python dict to stdout.
"""

import json
import os
import sys
from pathlib import Path
from pprint import pprint

JSON_DIR = Path("/home/jovyan/llm-addiction/sae_v3_analysis/results/json")

def load(name):
    with open(JSON_DIR / name) as f:
        return json.load(f)

def r(v, d=4):
    """Safe round."""
    if v is None: return None
    if isinstance(v, (int, float)): return round(v, d)
    return v

def best_layer(layers, metric="auc"):
    if not layers: return None
    return max(layers, key=lambda x: x.get(metric, 0))

def top5(layers, metric="auc"):
    s = sorted(layers, key=lambda x: x.get(metric, 0), reverse=True)
    return [(x["layer"], r(x[metric])) for x in s[:5]]

###############################################################################
print("=" * 100)
print("LOADING ALL 9 FILES...")
print("=" * 100)

f1 = load("all_analyses_20260306_091055.json")
f2 = load("extended_analyses_20260306_211214.json")
f3 = load("improved_v4_20260308_032435.json")
f4 = load("comprehensive_gemma_20260309_063511.json")
f5 = load("comprehensive_gemma_20260309_095339.json")
f6 = load("comprehensive_gemma_20260309_182926.json")
f7 = load("hidden_gaps_20260309_181059.json")
f8 = load("correlation_20260309_092256.json")
f9 = load("condition_v2_20260308_151953.json")

print("All files loaded. Keys per file:")
for i, (name, d) in enumerate([(1,f1),(2,f2),(3,f3),(4,f4),(5,f5),(6,f6),(7,f7),(8,f8),(9,f9)], 1):
    print(f"  File {name}: {list(d.keys())}")

###############################################################################
# RQ1: BANKRUPTCY PREDICTION
###############################################################################
print("\n" + "=" * 100)
print("RQ1: WHICH FEATURES/ACTIVATIONS CONSISTENTLY PREDICT BANKRUPTCY?")
print("=" * 100)

# --- 1A. SAE BK Decision-Point (Goal A from file1) ---
print("\n--- 1A. SAE BK Decision-Point Classification (all rounds) ---")
for paradigm in ["ic", "sm", "mw"]:
    layers = f1["goal_a_classification"][paradigm]["sae"]
    b = best_layer(layers)
    print(f"\n  {paradigm.upper()}: n_pos={b['n_pos']}, n_neg={b['n_neg']}")
    print(f"    Best: L{b['layer']}, AUC={r(b['auc'])}, std={r(b['auc_std'])}, F1={r(b['f1'])}, prec={r(b['precision'])}, recall={r(b['recall'])}, n_feat={b.get('n_features')}")
    print(f"    Top5: {top5(layers)}")
    print(f"    All layers AUC range: {r(min(x['auc'] for x in layers))} - {r(max(x['auc'] for x in layers))}")

# --- 1B. SAE R1 Classification (Goal B from file1) ---
print("\n--- 1B. SAE R1 Classification (first round only, balance-controlled) ---")
for paradigm in ["ic", "sm", "mw"]:
    layers = f1["goal_b_r1"][paradigm]["sae"]
    b = best_layer(layers)
    print(f"\n  {paradigm.upper()}: n_pos={b.get('n_pos')}, n_neg={b.get('n_neg')}")
    print(f"    Best: L{b['layer']}, AUC={r(b['auc'])}, std={r(b.get('auc_std',0))}, n_feat={b.get('n_features')}")
    print(f"    Top5: {top5(layers)}")
    print(f"    All layers AUC range: {r(min(x['auc'] for x in layers))} - {r(max(x['auc'] for x in layers))}")

# --- 1C. Balance-Matched SAE BK (Exp2a from file2) ---
print("\n--- 1C. Balance-Matched SAE BK (controls for balance confound) ---")
for paradigm in ["ic", "sm", "mw"]:
    layers = f2["exp2a_balance_matched"][paradigm]
    b = best_layer(layers)
    print(f"\n  {paradigm.upper()}: n_pos={b.get('n_pos')}, n_neg={b.get('n_neg')}")
    print(f"    Best: L{b['layer']}, AUC={r(b['auc'])}, std={r(b.get('auc_std',0))}, n_feat={b.get('n_features')}")
    print(f"    Top5: {top5(layers)}")

# --- 1D. Gemma Balance-Controlled R1 (from file3) ---
print("\n--- 1D. Balance-Controlled R1 SAE (improved v4 from file3) ---")
for paradigm in ["ic", "sm", "mw"]:
    if paradigm in f3.get("gemma_balance_controlled", {}):
        layers = f3["gemma_balance_controlled"][paradigm]
        if isinstance(layers, list):
            b = best_layer(layers)
            print(f"\n  {paradigm.upper()}: n_pos={b.get('n_pos')}, n_neg={b.get('n_neg')}")
            print(f"    Best: L{b['layer']}, AUC={r(b['auc'])}, std={r(b.get('auc_std',0))}")
            print(f"    Top5: {top5(layers)}")
        elif isinstance(layers, dict):
            print(f"\n  {paradigm.upper()}: {layers}")

# --- 1E. R1 Permutation Test (from file3) ---
print("\n--- 1E. R1 Permutation Tests (statistical significance) ---")
for paradigm in ["ic", "sm", "mw"]:
    if paradigm in f3.get("r1_permutation_test", {}):
        vals = f3["r1_permutation_test"][paradigm]
        if isinstance(vals, list):
            for v in vals:
                print(f"  {paradigm.upper()} L{v.get('layer')}: observed_AUC={r(v.get('observed_auc'))}, null_mean={r(v.get('null_mean'))}, null_std={r(v.get('null_std'))}, p={v.get('p_value')}, n_perm={v.get('n_permutations')}")
        elif isinstance(vals, dict):
            print(f"  {paradigm.upper()} L{vals.get('layer')}: observed_AUC={r(vals.get('observed_auc'))}, null_mean={r(vals.get('null_mean'))}, null_std={r(vals.get('null_std'))}, p={vals.get('p_value')}, n_perm={vals.get('n_permutations')}")

# --- 1F. Hidden State BK Classification (from file4) ---
print("\n--- 1F. Hidden State BK Decision-Point Classification ---")
for paradigm in ["ic", "sm", "mw"]:
    if paradigm in f4.get("hidden_bk", {}):
        layers = f4["hidden_bk"][paradigm]
        b = best_layer(layers)
        print(f"\n  {paradigm.upper()}: n_pos={b.get('n_pos')}, n_neg={b.get('n_neg')}")
        print(f"    Best: L{b['layer']}, AUC={r(b['auc'])}, std={r(b.get('auc_std',0))}")
        print(f"    Top5: {top5(layers)}")
        print(f"    All layers AUC range: {r(min(x['auc'] for x in layers))} - {r(max(x['auc'] for x in layers))}")

# --- 1G. Balance-Matched Hidden BK (from file7) ---
print("\n--- 1G. Balance-Matched Hidden State BK ---")
for paradigm in ["ic", "sm", "mw"]:
    if paradigm in f7.get("balance_matched_hidden", {}):
        layers = f7["balance_matched_hidden"][paradigm]
        b = best_layer(layers)
        print(f"\n  {paradigm.upper()}: n_pos={b.get('n_pos')}, n_neg={b.get('n_neg')}")
        print(f"    Best: L{b['layer']}, AUC={r(b['auc'])}")
        print(f"    Top5: {top5(layers)}")

# --- 1H. Risk Preference Classification (from file2 and file4/file6) ---
print("\n--- 1H. SAE Risk Preference Classification ---")
for paradigm in ["ic", "sm", "mw"]:
    if paradigm in f2.get("exp3_risk", {}):
        layers = f2["exp3_risk"][paradigm]
        b = best_layer(layers)
        print(f"\n  {paradigm.upper()} (SAE, file2): n_pos={b.get('n_pos')}, n_neg={b.get('n_neg')}")
        print(f"    Best: L{b['layer']}, AUC={r(b['auc'])}, std={r(b.get('auc_std',0))}")
        print(f"    Top5: {top5(layers)}")

print("\n--- 1I. Hidden State Risk Preference Classification (corrected, file6) ---")
for paradigm in ["ic", "sm", "mw"]:
    if paradigm in f6.get("hidden_risk", {}):
        layers = f6["hidden_risk"][paradigm]
        b = best_layer(layers)
        print(f"\n  {paradigm.upper()}: ")
        print(f"    Best: L{b['layer']}, AUC={r(b['auc'])}, std={r(b.get('auc_std',0))}")
        print(f"    Top5: {top5(layers)}")

# --- 1J. Correlation Analysis (from file8) ---
print("\n--- 1J. Feature-Outcome Correlation Analysis ---")
for corr_type in ["dp_correlation", "r1_correlation"]:
    print(f"\n  === {corr_type} ===")
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm in f8.get(corr_type, {}):
            p_data = f8[corr_type][paradigm]
            if isinstance(p_data, dict):
                print(f"\n  {paradigm.upper()}:")
                for k in ["n_significant", "total_features", "best_layer", "fdr_alpha", "n_pos", "n_neg"]:
                    if k in p_data:
                        print(f"    {k}: {p_data[k]}")
                # Top features
                if "top_features" in p_data and isinstance(p_data["top_features"], list):
                    print(f"    Top 10 features:")
                    for feat in p_data["top_features"][:10]:
                        print(f"      L{feat.get('layer')} F{feat.get('feature_idx')}: corr={r(feat.get('correlation'))}, p={feat.get('p_value')}, d={r(feat.get('cohens_d'))}")
                # Per-layer summary
                if "per_layer" in p_data:
                    sig_layers = [(l.get("layer"), l.get("n_significant")) for l in p_data["per_layer"] if l.get("n_significant", 0) > 0]
                    print(f"    Layers with significant features ({len(sig_layers)}): {sig_layers[:15]}...")
            elif isinstance(p_data, list):
                print(f"  {paradigm.upper()}: {len(p_data)} layer entries")

# --- 1K. Behavioral-SAE Linkage (from file8) ---
print("\n--- 1K. Behavioral-SAE Linkage ---")
bsl = f8.get("behavioral_sae_linkage", {})
pprint(bsl, width=120)

###############################################################################
# RQ2: DOMAIN-INVARIANT REPRESENTATIONS
###############################################################################
print("\n" + "=" * 100)
print("RQ2: ARE THERE DOMAIN-INVARIANT REPRESENTATIONS?")
print("=" * 100)

# --- 2A. Cross-Domain SAE Transfer (Goal C from file1) ---
print("\n--- 2A. Cross-Domain SAE Transfer (Goal C, file1) ---")
goal_c = f1.get("goal_c_cross_domain", {})
for direction, vals in goal_c.items():
    if isinstance(vals, list):
        for v in vals:
            print(f"  {direction} L{v.get('layer')}: AUC={r(v.get('auc'))}, std={r(v.get('auc_std',0))}, n_feat={v.get('n_features')}")
    elif isinstance(vals, dict):
        print(f"  {direction}: AUC={r(vals.get('auc'))}, std={r(vals.get('auc_std',0))}")

# --- 2B. Cross-Domain Bootstrap (from file3) ---
print("\n--- 2B. Cross-Domain Bootstrap (with CIs, file3) ---")
cdb = f3.get("cross_domain_bootstrap", {})
for direction, vals in cdb.items():
    if isinstance(vals, list):
        for v in vals:
            print(f"  {direction} L{v.get('layer')}: mean_AUC={r(v.get('mean_auc'))}, CI=[{r(v.get('ci_lower'))}, {r(v.get('ci_upper'))}], std={r(v.get('std_auc'))}, n_boot={v.get('n_bootstrap')}")
    elif isinstance(vals, dict):
        print(f"  {direction} L{vals.get('layer')}: mean_AUC={r(vals.get('mean_auc'))}, CI=[{r(vals.get('ci_lower'))}, {r(vals.get('ci_upper'))}], std={r(vals.get('std_auc'))}, n_boot={vals.get('n_bootstrap')}")

# --- 2C. Same-Layer Feature Overlap (from file3) ---
print("\n--- 2C. Same-Layer Feature Overlap (top-100 features per paradigm) ---")
slf = f3.get("same_layer_features", {})
for layer_key, vals in sorted(slf.items()):
    if isinstance(vals, dict):
        ic_sm = vals.get("ic_sm_overlap", vals.get("shared_features_ic_sm"))
        ic_mw = vals.get("ic_mw_overlap", vals.get("shared_features_ic_mw"))
        sm_mw = vals.get("sm_mw_overlap", vals.get("shared_features_sm_mw"))
        j_ic_sm = vals.get("ic_sm_jaccard")
        j_ic_mw = vals.get("ic_mw_jaccard")
        j_sm_mw = vals.get("sm_mw_jaccard")
        print(f"  {layer_key}: IC-SM overlap={ic_sm} (J={r(j_ic_sm)}), IC-MW overlap={ic_mw} (J={r(j_ic_mw)}), SM-MW overlap={sm_mw} (J={r(j_sm_mw)})")
        # Print actual shared features if available
        for pair in ["ic_sm", "ic_mw", "sm_mw"]:
            shared_key = f"shared_features_{pair}"
            if shared_key in vals and isinstance(vals[shared_key], list) and len(vals[shared_key]) > 0:
                print(f"    Shared {pair}: {vals[shared_key]}")

# --- 2D. Cross-Paradigm Feature Overlap from Correlation (from file8) ---
print("\n--- 2D. Cross-Paradigm Feature Overlap from Correlation Analysis (file8) ---")
cp_overlap = f8.get("cross_paradigm_overlap", {})
pprint(cp_overlap, width=120)

# --- 2E. Hidden State Cross-Domain Transfer (from file7) ---
print("\n--- 2E. Hidden State Cross-Domain Transfer (file7) ---")
hcd = f7.get("hidden_cross_domain", {})
for direction, vals in hcd.items():
    if isinstance(vals, list):
        for v in vals:
            print(f"  {direction} L{v.get('layer')}: mean_AUC={r(v.get('mean_auc'))}, CI=[{r(v.get('ci_lower'))}, {r(v.get('ci_upper'))}], std={r(v.get('std_auc'))}")
    elif isinstance(vals, dict):
        print(f"  {direction} L{vals.get('layer')}: mean_AUC={r(vals.get('mean_auc'))}, CI=[{r(vals.get('ci_lower'))}, {r(vals.get('ci_upper'))}]")

###############################################################################
# RQ3: CONDITION DIFFERENCES
###############################################################################
print("\n" + "=" * 100)
print("RQ3: DO REPRESENTATIONS DIFFER BY CONDITIONS?")
print("=" * 100)

# --- 3A. Condition Encoding (Exp4 from file2) ---
print("\n--- 3A. SAE Condition Encoding (multi-class classification, file2) ---")
exp4 = f2.get("exp4_condition_encoding", {})
for paradigm in ["ic", "sm", "mw"]:
    if paradigm not in exp4:
        continue
    cond_data = exp4[paradigm]
    if isinstance(cond_data, dict):
        for sub_key, sub_val in cond_data.items():
            if isinstance(sub_val, list):
                b = max(sub_val, key=lambda x: x.get("auc", x.get("accuracy", 0)))
                print(f"\n  {paradigm.upper()} {sub_key}:")
                print(f"    Best: L{b.get('layer')}, AUC={r(b.get('auc'))}, acc={r(b.get('accuracy'))}, n_classes={b.get('n_classes')}, chance={r(b.get('chance'))}")
                t5 = sorted(sub_val, key=lambda x: x.get("auc", x.get("accuracy", 0)), reverse=True)[:5]
                print(f"    Top5: {[(x.get('layer'), r(x.get('auc', x.get('accuracy')))) for x in t5]}")
    elif isinstance(cond_data, list):
        b = max(cond_data, key=lambda x: x.get("auc", x.get("accuracy", 0)))
        print(f"\n  {paradigm.upper()}:")
        print(f"    Best: L{b.get('layer')}, AUC={r(b.get('auc'))}, acc={r(b.get('accuracy'))}, n_classes={b.get('n_classes')}")
        t5 = sorted(cond_data, key=lambda x: x.get("auc", x.get("accuracy", 0)), reverse=True)[:5]
        print(f"    Top5: {[(x.get('layer'), r(x.get('auc', x.get('accuracy')))) for x in t5]}")
        # Print ALL layers for this
        print(f"    All layers:")
        for item in sorted(cond_data, key=lambda x: x.get("layer", 0)):
            print(f"      L{item.get('layer')}: AUC={r(item.get('auc'))}, acc={r(item.get('accuracy'))}, std={r(item.get('accuracy_std', item.get('auc_std')))}")

# --- 3B. Per-Condition SAE BK (from file4) ---
print("\n--- 3B. Per-Condition SAE BK Classification (file4) ---")
pc_sae_bk = f4.get("percondition_sae_bk", {})
for paradigm in ["ic", "sm", "mw"]:
    if paradigm not in pc_sae_bk:
        continue
    print(f"\n  {paradigm.upper()}:")
    cond_data = pc_sae_bk[paradigm]
    if isinstance(cond_data, dict):
        for cond_key in sorted(cond_data.keys()):
            layers = cond_data[cond_key]
            if isinstance(layers, list) and len(layers) > 0:
                b = best_layer(layers)
                print(f"    {cond_key}: Best L{b['layer']}, AUC={r(b['auc'])}, n_pos={b.get('n_pos')}, n_neg={b.get('n_neg')}")
                print(f"      Top5: {top5(layers)}")

# --- 3C. Per-Condition SAE Risk (from file6 corrected) ---
print("\n--- 3C. Per-Condition SAE Risk Classification (file6 corrected) ---")
pc_sae_risk = f6.get("percondition_sae_risk", {})
for paradigm in ["ic", "sm", "mw"]:
    if paradigm not in pc_sae_risk:
        continue
    print(f"\n  {paradigm.upper()}:")
    cond_data = pc_sae_risk[paradigm]
    if isinstance(cond_data, dict):
        for cond_key in sorted(cond_data.keys()):
            layers = cond_data[cond_key]
            if isinstance(layers, list) and len(layers) > 0:
                b = best_layer(layers)
                print(f"    {cond_key}: Best L{b['layer']}, AUC={r(b['auc'])}, n_pos={b.get('n_pos')}, n_neg={b.get('n_neg')}")

# --- 3D. Per-Condition Hidden BK (from file5 detailed) ---
print("\n--- 3D. Per-Condition Hidden State BK (file5 detailed) ---")
pc_hbk = f5.get("percondition_hidden_bk", {})
for paradigm in ["ic", "sm", "mw"]:
    if paradigm not in pc_hbk:
        continue
    print(f"\n  {paradigm.upper()}:")
    cond_data = pc_hbk[paradigm]
    if isinstance(cond_data, dict):
        for cond_key in sorted(cond_data.keys()):
            layers = cond_data[cond_key]
            if isinstance(layers, list) and len(layers) > 0:
                b = best_layer(layers)
                print(f"    {cond_key}: Best L{b['layer']}, AUC={r(b['auc'])}, n_pos={b.get('n_pos')}, n_neg={b.get('n_neg')}")
                print(f"      Top5: {top5(layers)}")

# --- 3E. Per-Condition Hidden BK R1 (from file7) ---
print("\n--- 3E. Per-Condition Hidden BK R1 (file7) ---")
pc_hbk_r1 = f7.get("percondition_hidden_bk_r1", {})
for paradigm in ["ic", "sm", "mw"]:
    if paradigm not in pc_hbk_r1:
        continue
    print(f"\n  {paradigm.upper()}:")
    cond_data = pc_hbk_r1[paradigm]
    if isinstance(cond_data, dict):
        for cond_key in sorted(cond_data.keys()):
            layers = cond_data[cond_key]
            if isinstance(layers, list) and len(layers) > 0:
                b = best_layer(layers)
                print(f"    {cond_key}: Best L{b['layer']}, AUC={r(b['auc'])}, n_pos={b.get('n_pos')}, n_neg={b.get('n_neg')}")

# --- 3F. Per-Condition Hidden Risk (from file7) ---
print("\n--- 3F. Per-Condition Hidden Risk (file7) ---")
pc_h_risk = f7.get("percondition_hidden_risk", {})
for paradigm in ["ic", "sm", "mw"]:
    if paradigm not in pc_h_risk:
        continue
    print(f"\n  {paradigm.upper()}:")
    cond_data = pc_h_risk[paradigm]
    if isinstance(cond_data, dict):
        for cond_key in sorted(cond_data.keys()):
            layers = cond_data[cond_key]
            if isinstance(layers, list) and len(layers) > 0:
                b = best_layer(layers)
                print(f"    {cond_key}: Best L{b['layer']}, AUC={r(b['auc'])}, n_pos={b.get('n_pos')}, n_neg={b.get('n_neg')}")

# --- 3G. Condition V2 with marginal effects (from file9) ---
print("\n--- 3G. Per-Condition SAE BK with Component Marginal Effects (file9) ---")
pprint(f9, width=120)

# --- 3H. Hidden Risk corrected per-paradigm (from file6) ---
print("\n--- 3H. Hidden Risk Corrected - Per Paradigm Summary (file6) ---")
for paradigm in ["ic", "sm", "mw"]:
    if paradigm in f6.get("hidden_risk", {}):
        layers = f6["hidden_risk"][paradigm]
        b = best_layer(layers)
        print(f"  {paradigm.upper()}: Best L{b['layer']}, AUC={r(b['auc'])}, std={r(b.get('auc_std',0))}")
        print(f"    Top5: {top5(layers)}")
        print(f"    All layers: {[(x['layer'], r(x['auc'])) for x in sorted(layers, key=lambda x: x['layer'])]}")

print("\n" + "=" * 100)
print("EXTRACTION COMPLETE")
print("=" * 100)
