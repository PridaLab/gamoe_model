import os

PROJECT_DIR = "/home/melisamc/Documents/ripple_composition"
SPLIT_ROOT = os.path.join(PROJECT_DIR, "experiments_filt_ripples/splits_priors")
GAMOE_OUT_ROOT = os.path.join(PROJECT_DIR, "experiments_filt_ripples/gamoe_allAblations_FRandWeights")
KNN_OUT_ROOT = os.path.join(PROJECT_DIR, "experiments_filt_ripples/knn_allAblations_FRandWeights")

# Full cell-type feature set (11 types)
FULL_FEATS = [
    "DeepPyr", "DeepPV", "DeepBist", "SupPyr", "SupPV", "SupBist",
    "DeepCCK", "DeepAxoAxonic", "SupCCK", "SupAxoAxonic", "OriensInt"
]

CELLTYPE_PAIRS = {
    "Pyr":       ["DeepPyr", "SupPyr"],
    "PV":        ["DeepPV", "SupPV"],
    "Bist":      ["DeepBist", "SupBist"],
    "CCK":       ["DeepCCK", "SupCCK"],
    "AxoAxonic": ["DeepAxoAxonic", "SupAxoAxonic"],
    "OriensInt": ["OriensInt"],
}

CELLTYPES_WITH_DEPTH = ["Pyr", "PV", "Bist", "CCK", "AxoAxonic"]
DEEP_TYPES = ["DeepPyr", "DeepPV", "DeepBist", "DeepCCK", "DeepAxoAxonic", "OriensInt"]
SUP_TYPES  = ["SupPyr", "SupPV", "SupBist", "SupCCK", "SupAxoAxonic"]

# Canonical ablation conditions
ABLATION_CONDS = {
    "All":             [],
    "Only_Pyr":        [f for f in FULL_FEATS if f not in CELLTYPE_PAIRS["Pyr"]],
    "No_Int":          [f for f in FULL_FEATS if f not in CELLTYPE_PAIRS["Pyr"]],
    "No_Pyr":          CELLTYPE_PAIRS["Pyr"],
    "No_PV":           CELLTYPE_PAIRS["PV"],
    "No_Bist":         CELLTYPE_PAIRS["Bist"],
    "No_CCK":          CELLTYPE_PAIRS["CCK"],
    "No_AxoAxonic":    CELLTYPE_PAIRS["AxoAxonic"],
    "No_OriensInt":    CELLTYPE_PAIRS["OriensInt"],
    "No_Module1_deep": ["DeepPV", "DeepBist"],
    "No_Module1_sup":  ["SupPV",  "SupBist"],
    "No_Module2_deep": ["DeepCCK", "DeepAxoAxonic", "OriensInt"],
    "No_Module2_sup":  ["SupCCK",  "SupAxoAxonic"],
    "No_Module1":      ["DeepPV", "SupPV", "DeepBist", "SupBist"],
    "No_Module2":      ["DeepCCK", "SupCCK", "DeepAxoAxonic", "SupAxoAxonic", "OriensInt"],
    "No_deep_withPyr": [f for f in DEEP_TYPES if f != "DeepPyr"],
    "No_deep_noPyr":   DEEP_TYPES,
    "No_sup_withPyr":  [f for f in SUP_TYPES if f != "SupPyr"],
    "No_sup_noPyr":    SUP_TYPES,
}

# Per-celltype depth ablations
for ct in CELLTYPES_WITH_DEPTH:
    d_feat, s_feat = CELLTYPE_PAIRS[ct]
    ABLATION_CONDS[f"No_{ct}_deep"] = [d_feat]
    ABLATION_CONDS[f"No_{ct}_sup"]  = [s_feat]

ABLATION_CONDS["No_Int_deep"] = ["SupPyr", "DeepPV", "DeepBist", "DeepCCK", "DeepAxoAxonic", "OriensInt", "SupPV", "SupBist", "SupCCK", "SupAxoAxonic"]
ABLATION_CONDS["No_Int_sup"]  = ["DeepPyr", "DeepPV", "DeepBist", "DeepCCK", "DeepAxoAxonic", "OriensInt", "SupPV", "SupBist", "SupCCK", "SupAxoAxonic"]

# Analysis feature groupings
GROUP5_FEATURES = ["All", "No_Pyr", "Only_Pyr", "No_PV", "No_Bist", "No_CCK", "No_AxoAxonic", "No_OriensInt"]
GROUP6_FEATURES = ["No_Module1", "No_Module2", "No_Module1_deep", "No_Module1_sup"]
GROUP7_FEATURES = ["No_deep_withPyr", "No_sup_withPyr", "No_deep_noPyr", "No_sup_noPyr"]
PAIRED_CELLTYPES = ["No_Int", "Pyr", "PV", "Bist", "CCK", "AxoAxonic"]

# Ablation paradigms
GAMOE_ABLAT_MODES = ["gate", "shuffle"]
KNN_ABLAT_MODES   = ["input_drop", "column_shuffle"]