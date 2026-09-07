import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, kruskal

from configs.ablation_config import (
    GROUP5_FEATURES, GROUP6_FEATURES, GROUP7_FEATURES, PAIRED_CELLTYPES,
    GAMOE_OUT_ROOT, KNN_OUT_ROOT
)
from src.stats.parametric import anova_with_posthoc, run_two_way_anova_paired
from src.stats.non_parametric import pairwise_rank_sum_bonferroni, safe_mwu, safe_ttest_ind
from src.utils.data_loaders import attach_celltype_depth
from src.utils.metrics import bonferroni_adjust, p_to_stars

def run_gamoe_stats():
    m_path = os.path.join(GAMOE_OUT_ROOT, "all_metrics_kfold.csv")
    if not os.path.exists(m_path):
        return

    dfm = pd.read_csv(m_path)
    out_dir = os.path.join(GAMOE_OUT_ROOT, "statistical_reports")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Parametric ANOVA & Post-hoc for groups 5, 6, 7 across all modes
    groups = {"group5_cellTypes": GROUP5_FEATURES, "group6_modules": GROUP6_FEATURES, "group7_depth": GROUP7_FEATURES}
    for g_name, f_list in groups.items():
        for inp in dfm["input_type"].unique():
            for mode in dfm["ablation_mode"].unique():
                sub = dfm[(dfm["input_type"] == inp) & (dfm["ablation_mode"] == mode) & (dfm["feature_set"].isin(f_list))]
                if sub.empty: continue
                r2_txt = anova_with_posthoc(sub, f_list, "R2_global")
                err_txt = anova_with_posthoc(sub, f_list, "mean_error")
                with open(os.path.join(out_dir, f"ANOVA_{g_name}_{inp}_{mode}.txt"), "w") as f:
                    f.write(f"--- R2_global ---\n{r2_txt}\n\n--- mean_error ---\n{err_txt}\n")

    # 2. Deep/Sup Two-way ANOVA and paired tests
    df_ds = attach_celltype_depth(dfm, PAIRED_CELLTYPES)
    with open(os.path.join(out_dir, "GAMoE_TwoWay_ANOVA_DeepSup.txt"), "w") as f:
        for mode in df_ds["ablation_mode"].unique():
            sub = df_ds[df_ds["ablation_mode"] == mode]
            aov, pdict = run_two_way_anova_paired(sub, PAIRED_CELLTYPES, "R2_global")
            f.write(f"=== Mode: {mode} (R2_global) ===\n{aov.to_string()}\n\n")

    # 3. Non-parametric Kruskal-Wallis + Mann-Whitney U suite
    with open(os.path.join(out_dir, "GAMoE_NonParametric_KW_MWU.txt"), "w") as f:
        for mode in df_ds["ablation_mode"].unique():
            for inp in df_ds["input_type"].unique():
                d0 = df_ds[(df_ds["ablation_mode"] == mode) & (df_ds["input_type"] == inp)]
                if d0.empty: continue
                # Deep KW
                deep_sub = d0[d0["depth_group"] == "deep"]
                d_dict = {ct: deep_sub[deep_sub["celltype"] == ct]["R2_global"].dropna().values for ct in deep_sub["celltype"].unique()}
                d_dict = {k: v for k, v in d_dict.items() if len(v) >= 2}
                if len(d_dict) >= 2:
                    h, p_kw = kruskal(*d_dict.values())
                    f.write(f"\nMode: {mode} | Input: {inp} | Deep KW H={h:.4g}, p={p_kw:.4g}\n")
                    ph = pairwise_rank_sum_bonferroni(d_dict)
                    f.write(ph.to_string() + "\n")

def run_knn_stats():
    m_path = os.path.join(KNN_OUT_ROOT, "all_metrics_kfold_knn.csv")
    if not os.path.exists(m_path):
        return
    dfm = pd.read_csv(m_path)
    out_dir = os.path.join(KNN_OUT_ROOT, "statistical_reports")
    os.makedirs(out_dir, exist_ok=True)

    # Paired t-tests: True vs Shuffled
    with open(os.path.join(out_dir, "KNN_True_vs_Shuffled_ttests.txt"), "w") as f:
        for inp in dfm["input_type"].unique():
            sub = dfm[(dfm["input_type"] == inp) & (dfm["ablation_mode"] == "input_drop")]
            f.write(f"\n=== Input: {inp} (True vs Shuffled) ===\n")
            for fs in sub["feature_set"].unique():
                tr = sub[(sub["feature_set"] == fs) & (sub["shuffle_type"] == "true")]
                sh = sub[(sub["feature_set"] == fs) & (sub["shuffle_type"] == "shuffled")]
                cf = sorted(set(tr["fold"]).intersection(set(sh["fold"])))
                if len(cf) >= 2:
                    r2_tr = [tr[tr["fold"] == k]["R2_global"].iloc[0] for k in cf]
                    r2_sh = [sh[sh["fold"] == k]["R2_global"].iloc[0] for k in cf]
                    t, p = ttest_rel(r2_tr, r2_sh)
                    f.write(f"{fs}: R2 t={t:.4g}, p={p:.4g}\n")

if __name__ == "__main__":
    run_gamoe_stats()
    run_knn_stats()
    print("Statistical reports written successfully.")