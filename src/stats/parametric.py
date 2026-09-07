import itertools
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind, ttest_rel
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def anova_with_posthoc(df_in, feature_sets, value_col, group_col="feature_set"):
    groups = []
    for fs in feature_sets:
        vals = df_in[df_in[group_col] == fs][value_col].dropna().values
        if len(vals) > 0:
            groups.append((fs, vals))

    if len(groups) < 2:
        return "Not enough groups for ANOVA."

    lines = []
    f_stat, p_anova = f_oneway(*[g[1] for g in groups])
    lines.append(f"ANOVA on {value_col}: F = {f_stat:.4g}, p = {p_anova:.4g}\n")
    m = len(groups)
    n_pairs = m * (m - 1) / 2
    alpha_corr = 0.05 / max(1, n_pairs)
    lines.append(f"Bonferroni alpha = {alpha_corr:.4g}\n")

    for (n1, v1), (n2, v2) in itertools.combinations(groups, 2):
        t_stat, p_raw = ttest_ind(v1, v2, equal_var=False)
        p_corr = p_raw * n_pairs
        lines.append(f"{n1} vs {n2}: t = {t_stat:.4g}, p_raw = {p_raw:.4g}, p_corr = {p_corr:.4g}")

    return "\n".join(lines)

def run_two_way_anova_paired(df_in, paired_cts, ycol):
    d = df_in[(df_in["celltype"].isin(paired_cts)) & (df_in["depth"].isin(["deep", "sup"]))].copy()
    d = d.dropna(subset=[ycol, "celltype", "depth"])
    d["celltype"] = d["celltype"].astype("category").cat.remove_unused_categories()
    d["depth"] = d["depth"].astype("category").cat.remove_unused_categories()
    model = smf.ols(f"{ycol} ~ C(celltype) * C(depth)", data=d).fit()
    aov = anova_lm(model, typ=2)
    p_cell = float(aov.loc["C(celltype)", "PR(>F)"]) if "C(celltype)" in aov.index else np.nan
    p_depth = float(aov.loc["C(depth)", "PR(>F)"]) if "C(depth)" in aov.index else np.nan
    p_int = float(aov.loc["C(celltype):C(depth)", "PR(>F)"]) if "C(celltype):C(depth)" in aov.index else np.nan
    return aov, {"celltype": p_cell, "depth": p_depth, "interaction": p_int}