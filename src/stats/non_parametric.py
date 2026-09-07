import itertools
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, ttest_ind
from src.utils.metrics import bonferroni_adjust, p_to_stars

def safe_mwu(x, y):
    x = np.asarray(x, dtype=float)[np.isfinite(x)]
    y = np.asarray(y, dtype=float)[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, len(x), len(y)
    u, p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
    return float(u), float(p), int(len(x)), int(len(y))

def safe_ttest_ind(x, y):
    x = np.asarray(x, dtype=float)[np.isfinite(x)]
    y = np.asarray(y, dtype=float)[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, len(x), len(y)
    t, p = ttest_ind(x, y, equal_var=False)
    return float(t), float(p), int(len(x)), int(len(y))

def pairwise_rank_sum_bonferroni(groups_dict):
    labels = [k for k, v in groups_dict.items() if len(np.asarray(v)) >= 2]
    rows = []
    for a, b in itertools.combinations(labels, 2):
        u, p, n1, n2 = safe_mwu(groups_dict[a], groups_dict[b])
        rows.append({"group1": a, "group2": b, "U": u, "p_raw": p, "n1": n1, "n2": n2})
    ph = pd.DataFrame(rows)
    if not ph.empty:
        ph["p_bonf"] = bonferroni_adjust(ph["p_raw"].values)
        ph["stars"]  = ph["p_bonf"].apply(p_to_stars)
    return ph