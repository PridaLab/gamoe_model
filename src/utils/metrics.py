import numpy as np
import pandas as pd

def corr_r2_scores(U_true, U_pred, eps=1e-12):
    U_true, U_pred = np.asarray(U_true), np.asarray(U_pred)
    D = U_true.shape[1]
    r_dims, r2_dims = [], []
    for d in range(D):
        x, y = U_true[:, d], U_pred[:, d]
        sx, sy = x.std(), y.std()
        r = 0.0 if (sx < eps or sy < eps) else np.corrcoef(x, y)[0, 1]
        r = 0.0 if np.isnan(r) else r
        r_dims.append(float(r))
        r2_dims.append(float(r**2))
    vars_ = U_true.var(axis=0)
    w = vars_ / (vars_.sum() + eps)
    r2_global = float((w * np.array(r2_dims)).sum())
    return r2_global, r2_dims, r_dims

def safe_weighted_mean(values, weights):
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    s = np.nansum(weights)
    return np.nan if (not np.isfinite(s) or s <= 0) else float(np.nansum(values * weights) / s)

def p_to_stars(p):
    if not np.isfinite(p): return "na"
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 5e-2: return "*"
    return "ns"

def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for k, idx in enumerate(order):
        val = min(1.0, pvals[idx] * (m - k))
        val = max(val, prev)
        adj[idx] = val
        prev = val
    return adj

def bonferroni_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    mask = np.isfinite(pvals)
    m = int(mask.sum())
    if m > 0:
        out[mask] = np.minimum(1.0, pvals[mask] * float(m))
    return out

def compute_error_weighted_covariates(df, covariate_cols=["freq", "amp", "entropy"]):
    records = []
    group_cols = [c for c in ["input_type", "ablation_mode", "feature_set", "fold", "shuffle_type"] if c in df.columns]
    for key, sub in df.groupby(group_cols):
        sub = sub.dropna(subset=["error"])
        if sub.empty:
            continue
        w = np.clip(sub["error"].values.astype(float), 0, None)
        for feat in covariate_cols:
            if feat not in sub.columns:
                continue
            x = sub[feat].values.astype(float)
            valid = ~np.isnan(x)
            x_v, w_v = x[valid], w[valid]
            if x_v.size == 0:
                continue
            wmean = np.average(x_v, weights=w_v) if w_v.sum() > 0 else x_v.mean()
            row = dict(zip(group_cols, key))
            row.update({
                "feature": feat, "weighted_mean": float(wmean),
                "weight_sum": float(w_v.sum()), "n_examples": int(len(x_v))
            })
            records.append(row)
    return records