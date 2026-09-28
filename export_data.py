import gzip
import json
import os
import numpy as np
import pandas as pd

# Paths to predictions
possible_paths = [
    "/home/melisamc/Documents/ripple_composition/experiments_filt_ripples/gamoe_CelltypeDeepSup/all_predictions_umap4d_kfold_with_covs.csv",
    os.path.join(
        "..",
        "experiments_filt_ripples",
        "gamoe_CelltypeDeepSup",
        "all_predictions_umap4d_kfold_with_covs.csv",
    ),
    "experiments_filt_ripples/gamoe_CelltypeDeepSup/all_predictions_umap4d_kfold_with_covs.csv",
    "all_predictions_umap4d_kfold_with_covs.csv",
]

csv_path = None
for p in possible_paths:
  if os.path.exists(p):
    csv_path = p
    break

if csv_path is None:
  raise FileNotFoundError(
      "Could not find all_predictions_umap4d_kfold_with_covs.csv"
  )

print(f"Loading predictions from: {csv_path}")
df = pd.read_csv(csv_path)

os.makedirs("docs", exist_ok=True)

# Filter to Fold 1 for the web app to maintain a clean 1-to-1 event mapping
if "fold" in df.columns:
  df_fold = df[df["fold"] == 1].copy()
  if df_fold.empty:
    df_fold = df.copy()
else:
  df_fold = df.copy()

# Extract distinct base ripples from baseline unablated set ("All", "gate")
baseline_df = df_fold[
    (df_fold["feature_set"] == "All") & (df_fold["ablation_mode"] == "gate")
].copy()
if baseline_df.empty:
  # Fallback to first available condition
  baseline_df = df_fold[
      df_fold["feature_set"] == df_fold["feature_set"].iloc[0]
  ].copy()

baseline_df = baseline_df.reset_index(drop=True)
n_ripples = len(baseline_df)

# Subsample if large to keep browser fast and responsive
MAX_RIPPLES = 4000
if n_ripples > MAX_RIPPLES:
  keep_indices = np.random.choice(n_ripples, MAX_RIPPLES, replace=False)
  keep_indices.sort()
else:
  keep_indices = np.arange(n_ripples)

# 1. Base Events: True coordinates and physical covariates
ripples_base = []
for idx in keep_indices:
  row = baseline_df.iloc[idx]
  ripples_base.append({
      "id": int(idx),
      "u1_true": float(row["UMAP1_true"]),
      "u2_true": float(row["UMAP2_true"]),
      "u1_pred_full": float(row["UMAP1_pred"]),
      "u2_pred_full": float(row["UMAP2_pred"]),
      "freq": float(row["freq"]),
      "amp": float(row["amp"]),
      "err_full": float(row["error"]),
  })

# 2. Ablations map: condition -> mode -> list of {u1_pred, u2_pred, err}
conditions = sorted(df_fold["feature_set"].unique().tolist())
modes = ["gate", "shuffle"]

ablations_map = {}
for cond in conditions:
  ablations_map[cond] = {}
  for m in modes:
    sub = df_fold[
        (df_fold["feature_set"] == cond) & (df_fold["ablation_mode"] == m)
    ].reset_index(drop=True)
    if len(sub) >= n_ripples:
      abl_records = []
      for idx in keep_indices:
        r = sub.iloc[idx]
        abl_records.append({
            "u1_pred": float(r["UMAP1_pred"]),
            "u2_pred": float(r["UMAP2_pred"]),
            "err": float(r["error"]),
        })
      ablations_map[cond][m] = abl_records

payload = {
    "conditions": conditions,
    "modes": modes,
    "ripples": ripples_base,
    "ablations": ablations_map,
}

output_json = "docs/data.json"
with open(output_json, "w") as f:
  json.dump(payload, f)

print(f"Exported {len(ripples_base)} events successfully to {output_json}!")
