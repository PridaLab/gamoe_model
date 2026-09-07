import os, re
import numpy as np
import pandas as pd
from scipy.io import loadmat

pat_ds = re.compile(r"^No_(?P<celltype>[^_]+)_(?P<depth>deep|sup)$")

def load_single_cov_vector(fold_dir, base_candidates):
    for base in base_candidates:
        p_csv = os.path.join(fold_dir, f"{base}.csv")
        if os.path.exists(p_csv):
            return pd.read_csv(p_csv).iloc[:, 0].to_numpy()
        p_mat = os.path.join(fold_dir, f"{base}.mat")
        if os.path.exists(p_mat):
            m = loadmat(p_mat)
            keys = [k for k in m.keys() if not k.startswith("__")]
            return np.asarray(m[keys[0]]).squeeze()
    raise FileNotFoundError(f"Covariate file not found in {fold_dir} for {base_candidates}")

def attach_celltype_depth(df_in, paired_cts, feature_set_col="feature_set"):
    df = df_in.copy()
    parsed = df[feature_set_col].astype(str).str.extract(pat_ds)
    df["celltype"] = parsed["celltype"]
    df["depth"] = parsed["depth"]
    df.loc[df["celltype"] == "Int", "celltype"] = "No_Int"
    df.loc[df[feature_set_col].astype(str) == "All", ["celltype", "depth"]] = ["All", "all"]
    df.loc[df[feature_set_col].astype(str) == "No_OriensInt", ["celltype", "depth"]] = ["OriensInt", "single"]
    df = df[df[feature_set_col].astype(str) != "No_Int"].copy()
    df["depth_group"] = df["depth"]
    df.loc[df["celltype"] == "OriensInt", "depth_group"] = "deep"
    keep = (
        (df["celltype"].isin(paired_cts) & df["depth"].isin(["deep", "sup"])) |
        (df["celltype"].isin(["All", "OriensInt"]) & df["depth"].isin(["all", "single"]))
    )
    return df[keep].copy()