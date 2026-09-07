import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from configs.ablation_config import GAMOE_OUT_ROOT, PAIRED_CELLTYPES

def plot_weighted_covariates():
    preds_path = os.path.join(GAMOE_OUT_ROOT, "all_predictions_umap4d_kfold_with_covs.csv")
    w_path = os.path.join(GAMOE_OUT_ROOT, "error_weighted_FAE_by_ablation.csv")
    if not (os.path.exists(preds_path) and os.path.exists(w_path)): return

    df_raw = pd.read_csv(preds_path)
    df_w = pd.read_csv(w_path)
    sns.set_theme(style="white")

    out_dir = os.path.join(GAMOE_OUT_ROOT, "covariate_plots")
    os.makedirs(out_dir, exist_ok=True)

    for mode in df_w["ablation_mode"].unique():
        for feat in ["freq", "amp"]:
            sub_w = df_w[(df_w["ablation_mode"] == mode) & (df_w["feature"] == feat)]
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)

            for idx, inp in enumerate(["frTable", "weights"]):
                sw = sub_w[sub_w["input_type"] == inp]
                if sw.empty: continue
                sns.boxplot(data=sw, x="feature_set", y="weighted_mean", ax=axes[idx], showfliers=False)
                axes[idx].set_title(f"Error-weighted {feat} ({mode}) — {inp}")
                axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha="right")

                # Overlay dashed unweighted true mean segments
                raw_sub = df_raw[(df_raw["ablation_mode"] == mode) & (df_raw["input_type"] == inp)]
                labels = [t.get_text() for t in axes[idx].get_xticklabels()]
                for fs in labels:
                    real_v = raw_sub[raw_sub["feature_set"] == fs][feat].dropna().values
                    if len(real_v) > 0:
                        xpos = labels.index(fs)
                        axes[idx].hlines(real_v.mean(), xpos - 0.3, xpos + 0.3, colors="k", linestyles="--", linewidth=1.2)

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"weighted_{feat}_{mode}.png"), dpi=300)
            plt.close()

if __name__ == "__main__":
    plot_weighted_covariates()
    print("Covariate plots generated.")