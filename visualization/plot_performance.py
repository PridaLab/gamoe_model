import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from configs.ablation_config import (
    ABLATION_CONDS, GROUP5_FEATURES, GROUP6_FEATURES, GROUP7_FEATURES,
    GAMOE_OUT_ROOT, KNN_OUT_ROOT
)

sns.set_theme(style="whitegrid")

def plot_gamoe_r2_summary():
    m_path = os.path.join(GAMOE_OUT_ROOT, "all_metrics_kfold.csv")
    if not os.path.exists(m_path): return
    dfm = pd.read_csv(m_path)

    for mode in dfm["ablation_mode"].unique():
        sub_mode = dfm[dfm["ablation_mode"] == mode].copy()
        order = [f for f in ABLATION_CONDS.keys() if f in sub_mode["feature_set"].unique()]
        sub_mode["feature_set"] = pd.Categorical(sub_mode["feature_set"], categories=order, ordered=True)

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
        for idx, inp in enumerate(["frTable", "weights"]):
            sub = sub_mode[sub_mode["input_type"] == inp]
            if not sub.empty:
                sns.boxplot(data=sub, x="feature_set", y="R2_global", ax=axes[idx], showfliers=False)
                axes[idx].set_title(f"GAMoE ({mode} ablation) — {inp}")
                axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        out_png = os.path.join(GAMOE_OUT_ROOT, f"R2_boxplot_{mode}_FRtop_WEIbottom.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"Saved: {out_png}")

def plot_knn_r2_comparison():
    m_path = os.path.join(KNN_OUT_ROOT, "all_metrics_kfold_knn.csv")
    if not os.path.exists(m_path): return
    dfm = pd.read_csv(m_path)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
    for idx, inp in enumerate(["frTable", "weights"]):
        sub = dfm[dfm["input_type"] == inp]
        if not sub.empty:
            sns.boxplot(data=sub, x="feature_set", y="R2_global", hue="ablation_mode", ax=axes[idx], showfliers=False)
            axes[idx].set_title(f"KNN Ablation Styles (input_drop vs column_shuffle) — {inp}")
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    out_png = os.path.join(KNN_OUT_ROOT, "R2_boxplot_knn_both_ablation_modes.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    plot_gamoe_r2_summary()
    plot_knn_r2_comparison()