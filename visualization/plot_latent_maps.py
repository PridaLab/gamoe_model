import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from configs.ablation_config import GAMOE_OUT_ROOT, ABLATION_CONDS

# Custom green gradient (0 to 3 range)
CUSTOM_GREEN = LinearSegmentedColormap.from_list(
    "custom_green_svg",
    ["#f1f4ec", "#d6e0c4", "#b5c796", "#86a25f", "#5f7f34"],
    N=256
)

def plot_umap_error_heatmaps():
    p_path = os.path.join(GAMOE_OUT_ROOT, "all_predictions_umap4d_kfold_with_covs.csv")
    if not os.path.exists(p_path): return
    df = pd.read_csv(p_path)

    out_dir = os.path.join(GAMOE_OUT_ROOT, "umap_error_plots_customGreen")
    os.makedirs(out_dir, exist_ok=True)

    df["error_clip"] = np.clip(df["error"].to_numpy(dtype=float), 0.0, 3.0)

    for (fs, inp, mode), g in df.groupby(["feature_set", "input_type", "ablation_mode"]):
        fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=200)
        ax.scatter(
            g["UMAP1_true"], g["UMAP2_true"], c=g["error_clip"],
            s=12, alpha=0.3, cmap=CUSTOM_GREEN, vmin=0.0, vmax=3.0,
            linewidths=0, rasterized=True
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values(): spine.set_visible(False)

        fname = f"umap_error_{fs}_{inp}_{mode}_0to3.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved custom UMAP error maps to {out_dir}")

def plot_overlays():
    p_path = os.path.join(GAMOE_OUT_ROOT, "all_predictions_umap4d_kfold_with_covs.csv")
    if not os.path.exists(p_path): return
    df = pd.read_csv(p_path)

    out_dir = os.path.join(GAMOE_OUT_ROOT, "umap_overlays")
    os.makedirs(out_dir, exist_ok=True)

    for (inp, mode), sub_all in df.groupby(["input_type", "ablation_mode"]):
        for fs in ["All", "No_Pyr", "No_PV", "No_Module1"]:
            sub = sub_all[sub_all["feature_set"] == fs]
            if sub.empty: continue
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(sub["UMAP1_true"], sub["UMAP2_true"], s=4, c="black", alpha=0.3, label="true")
            ax.scatter(sub["UMAP1_pred"], sub["UMAP2_pred"], s=4, c="red", alpha=0.3, label="pred")
            ax.set_title(f"{fs} ({mode}) - {inp}")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"overlay_{fs}_{inp}_{mode}.png"), dpi=200)
            plt.close()

if __name__ == "__main__":
    plot_umap_error_heatmaps()
    plot_overlays()