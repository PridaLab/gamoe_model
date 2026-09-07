import os
import numpy as np
import pandas as pd

from configs.ablation_config import (
    FULL_FEATS, ABLATION_CONDS, KNN_ABLAT_MODES, SPLIT_ROOT, KNN_OUT_ROOT
)
from src.models.knn import run_knn_ablation_fold
from src.utils.data_loaders import load_single_cov_vector
from src.utils.metrics import compute_error_weighted_covariates

N_FOLDS = 10
INPUT_CONFIGS = [{"input_type": "frTable"}, {"input_type": "weights"}]

def main():
    os.makedirs(KNN_OUT_ROOT, exist_ok=True)
    all_preds, all_metrics = [], []

    for fold in range(1, N_FOLDS + 1):
        fold_dir = os.path.join(SPLIT_ROOT, f"fold_{fold}")
        print(f"KNN Processing Fold {fold}/{N_FOLDS}...")

        U_tr = pd.read_csv(os.path.join(fold_dir, "umap4_train.csv")).to_numpy()
        U_te = pd.read_csv(os.path.join(fold_dir, "umap4_test_eval.csv")).to_numpy()

        for inp_cfg in INPUT_CONFIGS:
            inp = inp_cfg["input_type"]
            prefix = "frTable" if inp == "frTable" else "weights"
            X_tr_full = pd.read_csv(os.path.join(fold_dir, f"{prefix}_train.csv"))[FULL_FEATS]
            X_te_full = pd.read_csv(os.path.join(fold_dir, f"{prefix}_test_eval.csv"))[FULL_FEATS]

            # Both ablation styles: input_drop AND column_shuffle
            for mode in KNN_ABLAT_MODES:
                p_list, m_list = run_knn_ablation_fold(
                    X_tr_full, X_te_full, U_tr, U_te, FULL_FEATS, ABLATION_CONDS,
                    ablation_mode=mode, input_type=inp, fold=fold
                )
                all_preds.extend(p_list)
                all_metrics.extend(m_list)

    df_preds = pd.concat(all_preds, ignore_index=True)
    df_metrics = pd.concat(all_metrics, ignore_index=True)

    # Attach covariates
    for col in ["freq", "amp"]:
        df_preds[col] = np.nan

    for fold in sorted(df_preds["fold"].unique()):
        f_dir = os.path.join(SPLIT_ROOT, f"fold_{fold}")
        mask_f = (df_preds["fold"] == fold)
        idx_f = df_preds.index[mask_f].sort_values()
        n_rows = len(idx_f)

        for col in ["amp", "freq"]:
            vec = load_single_cov_vector(f_dir, [f"{col}_test_eval", f"{col}_eval_test", f"{col}_test", col])
            reps = max(1, n_rows // len(vec))
            df_preds.loc[idx_f, col] = np.tile(vec, reps)[:n_rows]

    p_path = os.path.join(KNN_OUT_ROOT, "all_predictions_umap4d_kfold_knn_with_covs.csv")
    m_path = os.path.join(KNN_OUT_ROOT, "all_metrics_kfold_knn.csv")
    df_preds.to_csv(p_path, index=False)
    df_metrics.to_csv(m_path, index=False)

    df_w = pd.DataFrame(compute_error_weighted_covariates(df_preds, covariate_cols=["freq", "amp"]))
    df_w.to_csv(os.path.join(KNN_OUT_ROOT, "error_weighted_FA_by_ablation_knn.csv"), index=False)
    print(f"Saved KNN results to {KNN_OUT_ROOT}")

if __name__ == "__main__":
    main()
