import math
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from src.utils.metrics import corr_r2_scores

def build_knn(n_neighbors=15, weights="distance"):
    return KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        metric="minkowski",
        p=2
    )

def run_knn_ablation_fold(
    X_tr_full, X_te_full, U_tr, U_te, full_feats, ablation_conds,
    ablation_mode, input_type, fold, n_neighbors=15
):
    """
    Executes KNN under:
      - 'input_drop': drops columns from training and testing
      - 'column_shuffle': shuffles selected training columns across samples
    """
    all_preds, all_metrics = [], []
    feat_to_idx = {f: i for i, f in enumerate(full_feats)}

    for cond_name, feats_ablate in ablation_conds.items():
        if ablation_mode == "input_drop":
            feats_keep = [f for f in full_feats if f not in feats_ablate]
            X_tr_base = X_tr_full[feats_keep].to_numpy()
            X_te = X_te_full[feats_keep].to_numpy()

            for shuffle_type in ["true", "shuffled"]:
                X_tr = X_tr_base.copy()
                if shuffle_type == "shuffled":
                    X_tr = X_tr[np.random.permutation(X_tr.shape[0])]

                scX = StandardScaler().fit(X_tr)
                knn = build_knn(n_neighbors=n_neighbors)
                knn.fit(scX.transform(X_tr), U_tr)
                U_pred = knn.predict(scX.transform(X_te))

                R2_g, r2_d, r_d = corr_r2_scores(U_te, U_pred)
                err = np.linalg.norm(U_te - U_pred, axis=1)

                dfp = pd.DataFrame({
                    "UMAP1_true": U_te[:, 0], "UMAP2_true": U_te[:, 1],
                    "UMAP3_true": U_te[:, 2], "UMAP4_true": U_te[:, 3],
                    "UMAP1_pred": U_pred[:, 0], "UMAP2_pred": U_pred[:, 1],
                    "UMAP3_pred": U_pred[:, 2], "UMAP4_pred": U_pred[:, 3],
                    "error": err, "feature_set": cond_name, "input_type": input_type,
                    "ablation_mode": "input_drop", "shuffle_type": shuffle_type, "fold": fold
                })
                dfm = pd.DataFrame({
                    "feature_set": [cond_name], "input_type": [input_type], "ablation_mode": ["input_drop"],
                    "shuffle_type": [shuffle_type], "R2_global": [R2_g],
                    "RMSE": [math.sqrt(mean_squared_error(U_te, U_pred))],
                    "mean_error": [float(err.mean())], "fold": [fold]
                })
                all_preds.append(dfp)
                all_metrics.append(dfm)

        elif ablation_mode == "column_shuffle":
            X_tr = X_tr_full[full_feats].to_numpy().copy()
            X_te = X_te_full[full_feats].to_numpy().copy()

            for f in feats_ablate:
                j = feat_to_idx[f]
                perm = np.random.permutation(X_tr.shape[0])
                X_tr[:, j] = X_tr[:, j][perm]

            scX = StandardScaler().fit(X_tr)
            knn = build_knn(n_neighbors=n_neighbors)
            knn.fit(scX.transform(X_tr), U_tr)
            U_pred = knn.predict(scX.transform(X_te))

            R2_g, r2_d, r_d = corr_r2_scores(U_te, U_pred)
            err = np.linalg.norm(U_te - U_pred, axis=1)

            dfp = pd.DataFrame({
                "UMAP1_true": U_te[:, 0], "UMAP2_true": U_te[:, 1],
                "UMAP3_true": U_te[:, 2], "UMAP4_true": U_te[:, 3],
                "UMAP1_pred": U_pred[:, 0], "UMAP2_pred": U_pred[:, 1],
                "UMAP3_pred": U_pred[:, 2], "UMAP4_pred": U_pred[:, 3],
                "error": err, "feature_set": cond_name, "input_type": input_type,
                "ablation_mode": "column_shuffle", "shuffle_type": "none", "fold": fold
            })
            dfm = pd.DataFrame({
                "feature_set": [cond_name], "input_type": [input_type], "ablation_mode": ["column_shuffle"],
                "shuffle_type": ["none"], "R2_global": [R2_g],
                "RMSE": [math.sqrt(mean_squared_error(U_te, U_pred))],
                "mean_error": [float(err.mean())], "fold": [fold]
            })
            all_preds.append(dfp)
            all_metrics.append(dfm)

    return all_preds, all_metrics
