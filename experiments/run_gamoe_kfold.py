import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from configs.ablation_config import (
    FULL_FEATS, ABLATION_CONDS, GAMOE_ABLAT_MODES, SPLIT_ROOT, GAMOE_OUT_ROOT
)
from src.models.moe import GatedMoE, eval_gamoe_ablations
from src.geometry.priors import build_prior
from src.utils.data_loaders import load_single_cov_vector
from src.utils.metrics import compute_error_weighted_covariates

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, PATIENCE, LR, WD, BATCH = 50, 25, 1e-3, 1e-4, 1024
ALPHA_FUSION, N_FOLDS, HIDDEN_EXPERT = 2.0, 10, 11
INPUT_CONFIGS = [{"input_type": "frTable"}, {"input_type": "weights"}]

def train_on_full_split(fold_dir, input_type):
    prefix = "frTable" if input_type == "frTable" else "weights"
    X_tr_np = pd.read_csv(os.path.join(fold_dir, f"{prefix}_train.csv"))[FULL_FEATS].to_numpy()
    X_te_np = pd.read_csv(os.path.join(fold_dir, f"{prefix}_test_eval.csv"))[FULL_FEATS].to_numpy()
    U_tr = pd.read_csv(os.path.join(fold_dir, "umap4_train.csv")).to_numpy()
    U_te = pd.read_csv(os.path.join(fold_dir, "umap4_test_eval.csv")).to_numpy()

    prior_full = pd.read_csv(os.path.join(fold_dir, f"{prefix}_prior.csv"))
    W_prior = prior_full[FULL_FEATS].to_numpy()
    U_prior = pd.read_csv(os.path.join(fold_dir, "umap4_prior.csv")).to_numpy()

    prior_fn = build_prior(U_prior, W_prior)
    Q_tr, Q_te = prior_fn(U_tr), prior_fn(U_te)

    scX = StandardScaler().fit(X_tr_np)
    scY = StandardScaler().fit(U_tr)

    Xt = torch.tensor(scX.transform(X_tr_np), dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(scX.transform(X_te_np), dtype=torch.float32, device=DEVICE)
    Yt = torch.tensor(scY.transform(U_tr), dtype=torch.float32, device=DEVICE)
    Yv = torch.tensor(scY.transform(U_te), dtype=torch.float32, device=DEVICE)

    Pf_tr = torch.tensor(Q_tr, dtype=torch.float32, device=DEVICE) ** ALPHA_FUSION
    Pf_tr /= (Pf_tr.sum(1, keepdim=True) + 1e-12)
    Pf_te = torch.tensor(Q_te, dtype=torch.float32, device=DEVICE) ** ALPHA_FUSION
    Pf_te /= (Pf_te.sum(1, keepdim=True) + 1e-12)

    model = GatedMoE(n_experts=len(FULL_FEATS), out_dim=U_tr.shape[1], hidden=HIDDEN_EXPERT).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    mse = nn.MSELoss()
    best_val, bad = np.inf, 0

    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(Xt.size(0), device=DEVICE)
        for b in range(0, len(idx), BATCH):
            ib = idx[b:b+BATCH]
            opt.zero_grad()
            loss = mse(model(Xt[ib], Pf_tr[ib]), Yt[ib])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val = mse(model(Xv, Pf_te), Yv).item()
        if val < best_val:
            best_val, bad = val, 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break

    return model, scY, Xv, U_te, Pf_te

def main():
    os.makedirs(GAMOE_OUT_ROOT, exist_ok=True)
    all_preds, all_metrics = [], []

    for fold in range(1, N_FOLDS + 1):
        fold_dir = os.path.join(SPLIT_ROOT, f"fold_{fold}")
        print(f"GAMoE Processing Fold {fold}/{N_FOLDS}...")

        for input_cfg in INPUT_CONFIGS:
            inp = input_cfg["input_type"]
            model, scY, Xv, U_te, Pf_te = train_on_full_split(fold_dir, inp)

            # Both ablation styles: gate masking AND gate shuffling
            for mode in GAMOE_ABLAT_MODES:
                p_list, m_list = eval_gamoe_ablations(
                    model, scY, Xv, U_te, Pf_te, FULL_FEATS, ABLATION_CONDS,
                    ablation_mode=mode, input_type=inp, device=DEVICE
                )
                for p, m in zip(p_list, m_list):
                    p["fold"] = fold
                    m["fold"] = fold
                    all_preds.append(p)
                    all_metrics.append(m)

    df_preds = pd.concat(all_preds, ignore_index=True)
    df_metrics = pd.concat(all_metrics, ignore_index=True)

    # Attach split priors covariates (freq, amp, entropy)
    for col in ["freq", "amp", "entropy"]:
        df_preds[col] = np.nan

    for fold in sorted(df_preds["fold"].unique()):
        f_dir = os.path.join(SPLIT_ROOT, f"fold_{fold}")
        mask_f = (df_preds["fold"] == fold)
        idx_f = df_preds.index[mask_f].sort_values()
        n_rows = len(idx_f)

        for col in ["amp", "freq", "entropy"]:
            vec = load_single_cov_vector(f_dir, [f"{col}_test_eval", f"{col}_eval_test", f"{col}_test", col])
            reps = max(1, n_rows // len(vec))
            df_preds.loc[idx_f, col] = np.tile(vec, reps)[:n_rows]

    # Write persistent CSVs
    p_path = os.path.join(GAMOE_OUT_ROOT, "all_predictions_umap4d_kfold_with_covs.csv")
    m_path = os.path.join(GAMOE_OUT_ROOT, "all_metrics_kfold.csv")
    df_preds.to_csv(p_path, index=False)
    df_metrics.to_csv(m_path, index=False)

    # Error-weighted table
    df_w = pd.DataFrame(compute_error_weighted_covariates(df_preds))
    df_w.to_csv(os.path.join(GAMOE_OUT_ROOT, "error_weighted_FAE_by_ablation.csv"), index=False)
    print(f"Saved GAMoE results to {GAMOE_OUT_ROOT}")

if __name__ == "__main__":
    main()
