import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from src.utils.metrics import corr_r2_scores

class ExpertLinearProj(nn.Module):
    def __init__(self, in_dim=1, out_dim=4, hidden=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.ReLU(),
            nn.Linear(hidden, out_dim, bias=True),
        )

    def forward(self, x):
        return self.net(x)

class GatedMoE(nn.Module):
    def __init__(self, n_experts=11, out_dim=4, hidden=11):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([
            ExpertLinearProj(1, out_dim, hidden=hidden) for _ in range(n_experts)
        ])

    def forward(self, x, gates):
        outs = [exp(x[:, j:j+1]) for j, exp in enumerate(self.experts)]
        outs = torch.stack(outs, dim=2)
        return (outs * gates.unsqueeze(1)).sum(2)

def eval_gamoe_ablations(
    model, scY, Xv_all, U_test, Pf_te, full_feats, ablation_conds,
    ablation_mode, input_type, arch="linear", use_csd=False, device="cpu"
):
    all_preds, all_metrics = [], []
    feat_to_idx = {f: i for i, f in enumerate(full_feats)}
    n_experts = len(full_feats)

    for cond_name, feats_to_ablate in ablation_conds.items():
        Pf_mod = Pf_te.clone()

        if ablation_mode == "gate":
            # Mask gates: set selected weights to zero and renormalize
            mask = torch.ones(n_experts, dtype=torch.float32, device=device)
            for f in feats_to_ablate:
                mask[feat_to_idx[f]] = 0.0
            Pf_mod = Pf_mod * mask[None, :]

        elif ablation_mode == "shuffle":
            # Shuffle selected gate columns across test samples
            if len(feats_to_ablate) > 0:
                perm = torch.randperm(Pf_mod.size(0), device=device)
                for f in feats_to_ablate:
                    j = feat_to_idx[f]
                    Pf_mod[:, j] = Pf_mod[perm, j]

        elif ablation_mode == "input":
            # Zero out standardized inputs directly
            Xv_in = Xv_all.clone()
            for f in feats_to_ablate:
                Xv_in[:, feat_to_idx[f]] = 0.0
            Xv_all_eval = Xv_in
        else:
            raise ValueError(f"Unknown ablation mode: {ablation_mode}")

        if ablation_mode != "input":
            Pf_mod = Pf_mod / (Pf_mod.sum(1, keepdim=True) + 1e-12)
            Xv_all_eval = Xv_all

        model.eval()
        with torch.no_grad():
            Y_pred_scaled = model(Xv_all_eval, Pf_mod).cpu().numpy()
        Y_pred = scY.inverse_transform(Y_pred_scaled)

        R2_global, r2_dims, r_dims = corr_r2_scores(U_test, Y_pred)
        err = np.linalg.norm(U_test - Y_pred, axis=1)
        rmse = math.sqrt(mean_squared_error(U_test, Y_pred))

        dfp = pd.DataFrame({
            "UMAP1_true": U_test[:, 0], "UMAP2_true": U_test[:, 1],
            "UMAP3_true": U_test[:, 2], "UMAP4_true": U_test[:, 3],
            "UMAP1_pred": Y_pred[:, 0], "UMAP2_pred": Y_pred[:, 1],
            "UMAP3_pred": Y_pred[:, 2], "UMAP4_pred": Y_pred[:, 3],
            "error": err
        })
        dfp["feature_set"]       = cond_name
        dfp["train_feature_set"] = "All"
        dfp["arch"]              = arch
        dfp["use_csd"]           = use_csd
        dfp["input_type"]        = input_type
        dfp["ablation_mode"]     = ablation_mode

        dfm = pd.DataFrame({
            "train_feature_set": ["All"],
            "feature_set":       [cond_name],
            "arch":              [arch],
            "use_csd":           [use_csd],
            "input_type":        [input_type],
            "ablation_mode":     [ablation_mode],
            "R2_global":         [R2_global],
            "RMSE":              [rmse],
            "mean_error":        [float(err.mean())],
            **{f"R2_dim{i+1}": [r2_dims[i]] for i in range(4)},
            **{f"r_dim{i+1}":  [r_dims[i]]  for i in range(4)}
        })

        all_preds.append(dfp)
        all_metrics.append(dfm)

    return all_preds, all_metrics
