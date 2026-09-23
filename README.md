# GaMoE: Geometry-Aware Mixture of Experts for Neural Manifold Composition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PridaLab/gamoe_model/blob/main/notebooks/gamoe_simulated_example.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular PyTorch framework implementing **Geometry-Aware Mixture of Experts (GaMoE)** to factorize high-dimensional neural ensemble dynamics (firing rates and synaptic weights) into low-dimensional topological manifolds of hippocampal sharp-wave ripples under systematic cell-type, laminar, and module ablation paradigms.

---

## 🔬 Model Overview

GaMoE decomposes population activity by allocating an independent, constrained bottleneck expert network $f_j(x_j)$ ($1 \to 11 \to \text{ReLU} \to D$) to each of the $N=11$ CA1 neuronal subpopulations (pyramidal cells and interneuron subtypes across deep and superficial layers). Experts are dynamically recombined via geometry-aware gating coefficients $g_j(u)$ derived from Gaussian density priors fitted over the underlying manifold coordinates:

<p align="center">
  <img src="docs/images/gamoe_computational_graph.png" alt="GaMoE Computational Graph" width="85%" />
</p>

$$\hat{\mathbf{u}}_i = \sum_{j=1}^{N} g_{ij}(\mathbf{u}_i) \cdot f_j(x_{ij})$$

where the gating distribution is normalized such that:

$$g_j(u) \propto \mathcal{N}(u; \mu_j, \Sigma_j)^\alpha, \qquad \sum_{j=1}^{N} g_{ij} = 1$$

---

## 🧪 Cell-Type Ablation Simulations

To assess the functional contribution of specific cell assemblies, ablations are executed **strictly at test time**, leaving the trained expert weights completely unchanged. Ablations operate directly on the gating distributions rather than zeroing input features:

### 1. Gate Drop / Masking (`gate`)
For a target cell-type subpopulation (e.g., Pyramidal Cells $\text{PCs} = \{\text{DeepPyr}, \text{SupPyr}\}$), the corresponding gates are set to zero and the remaining active gates are renormalized:

$$g'_{ij} = \begin{cases}  0, & \text{if } j \in \text{ablated group} \\ \frac{g_{ij}}{\sum_{k \notin \text{ablated group}} g_{ik}}, & \text{otherwise} \end{cases}$$

This manipulation evaluates how much the trained model relies on the corresponding experts for the reconstruction of the ripple embedding.

### 2. Gate Shuffling (`shuffle`)
For a given cell type $j$, its gate coefficients are permuted across events, $\{g_{ij}\}_i$, while leaving all other experts' gates intact:

$$g'_{ij} = \text{permute}(\{g_{ij}\}_i), \qquad g''_{ij} = \frac{g'_{ij}}{\sum_{k=1}^N g'_{ik}}$$

This preserves the marginal distribution of gate strengths for that type but disrupts the spatial mapping between gate mass and ripple cloud position, probing the necessity of geometry-specific routing for each cell type.

### 3. Input Zeroing (`input`)
Directly sets the standardized input features to zero ($x_{ij} = 0$) for target cell types while leaving the geometric gating distribution intact.

---

## ⚖️ Control Models & Ripple Decoding

To interpret GaMoE performance and test whether reconstruction depends on the geometry of the cloud rather than on the mere presence of multiple experts, we benchmark against:

* **Agnostic MoE ($\mathbf{aMoE}$)**: Evaluates the same trained experts under globally shuffled gates across events, thereby disrupting the alignment between gates and ripple cloud geometry while preserving gate statistics.
* **Feature-Space MoE ($\mathbf{fMoE}$)**: Reconstructs 3-dimensional ripple features (Frequency, Amplitude, and Entropy; FAE space) directly from mean weighted activity.
* **$k$-Nearest-Neighbors Regressor ($\mathbf{KNN}$)**: Predicts UMAP coordinates directly from the mean weighted activity of the 11 cell types without geometric priors or modular factorization. Analogous cell-type ablations (feature drop and column shuffle) are applied at the level of the input features.

Together, GaMoE and KNN address complementary questions: GaMoE reveals how ripple cloud-aligned components can be reconstructed from cell-type inputs under geometric constraints, whereas KNN assesses how much of the cloud structure can be decoded directly from those inputs without such constraints.

---

## 📈 Error-Weighted Ripple Metrics

To ground latent reconstruction errors in physiological ripple dynamics, we compute error-weighted averages of ripple frequency ($f_i$) and amplitude ($a_i$) for each model, fold, and ablation condition:

$$\langle f \rangle_e = \frac{\sum_i e_i \cdot f_i}{\sum_i e_i}, \qquad \langle a \rangle_e = \frac{\sum_i e_i \cdot a_i}{\sum_i e_i}$$

where $e_i = \Vert{}\hat{\mathbf{u}}_i - \mathbf{u}_i\Vert{}_2$ is the point-wise Euclidean reconstruction error in UMAP space, ensuring that events with larger reconstruction errors contribute more strongly to the summary value. For comparison, unweighted baseline means are computed across the same events:

$$\langle f \rangle = \frac{1}{M}\sum_{i=1}^{M} f_i, \qquad \langle a \rangle = \frac{1}{M}\sum_{i=1}^{M} a_i$$

---

## 🎯 Performance Metrics

All models are evaluated using **10-fold cross-validation**, reporting mean $\pm$ SD across folds for:

1. **Per-Dimension $R^2$**: Squared Pearson correlation between true and predicted UMAP coordinates for each manifold dimension ($d \in \{1, 2, 3, 4\}$):
   $$R^2_d = \left( \frac{\text{Cov}(\mathbf{u}_d, \hat{\mathbf{u}}_d)}{\sigma_{\mathbf{u}_d} \sigma_{\hat{\mathbf{u}}_d}} \right)^2$$
2. **Global $R^2$**: Variance-weighted average across all dimensions, prioritizing axes with larger variance:
   $$R^2_{\text{global}} = \sum_{d=1}^{D} w_d R^2_d, \quad \text{where } w_d = \frac{\text{Var}(\mathbf{u}_d)}{\sum_{k=1}^D \text{Var}(\mathbf{u}_k)}$$
3. **Point-wise Euclidean Error**: $e_i = \Vert{}\hat{\mathbf{u}}_i - \mathbf{u}_i\Vert{}_2$ for each event.
4. **Topological Structure Index (SI)**: Quantifies whether raw feature activations, gating probabilities, or prediction errors exhibit localized topological organization across the latent manifold.

<p align="center">
  <img src="docs/images/r2_ablation_summary.png" alt="R2 Summary Across Ablations" width="85%" />
</p>

---

## 📁 Repository Structure

```text
gamoe_model/
├── configs/
│   ├── __init__.py
│   └── ablation_config.py      # Feature sets, cell-type pairs, ablation conditions
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── moe.py              # ExpertLinearProj, GatedMoE architectures
│   │   └── knn.py              # KNN baseline wrapper (drop & shuffle)
│   ├── geometry/
│   │   ├── prior.py            # Gaussian density prior & gate builders
│   │   └── structure_index.py  # Structure Index (SI) computation
│   ├── stats/
│   │   ├── parametric.py       # One-way/Two-way ANOVA + Bonferroni post-hoc
│   │   └── non_parametric.py   # Kruskal-Wallis + pairwise Mann-Whitney U
│   └── utils/
│       ├── data_loaders.py     # CSV/MAT loaders, covariate aligners
│       └── metrics.py          # Variance-weighted R², RMSE, error-weighted means
├── experiments/
│   ├── run_gamoe_kfold.py      # K-fold GaMoE training and dual-mode ablations
│   └── run_knn_kfold.py        # K-fold KNN ablation & shuffle training
├── analysis/
│   ├── compute_statistics.py   # Parametric & non-parametric statistical reports
│   └── compute_si_metrics.py   # Structure Index evaluation on latent predictions
├── visualization/
│   ├── plot_performance.py     # R² and Euclidean error boxplots
│   ├── plot_latent_maps.py     # UMAP scatter, error projections, overlays
│   └── plot_covariates.py      # Error-weighted freq/amp summaries
├── notebooks/
│   └── gamoe_simulated_example.ipynb # Interactive demo with simulated data
├── docs/
│   └── images/                 # Embedded README figures
└── requirements.txt
```
---
## Citation

If you use this codebase or model in your research, please cite:

```bibtex
@software{gamoe_model2026,
  author = {Teresa Jurado-Parras#, Melisa Maidana-Capitan#, Candela Sanchez-Bellot*, Eloy Parra-Barrero*, Elena Cid, Enrique R. Sebastian, and Liset M. de la Prida},
  title = {Cell-type-resolved microcircuit dissection reveals inhibitory modules underlying ripple variability},
  url = {[https://github.com/PridaLab/gamoe_model](https://github.com/PridaLab/gamoe_model)},
  year = {2026}
}
```
--
## Running steps

# 1. Train GAMoE across gate-masking and gate-shuffling
python -m experiments.run_gamoe_kfold

# 2. Train KNN across feature-drop and column-shuffling
python -m experiments.run_knn_kfold

# 3. Run complete statistical testing (ANOVA, KW, MWU, paired t-tests)
python -m analysis.compute_statistics

# 4. Generate all figures
python -m visualization.plot_performance
python -m visualization.plot_latent_maps
python -m visualization.plot_covariates

---
## Licence 

