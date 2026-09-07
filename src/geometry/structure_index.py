import warnings, copy
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix
from sklearn.metrics import pairwise_distances
from sklearn.manifold import Isomap
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

def cloud_overlap_neighbors(cloud1, cloud2, k, distance_metric):
    cloud_all = np.vstack((cloud1, cloud2)).astype('float32')
    idx_sep = cloud1.shape[0]
    if distance_metric == 'euclidean':
        knn = NearestNeighbors(n_neighbors=k, metric="minkowski", p=2).fit(cloud_all)
        I = knn.kneighbors(return_distance=False)
    elif distance_metric == 'geodesic':
        model_iso = Isomap(n_components=1)
        dist_mat = model_iso.fit_transform(cloud_all)
        knn = NearestNeighbors(n_neighbors=k, metric="precomputed").fit(model_iso.dist_matrix_)
        I = knn.kneighbors(return_distance=False)
    overlap_1_2 = np.sum(I[:idx_sep, :] >= idx_sep) / (cloud1.shape[0] * k)
    overlap_2_1 = np.sum(I[idx_sep:, :] < idx_sep) / (cloud2.shape[0] * k)
    return overlap_1_2, overlap_2_1

def compute_structure_index(data, label, n_bins=10, dims=None, **kwargs):
    if label.ndim == 1:
        label = label.reshape(-1, 1)
    if dims is None:
        dims = list(range(data.shape[1]))
    data = data[:, dims]
    valid_mask = ~(np.any(np.isnan(data), axis=1) | np.any(np.isnan(label), axis=1))
    data = data[valid_mask]
    label = label[valid_mask]

    min_l = np.percentile(label, 5, axis=0)
    max_l = np.percentile(label, 95, axis=0)
    n_neighbors = kwargs.get("n_neighbors", 15)
    distance_metric = kwargs.get("distance_metric", "euclidean")
    num_shuffles = kwargs.get("num_shuffles", 0)

    edges = [np.linspace(min_l[i], max_l[i], n_bins + 1) for i in range(label.shape[1])]
    bin_label = np.digitize(label[:, 0], edges[0]) - 1
    valid_bins = (bin_label >= 0) & (bin_label < n_bins)
    data, bin_label = data[valid_bins], bin_label[valid_bins]
    ubins = np.unique(bin_label)
    nb = len(ubins)
    if nb < 2:
        return 0.0, None, np.zeros((1, 1)), np.zeros(num_shuffles)

    overlap_mat = np.zeros((nb, nb))
    for a in range(nb):
        A = data[bin_label == ubins[a]]
        for b in range(a + 1, nb):
            B = data[bin_label == ubins[b]]
            ov_ab, ov_ba = cloud_overlap_neighbors(A, B, n_neighbors, distance_metric)
            overlap_mat[a, b] = ov_ab
            overlap_mat[b, a] = ov_ba

    deg = np.nansum(overlap_mat, axis=1)
    SI = np.clip(2 * ((1 - np.mean(deg) / (nb - 1)) - 0.5), 0, 1)
    return float(SI), (bin_label, None), overlap_mat, np.zeros(num_shuffles)