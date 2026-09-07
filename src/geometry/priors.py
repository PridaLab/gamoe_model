import numpy as np

def weighted_mean_cov(X, w):
    w = np.clip(w, 0, None)
    s = w.sum() + 1e-12
    m = (X * w[:, None]).sum(0) / s
    diff = (X - m) * np.sqrt(w[:, None])
    C = (diff.T @ diff) / s + 1e-6 * np.eye(X.shape[1])
    return m, C

def gaussian_pdf(X, mean, cov):
    inv, det = np.linalg.inv(cov), np.linalg.det(cov)
    diff = X - mean
    norm = 1.0 / np.sqrt(((2 * np.pi) ** X.shape[1]) * det)
    ex = np.einsum("ni,ij,nj->n", diff, inv, diff)
    return norm * np.exp(-0.5 * ex)

def build_prior(U, W):
    comps = []
    for j in range(W.shape[1]):
        wj = W[:, j]
        if np.allclose(wj.sum(), 0):
            m, C = U.mean(0), np.cov(U.T) + 1e-6 * np.eye(U.shape[1])
        else:
            m, C = weighted_mean_cov(U, wj)
        comps.append((m, C))

    def prior(U2):
        D = np.zeros((U2.shape[0], W.shape[1]))
        for j, (m, C) in enumerate(comps):
            D[:, j] = gaussian_pdf(U2, m, C)
        D += 1e-12
        return D / D.sum(1, keepdims=True)

    return prior