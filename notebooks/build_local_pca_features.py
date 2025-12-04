#!/usr/bin/env python
"""
build_local_pca_features.py

Helper functions to compute local PCA / shape features and tangent
directions for a set of 3D points.

Designed to be imported from:
  - inspect_local_pca.py  (sanity plots)
  - dataset.py            (to append features to hit.x later)
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_local_pca_features(pos, k=12):
    """
    Compute local PCA-based shape and direction features for each point.

    Parameters
    ----------
    pos : np.ndarray, shape (N, 3)
        3D coordinates of hits/spacepoints in *one event*.
    k : int, optional
        Number of nearest neighbours (including the hit itself) to use
        when building the local covariance matrix. Typical values: 8–20.

    Returns
    -------
    features : dict of str -> np.ndarray (shape (N,))
        Keys:
          - 'linearity'
          - 'planarity'
          - 'sphericity'
          - 'tx', 'ty', 'tz'  (components of local tangent direction)

        All arrays are float32.
    """
    pos = np.asarray(pos, dtype=np.float32)
    N = pos.shape[0]
    if N < 4:
        # Not enough points to do meaningful PCA; return zeros.
        zeros = np.zeros(N, dtype=np.float32)
        return {
            "linearity": zeros.copy(),
            "planarity": zeros.copy(),
            "sphericity": zeros.copy(),
            "tx": zeros.copy(),
            "ty": zeros.copy(),
            "tz": zeros.copy(),
        }

    k = int(k)
    k = max(4, min(k, N))  # clamp

    # Nearest neighbours in 3D
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(pos)
    _, idx = nbrs.kneighbors(pos)  # idx: (N, k)

    linearity = np.zeros(N, dtype=np.float32)
    planarity = np.zeros(N, dtype=np.float32)
    sphericity = np.zeros(N, dtype=np.float32)
    tx = np.zeros(N, dtype=np.float32)
    ty = np.zeros(N, dtype=np.float32)
    tz = np.zeros(N, dtype=np.float32)

    eps = 1e-8

    for i in range(N):
        pts = pos[idx[i]]               # (k, 3)
        pts_centered = pts - pts.mean(axis=0, keepdims=True)

        # Covariance matrix (3x3)
        cov = np.cov(pts_centered, rowvar=False)

        # Eigen-decomposition; eigh returns ascending eigenvalues
        vals, vecs = np.linalg.eigh(cov)
        # Sort so that λ1 >= λ2 >= λ3
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        l1, l2, l3 = vals
        denom = float(l1) + eps

        # Shape descriptors
        linearity[i] = (l1 - l2) / denom
        planarity[i] = (l2 - l3) / denom
        sphericity[i] = l3 / denom

        # Local tangent direction: eigenvector for λ1
        v1 = vecs[:, 0]
        v1 = v1 / (np.linalg.norm(v1) + eps)

        tx[i], ty[i], tz[i] = v1.astype(np.float32)

    return {
        "linearity": linearity,
        "planarity": planarity,
        "sphericity": sphericity,
        "tx": tx,
        "ty": ty,
        "tz": tz,
    }
