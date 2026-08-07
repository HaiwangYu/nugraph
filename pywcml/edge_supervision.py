"""Legacy point-level edge-supervision helpers.

These arrays are retained for diagnostic compatibility. They are not inputs to
the reconstruction-only NuGraph candidate topology.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _unique_pairs(pairs: np.ndarray) -> np.ndarray:
    """Deduplicate undirected pairs."""

    if pairs.size == 0:
        return pairs.astype(np.int64, copy=False)
    a = np.minimum(pairs[:, 0], pairs[:, 1])
    b = np.maximum(pairs[:, 0], pairs[:, 1])
    return np.unique(np.stack([a, b], axis=1).astype(np.int64, copy=False), axis=0)


def mine_hard_negatives_radius_mm(
    xyz_mm: np.ndarray,
    tid_points: np.ndarray,
    n_target: int,
    radius_mm: float = 60.0,
    seed: int = 123,
) -> np.ndarray:
    """Return up to ``n_target`` different-TID pairs within ``radius_mm``."""

    tid = tid_points.astype(np.int64, copy=False)
    labeled = np.where(tid != -1)[0].astype(np.int64, copy=False)
    if labeled.size < 2 or n_target <= 0:
        return np.empty((0, 2), dtype=np.int64)

    points = xyz_mm[labeled].astype(np.float32, copy=False)
    neighbors = NearestNeighbors(radius=float(radius_mm), algorithm="ball_tree")
    neighbors.fit(points)

    candidates = []
    for local_index, local_neighbors in enumerate(neighbors.radius_neighbors(points, return_distance=False)):
        if local_neighbors.size == 0:
            continue
        source = labeled[local_index]
        source_tid = tid[source]
        for neighbor_index in local_neighbors:
            target = labeled[neighbor_index]
            if target == source or source_tid == tid[target]:
                continue
            candidates.append((source, target) if source < target else (target, source))

    if not candidates:
        return np.empty((0, 2), dtype=np.int64)
    result = _unique_pairs(np.asarray(candidates, dtype=np.int64))
    if result.shape[0] > n_target:
        rng = np.random.default_rng(seed)
        result = result[rng.choice(result.shape[0], size=n_target, replace=False)]
    return result


def mine_random_negatives(
    tid_points: np.ndarray,
    n_target: int,
    seed: int = 123,
) -> np.ndarray:
    """Return best-effort random different-TID pairs; never raise if scarce."""

    tid = tid_points.astype(np.int64, copy=False)
    labeled = np.where(tid != -1)[0].astype(np.int64, copy=False)
    if labeled.size < 2 or n_target <= 0:
        return np.empty((0, 2), dtype=np.int64)

    labeled_tid = tid[labeled]
    unique_tid = np.unique(labeled_tid)
    unique_tid = unique_tid[unique_tid != -1]
    if unique_tid.size < 2:
        return np.empty((0, 2), dtype=np.int64)

    buckets = {value: labeled[labeled_tid == value] for value in unique_tid}
    rng = np.random.default_rng(seed)
    result = np.empty((n_target, 2), dtype=np.int64)
    count = 0
    tries = 0
    max_tries = max(1000, 20 * n_target)
    while count < n_target and tries < max_tries:
        tries += 1
        first_tid, second_tid = rng.choice(unique_tid, size=2, replace=False)
        first = rng.choice(buckets[first_tid])
        second = rng.choice(buckets[second_tid])
        if first == second:
            continue
        result[count] = (first, second) if first < second else (second, first)
        count += 1

    if count == 0:
        return np.empty((0, 2), dtype=np.int64)
    result = _unique_pairs(result[:count])
    if result.shape[0] < n_target:
        extra = mine_random_negatives(tid_points, n_target - result.shape[0], seed=seed + 999)
        if extra.shape[0] > 0:
            result = _unique_pairs(np.concatenate([result, extra], axis=0))
    if result.shape[0] > n_target:
        rng = np.random.default_rng(seed + 17)
        result = result[rng.choice(result.shape[0], size=n_target, replace=False)]
    return result


def build_edge_supervision(
    xyz_mm: np.ndarray,
    tid_points: np.ndarray,
    pos_pairs: np.ndarray,
    balance_edges: bool = True,
    neg_radius_mm: float = 60.0,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    """Build legacy point ``edge_index`` and same-instance ``edge_y``."""

    tid = tid_points.astype(np.int64, copy=False)
    if pos_pairs is None or len(pos_pairs) == 0:
        pos_pairs = np.empty((0, 2), dtype=np.int64)
    else:
        pos_pairs = _unique_pairs(np.asarray(pos_pairs, dtype=np.int64))
        source, target = pos_pairs[:, 0], pos_pairs[:, 1]
        labelable_same = (
            (tid[source] != -1)
            & (tid[target] != -1)
            & (tid[source] == tid[target])
        )
        pos_pairs = pos_pairs[labelable_same]

    n_pos = int(pos_pairs.shape[0])
    if n_pos == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.int8)

    n_target_neg = n_pos if balance_edges else max(1, n_pos // 2)
    neg_pairs = mine_hard_negatives_radius_mm(
        xyz_mm=xyz_mm,
        tid_points=tid,
        n_target=n_target_neg,
        radius_mm=float(neg_radius_mm),
        seed=seed,
    )
    if neg_pairs.shape[0] < n_target_neg:
        extra = mine_random_negatives(tid, n_target_neg - neg_pairs.shape[0], seed=seed + 1)
        if extra.shape[0] > 0:
            neg_pairs = _unique_pairs(
                np.concatenate([neg_pairs, extra], axis=0) if neg_pairs.size else extra
            )

    n_neg = int(neg_pairs.shape[0])
    if n_neg:
        source = np.concatenate([pos_pairs[:, 0], neg_pairs[:, 0]])
        target = np.concatenate([pos_pairs[:, 1], neg_pairs[:, 1]])
        labels = np.concatenate(
            [np.ones(n_pos, dtype=np.int8), np.zeros(n_neg, dtype=np.int8)]
        )
    else:
        source, target = pos_pairs[:, 0], pos_pairs[:, 1]
        labels = np.ones(n_pos, dtype=np.int8)

    return (
        np.stack([source, target], axis=0).astype(np.int64, copy=False),
        labels.astype(np.int8, copy=False),
    )


__all__ = [
    "build_edge_supervision",
    "mine_hard_negatives_radius_mm",
    "mine_random_negatives",
]
