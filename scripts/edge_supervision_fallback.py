# filename: edge_supervision_fallback.py
"""
Robust edge supervision builder for NuGraph instance training.

Guarantees:
- No RuntimeError due to "insufficient hard negatives" or "no candidate negatives".
- If balance_edges=True, tries to match #neg == #pos.
  If hard negatives within radius are insufficient, fills remaining negatives by global random negatives.
- Returned edges are always labelable (tid!=-1 on endpoints).
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
    out = np.stack([a, b], axis=1).astype(np.int64, copy=False)
    # numpy unique rows trick
    out = np.unique(out, axis=0)
    return out


def mine_hard_negatives_radius_mm(
    xyz_mm: np.ndarray,
    tid_points: np.ndarray,
    n_target: int,
    radius_mm: float = 60.0,
    seed: int = 123,
) -> np.ndarray:
    """
    Mine negative pairs (u,v) such that:
    - tid[u] != -1 and tid[v] != -1
    - tid[u] != tid[v]
    - ||x_u - x_v|| <= radius_mm

    Returns up to n_target pairs. NEVER raises.
    """
    tid = tid_points.astype(np.int64, copy=False)
    labeled = np.where(tid != -1)[0].astype(np.int64, copy=False)
    if labeled.size < 2 or n_target <= 0:
        return np.empty((0, 2), dtype=np.int64)

    X = xyz_mm[labeled].astype(np.float32, copy=False)

    # Build radius neighbor graph
    nn = NearestNeighbors(radius=float(radius_mm), algorithm="ball_tree")
    nn.fit(X)
    neigh = nn.radius_neighbors(X, return_distance=False)

    cand = []
    for i_loc, nb in enumerate(neigh):
        if nb.size == 0:
            continue
        ui = labeled[i_loc]
        ti = tid[ui]
        # Only consider cross-tid neighbors
        for j_loc in nb:
            vj = labeled[j_loc]
            if vj == ui:
                continue
            if ti == tid[vj]:
                continue
            a, b = (ui, vj) if ui < vj else (vj, ui)
            cand.append((a, b))

    if not cand:
        return np.empty((0, 2), dtype=np.int64)

    cand = np.array(cand, dtype=np.int64)
    cand = _unique_pairs(cand)

    # If too many, subsample deterministically
    if cand.shape[0] > n_target:
        rng = np.random.default_rng(seed)
        idx = rng.choice(cand.shape[0], size=n_target, replace=False)
        cand = cand[idx]

    return cand


def mine_random_negatives(
    tid_points: np.ndarray,
    n_target: int,
    seed: int = 123,
) -> np.ndarray:
    """
    Global random negatives: pick u,v from different tids, both tid!=-1.
    NEVER raises; may return fewer if impossible.
    """
    tid = tid_points.astype(np.int64, copy=False)
    labeled = np.where(tid != -1)[0].astype(np.int64, copy=False)
    if labeled.size < 2 or n_target <= 0:
        return np.empty((0, 2), dtype=np.int64)

    t = tid[labeled]
    uniq = np.unique(t)
    uniq = uniq[uniq != -1]
    if uniq.size < 2:
        return np.empty((0, 2), dtype=np.int64)

    buckets = {uu: labeled[t == uu] for uu in uniq}
    rng = np.random.default_rng(seed)

    out = np.empty((n_target, 2), dtype=np.int64)
    got = 0
    # Try a bounded number of attempts to avoid infinite loops in pathological cases
    max_tries = max(1000, 20 * n_target)
    tries = 0
    while got < n_target and tries < max_tries:
        tries += 1
        t1, t2 = rng.choice(uniq, size=2, replace=False)
        u = rng.choice(buckets[t1])
        v = rng.choice(buckets[t2])
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        out[got] = (a, b)
        got += 1

    if got == 0:
        return np.empty((0, 2), dtype=np.int64)

    out = out[:got]
    out = _unique_pairs(out)

    # If we lost some due to dedup, top up once more (optional)
    if out.shape[0] < n_target:
        need = n_target - out.shape[0]
        extra = mine_random_negatives(tid_points, need, seed=seed + 999)
        if extra.shape[0] > 0:
            out = _unique_pairs(np.concatenate([out, extra], axis=0))

    # Cap to n_target
    if out.shape[0] > n_target:
        rng = np.random.default_rng(seed + 17)
        idx = rng.choice(out.shape[0], size=n_target, replace=False)
        out = out[idx]

    return out


def build_edge_supervision(
    xyz_mm: np.ndarray,
    tid_points: np.ndarray,
    pos_pairs: np.ndarray,
    balance_edges: bool = True,
    neg_radius_mm: float = 60.0,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (edge_index, edge_y).

    Inputs:
      - xyz_mm: (N,3) float, positions in mm
      - tid_points: (N,) int, truth instance id per point (=-1 means unlabeled)
      - pos_pairs: (P,2) int, positive edges (u,v) with same tid, labelable
      - balance_edges: if True, target #neg == #pos (best-effort)
      - neg_radius_mm: for hard negatives
    """
    tid = tid_points.astype(np.int64, copy=False)

    # Sanitize pos_pairs: ensure undirected unique, labelable, same tid
    if pos_pairs is None or len(pos_pairs) == 0:
        pos_pairs = np.empty((0, 2), dtype=np.int64)
    else:
        pos_pairs = np.asarray(pos_pairs, dtype=np.int64)
        pos_pairs = _unique_pairs(pos_pairs)
        uP, vP = pos_pairs[:, 0], pos_pairs[:, 1]
        mP = (tid[uP] != -1) & (tid[vP] != -1) & (tid[uP] == tid[vP])
        pos_pairs = pos_pairs[mP]

    n_pos = int(pos_pairs.shape[0])

    # If no positives, return empty supervision (caller can decide to skip)
    if n_pos == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.int8)

    # Target negatives
    n_target_neg = n_pos if balance_edges else max(1, n_pos // 2)

    # 1) Hard negatives within radius
    neg_pairs = mine_hard_negatives_radius_mm(
        xyz_mm=xyz_mm,
        tid_points=tid,
        n_target=n_target_neg,
        radius_mm=float(neg_radius_mm),
        seed=seed,
    )

    # 2) Fill remainder with global random negatives
    if neg_pairs.shape[0] < n_target_neg:
        need = n_target_neg - neg_pairs.shape[0]
        extra = mine_random_negatives(tid, need, seed=seed + 1)
        if extra.shape[0] > 0:
            neg_pairs = _unique_pairs(
                np.concatenate([neg_pairs, extra], axis=0) if neg_pairs.size else extra
            )

    # If still short, proceed "as balanced as possible"
    n_neg = int(neg_pairs.shape[0])

    u_all = np.concatenate([pos_pairs[:, 0], neg_pairs[:, 0]]) if n_neg else pos_pairs[:, 0]
    v_all = np.concatenate([pos_pairs[:, 1], neg_pairs[:, 1]]) if n_neg else pos_pairs[:, 1]
    y_all = np.concatenate([np.ones(n_pos, dtype=np.int8), np.zeros(n_neg, dtype=np.int8)]) if n_neg else np.ones(n_pos, dtype=np.int8)

    edge_index = np.stack([u_all, v_all], axis=0).astype(np.int64, copy=False)
    edge_y = y_all.astype(np.int8, copy=False)

    return edge_index, edge_y
