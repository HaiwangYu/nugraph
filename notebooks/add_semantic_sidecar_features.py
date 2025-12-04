#!/usr/bin/env python3
import argparse
import sys
from typing import Tuple, Optional

import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_bounds(h5_path: str, max_events: Optional[int] = None) -> Tuple[float, float, float, float, float, float]:
    """Scan the file once to infer global x,y,z bounds from sp/pos."""
    x_min = y_min = z_min = np.inf
    x_max = y_max = z_max = -np.inf
    total_hits = 0

    with h5py.File(h5_path, "r") as f:
        ds = f["dataset"]
        keys = list(ds.keys())
        if max_events is not None:
            keys = keys[:max_events]

        for k in keys:
            rec = ds[k][()]
            pos = np.asarray(rec["sp/pos"])
            if pos.size == 0:
                continue
            x = pos[:, 0]
            y = pos[:, 1]
            z = pos[:, 2]

            x_min = min(x_min, float(x.min()))
            x_max = max(x_max, float(x.max()))
            y_min = min(y_min, float(y.min()))
            y_max = max(y_max, float(y.max()))
            z_min = min(z_min, float(z.min()))
            z_max = max(z_max, float(z.max()))

            total_hits += pos.shape[0]

    print(f"[Bounds] x:[{x_min:.2f}, {x_max:.2f}]  "
          f"y:[{y_min:.2f}, {y_max:.2f}]  "
          f"z:[{z_min:.2f}, {z_max:.2f}]")
    print(f"[Bounds] total hits scanned: {total_hits}")
    return x_min, x_max, y_min, y_max, z_min, z_max


def compute_distances_to_walls(
    pos: np.ndarray,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    z_min: float, z_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    pos: (N, 3) array of [x,y,z].

    Returns:
      d_wall: distance to nearest wall (min over x,y,z faces)
      d_top : distance to TOP wall in y (y_max - y)
    """
    x = pos[:, 0]
    y = pos[:, 1]
    z = pos[:, 2]

    dx = np.minimum(x - x_min, x_max - x)
    dy = np.minimum(y - y_min, y_max - y)
    dz = np.minimum(z - z_min, z_max - z)

    d_wall = np.minimum(np.minimum(dx, dy), dz)
    d_top = y_max - y

    return d_wall.astype(np.float32), d_top.astype(np.float32)


def compute_local_pca_features(
    pos: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each hit, compute local PCA in its k-NN neighborhood.

    Returns:
      linearity  : (N,)  (λ1 - λ2) / λ1
      sphericity : (N,)  λ3 / λ1
      ty         : (N,)  y-component of principal direction
      tz         : (N,)  z-component of principal direction
    """
    N = pos.shape[0]
    if N == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    k_eff = min(k, N)
    nbrs = NearestNeighbors(
        n_neighbors=k_eff,
        algorithm="kd_tree",
    )
    nbrs.fit(pos)
    _, indices = nbrs.kneighbors(pos, return_distance=True)  # (N, k_eff)

    linearity = np.zeros(N, dtype=np.float32)
    sphericity = np.zeros(N, dtype=np.float32)
    ty = np.zeros(N, dtype=np.float32)
    tz = np.zeros(N, dtype=np.float32)

    for i in range(N):
        neigh_idx = indices[i]
        pts = pos[neigh_idx]  # (k_eff, 3)
        if pts.shape[0] < 3:
            continue

        # Center
        c = pts.mean(axis=0, keepdims=True)
        X = pts - c

        # Covariance
        C = X.T @ X / float(X.shape[0])

        # Eigen-decomposition
        vals, vecs = np.linalg.eigh(C)  # vals ascending
        # Sort descending
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        lam1, lam2, lam3 = vals
        denom = lam1 if lam1 > 1e-9 else 1e-9

        linearity[i] = (lam1 - lam2) / denom
        sphericity[i] = lam3 / denom

        # Principal direction
        v1 = vecs[:, 0]  # (3,)
        # Normalize sign not important; we care about absolute orientation
        ty[i] = float(v1[1])
        tz[i] = float(v1[2])

    return linearity, sphericity, ty, tz


def main() -> int:
    ap = argparse.ArgumentParser(description="Add semantic sidecar features (distance + local PCA) to HDF5.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input H5 file (NuGraph-style)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output H5 file with sidecar features")
    ap.add_argument("--k", type=int, default=12, help="k for local PCA neighbors")
    ap.add_argument("--max-events", type=int, default=None, help="Optional limit on number of events to process")
    args = ap.parse_args()

    in_path = args.in_path
    out_path = args.out_path
    k = args.k

    print(f"[Info] Input : {in_path}")
    print(f"[Info] Output: {out_path}")
    print(f"[Info] k for local PCA: {k}")

    # 1) Infer global bounds
    x_min, x_max, y_min, y_max, z_min, z_max = compute_bounds(in_path)

    # 2) Copy input file -> output file and add sidecar group
    with h5py.File(in_path, "r") as fin, h5py.File(out_path, "w") as fout:
        # Copy attributes
        for k_attr, v in fin.attrs.items():
            fout.attrs[k_attr] = v

        # Copy all top-level groups/datasets as-is
        for name, obj in fin.items():
            fin.copy(obj, fout, name=name)
            print(f"[Copy] top-level '{name}'")

        ds_in = fin["dataset"]
        keys = list(ds_in.keys())
        total_events = len(keys)
        print(f"[Info] Total events in file: {total_events}")

        if args.max_events is not None:
            keys = keys[: args.max_events]
        print(f"[Info] Events to process   : {len(keys)}")

        # Create sidecar group
        sem_grp = fout.require_group("sem_features")
        sp_grp = sem_grp.require_group("sp")

        # 3) Per-event processing
        for i, key in enumerate(keys):
            print(f"[Proc] Event {i}/{len(keys)}  key={key}")

            rec = ds_in[key][()]  # scalar compound
            pos = np.asarray(rec["sp/pos"])
            if pos.size == 0:
                # still create an empty dataset so indexing stays consistent
                sp_grp.create_dataset(key, data=np.zeros((0, 6), dtype=np.float32))
                continue

            # Distances
            d_wall, d_top = compute_distances_to_walls(
                pos, x_min, x_max, y_min, y_max, z_min, z_max
            )

            # Local PCA features
            linearity, sphericity, ty, tz = compute_local_pca_features(pos, k=k)

            # Stack into (N, 6)
            sem_features = np.stack(
                [d_wall, d_top, linearity, sphericity, ty, tz],
                axis=-1,
            ).astype(np.float32)

            # Sanity check: N must match sp/features first dim
            n_sp = np.asarray(rec["sp/features"]).shape[0]
            if sem_features.shape[0] != n_sp:
                print(
                    f"[Warn] Event {key}: sem_features N={sem_features.shape[0]} "
                    f"!= sp/features N={n_sp}; truncating to min."
                )
                n = min(sem_features.shape[0], n_sp)
                sem_features = sem_features[:n, :]

            # Write sidecar dataset for this event
            sp_grp.create_dataset(
                key,
                data=sem_features,
                compression="gzip",
                shuffle=True,
                dtype="f4",
            )

    print("[Done] Wrote sidecar features to", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
