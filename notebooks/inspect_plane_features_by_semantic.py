# filename: inspect_plane_features_by_semantic.py
#!/usr/bin/env python
"""
Inspect per-plane features (u/x, v/x, y/x) and see how they differ
between neutrino and cosmic classes, using u/v/y/y_semantic and
semantic_classes from the H5.

This is meant to answer:
  - "Did vertex + sidecar actually get baked into u/v/y/x?"
  - "Do those extra columns show any separation between nu and cosmic?"

Usage:
  python inspect_plane_features_by_semantic.py \
      --data-path /path/to/23334072_nug4_vertex_test.h5 \
      --plane u \
      --n-events 50

Plane choices: u, v, y, or 'all' to inspect each plane in turn.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-path",
        required=True,
        help="NuGraph HDF5 file (e.g. 23334072_nug4_vertex_test.h5)",
    )
    p.add_argument(
        "--plane",
        type=str,
        default="u",
        help="Plane to inspect: 'u', 'v', 'y', or 'all'. Default: u",
    )
    p.add_argument(
        "--n-events",
        type=int,
        default=50,
        help="Max number of events to scan from the train split.",
    )
    return p.parse_args()


def _get_semantic_indices(semantic_classes):
    """Return (nu_idx, cosmic_idx) based on semantic_classes list."""
    try:
        nu_idx = int(semantic_classes.index("nu"))
    except ValueError:
        raise RuntimeError(
            "Could not find 'nu' in semantic_classes. "
            f"Got: {semantic_classes}"
        )

    cosmic_idx = None
    for name in ("cosmic", "cos", "cosmic_other", "other"):
        if name in semantic_classes:
            cosmic_idx = int(semantic_classes.index(name))
            break
    if cosmic_idx is None:
        raise RuntimeError(
            "Could not find a cosmic-like class name in semantic_classes. "
            f"Got: {semantic_classes}"
        )

    return nu_idx, cosmic_idx


def inspect_plane(f, keys, plane_name, nu_idx, cosmic_idx):
    """
    Accumulate stats for one plane (u/v/y) across selected events.
    """
    feat_key = f"{plane_name}/x"
    ysem_key = f"{plane_name}/y_semantic"

    # Find feature dimensionality from the first event that has this plane
    n_feat = None
    for key in keys:
        rec = f["dataset"][key][()]
        if feat_key in rec.dtype.names and ysem_key in rec.dtype.names:
            arr = np.asarray(rec[feat_key])
            if arr.size == 0:
                continue
            n_feat = arr.shape[1]
            break

    if n_feat is None:
        print(f"[{plane_name}-plane] No usable events found with {feat_key}. Skipping.")
        return

    # Decompose feature indices into base + vertex + sidecar
    base_dim = 5  # [Qtot, mean_err, nhits, pitch_min, pitch_max]
    vertex_dim = 4  # [vtx_dist, vtx_dx, vtx_dy, vtx_dz]
    if n_feat < base_dim:
        raise RuntimeError(
            f"[{plane_name}-plane] Expected at least {base_dim} features, found {n_feat}."
        )
    if n_feat < base_dim + vertex_dim:
        print(
            f"[{plane_name}-plane] WARNING: only {n_feat} features, "
            f"so some or all vertex dims may be missing."
        )
        vertex_dim = max(0, n_feat - base_dim)
    sidecar_dim = max(0, n_feat - base_dim - vertex_dim)

    print(f"\n[{plane_name}-plane] Feature dimension F = {n_feat}")
    print(f"  base_dim   = {base_dim}  (0..{base_dim-1})")
    print(f"  vertex_dim = {vertex_dim} ({base_dim}..{base_dim+vertex_dim-1})")
    if sidecar_dim > 0:
        print(
            f"  sidecar_dim = {sidecar_dim} "
            f"({base_dim+vertex_dim}..{n_feat-1})"
        )
    else:
        print("  sidecar_dim = 0 (no extra sidecar features detected)")

    nu_vals = [[] for _ in range(n_feat)]
    cos_vals = [[] for _ in range(n_feat)]

    # Loop over events and accumulate per-feature nu/cos distributions
    for key in keys:
        rec = f["dataset"][key][()]
        if feat_key not in rec.dtype.names or ysem_key not in rec.dtype.names:
            continue

        feat = np.asarray(rec[feat_key])      # (N_nodes, F)
        ysem = np.asarray(rec[ysem_key])      # (N_nodes,)

        if feat.size == 0 or ysem.size == 0:
            continue
        if feat.shape[0] != ysem.shape[0]:
            print(
                f"[{plane_name}-plane] WARN: {key} has feat.shape[0]={feat.shape[0]} "
                f"!= y_semantic.shape[0]={ysem.shape[0]} -> skipping"
            )
            continue

        mask_nu = (ysem == nu_idx)
        mask_cos = (ysem == cosmic_idx)

        if not mask_nu.any() and not mask_cos.any():
            continue

        for j in range(n_feat):
            col = feat[:, j]

            if mask_nu.any():
                vals_nu = col[mask_nu]
                vals_nu = vals_nu[np.isfinite(vals_nu)]
                if vals_nu.size > 0:
                    nu_vals[j].append(vals_nu)

            if mask_cos.any():
                vals_cos = col[mask_cos]
                vals_cos = vals_cos[np.isfinite(vals_cos)]
                if vals_cos.size > 0:
                    cos_vals[j].append(vals_cos)

    # Print stats
    print(
        f"\n[{plane_name}-plane] Stats per feature column (nu vs cosmic):\n"
        "col | kind      | n     mean      std       min       max"
    )

    def _kind(j: int) -> str:
        if j < base_dim:
            return "base   "
        elif j < base_dim + vertex_dim:
            return "vertex "
        else:
            return "sidecar"

    def _fmt(arr):
        if arr.size == 0:
            return (0, np.nan, np.nan, np.nan, np.nan)
        return (
            arr.size,
            float(arr.mean()),
            float(arr.std()),
            float(arr.min()),
            float(arr.max()),
        )

    for j in range(n_feat):
        arr_nu = np.concatenate(nu_vals[j]) if nu_vals[j] else np.array([], dtype=float)
        arr_cos = np.concatenate(cos_vals[j]) if cos_vals[j] else np.array([], dtype=float)

        n_nu, m_nu, s_nu, mn_nu, mx_nu = _fmt(arr_nu)
        n_c, m_c, s_c, mn_c, mx_c = _fmt(arr_cos)

        print(
            f"{j:3d} | { _kind(j) } nu   "
            f"{n_nu:6d} {m_nu:9.4g} {s_nu:9.4g} {mn_nu:9.4g} {mx_nu:9.4g}"
        )
        print(
            f"    | { _kind(j) } cos  "
            f"{n_c:6d} {m_c:9.4g} {s_c:9.4g} {mn_c:9.4g} {mx_c:9.4g}"
        )


def main():
    args = parse_args()
    path = Path(args.data_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        # semantic class names
        if "semantic_classes" not in f:
            raise RuntimeError("File has no 'semantic_classes' dataset.")

        semantic_classes = f["semantic_classes"].asstr()[()].tolist()
        print("[Info] semantic_classes:", semantic_classes)
        nu_idx, cosmic_idx = _get_semantic_indices(semantic_classes)
        print(f"[Info] Using nu_idx={nu_idx}, cosmic_idx={cosmic_idx}")

        # which events to scan?
        if "samples/train" in f:
            keys = f["samples/train"][()]
        else:
            keys = list(f["dataset"].keys())
        keys = [k.decode() if isinstance(k, bytes) else k for k in keys]

        if not keys:
            raise RuntimeError("No events found in file.")

        keys = keys[: args.n_events]
        print(f"[Info] Inspecting {len(keys)} train events from {path.name}")

        planes = ["u", "v", "y"] if args.plane == "all" else [args.plane]

        for plane_name in planes:
            inspect_plane(f, keys, plane_name, nu_idx, cosmic_idx)


if __name__ == "__main__":
    main()
