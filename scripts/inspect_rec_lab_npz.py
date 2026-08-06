#!/usr/bin/env python
# filename: inspect_rec_lab_npz.py
"""
Quick inspection of a labeled rec NPZ file (rec-lab-apa*-*.npz).

Prints:
  - list of keys
  - shapes/dtypes
  - basic stats for is_nu / origin_label / vertex-distance features
"""

import sys
import numpy as np

def main(path: str) -> None:
    print(f"Loading NPZ: {path}")
    data = np.load(path)
    keys = sorted(list(data.files))
    print("\n=== Keys in file ===")
    for k in keys:
        arr = data[k]
        print(f"  {k:20s} shape={arr.shape} dtype={arr.dtype}")

    # --- Core geometry / points ---
    if "points" in data:
        pts = data["points"]
        print("\n[points]")
        print(f"  shape: {pts.shape}")
        if pts.size:
            print(f"  first row: {pts[0]}")
        # remind ourselves of columns
        print("  (remember: typically [x, y, z, charge, blob_idx, cluster_idx, ...])")

    # --- Original is_nu labels ---
    if "is_nu" in data:
        is_nu = data["is_nu"]
        print("\n[is_nu]")
        print(f"  shape: {is_nu.shape}, dtype={is_nu.dtype}")
        uniq, counts = np.unique(is_nu, return_counts=True)
        print("  unique values / counts:")
        for u, c in zip(uniq, counts):
            print(f"    {u:4d}: {c}")

    # --- Origin labels (nu / cosmic / other) ---
    if "origin_label" in data:
        ori = data["origin_label"]
        print("\n[origin_label]  (e.g. 0=nu,1=cosmic,2=other)")
        print(f"  shape: {ori.shape}, dtype={ori.dtype}")
        uniq, counts = np.unique(ori, return_counts=True)
        print("  unique values / counts:")
        for u, c in zip(uniq, counts):
            print(f"    {u:4d}: {c}")

    # --- Distance to true nu vertex ---
    if "dist_to_nu_vtx" in data:
        d = data["dist_to_nu_vtx"]
        print("\n[dist_to_nu_vtx]  (cm)")
        print(f"  shape: {d.shape}, dtype={d.dtype}")
        print(f"  min / max: {d.min():.3f} / {d.max():.3f}")
        print(f"  mean / std: {d.mean():.3f} / {d.std():.3f}")

    if "dz_from_nu_vtx" in data:
        dz = data["dz_from_nu_vtx"]
        print("\n[dz_from_nu_vtx]  (cm)")
        print(f"  shape: {dz.shape}, dtype={dz.dtype}")
        print(f"  min / max: {dz.min():.3f} / {dz.max():.3f}")
        print(f"  mean / std: {dz.mean():.3f} / {dz.std():.3f}")

    # --- Stored vertex position (if present) ---
    for key in ("nu_vtx", "nu_vertex", "nu_vertex_xyz"):
        if key in data:
            vtx = data[key]
            print(f"\n[{key}]")
            print(f"  shape: {vtx.shape}, dtype={vtx.dtype}")
            if vtx.size:
                print(f"  first entry: {vtx[0]}")
            break

    # --- Sanity check alignment ---
    if "points" in data and "is_nu" in data:
        pts = data["points"]
        is_nu = data["is_nu"]
        if pts.shape[0] == is_nu.shape[0]:
            print("\n[Sanity] points and is_nu lengths match ✅")
        else:
            print("\n[Sanity] WARNING: points.shape[0] != is_nu.shape[0] ❌")

    print("\nDone.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_rec_lab_npz.py /path/to/rec-lab-apa0-0.npz")
        sys.exit(1)
    main(sys.argv[1])
