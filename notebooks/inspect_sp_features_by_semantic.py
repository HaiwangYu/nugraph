#!/usr/bin/env python
"""
inspect_sp_features_by_semantic.py

Inspect the spacepoint feature columns (sp/features) and see how they differ
between neutrino and cosmic classes, using sp/y_semantic and semantic_classes.

This is meant to answer:
  - "Do the columns that (should) contain vertex/dist info actually encode
     something useful for nu vs cosmic semantics?"

Usage:
  python inspect_sp_features_by_semantic.py \
      --data-path /lus/eagle/projects/neutrinoGPU/abhat/sbnd/clustering/23334072_nug4_vertex.h5 \
      --n-events 200
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
        help="NuGraph HDF5 file (e.g. 23334072_nug4_vertex.h5)",
    )
    p.add_argument(
        "--n-events",
        type=int,
        default=200,
        help="Max number of events to scan.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.data_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        # --- semantic class names ---
        if "semantic_classes" not in f:
            raise RuntimeError("File has no 'semantic_classes' dataset.")

        semantic_classes = f["semantic_classes"].asstr()[()].tolist()
        print("[Info] semantic_classes:", semantic_classes)

        try:
            nu_idx = int(semantic_classes.index("nu"))
        except ValueError:
            raise RuntimeError(
                "Could not find 'nu' in semantic_classes. "
                f"Got: {semantic_classes}"
            )

        # Try to find 'cosmic'-like class
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

        print(f"[Info] Using nu_idx={nu_idx}, cosmic_idx={cosmic_idx}")

        # --- which events to scan? ---
        if "samples/train" in f:
            keys = f["samples/train"][()]
        else:
            keys = list(f["dataset"].keys())

        keys = [k.decode() if isinstance(k, bytes) else k for k in keys]
        if len(keys) == 0:
            raise RuntimeError("No events found in file.")

        keys = keys[: args.n_events]
        print(f"[Info] Inspecting {len(keys)} events from {path.name}")

        # --- find feature dimensionality from first event ---
        first_rec = f["dataset"][keys[0]][()]
        if "sp/features" not in first_rec.dtype.names:
            raise RuntimeError("Record has no 'sp/features' field.")

        feat0 = np.asarray(first_rec["sp/features"])
        if feat0.ndim != 2:
            raise RuntimeError(
                f"sp/features field has shape {feat0.shape}, expected (N_sp, F)."
            )

        n_feat = feat0.shape[1]
        print(f"[Info] sp/features has dimension F = {n_feat}")

        # accumulators: we will keep concatenated arrays per class for simplicity
        nu_vals = [[] for _ in range(n_feat)]
        cos_vals = [[] for _ in range(n_feat)]

        # --- loop over events ---
        for key in keys:
            rec = f["dataset"][key][()]

            feat = np.asarray(rec["sp/features"])      # (N_sp, F)
            ysem = np.asarray(rec["sp/y_semantic"])    # (N_sp,)

            if feat.shape[0] == 0:
                continue
            if ysem.shape[0] != feat.shape[0]:
                # Something's wrong with alignment
                print(f"[Warn] {key}: sp/features and sp/y_semantic length mismatch: "
                      f"{feat.shape[0]} vs {ysem.shape[0]}")
                continue

            mask_nu = (ysem == nu_idx)
            mask_cos = (ysem == cosmic_idx)

            if not mask_nu.any() and not mask_cos.any():
                continue

            # per-feature collection
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

        # --- compute stats ---
        print("\n[Stats per sp/features column (nu vs cosmic)]")
        print("col | n_nu   mean_nu   std_nu    min_nu    max_nu  ||  "
              "n_cos  mean_cos  std_cos   min_cos   max_cos")

        for j in range(n_feat):
            if nu_vals[j]:
                arr_nu = np.concatenate(nu_vals[j])
            else:
                arr_nu = np.array([], dtype=float)

            if cos_vals[j]:
                arr_cos = np.concatenate(cos_vals[j])
            else:
                arr_cos = np.array([], dtype=float)

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

            n_nu, m_nu, s_nu, mn_nu, mx_nu = _fmt(arr_nu)
            n_c, m_c, s_c, mn_c, mx_c = _fmt(arr_cos)

            print(
                f"{j:3d} | "
                f"{n_nu:6d} {m_nu:9.4g} {s_nu:9.4g} {mn_nu:9.4g} {mx_nu:9.4g}  ||  "
                f"{n_c:6d} {m_c:9.4g} {s_c:9.4g} {mn_c:9.4g} {mx_c:9.4g}"
            )


if __name__ == "__main__":
    main()
