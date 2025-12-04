#!/usr/bin/env python
"""
inspect_vertex_in_h5.py

Inspect the raw HDF5 'dataset' records and print stats for any fields that
look vertex-related (names containing 'vtx' or 'vertex'), per plane.

Usage:
  python inspect_vertex_in_h5.py --data-path /path/to/23334072_nug4_vertex.h5 \
                                 --n-events 100

This does NOT go through NuGraphDataModule/transform; it inspects the
HDF5 file directly so we stop guessing.
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
        default=50,
        help="How many events to sample for stats (max).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.data_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        # Figure out which event keys to use
        if "samples/train" in f:
            sample_keys = f["samples/train"][()]
        else:
            # fallback: use all dataset keys
            sample_keys = list(f["dataset"].keys())

        # h5py returns bytes sometimes; make them str
        sample_keys = [k.decode() if isinstance(k, bytes) else k for k in sample_keys]
        if len(sample_keys) == 0:
            raise RuntimeError("No samples found in file.")

        sample_keys = sample_keys[: args.n_events]
        print(f"[Info] Inspecting {len(sample_keys)} events from {path.name}")

        # Look at the dtype of one dataset record to find all vtx-like fields
        first_rec = f["dataset"][sample_keys[0]][()]
        field_names = list(first_rec.dtype.names)

        # only keep fields that look vertex related
        vtx_fields = [n for n in field_names if ("vtx" in n.lower()) or ("vertex" in n.lower())]
        if not vtx_fields:
            print("No fields with 'vtx' or 'vertex' in their name were found in dataset dtype.")
            print("Available fields include (first 50):")
            print(field_names[:50])
            return

        print("\n[Info] Found vertex-like fields in dataset dtype:")
        for n in vtx_fields:
            print("  -", n)

        # Now accumulate stats per field (and per plane if applicable)
        # Many schemas use names like 'u_vtx_dist', 'v_vtx_dist', 'y_vtx_dist', etc.
        stats = {}

        for key in sample_keys:
            rec = f["dataset"][key][()]  # scalar compound

            for fld in vtx_fields:
                arr = rec[fld]

                # Some fields may be arrays per plane, some scalar; normalize to 1D
                arr = np.asarray(arr).ravel()
                if arr.size == 0:
                    continue

                if fld not in stats:
                    stats[fld] = []
                stats[fld].append(arr)

        # Print stats
        print("\n[Stats over sampled events]")
        for fld, chunks in stats.items():
            vals = np.concatenate(chunks) if chunks else np.array([], dtype=float)
            if vals.size == 0:
                print(f"  {fld}: no data collected.")
                continue

            finite = np.isfinite(vals)
            if not finite.any():
                print(f"  {fld}: all values are non-finite (NaN/Inf).")
                continue

            vals_f = vals[finite]
            print(
                f"  {fld}: n={vals_f.size}, "
                f"mean={vals_f.mean():.4g}, std={vals_f.std():.4g}, "
                f"min={vals_f.min():.4g}, max={vals_f.max():.4g}"
            )


if __name__ == "__main__":
    main()
