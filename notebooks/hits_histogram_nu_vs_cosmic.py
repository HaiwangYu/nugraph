#!/usr/bin/env python3
"""
hits_histogram_strict01.py

Count per-event hits with a STRICT label mapping:
  neutrino = 1
  cosmic   = 0
All other labels (e.g., -1 non-trackable, -2 no match) are ignored.

Sums across U/V/Y plane label arrays: {u,v,y}/y_semantic

Outputs summary stats, a PNG histogram, and (optionally) a CSV of counts.

Example:
  python hits_histogram_strict01.py /path/to/23334072.h5 --max-events 1000 \
    --out-png nu_cosmic_hist_strict01.png --out-csv nu_cosmic_counts_strict01.csv
"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

try:
    import h5py
except ImportError:
    print("This script requires h5py. Try: pip install h5py", file=sys.stderr)
    sys.exit(1)

import matplotlib.pyplot as plt


def list_event_keys(h: h5py.File) -> List[str]:
    """Return sorted list of scalar compound datasets under '/dataset'."""
    out = []
    if "dataset" not in h:
        return out
    for k in sorted(h["dataset"].keys()):
        d = h["dataset"][k]
        if isinstance(d, h5py.Dataset) and d.shape == ():
            out.append(f"dataset/{k}")
    return out


def count_plane(labels: np.ndarray) -> Tuple[int, int]:
    """
    STRICT mapping:
      neutrino = 0
      cosmic   = 1
    Ignore others.
    Returns (nu_hits, cosmic_hits) for a single plane.
    """
    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)
    # cast to int
    labels = labels.astype(np.int64, copy=False)
    nu_hits = int(np.count_nonzero(labels == 0))
    co_hits = int(np.count_nonzero(labels != 0))
    return nu_hits, co_hits


def count_event_strict01(rec: np.void) -> Tuple[int, int, dict]:
    """
    Sum across planes. Return (nu_total, cosmic_total, label_counts_all)
    label_counts_all is a small dict with counts of {-2, -1, 0, 1} seen across all planes.
    """
    nu_total = 0
    co_total = 0
    label_counts = { -2: 0, -1: 0, 0: 0, 1: 0 }

    for plane in ("u", "v", "y"):
        key = f"{plane}/y_semantic"
        if key not in rec.dtype.names:
            continue
        arr = rec[key]
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        arr = arr.astype(np.int64, copy=False)

        # strict counts
        nnu, nco = count_plane(arr)
        nu_total += nnu
        co_total += nco

        # keep a simple label distribution (only for common labels)
        for lab in (-2, -1, 0, 1):
            label_counts[lab] += int(np.count_nonzero(arr == lab))

    return nu_total, co_total, label_counts


def summarize(a: np.ndarray) -> dict:
    return {
        "N": int(a.size),
        "min": float(a.min()),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=" Distribution of neutrino vs cosmic hits per event")
    ap.add_argument("h5file", help="Path to Haiwang-style NuGraph HDF5")
    ap.add_argument("--max-events", type=int, default=None, help="Max events to process")
    ap.add_argument("--out-png", default="nu_cosmic_hist_strict01.png", help="Histogram PNG output")
    ap.add_argument("--out-csv", default=None, help="Optional CSV with per-event counts")
    ap.add_argument("--bins", type=int, default=60, help="Histogram bins")
    ap.add_argument("--show", action="store_true", help="Show plot interactively")
    args = ap.parse_args()

    if not os.path.exists(args.h5file):
        print(f"File not found: {args.h5file}", file=sys.stderr)
        sys.exit(1)

    nu_counts: List[int] = []
    co_counts: List[int] = []
    keys_kept: List[str] = []
    # optional sanity snapshot of the first few label distributions
    sanity_snapshots: List[tuple] = []

    with h5py.File(args.h5file, "r") as h:
        keys = list_event_keys(h)
        if not keys:
            print("No scalar compound datasets under '/dataset'.", file=sys.stderr)
            sys.exit(2)
        if args.max_events is not None:
            keys = keys[: args.max_events]

        for i, ds_key in enumerate(keys, 1):
            try:
                rec = h[ds_key][()]  # numpy.void
            except Exception as e:
                print(f"Warning: failed to read {ds_key}: {e}", file=sys.stderr)
                continue

            nnu, nco, lab_counts = count_event_strict01(rec)
            nu_counts.append(nnu)
            co_counts.append(nco)
            keys_kept.append(ds_key)

            # keep a few label distributions to print later (sanity check)
            if len(sanity_snapshots) < 5:
                sanity_snapshots.append((ds_key, lab_counts))

            if i % 1000 == 0:
                print(f"...processed {i} events", file=sys.stderr)

    if not nu_counts:
        print("No events processed.", file=sys.stderr)
        sys.exit(3)

    nu_arr = np.asarray(nu_counts, dtype=np.int64)
    co_arr = np.asarray(co_counts, dtype=np.int64)

    # Sanity: show first few per-event label distributions (sum across planes)
    print("\n=== Sanity: first few per-event label distributions (U+V+Y) ===")
    for k, lab in sanity_snapshots:
        print(f"{k}: {{-2: {lab[-2]}, -1: {lab[-1]}, 0: {lab[0]}, 1: {lab[1]}}}")

    print("\n=== Summary: neutrino(1) hits per event (STRICT) ===")
    for k, v in summarize(nu_arr).items():
        print(f"{k:>6s}: {v}")

    print("\n=== Summary: cosmic(0) hits per event (STRICT) ===")
    for k, v in summarize(co_arr).items():
        print(f"{k:>6s}: {v}")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["event_key", "nu_hits_1", "cosmic_hits_0"])
            for k, nnu, nco in zip(keys_kept, nu_arr, co_arr):
                w.writerow([k, int(nnu), int(nco)])
        print(f"\nWrote per-event counts CSV: {args.out_csv}")

    # Plot histograms
    plt.figure(figsize=(9, 5.5))
    plt.hist(nu_arr, bins=args.bins, alpha=0.6, label="Neutrino (label 1) hits / event")
    plt.hist(co_arr, bins=args.bins, alpha=0.6, label="Cosmic (label 0) hits / event")
    plt.xlabel("Hits per event")
    plt.ylabel("Count of events")
    plt.xlim(-10,500)
    plt.title("Neutrino (1) vs Cosmic (0) hits per event")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=500)
    print(f"Wrote histogram PNG: {args.out_png}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
