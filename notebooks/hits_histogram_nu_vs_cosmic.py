#!/usr/bin/env python3
"""
hits_histogram_nu_vs_cosmic.py

Count per-event hits with a STRICT label mapping based on the file's
'semantic_classes' order. We look up the indices for 'nu' and 'cosmic'
inside the HDF5 to avoid hard-coding.

Ignores other labels (e.g., -1 non-trackable, -2 no match).

Sums across U/V/Y plane label arrays: {u,v,y}/y_semantic
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


def resolve_label_indices(h: h5py.File) -> Tuple[int, int]:
    """
    Read semantic_classes from file and return (nu_idx, cosmic_idx).
    Falls back to (0,1) if metadata is missing, with a warning.
    """
    nu_idx, cosmic_idx = 0, 1
    try:
        classes = h["semantic_classes"].asstr()[()].tolist()
        if not isinstance(classes, list):
            classes = list(classes)
        classes = [str(c) for c in classes]
        assert "nu" in classes and "cosmic" in classes
        nu_idx = classes.index("nu")
        cosmic_idx = classes.index("cosmic")
        print(f"[Mapping] semantic_classes = {classes}", file=sys.stderr)
        print(f"[Mapping] 'nu' -> {nu_idx}, 'cosmic' -> {cosmic_idx}", file=sys.stderr)
    except Exception as e:
        print(f"[Mapping] WARNING: could not read semantic_classes, "
              f"falling back to nu=0, cosmic=1 ({e})", file=sys.stderr)
    return nu_idx, cosmic_idx


def count_plane(labels: np.ndarray, nu_idx: int, cosmic_idx: int) -> Tuple[int, int]:
    """
    STRICT counting on a single plane:
      neutrino = nu_idx
      cosmic   = cosmic_idx
    Ignore others (incl. -1/-2).
    Returns (nu_hits, cosmic_hits).
    """
    if not isinstance(labels, np.ndarray):
        labels = np.asarray(labels)
    labels = labels.astype(np.int64, copy=False)
    nu_hits = int(np.count_nonzero(labels == nu_idx))
    co_hits = int(np.count_nonzero(labels == cosmic_idx))
    return nu_hits, co_hits


def count_event(rec: np.void, nu_idx: int, cosmic_idx: int) -> Tuple[int, int, dict]:
    """
    Sum across planes. Return (nu_total, cosmic_total, label_counts_all)
    label_counts_all includes counts of {-2, -1, nu_idx, cosmic_idx}.
    """
    nu_total = 0
    co_total = 0
    label_counts = { -2: 0, -1: 0, nu_idx: 0, cosmic_idx: 0 }

    for plane in ("u", "v", "y"):
        key = f"{plane}/y_semantic"
        if key not in rec.dtype.names:
            continue
        arr = rec[key]
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        arr = arr.astype(np.int64, copy=False)

        # strict counts (ignore everything else)
        nnu, nco = count_plane(arr, nu_idx, cosmic_idx)
        nu_total += nnu
        co_total += nco

        # keep simple label distribution for sanity
        for lab in (-2, -1, nu_idx, cosmic_idx):
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
    ap = argparse.ArgumentParser(description="Distribution of neutrino vs cosmic hits per event")
    ap.add_argument("h5file", help="Path to NuGraph HDF5")
    ap.add_argument("--max-events", type=int, default=None, help="Max events to process")
    ap.add_argument("--out-png", default="nu_cosmic_hist.png", help="Histogram PNG output")
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
    sanity_snapshots: List[tuple] = []

    with h5py.File(args.h5file, "r") as h:
        nu_idx, cosmic_idx = resolve_label_indices(h)
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

            nnu, nco, lab_counts = count_event(rec, nu_idx, cosmic_idx)
            nu_counts.append(nnu)
            co_counts.append(nco)
            keys_kept.append(ds_key)

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
        print(f"{k}: {{-2: {lab.get(-2,0)}, -1: {lab.get(-1,0)}, "
              f"nu({nu_idx}): {lab.get(nu_idx,0)}, cosmic({cosmic_idx}): {lab.get(cosmic_idx,0)}}}")

    print(f"\n=== Summary: neutrino (label {nu_idx}) hits per event (STRICT) ===")
    for k, v in summarize(nu_arr).items():
        print(f"{k:>6s}: {v}")

    print(f"\n=== Summary: cosmic (label {cosmic_idx}) hits per event (STRICT) ===")
    for k, v in summarize(co_arr).items():
        print(f"{k:>6s}: {v}")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["event_key", f"nu_hits_label_{nu_idx}", f"cosmic_hits_label_{cosmic_idx}"])
            for k, nnu, nco in zip(keys_kept, nu_arr, co_arr):
                w.writerow([k, int(nnu), int(nco)])
        print(f"\nWrote per-event counts CSV: {args.out_csv}")

    # Plot histograms
    plt.figure(figsize=(9, 5.5))
    plt.hist(nu_arr, bins=args.bins, alpha=0.6, label=f"Neutrino (label {nu_idx}) hits / event")
    plt.hist(co_arr, bins=args.bins, alpha=0.6, label=f"Cosmic (label {cosmic_idx}) hits / event")
    plt.xlabel("Hits per event")
    plt.ylabel("Count of events")
    # plt.xlim(-10, 500)
    plt.title(f"Neutrino (label {nu_idx}) vs Cosmic (label {cosmic_idx}) hits per event")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=500)
    print(f"Wrote histogram PNG: {args.out_png}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
