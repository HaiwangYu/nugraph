#!/usr/bin/env python3
"""
peek_nugraph_record.py — Inspect Haiwang-style NuGraph event records in HDF5.

This script:
  1) Lists event dataset keys under '/dataset'
  2) Loads one dataset (one event) into a dict[str, np.ndarray | scalar]
  3) Prints a clean, readable shape summary
  4) (Optional) Exports that single event to NPZ for quick experiments

Example:
  python peek_nugraph_record.py /scratch/7DayLifetime/yuhw/wirecell/nugraph/data/23334072.h5
  python peek_nugraph_record.py /scratch/.../23334072.h5 --key dataset/23334072_113_rec-lab-apa1-4
  python peek_nugraph_record.py /scratch/.../23334072.h5 --export foo_event.npz
"""

import argparse
import os
import sys
from typing import Dict, Any, Tuple

import numpy as np

try:
    import h5py
except ImportError:
    print("This script requires h5py. Try: pip install h5py", file=sys.stderr)
    sys.exit(1)


def list_event_keys(h: h5py.File) -> list:
    """Return sorted list of dataset keys under '/dataset'."""
    keys = []
    if "dataset" not in h:
        return keys
    grp = h["dataset"]
    for k in sorted(grp.keys()):
        obj = grp.get(k, getlink=False)
        if isinstance(obj, h5py.Dataset):
            # One event per dataset (compound scalar)
            keys.append(f"dataset/{k}")
    return keys


def load_event_record(h: h5py.File, ds_key: str) -> Dict[str, Any]:
    """
    Load a single event dataset (compound scalar) into a flat dict:
      field_name -> numpy array or Python scalar.
    Works with fields like 'u/pos', 'v/x', 'evt/y', etc.
    """
    if ds_key not in h:
        # try without the leading slash if user passed that
        ds_key = ds_key.lstrip("/")
        if ds_key not in h:
            raise KeyError(f"Dataset not found: {ds_key}")

    dset = h[ds_key]
    if dset.shape != ():
        raise ValueError(f"Expected a scalar compound dataset; got shape={dset.shape}")

    # Read the scalar compound into a numpy.void
    rec = dset[()]  # numpy.void with fields
    out: Dict[str, Any] = {}

    # dset.dtype.names are the field names like 'u/pos', 'v/x', 'evt/y', etc.
    for name in dset.dtype.names:
        val = rec[name]
        # Normalize scalars to Python ints/floats for readability
        if isinstance(val, np.ndarray):
            out[name] = val
        else:
            # numpy scalar or Python scalar
            if hasattr(val, "item"):
                out[name] = val.item()
            else:
                out[name] = val
    return out


def summarize_event(event: Dict[str, Any]) -> str:
    """
    Produce a readable, grouped summary of the event dict:
      - metadata/*
      - sp/* (spacepoints)
      - per-plane (u, v, y)
      - graph edges (*_plane_*/*edge_index, *_nexus_sp/*edge_index)
      - evt/*
    """
    # Helper: group keys by top-prefix before first '/'
    groups = {}
    for k in event.keys():
        if "/" in k:
            top = k.split("/", 1)[0]
        else:
            # keys like 'evt' might be top-level without slash in summary, but we expect 'evt/...'
            top = k
        groups.setdefault(top, []).append(k)

    lines = []
    def add(title):
        lines.append(f"\n[{title}]")

    # metadata
    if "metadata" in groups:
        add("metadata")
        for k in sorted(groups["metadata"]):
            v = event[k]
            lines.append(f"  {k:28s} = {v}")

    # spacepoints
    if "sp" in groups:
        add("spacepoints (sp)")
        for k in sorted(groups["sp"]):
            v = event[k]
            if isinstance(v, np.ndarray):
                lines.append(f"  {k:28s} shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                lines.append(f"  {k:28s} = {v}")

    # planes u, v, y
    for plane in ("u", "v", "y"):
        if plane in groups:
            add(f"plane '{plane}'")
            for k in sorted(groups[plane]):
                v = event[k]
                if isinstance(v, np.ndarray):
                    lines.append(f"  {k:28s} shape={tuple(v.shape)} dtype={v.dtype}")
                else:
                    lines.append(f"  {k:28s} = {v}")

    # edges (catch anything with '/edge_index')
    edge_keys = [k for k in event.keys() if k.endswith("/edge_index")]
    if edge_keys:
        add("graph edges (edge_index)")
        for k in sorted(edge_keys):
            v = event[k]
            if isinstance(v, np.ndarray):
                lines.append(f"  {k:28s} shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                lines.append(f"  {k:28s} = {v}")

    # evt/*
    evt_keys = [k for k in event.keys() if k.startswith("evt/")]
    if evt_keys:
        add("event-level (evt)")
        for k in sorted(evt_keys):
            v = event[k]
            if isinstance(v, np.ndarray):
                lines.append(f"  {k:28s} shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                lines.append(f"  {k:28s} = {v}")

    # any leftovers
    leftovers = sorted(set(event.keys()) - set(sum(groups.values(), [])))
    leftovers = [k for k in leftovers if not k.startswith("evt/")]
    if leftovers:
        add("other")
        for k in leftovers:
            v = event[k]
            if isinstance(v, np.ndarray):
                lines.append(f"  {k:28s} shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                lines.append(f"  {k:28s} = {v}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Peek structured NuGraph event datasets")
    ap.add_argument("h5file", help="Path to Haiwang-style NuGraph HDF5")
    ap.add_argument("--key", help="Full dataset key, e.g., dataset/23334072_113_rec-lab-apa1-4")
    ap.add_argument("--export", help="Optional path to save the loaded event as NPZ")
    args = ap.parse_args()

    if not os.path.exists(args.h5file):
        print(f"File not found: {args.h5file}", file=sys.stderr)
        sys.exit(1)

    with h5py.File(args.h5file, "r") as h:
        keys = list_event_keys(h)
        if not keys:
            print("No datasets found under '/dataset'.")
            sys.exit(0)

        print("=== Available event datasets (first 20) ===")
        for k in keys[:20]:
            print(" ", k)
        if len(keys) > 20:
            print(f"  ... and {len(keys) - 20} more")

        target = args.key or keys[0]
        if target not in h:
            # tolerate leading slash variations
            t2 = target.lstrip("/")
            if t2 in h:
                target = t2
            else:
                print(f"\nRequested key not found: {target}\n", file=sys.stderr)
                sys.exit(1)

        print(f"\n=== Loading event: {target} ===")
        event = load_event_record(h, target)

    # Print a concise summary
    print(summarize_event(event))

    # Optional export for notebook tinkering
    if args.export:
        # Flatten names: replace '/' with '__' for NPZ keys
        flat = {}
        for k, v in event.items():
            kk = k.replace("/", "__")
            flat[kk] = v
        np.savez_compressed(args.export, **flat)
        print(f"\nSaved event to: {args.export}")


if __name__ == "__main__":
    main()
