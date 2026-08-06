#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser(
        description="Inspect a rec-lab-apa*.npz file for available arrays (e.g. vertex info)."
    )
    ap.add_argument("file", help="Path to rec-lab-apa*.npz")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")

    data = np.load(path, allow_pickle=True)
    print(f"=== {path} ===")
    print("Keys:")
    for k in data.files:
        arr = data[k]
        shape = getattr(arr, "shape", None)
        dtype = getattr(arr, "dtype", None)
        print(f"  - {k}: shape={shape}, dtype={dtype}")

    # Look for anything that smells like a vertex
    print("\nHeuristic search for vertex-like keys:")
    candidates = []
    for k in data.files:
        kl = k.lower()
        if ("vtx" in kl) or ("vertex" in kl) or ("nu_v" in kl) or ("nuvtx" in kl):
            candidates.append(k)

    if not candidates:
        print("  (No obvious vertex-related keys found.)")
    else:
        for k in candidates:
            arr = data[k]
            print(f"  * {k}: shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)}")
            # If it's small, print the actual values
            if hasattr(arr, "size") and arr.size <= 20:
                print(f"    values = {arr}")

if __name__ == "__main__":
    main()
