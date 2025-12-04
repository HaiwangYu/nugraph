#!/usr/bin/env python3
import h5py
import numpy as np
import sys

H5 = "/lus/eagle/projects/neutrinoGPU/abhat/sbnd/clustering/23334072_nug4_vertex.h5"

def main():
    h5file = sys.argv[1] if len(sys.argv) > 1 else H5

    with h5py.File(h5file, "r") as f:
        keys = list(f["dataset"].keys())
        print(f"Total events: {len(keys)}")
        max_check = min(20, len(keys))

        for i, k in enumerate(keys[:max_check]):
            rec = f["dataset"][k][()]  # scalar compound

            if "sp/y_instance" in rec.dtype.names:
                y_inst = np.asarray(rec["sp/y_instance"])
            else:
                y_inst = None

            if "sp/features" in rec.dtype.names:
                feats = np.asarray(rec["sp/features"])
            else:
                feats = None

            print(f"\nEvent {i} ({k}):")

            if y_inst is not None and y_inst.size > 0:
                u = np.unique(y_inst)
                print(f"  sp/y_instance uniques: {u[:20]} (len={len(u)})")
            else:
                print("  sp/y_instance: MISSING or empty")

            if feats is not None and feats.size > 0:
                print(f"  sp.features shape: {feats.shape}")
                if feats.shape[1] >= 6:
                    charge = feats[:, 0]
                    cluster = feats[:, 1]
                    vtx_dist = feats[:, 2]
                    print(f"    charge  min/max: {charge.min():.2f} / {charge.max():.2f}")
                    print(f"    cluster uniques (first 20): {np.unique(cluster)[:20]}")
                    print(f"    vtx_dist min/max: {vtx_dist.min():.2f} / {vtx_dist.max():.2f}")
            else:
                print("  sp.features: MISSING or empty")

if __name__ == "__main__":
    main()
