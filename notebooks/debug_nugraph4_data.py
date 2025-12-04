# filename: debug_nugraph4_data.py
#!/usr/bin/env python
"""
Debug NuGraph4 data pipeline:
- Inspect H5 structure
- Verify 'nu' label index
- Check per-plane semantic/instance arrays
- Check what NuGraphDataset.get() actually returns for hit store
"""

import argparse
from collections import Counter

import h5py
import numpy as np
import torch

from nugraph.data.dataset import NuGraphDataset  # your dataset.py
from pynuml.data import NuGraphData              # for consistency


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True, help="NuGraph HDF5 file")
    p.add_argument("--split", default="train",
                   choices=["train", "validation", "test"],
                   help="Which split to inspect")
    p.add_argument("--num-events", type=int, default=3,
                   help="How many events to dump")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[INFO] Opening HDF5: {args.data_path}")

    with h5py.File(args.data_path, "r") as f:
        # --- 1. semantic classes & 'nu' index ---
        if "semantic_classes" not in f:
            raise RuntimeError("HDF5 missing 'semantic_classes' dataset.")

        semantic_classes = f["semantic_classes"].asstr()[()].tolist()
        print(f"[INFO] semantic_classes: {semantic_classes}")

        if "nu" not in semantic_classes:
            raise RuntimeError(f"'nu' not found in semantic_classes={semantic_classes}")
        nu_index = int(semantic_classes.index("nu"))
        print(f"[INFO] 'nu' index = {nu_index}")

        # --- 2. sample list for requested split ---
        split_key = f"samples/{args.split}"
        if split_key not in f:
            raise RuntimeError(f"HDF5 missing split list: {split_key}")
        sample_names = f[split_key].asstr()[()]
        sample_names = np.asarray(sample_names, dtype=str)
        print(f"[INFO] {args.split} split has {len(sample_names)} samples")

        # --- 3. build a dataset with NO transform (raw) ---
        ds = NuGraphDataset(filename=args.data_path,
                            samples=list(sample_names),
                            transform=None)

        n_show = min(args.num_events, len(sample_names))
        print(f"[INFO] Will inspect first {n_show} events from {args.split}")

        for i in range(n_show):
            name = sample_names[i]
            print("\n" + "=" * 80)
            print(f"[EVENT {i}] name = {name}")

            # ---- 3a. Raw scalar record from H5 ----
            rec = f[f"dataset/{name}"][()]  # numpy.void
            print("[RAW] rec.dtype.names:")
            print("      ", rec.dtype.names)

            # Per-plane semantic / instance arrays (if present)
            for pl in ("u", "v", "y"):
                sem_key = f"{pl}/y_semantic"
                inst_key = f"{pl}/y_instance"
                print(f"[RAW] Plane {pl.upper()}:")
                if sem_key in rec.dtype.names:
                    arr = np.asarray(rec[sem_key])
                    uniq, cnt = np.unique(arr, return_counts=True)
                    print(f"   {sem_key}: len={len(arr)}, uniques={dict(zip(uniq.tolist(), cnt.tolist()))}")
                else:
                    print(f"   {sem_key}: MISSING")

                if inst_key in rec.dtype.names:
                    arr = np.asarray(rec[inst_key])
                    uniq, cnt = np.unique(arr, return_counts=True)
                    print(f"   {inst_key}: len={len(arr)}, uniques={dict(zip(uniq.tolist(), cnt.tolist()))}")
                else:
                    print(f"   {inst_key}: MISSING")

            # ---- 3b. NuGraphDataset.get() -> NuGraphData ----
            data = ds.get(i)

            print("[DATA] node stores:", [s._key for s in data.stores])  # type: ignore[attr-defined]

            # Prefer 'hit' store if it exists, otherwise fall back to planes
            if "hit" in data.node_stores:
                hit = data["hit"]
                print("[HIT] pos.shape:", tuple(hit.pos.shape))
                if hasattr(hit, "x"):
                    print("[HIT] x.shape:", tuple(hit.x.shape))
                else:
                    print("[HIT] x: MISSING")

                # plane metadata
                if hasattr(hit, "plane"):
                    plane = hit.plane
                    uniq_planes, cnt = torch.unique(plane, return_counts=True)
                    plane_counts = dict(zip(uniq_planes.tolist(), cnt.tolist()))
                    print("[HIT] plane unique values & counts:", plane_counts)

                    # sanity: compare per-plane counts with raw rec lengths
                    for pl_idx, pl_name in enumerate(("u", "v", "y")):
                        mask = (plane == pl_idx)
                        n_hits_plane = int(mask.sum().item())
                        print(f"      plane {pl_name.upper()} mask hits = {n_hits_plane}")
                        for key in (f"{pl_name}/y_semantic", f"{pl_name}/y_instance"):
                            if key in rec.dtype.names:
                                arr_len = len(np.asarray(rec[key]))
                                print(f"         raw {key} len = {arr_len}")
                            else:
                                print(f"         raw {key} MISSING")
                else:
                    print("[HIT] plane attribute MISSING")

                # semantic labels
                if hasattr(hit, "y_semantic"):
                    y_sem = hit.y_semantic
                    uniq, cnt = torch.unique(y_sem, return_counts=True)
                    print("[HIT] y_semantic uniques & counts:",
                          {int(u): int(c) for u, c in zip(uniq, cnt)})
                else:
                    print("[HIT] y_semantic: MISSING")

                # instance labels attached by dataset.py
                if hasattr(hit, "y_instance"):
                    y_inst = hit.y_instance
                    uniq, cnt = torch.unique(y_inst, return_counts=True)
                    print("[HIT] y_instance uniques & counts:",
                          {int(u): int(c) for u, c in zip(uniq, cnt)})
                    frac_unlabeled = float((y_inst < 0).sum().item()) / float(y_inst.numel())
                    print(f"[HIT] y_instance: frac unlabeled (== -1) = {frac_unlabeled:.3f}")
                else:
                    print("[HIT] y_instance: MISSING")

            else:
                print("[DATA] No 'hit' store; dumping per-plane stores instead")
                for pl_name in ("u", "v", "y"):
                    if pl_name in data.node_stores:
                        store = data[pl_name]
                        print(f"[{pl_name.upper()}] pos.shape:", tuple(store.pos.shape))
                        if hasattr(store, "x"):
                            print(f"[{pl_name.upper()}] x.shape:", tuple(store.x.shape))
                        if hasattr(store, "y_semantic"):
                            y_sem = store.y_semantic
                            uniq, cnt = torch.unique(y_sem, return_counts=True)
                            print(f"[{pl_name.upper()}] y_semantic uniques & counts:",
                                  {int(u): int(c) for u, c in zip(uniq, cnt)})
                        if hasattr(store, "y_instance"):
                            y_inst = store.y_instance
                            uniq, cnt = torch.unique(y_inst, return_counts=True)
                            print(f"[{pl_name.upper()}] y_instance uniques & counts:",
                                  {int(u): int(c) for u, c in zip(uniq, cnt)})

    print("\n[INFO] Debug scan complete.")


if __name__ == "__main__":
    main()
