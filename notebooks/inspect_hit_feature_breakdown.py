#!/usr/bin/env python
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from nugraph.data import H5DataModule
from nugraph.models import NuGraph4

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--n-events", type=int, default=1)
    args = p.parse_args()

    data_path = Path(args.data_path)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)

    # 1) Load a small datamodule (with full Transform)
    dm = H5DataModule(
        data_path=str(data_path),
        model=NuGraph4,
        batch_size=1,
        num_workers=0,
        shuffle="random",
        balance_frac=0.0,
    )
    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    print("Node types in batch:", list(batch.node_types))
    hit = batch["hit"]
    sp  = batch["sp"]

    print("\n[hit] x shape:", hit.x.shape)
    print("[hit] first 5 rows of x:")
    print(hit.x[:5])

    # 2) Grab the raw H5 record for the same event
    with h5py.File(str(data_path), "r") as f:
        # we only care about the first train sample
        key = f["samples/train"][0].decode()
        print(f"\n[Info] Using raw event key: {key}")
        rec = f["dataset"][key][()]

        # plane-level features & positions
        u_x = np.asarray(rec["u/x"])   # (N_u, 5)
        v_x = np.asarray(rec["v/x"])   # (N_v, 5)
        y_x = np.asarray(rec["y/x"])   # (N_y, 5)

        u_pos = np.asarray(rec["u/pos"])  # (N_u, 2)
        v_pos = np.asarray(rec["v/pos"])  # (N_v, 2)
        y_pos = np.asarray(rec["y/pos"])  # (N_y, 2)

        # blob-level features (what we turned into sidecar)
        sp_feat = np.asarray(rec["sp/features"])   # (N_sp, 6)
        print("[raw] sp/features shape:", sp_feat.shape)

    # 3) For a few hits, print everything we can
    plane = hit.plane.cpu().numpy()   # (N_hits,)
    to_sp  = hit.to_sp.cpu().numpy()  # (N_hits,) blob index, if Transform exposes it; if not, skip this

    print("\nInspecting first 10 hits:")
    for i in range(min(10, hit.x.size(0))):
        x_vec = hit.x[i].cpu().numpy()
        p = plane[i]
        b = to_sp[i] if "to_sp" in hit else -1

        if p == 0:
            src_x   = u_x
            src_pos = u_pos
        elif p == 1:
            src_x   = v_x
            src_pos = v_pos
        else:
            src_x   = y_x
            src_pos = y_pos

        # crude index mapping: assume hits are ordered like plane nodes;
        # if Transform uses a different ordering, this needs to be adjusted.
        plane_idx = np.sum(plane[:i] == p) - 1
        plane_idx = max(0, min(plane_idx, src_x.shape[0]-1))

        print(f"\nHit {i}:")
        print(f"  plane = {p}, approx plane_idx = {plane_idx}, blob_idx = {b}")
        print(f"  hit.x = {x_vec}")

        print(f"  plane.pos[{plane_idx}] = {src_pos[plane_idx]}")
        print(f"  plane.x[{plane_idx}]   = {src_x[plane_idx]}")

        if 0 <= b < sp_feat.shape[0]:
            print(f"  sp.features[{b}]      = {sp_feat[b]}")
        else:
            print("  sp.features: blob_idx out of range")

if __name__ == "__main__":
    main()
