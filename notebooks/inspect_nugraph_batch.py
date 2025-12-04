#!/usr/bin/env python
import torch
from nugraph.data import H5DataModule
from nugraph.models import NuGraph4

DATA = "/lus/eagle/projects/neutrinoGPU/abhat/sbnd/clustering/23334072_nug4_vertex.h5"

def main():
    dm = H5DataModule(
        data_path=DATA,
        model=NuGraph4,
        batch_size=4,
        num_workers=0,
        shuffle="random",
        balance_frac=0.0,
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    print("Node types in batch:", list(batch.node_types))

    # "hit" store: where NuGraph4 takes its x, pid/y_instance from
    h = batch["hit"]
    print("\n[hit] x shape:", h.x.shape)
    print("[hit] has y_instance?", hasattr(h, "y_instance"))
    if hasattr(h, "y_instance"):
        print("[hit] y_instance uniques (first 20):", h.y_instance.unique()[:20])
    print("[hit] has pid?", hasattr(h, "pid"))
    if hasattr(h, "pid"):
        print("[hit] pid uniques (first 20):", h.pid.unique()[:20])

    # Check sp store for the new features
    sp = batch["sp"]
    print("\n[sp] x shape:", sp.x.shape)
    print("[sp] y_semantic uniques:", sp.y_semantic.unique())
    if hasattr(sp, "y_instance"):
        print("[sp] y_instance uniques (first 20):", sp.y_instance.unique()[:20])

if __name__ == "__main__":
    main()
