#!/usr/bin/env python3
import inspect
import h5py
import torch

import nugraph.data as nd
from nugraph.data import NuGraphDataset, NuGraphDataModule
from nugraph.models import NuGraph4

# DATA_PATH = "/lus/eagle/projects/neutrinoGPU/abhat/sbnd/clustering/23334072_3d_ppedges.h5"
DATA_PATH = "/lus/eagle/projects/neutrinoGPU/abhat/sbnd/clustering/23334072_nug4_vertex.h5"

try:
    print("NuGraphDataset module:", inspect.getfile(nd.NuGraphDataset))
except Exception as e:
    print("NuGraphDataset module lookup failed:", e)

# -----------------------------
# Raw sample, no transform
# -----------------------------
with h5py.File(DATA_PATH) as f:
    train_keys = f["samples/train"].asstr()[()]
ds = NuGraphDataset(DATA_PATH, train_keys, transform=None)
raw = ds.get(0)
print("\n[RAW] node types:", raw.node_types)
print("[RAW] hit keys:", raw["hit"].keys() if "hit" in raw.node_types else "missing")
for plane in ("u", "v", "y"):
    if plane in raw.node_types:
        p = raw[plane]
        print(f"[RAW] {plane}: keys={p.keys()}, y_instance? {hasattr(p, 'y_instance')}")
        if hasattr(p, "y_instance"):
            print(f"       uniques={torch.unique(p.y_instance)[:10]}")
    else:
        print(f"[RAW] {plane}: missing")

# -----------------------------
# With model transform
# -----------------------------
dm = NuGraphDataModule(
    data_path=DATA_PATH,
    model=NuGraph4,  # requires transform to keep labels if you want them
    batch_size=1,
    num_workers=0,
    shuffle="random",
)
h = next(iter(dm.train_dataloader()))["hit"]
print("\n[TRANSFORMED] hit keys:", h.keys())
for key in ("pid", "y_instance", "y_semantic"):
    if hasattr(h, key):
        t = getattr(h, key)
        print(f"[TRANSFORMED] {key}: shape={tuple(t.shape)}, uniques={torch.unique(t)[:10]}")
    else:
        print(f"[TRANSFORMED] {key}: MISSING")
