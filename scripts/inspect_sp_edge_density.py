#!/usr/bin/env python3
import sys
import h5py
import numpy as np

path = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 500

def get(row, names):
    for n in names:
        if n in row.dtype.names:
            return row[n]
    return None

vals = []

with h5py.File(path, "r") as f:
    evs = list(f["dataset"].keys())[:limit]

    for ev in evs:
        row = f["dataset"][ev][()]

        yinst = get(row, ["sp/y_instance"])
        edge_y = get(row, ["sp/edge_y"])
        edge_labelable = get(row, ["sp/edge_labelable"])
        edge_index = get(row, ["sp/edge_label_index", "sp/edge_index"])

        if yinst is None or edge_y is None or edge_labelable is None or edge_index is None:
            continue

        yinst = np.asarray(yinst)
        edge_y = np.asarray(edge_y)
        edge_labelable = np.asarray(edge_labelable)
        edge_index = np.asarray(edge_index)

        n_sp = len(yinst)
        n_edges = edge_index.shape[1] if edge_index.ndim == 2 else len(edge_y)
        n_labelable = int(edge_labelable.sum())
        n_pos = int(((edge_y == 1) & (edge_labelable == 1)).sum())
        n_neg = int(((edge_y == 0) & (edge_labelable == 1)).sum())
        pos_frac = n_pos / max(n_labelable, 1)

        valid_inst = yinst[yinst >= 0]
        n_inst = len(np.unique(valid_inst)) if valid_inst.size else 0

        vals.append((n_sp, n_inst, n_edges, n_labelable, n_pos, n_neg, pos_frac))

arr = np.asarray(vals, dtype=float)
print("file:", path)
print("events:", len(vals))

for i, name in enumerate(["n_sp", "n_inst", "n_edges", "n_labelable", "n_pos", "n_neg", "pos_frac"]):
    print(
        f"{name:12s}",
        "mean", float(arr[:, i].mean()),
        "p50", float(np.percentile(arr[:, i], 50)),
        "p90", float(np.percentile(arr[:, i], 90)),
        "max", float(arr[:, i].max()),
    )
