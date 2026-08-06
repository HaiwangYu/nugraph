#!/usr/bin/env python3
import sys
import h5py
import numpy as np

try:
    from sklearn.metrics import adjusted_rand_score, completeness_score
except Exception as e:
    raise SystemExit(f"Need sklearn for metrics: {e}")

path = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

def get(row, names):
    for n in names:
        if n in row.dtype.names:
            return row[n]
    return None

class DSU:
    def __init__(self, n):
        self.p = np.arange(n, dtype=np.int64)
        self.r = np.zeros(n, dtype=np.int8)

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(int(a)), self.find(int(b))
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1

aris = []
comps = []
frag_counts = []
largest_fracs = []
edge_pos_recalls = []

with h5py.File(path, "r") as f:
    evs = list(f["dataset"].keys())[:limit]

    for ev in evs:
        row = f["dataset"][ev][()]

        y_true = get(row, ["sp/y_instance"])
        edge_index = get(row, ["sp/edge_label_index", "sp/edge_index"])
        edge_y = get(row, ["sp/edge_y"])
        edge_labelable = get(row, ["sp/edge_labelable"])

        if y_true is None or edge_index is None or edge_y is None or edge_labelable is None:
            continue

        y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
        edge_index = np.asarray(edge_index, dtype=np.int64)
        edge_y = np.asarray(edge_y, dtype=np.int64).reshape(-1)
        edge_labelable = np.asarray(edge_labelable, dtype=np.int64).reshape(-1)

        n = len(y_true)
        if n == 0 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
            continue

        valid = y_true >= 0
        if valid.sum() < 2:
            continue

        dsu = DSU(n)

        pos = (edge_labelable == 1) & (edge_y == 1)
        ei_pos = edge_index[:, pos]
        for a, b in ei_pos.T:
            if 0 <= a < n and 0 <= b < n:
                dsu.union(a, b)

        pred = np.array([dsu.find(i) for i in range(n)], dtype=np.int64)

        vt = y_true[valid]
        vp = pred[valid]

        aris.append(adjusted_rand_score(vt, vp))
        comps.append(completeness_score(vt, vp))

        # fragmentation: number of predicted components per true instance
        frag = []
        largest = []
        for tid in np.unique(vt):
            m = vt == tid
            pcs, cnts = np.unique(vp[m], return_counts=True)
            frag.append(len(pcs))
            largest.append(cnts.max() / cnts.sum())

        frag_counts.append(float(np.mean(frag)))
        largest_fracs.append(float(np.mean(largest)))

        # positive candidate recall among same-truth node pairs is too expensive all-pairs,
        # so use a proxy: fraction of truth instances with one connected component.
        one_comp = np.mean([x == 1 for x in frag])
        edge_pos_recalls.append(float(one_comp))

def summarize(x):
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "p50": float(np.percentile(x, 50)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }

print("file:", path)
print("events used:", len(aris))
print("oracle_ARI:", summarize(aris))
print("oracle_completeness:", summarize(comps))
print("mean_components_per_truth_instance:", summarize(frag_counts))
print("mean_largest_component_fraction_per_truth_instance:", summarize(largest_fracs))
print("fraction_truth_instances_unfragmented_proxy:", summarize(edge_pos_recalls))
