#!/usr/bin/env python
import numpy as np
import json
from scipy.spatial import cKDTree

def load_npz(rec_lab):
    d = np.load(rec_lab, allow_pickle=True)
    pts_cm = d["points"][:, :3].astype(float) / 10.0
    y = d["is_nu"].astype(int)
    vtx = d["nu_vtx"].astype(float)
    return d, pts_cm, y, vtx

def load_tru(tru_json):
    t = json.load(open(tru_json))
    tru_xyz = np.column_stack([t["x"], t["y"], t["z"]]).astype(float)
    q = np.asarray(t["q"]).astype(int)
    return tru_xyz, q

def summarize(name, arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return f"{name}: empty"
    return f"{name}: min/med/mean/max = {arr.min():.3f} / {np.median(arr):.3f} / {arr.mean():.3f} / {arr.max():.3f}"

def check(entry, apa="apa0"):
    rec_lab = f"rec-lab-{apa}-{entry}.npz"
    tru_json = f"tru-{apa}-{entry}.json"

    d, pts_cm, y, vtx = load_npz(rec_lab)
    tru_xyz, q = load_tru(tru_json)

    # bbox sanity
    lo = pts_cm.min(axis=0); hi = pts_cm.max(axis=0)
    inside = np.all((vtx >= lo) & (vtx <= hi))

    # nearest distances to vertex
    dist_all = np.linalg.norm(pts_cm - vtx[None, :], axis=1)
    dist_nu  = dist_all[y == 1]
    dist_co  = dist_all[y == 0]

    # nearest ν-hit distance (crucial)
    min_nu = float(dist_nu.min()) if dist_nu.size else float("nan")

    # truth near vertex sanity
    tru_dist = np.linalg.norm(tru_xyz - vtx[None, :], axis=1)
    # how many truth points within R of vertex?
    forR = [1, 2, 5, 10, 20, 50]
    frac_tru = {R: float((tru_dist <= R).mean()) for R in forR}

    # also check if q==1 truth exists near vertex
    tru_q1 = tru_dist[q == 1]
    frac_tru_q1 = {R: float((tru_q1 <= R).mean()) if tru_q1.size else 0.0 for R in forR}

    # reco->truth NN distances (overall)
    tree = cKDTree(tru_xyz)
    nn_d, _ = tree.query(pts_cm, k=1, workers=-1)

    print(f"\n=== ENTRY {entry} ({apa}) ===")
    print("run/subrun/event:", int(d["runNo"][0]), int(d["subRunNo"][0]), int(d["eventNo"][0]))
    print("nu_vtx:", vtx.tolist(), "inside_bbox:", bool(inside))
    print("counts is_nu:", {k:int(v) for k,v in zip(*np.unique(y, return_counts=True))})
    print(summarize("dist_to_vtx ALL", dist_all))
    print(summarize("dist_to_vtx NU ", dist_nu))
    print(summarize("dist_to_vtx CO ", dist_co))
    print("min dist_to_vtx among NU hits:", min_nu)
    print(summarize("reco->truth NN dist", nn_d))

    print("truth frac within R cm of vtx:", frac_tru)
    print("truth(q==1) frac within R cm of vtx:", frac_tru_q1)

if __name__ == "__main__":
    # focus on the weird ones first
    for e in [0, 9, 11]:
        check(e, "apa0")
