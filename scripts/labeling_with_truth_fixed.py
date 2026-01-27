#!/usr/bin/env python3
"""
Label WireCell rec-*.npz with:
  - is_nu (from tru-*.json nearest-neighbor truth match on xyz)
  - origin_label (0=nu, 1=cosmic/other, 2=unknown/unmatched)
  - nu_vtx (GENIE beam-ν vertex from CellTree: mc_nu_pos*)
  - nu_vtx_found (1/0)
  - vtx_dist, vtx_dx, vtx_dy, vtx_dz (per-hit geometry to ν vertex in cm)

AND (optional) instance truth from CellTree SimIDE (trackId) via ctpc_* plane hits:
  - truth_tid_f{0,1}p0, truth_tid_f{0,1}p1, truth_tid_f{0,1}p2   (per-plane ctpc hit -> truth trackId; -1 unlabeled)
  - truth_conf_f{0,1}p0, truth_conf_f{0,1}p1, truth_conf_f{0,1}p2 (per-plane confidence [0..1])
  - truth_tid_points_direct  (per-point truth trackId via direct point<->ctpc matching; -1 unlabeled)
  - truth_tid_points_direct_support (per-point direct-support count)
  - truth_blob_tid, truth_blob_purity, truth_blob_support (blob-level tid and purity/support)
  - truth_tid_points (per-point after blob-propagation; -1 unlabeled)

AND (optional) edge supervision:
  - edge_index (2, E) int64
  - edge_y     (E,)  int8  (1=same instance, 0=different)

Key fix:
  * APA0 reco contains ctpc_f0p{0,1,2}
  * APA1 reco contains ctpc_f1p{0,1,2}
This script auto-detects which family exists in each rec NPZ and labels accordingly.

FIX: Negative GEANT trackIds (like -169, -58157) are now remapped to positive
unique values instead of being wiped to -1. This preserves instance labels for
secondary particles.
"""

import os
import sys
import json
import argparse
import numpy as np

try:
    import uproot
except Exception:
    uproot = None

# Robust edge supervision (never fails due to insufficient negatives)
from edge_supervision_fallback import build_edge_supervision as _build_edge_supervision_fallback

from numpy.linalg import svd
from sklearn.neighbors import NearestNeighbors


# -----------------------------
# Utilities
# -----------------------------
def _as_intlike(a):
    return np.asarray(a).astype(np.int64, copy=False)


def _as_float(a):
    return np.asarray(a).astype(np.float32, copy=False)


def remap_negative_trackids(tid: np.ndarray) -> np.ndarray:
    """
    Remap negative trackIds (except -1) to unique positive IDs.

    Keeps:
      -1 as -1 (unlabeled)

    Maps:
      tid < -1  --> new positive IDs, stable within this array
    """
    tid = tid.copy()

    neg = tid < -1
    if not np.any(neg):
        return tid

    # start new IDs above the current max positive
    max_pos = int(tid[tid >= 0].max()) if np.any(tid >= 0) else 0
    neg_vals = np.unique(tid[neg])

    mapping = {int(v): (max_pos + 1 + i) for i, v in enumerate(neg_vals)}

    # vectorized remap
    for v, newv in mapping.items():
        tid[tid == v] = newv

    return tid



def merge_trunk_split_tids(points_xyz_mm: np.ndarray,
                          truth_tid_points: np.ndarray,
                          *,
                          min_pts: int = 300,
                          min_len_mm: float = 800.0,
                          angle_max_deg: float = 2.0,
                          mednn_max_mm: float = 15.0):
    """
    Returns:
      tid_merged: per-point merged truth tids (same shape as truth_tid_points)
      merges: list of merged pairs (t1, t2, angle_deg, mednn_mm)
      components: list of components (each list of tids) with size>1
    """
    P = points_xyz_mm.astype(float)
    tid = truth_tid_points.astype(int)

    m = tid >= 0
    Pm = P[m]
    tidm = tid[m]

    tids, cnt = np.unique(tidm, return_counts=True)
    keep = [int(t) for t, c in zip(tids, cnt) if c >= min_pts]
    if len(keep) < 2:
        return tid.copy(), [], []

    def pca_axis_len(X):
        Xc = X - X.mean(0, keepdims=True)
        _, _, Vt = svd(Xc, full_matrices=False)
        v = Vt[0]
        v = v / (np.linalg.norm(v) + 1e-12)
        proj = Xc @ v
        L = float(proj.max() - proj.min())
        return v, L

    axes = {}
    lens = {}
    clouds = {}
    for t in keep:
        X = Pm[tidm == t]
        v, L = pca_axis_len(X)
        axes[t] = v
        lens[t] = L
        clouds[t] = X

    def angdeg(v1, v2):
        c = abs(float(np.dot(v1, v2)))
        c = max(-1.0, min(1.0, c))
        return float(np.degrees(np.arccos(c)))

    # union-find
    parent = list(range(len(keep)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    merges = []
    for i, t1 in enumerate(keep):
        for j in range(i + 1, len(keep)):
            t2 = keep[j]

            if min(lens[t1], lens[t2]) < min_len_mm:
                continue

            a = angdeg(axes[t1], axes[t2])
            if a > angle_max_deg:
                continue

            X1, X2 = clouds[t1], clouds[t2]
            A, B = (X1, X2) if len(X1) >= len(X2) else (X2, X1)
            dist = NearestNeighbors(n_neighbors=1).fit(A).kneighbors(B, return_distance=True)[0].ravel()
            md = float(np.median(dist))

            if md < mednn_max_mm:
                union(i, j)
                merges.append((t1, t2, a, md))

    # build components
    comps = {}
    for i, t in enumerate(keep):
        r = find(i)
        comps.setdefault(r, []).append(t)

    components = [sorted(ts) for ts in comps.values() if len(ts) > 1]

    # representative = min tid in component
    rep = {}
    for ts in comps.values():
        rtid = min(ts)
        for t in ts:
            rep[t] = rtid

    tid_merged = tid.copy()
    for t, rt in rep.items():
        tid_merged[tid_merged == t] = rt

    return tid_merged, merges, components



# -----------------------------
# Semantics: Truth labeling (TRU JSON)
# -----------------------------
def get_isnu_labels(truth_file, rec_file, max_distance_cm=5.0, z_offset_cm=0.0, tagging_alg="blob", blob_grow_cm=20.0):
    """
    Nearest-neighbor match rec points -> truth points, assign truth_data['q'].
    -2 if nearest truth point farther than max_distance_cm.

    tagging_alg:
      point   : no propagation
      blob    : blob : promote only points within blob_grow_cm of ν-seed points inside the blob
      cluster : if any hit in cluster has label==1, set whole cluster to 1
    """


    with open(truth_file, "r") as f:
        truth_data = json.load(f)

    g2f = np.load(rec_file, allow_pickle=True)
    points = g2f["points"]

    # rec points in mm; convert to cm
    x = points[:, 0] / 10.0
    y = points[:, 1] / 10.0
    z = points[:, 2] / 10.0 + float(z_offset_cm)

    bidx = points[:, 4].astype(np.int64, copy=False)
    cidx = points[:, 5].astype(np.int64, copy=False)

    rec_xyz = np.column_stack([x, y, z]).astype(np.float32, copy=False)

    tru_xyz = np.column_stack([truth_data["x"], truth_data["y"], truth_data["z"]]).astype(np.float32, copy=False)
    tru_q = np.asarray(truth_data["q"]).astype(np.int16, copy=False)

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(tru_xyz)

    distances, indices = knn.kneighbors(rec_xyz)  # distances in cm
    distances = distances.reshape(-1)
    indices = indices.reshape(-1)

    isnu = np.full(len(rec_xyz), -2, dtype=np.int16)
    ok = distances <= float(max_distance_cm)
    isnu[ok] = tru_q[indices[ok]]

    if tagging_alg == "blob":
        # Local growth inside blob: do NOT flip the whole blob.
        # Only promote points within blob_grow_cm of nu-seed points in that blob.
        R = float(blob_grow_cm)

        isnu0 = isnu.copy()  # raw NN labels (seeds)

        for b in np.unique(bidx):
            m = (bidx == b)
            if not np.any(m):
                continue

            seeds = m & (isnu0 == 1)
            if not np.any(seeds):
                continue

            X_blob = rec_xyz[m]     # (Nb,3) in cm
            X_seed = rec_xyz[seeds] # (Ns,3) in cm

            knn2 = NearestNeighbors(n_neighbors=1)
            knn2.fit(X_seed)
            d, _ = knn2.kneighbors(X_blob)  # cm to nearest seed
            d = d.reshape(-1)

            grow = d <= R
            idx_blob = np.where(m)[0]
            isnu[idx_blob[grow]] = 1
    elif tagging_alg == "cluster":
        for c in np.unique(cidx):
            m = (cidx == c)
            if np.any(isnu[m] == 1):
                isnu[m] = 1
    elif tagging_alg == "point":
        pass
    else:
        raise ValueError(f"Unknown tagging_alg={tagging_alg} (expected point/blob/cluster)")

    return isnu


# -----------------------------
# CellTree ν-vertex extraction
# -----------------------------
def _open_celltree(tree_path):
    if uproot is None:
        raise RuntimeError("uproot is not available; install it or load it in your env.")
    if not os.path.exists(tree_path):
        raise FileNotFoundError(tree_path)

    f = uproot.open(tree_path)
    if "Event/Sim" not in f:
        keys = list(f.keys())
        raise KeyError(f"Could not find 'Event/Sim' in {tree_path}. Keys include: {keys[:20]}")
    return f["Event/Sim"]


def _read_one(tree, entry, branches):
    arrays = tree.arrays(branches, entry_start=entry, entry_stop=entry + 1, library="np")
    out = {}
    for k, v in arrays.items():
        if isinstance(v, np.ndarray) and v.shape[0] == 1:
            out[k] = v[0]
        else:
            out[k] = v
    return out


def get_beam_nu_vertex_from_celltree(celltree_path, entry):
    """
    Return (vx, vy, vz, found_flag). Units: cm.

    Try:
      1) mc_nu_pos (vector)
      2) mc_nu_pos_x/y/z
    """
    tree = _open_celltree(celltree_path)
    all_branches = set(tree.keys())

    if "mc_nu_pos" in all_branches:
        d = _read_one(tree, entry, ["mc_nu_pos"])
        v = d["mc_nu_pos"]
        try:
            v = np.asarray(v, dtype=float).reshape(-1)
            if v.size >= 3:
                return float(v[0]), float(v[1]), float(v[2]), True
        except Exception:
            pass

    cand = ("mc_nu_pos_x", "mc_nu_pos_y", "mc_nu_pos_z")
    if all(k in all_branches for k in cand):
        d = _read_one(tree, entry, list(cand))
        try:
            return float(d[cand[0]]), float(d[cand[1]]), float(d[cand[2]]), True
        except Exception:
            pass

    return 0.0, 0.0, 0.0, False


# -----------------------------
# Semantics gating logic
# -----------------------------
def truth_q1_fraction_within_R(tru_file, vtx_cm, R_cm):
    with open(tru_file, "r") as f:
        t = json.load(f)
    q = np.asarray(t["q"]).astype(int)
    if np.sum(q == 1) == 0:
        return 0.0
    xyz = np.column_stack([t["x"], t["y"], t["z"]]).astype(float)
    d = np.linalg.norm(xyz[q == 1] - np.asarray(vtx_cm, dtype=float)[None, :], axis=1)
    return float(np.mean(d <= float(R_cm)))


def apply_reco_vtx_gate(isnu, dist_cm, gate_cm):
    gate_cm = float(gate_cm)
    m = (isnu == 1) & (dist_cm > gate_cm)
    if np.any(m):
        isnu = isnu.copy()
        isnu[m] = 0
    return isnu


# -----------------------------
# Instance: SimIDE reading
# -----------------------------
def read_simide_arrays(celltree_path, entry):
    """
    Return dict of SimIDE arrays for one entry:
      channelIdY, tdc, x, y, z, trackId
    Shapes: (Nsimide,)
    """
    tree = _open_celltree(celltree_path)
    branches = [
        "simide_size",
        "simide_channelIdY",
        "simide_tdc",
        "simide_x",
        "simide_y",
        "simide_z",
        "simide_trackId",
    ]
    avail = set(tree.keys())
    missing = [b for b in branches if b not in avail]
    if missing:
        raise KeyError(f"Missing branches in celltree: {missing}")

    d = tree.arrays(branches, entry_start=entry, entry_stop=entry + 1, library="np")
    out = {}
    for k in branches:
        out[k] = d[k][0]
    n = int(out["simide_size"])
    for k in ["simide_channelIdY", "simide_tdc", "simide_x", "simide_y", "simide_z", "simide_trackId"]:
        if len(out[k]) != n:
            raise RuntimeError(f"SimIDE length mismatch for {k}: got {len(out[k])}, expected {n}")
    return {
        "channel": _as_intlike(out["simide_channelIdY"]),
        "tdc": _as_intlike(out["simide_tdc"]),
        "x_cm": _as_float(out["simide_x"]),
        "y_cm": _as_float(out["simide_y"]),
        "z_cm": _as_float(out["simide_z"]),
        "trackId": _as_intlike(out["simide_trackId"]),
    }


# -----------------------------
# Instance: ctpc -> SimIDE matching
# -----------------------------
def ctpc_col_candidates(ctpc):
    """
    Guess (chan_col, tick_col) from ctpc array by int-likeness and value ranges.
    """
    ncol = ctpc.shape[1]
    cols = list(range(ncol))

    def intlike_frac(col):
        v = ctpc[:, col]
        if not np.issubdtype(v.dtype, np.number):
            return 0.0
        r = np.abs(v - np.round(v))
        return float(np.mean(r < 1e-3))

    # tick candidates
    tick_best = None
    tick_score = -1
    for col in cols:
        v = ctpc[:, col]
        if not np.issubdtype(v.dtype, np.number):
            continue
        mn, mx = float(np.min(v)), float(np.max(v))
        if mn < -1e-3:
            continue
        if mx > 50000:
            continue
        sc = intlike_frac(col)
        sc2 = sc + 0.02 * (col / max(1, ncol - 1))
        if sc2 > tick_score:
            tick_score = sc2
            tick_best = col
    if tick_best is None:
        tick_best = min(6, ncol - 1)

    # channel candidates
    chan_best = None
    chan_score = -1
    for col in cols:
        v = ctpc[:, col]
        if not np.issubdtype(v.dtype, np.number):
            continue
        mn, mx = float(np.min(v)), float(np.max(v))
        if mx < 50 or mx > 20000:
            continue
        sc = intlike_frac(col)
        pref = 0.15 if col in (4, 5) else 0.0
        sc2 = sc + pref
        if sc2 > chan_score:
            chan_score = sc2
            chan_best = col
    if chan_best is None:
        chan_best = min(4, ncol - 1)

    return chan_best, tick_best


def match_ctpc_to_simide_trackid(
    ctpc,
    sim_channel,
    sim_tdc,
    sim_trackid,
    shift,
    dtdc_win=1,
    conf_thr=0.7,
    chan_col=None,
    tick_col=None,
):
    """
    Per ctpc hit:
      target_tdc = round(ctpc_tick) + shift
      match SimIDE entries with same channel and |sim_tdc - target_tdc| <= dtdc_win
      tid = mode(trackId), conf = frac(mode)
      if conf < conf_thr -> tid=-1
    """
    if chan_col is None or tick_col is None:
        chan_col2, tick_col2 = ctpc_col_candidates(ctpc)
        if chan_col is None:
            chan_col = chan_col2
        if tick_col is None:
            tick_col = tick_col2

    ctpc_chan = np.round(ctpc[:, chan_col]).astype(np.int64, copy=False)
    ctpc_tick = np.round(ctpc[:, tick_col]).astype(np.int64, copy=False)
    target_tdc = ctpc_tick + int(shift)

    from collections import defaultdict
    ch2idx = defaultdict(list)
    for i, ch in enumerate(sim_channel):
        ch2idx[int(ch)].append(i)

    tid = np.full(ctpc.shape[0], -1, dtype=np.int32)
    conf = np.zeros(ctpc.shape[0], dtype=np.float32)

    win = int(dtdc_win)
    thr = float(conf_thr)

    for i in range(ctpc.shape[0]):
        ch = int(ctpc_chan[i])
        inds = ch2idx.get(ch, None)
        if not inds:
            continue
        t0 = int(target_tdc[i])

        cand = []
        for j in inds:
            dt = int(sim_tdc[j]) - t0
            if -win <= dt <= win:
                cand.append(int(sim_trackid[j]))
        if not cand:
            continue
        if len(cand) == 1:
            tid[i] = np.int32(cand[0])
            conf[i] = 1.0
            continue

        cand = np.asarray(cand, dtype=np.int64)
        u, c = np.unique(cand, return_counts=True)
        k = int(np.argmax(c))
        best_tid = int(u[k])
        best_conf = float(c[k] / np.sum(c))

        if best_conf >= thr:
            tid[i] = np.int32(best_tid)
            conf[i] = np.float32(best_conf)

    return tid, conf


def detect_ctpc_family(out_dict):
    """
    Determine whether this rec NPZ contains ctpc_f0p* or ctpc_f1p*.
    Returns 'f0' or 'f1'. Raises if neither or both are present.
    """
    has_f0 = any((f"ctpc_f0p{i}" in out_dict) for i in (0, 1, 2))
    has_f1 = any((f"ctpc_f1p{i}" in out_dict) for i in (0, 1, 2))
    if has_f0 and has_f1:
        raise RuntimeError("Reco contains BOTH ctpc_f0p* and ctpc_f1p*; ambiguous.")
    if not has_f0 and not has_f1:
        raise RuntimeError("Reco contains neither ctpc_f0p* nor ctpc_f1p*; cannot instance-label.")
    return "f0" if has_f0 else "f1"


# -----------------------------
# Instance: ctpc -> points (direct) geometric matching
# -----------------------------
def build_truth_tid_points_direct(points_mm, ctpc_by_plane, tid_by_plane, max_dist_mm=5.0):
    """
    DIRECT point labeling (forced):

    Use ONLY collection plane (p2) and ONLY XZ to map ctpc hits -> rec points.

    Empirically validated:
      - ctpc_{f0,f1}p2 columns [0,1] are (x,z) in mm
      - rec['points'] columns [0,2] are (x,z) in mm

    This avoids p0/p1 wire-space rotations and eliminates large fractions of -1 labeling.
    """


    n_points = points_mm.shape[0]
    tid_points = np.full(n_points, -1, dtype=np.int32)
    support = np.zeros(n_points, dtype=np.int16)

    # We keep meta for debugging / sanity checks
    meta = {}

    # points: use XZ (mm)
    pts_xyz = points_mm[:, :3].astype(np.float32, copy=False)
    P_xz = pts_xyz[:, [0, 2]]  # (x,z)

    # ---- choose p2 only (for whichever family exists) ----
    p2_name = None
    for cand in ("f0p2", "f1p2"):
        if cand in ctpc_by_plane:
            p2_name = cand
            break

    if p2_name is None:
        meta["p2"] = {"used": 0, "pair": "xz", "median_mm": None, "within_frac": 0.0, "reason": "no_p2"}
        return tid_points, support, meta

    ctpc = ctpc_by_plane[p2_name]
    tid_ct = tid_by_plane[p2_name]
    m = (tid_ct != -1)

    if not np.any(m):
        meta["p2"] = {"used": 0, "pair": "xz", "median_mm": None, "within_frac": 0.0, "reason": "no_labeled_ctpc"}
        return tid_points, support, meta

    ct = ctpc[m]
    tids = tid_ct[m].astype(np.int32, copy=False)

    # ctpc p2: use columns [0,1] as (x,z) in mm
    if ct.shape[1] < 2:
        meta["p2"] = {"used": 0, "pair": "xz", "median_mm": None, "within_frac": 0.0, "reason": "ctpc_cols<2"}
        return tid_points, support, meta

    C_xz = ct[:, [0, 1]].astype(np.float32, copy=False)

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(P_xz)
    dist, idx = knn.kneighbors(C_xz)
    dist = dist.reshape(-1)
    idx = idx.reshape(-1).astype(np.int64, copy=False)

    ok = dist <= float(max_dist_mm)

    meta["p2"] = {
        "used": int(np.sum(ok)),
        "pair": "xz",
        "median_mm": float(np.median(dist)) if dist.size else None,
        "within_frac": float(np.mean(ok)) if dist.size else 0.0,
        "p2_name": p2_name,
    }

    if not np.any(ok):
        return tid_points, support, meta

    # Winner-take-all per point with support counting
    for pidx, tidv in zip(idx[ok], tids[ok]):
        pidx = int(pidx)
        tidv = int(tidv)
        if tid_points[pidx] == -1:
            tid_points[pidx] = np.int32(tidv)
            support[pidx] = np.int16(1)
        else:
            if int(tid_points[pidx]) == tidv:
                support[pidx] = np.int16(int(support[pidx]) + 1)

    return tid_points, support, meta


# -----------------------------
# Instance: blob propagation + purity
# -----------------------------
def blob_propagate_truth(points, tid_points_direct, purity_thr=0.85, min_blob_support=1):
    blob_id = points[:, 4].astype(np.int64, copy=False)
    n_blobs = int(blob_id.max()) + 1 if blob_id.size else 0

    blob_tid = np.full(n_blobs, -1, dtype=np.int32)
    blob_purity = np.zeros(n_blobs, dtype=np.float32)
    blob_support = np.zeros(n_blobs, dtype=np.int16)

    for b in range(n_blobs):
        m = (blob_id == b)
        if not np.any(m):
            continue
        tids = tid_points_direct[m]
        tids = tids[tids != -1]
        if tids.size < int(min_blob_support):
            continue
        u, c = np.unique(tids, return_counts=True)
        k = int(np.argmax(c))
        best_tid = int(u[k])
        support = int(np.sum(c))
        purity = float(c[k] / support) if support > 0 else 0.0

        blob_support[b] = np.int16(support)
        blob_purity[b] = np.float32(purity)

        if purity >= float(purity_thr):
            blob_tid[b] = np.int32(best_tid)

    tid_points = np.full(points.shape[0], -1, dtype=np.int32)
    for b in range(n_blobs):
        if blob_tid[b] == -1:
            continue
        tid_points[blob_id == b] = blob_tid[b]

    return tid_points, blob_tid, blob_purity, blob_support


# -----------------------------
# Edge supervision (robust)
# -----------------------------
def write_edge_supervision_to_out(
    out,
    write_edge_sup: bool = False,
    balance_edges: bool = False,
    neg_radius_mm: float = 60.0,
    seed: int = 123,
):
    """
    Writes out['edge_index'], out['edge_y'] using fallback builder.

    Definitions:
      - labelable node: truth_tid_points >= 0
      - positive edge: endpoints labelable AND same tid
      - negative edge: endpoints labelable AND different tid

    Behavior:
      - If balance_edges=False:
          write ONLY labelable ppedges + their labels (no extra edges sampled)
      - If balance_edges=True:
          build a balanced pos/neg set using _build_edge_supervision_fallback,
          but FIRST augment positives with a within-tid kNN "backbone" to reduce
          fragmentation (stitch across small gaps).
    """
    if not write_edge_sup:
        return

    # ----------------------------
    # Required inputs
    # ----------------------------
    if "ppedges" not in out:
        raise RuntimeError("write_edge_sup requested but 'ppedges' not found in NPZ.")
    if "truth_tid_points" not in out:
        raise RuntimeError("write_edge_sup requested but 'truth_tid_points' not present.")
    if "points" not in out:
        raise RuntimeError("write_edge_sup requested but 'points' not present.")

    ppedges = out["ppedges"]
    if ppedges.ndim != 2 or ppedges.shape[1] < 2:
        raise RuntimeError(f"Unexpected ppedges shape: {ppedges.shape}")

    xyz_mm = out["points"][:, :3]
    if xyz_mm.ndim != 2 or xyz_mm.shape[1] != 3:
        raise RuntimeError(f"Unexpected points[:, :3] shape: {xyz_mm.shape}")

    tid = out["truth_tid_points"].astype(np.int64, copy=False)
    if tid.shape[0] != xyz_mm.shape[0]:
        raise RuntimeError("truth_tid_points length != n_points")

    # ppedges might be shape (E,2) or (E,3); we only use first 2 cols
    u = ppedges[:, 0].astype(np.int64, copy=False)
    v = ppedges[:, 1].astype(np.int64, copy=False)

    # ----------------------------
    # Labelable ppedges only (tid >= 0)
    # ----------------------------
    valid = (tid >= 0)
    m_lab = valid[u] & valid[v]
    uL = u[m_lab]
    vL = v[m_lab]

    if uL.size == 0:
        # write empty to make downstream robust
        out["edge_index"] = np.empty((2, 0), dtype=np.int64)
        out["edge_y"] = np.empty((0,), dtype=np.int8)
        return

    yL = (tid[uL] == tid[vL]).astype(np.int8, copy=False)

    # If not balancing: write only labelable ppedges
    if not balance_edges:
        out["edge_index"] = np.stack([uL, vL], axis=0).astype(np.int64, copy=False)
        out["edge_y"] = yL.astype(np.int8, copy=False)
        return

    # ----------------------------
    # Build positives from labelable ppedges
    # ----------------------------
    pos_mask = (yL == 1)
    if not np.any(pos_mask):
        # Cannot build balanced set without positives; fall back to raw labelable ppedges
        out["edge_index"] = np.stack([uL, vL], axis=0).astype(np.int64, copy=False)
        out["edge_y"] = yL.astype(np.int8, copy=False)
        return

    pos_pairs = np.stack([uL[pos_mask], vL[pos_mask]], axis=1).astype(np.int64, copy=False)

    # ----------------------------
    # AUGMENT POSITIVES: within-tid kNN backbone (reduces fragmentation)
    # ----------------------------
    # Tunables (conservative defaults)
    K_POS = 2                 # add up to 2 neighbors per node (within same tid)
    RADIUS_MM = 20.0          # only connect within 2.0 cm
    MIN_NODES_PER_TID = 400   # only augment big instances
    MAX_NODES_PER_TID = 3000  # cap to bound runtime

    try:
        from scipy.spatial import cKDTree
    except Exception as e:
        cKDTree = None

    if cKDTree is not None:
        rng = np.random.default_rng(int(seed))

        # store undirected unique pairs (i<j)
        pos_set = set()
        if pos_pairs.size:
            for a, b in pos_pairs.tolist():
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                pos_set.add((int(a), int(b)))

        tids = np.unique(tid[tid >= 0])
        xyz_f = xyz_mm.astype(np.float32, copy=False)

        for t in tids.tolist():
            idx = np.where(tid == t)[0]
            n = int(idx.size)
            if n < MIN_NODES_PER_TID:
                continue

            if n > MAX_NODES_PER_TID:
                idx = rng.choice(idx, size=MAX_NODES_PER_TID, replace=False)
                n = int(idx.size)

            X = xyz_f[idx]
            # If X is tiny after cap (shouldn't be), skip
            if n < 2:
                continue

            tree = cKDTree(X)

            # query k nearest INCLUDING self at dist=0
            k = min(K_POS + 1, n)
            dists, nbrs = tree.query(X, k=k, workers=-1)

            # add edges within radius
            for i_local in range(n):
                src = int(idx[i_local])
                for j in range(1, k):  # skip self
                    if float(dists[i_local, j]) > RADIUS_MM:
                        continue
                    dst = int(idx[int(nbrs[i_local, j])])
                    if src == dst:
                        continue
                    a, b = (src, dst) if src < dst else (dst, src)
                    pos_set.add((a, b))

        if pos_set:
            pos_pairs = np.array(list(pos_set), dtype=np.int64)
        else:
            pos_pairs = np.empty((0, 2), dtype=np.int64)

    # If augmentation somehow wiped out positives, fall back
    if pos_pairs.shape[0] == 0:
        out["edge_index"] = np.stack([uL, vL], axis=0).astype(np.int64, copy=False)
        out["edge_y"] = yL.astype(np.int8, copy=False)
        return

    # ----------------------------
    # Now build balanced edges (pos + neg) using your existing fallback builder
    # ----------------------------
    edge_index, edge_y = _build_edge_supervision_fallback(
        xyz_mm=xyz_mm.astype(np.float32, copy=False),
        tid_points=tid,
        pos_pairs=pos_pairs,
        balance_edges=True,
        neg_radius_mm=float(neg_radius_mm),
        seed=int(seed),
    )

    out["edge_index"] = edge_index.astype(np.int64, copy=False)
    out["edge_y"] = edge_y.astype(np.int8, copy=False)


# -----------------------------
# Core per-entry processing
# -----------------------------
def process_entry(
    entry,
    tru_prefix,
    rec_prefix,
    out_prefix,
    celltree_path,
    max_distance_cm=5.0,
    z_offset_cm=0.0,
    tagging_alg="blob",
    blob_grow_cm=20.0,
    vtx_gate_reco_cm=80.0,
    use_truth_gate=False,
    vtx_gate_truth_cm=50.0,
    # instance knobs
    do_instance=True,
    shift_p0=2992,
    shift_p1=2992,
    shift_p2=2992,
    dtdc_win=1,
    conf_thr=0.7,
    purity_thr=0.85,
    min_blob_support=1,
    max_dist_mm=5.0,
    # edge knobs
    write_edge_sup=False,
    balance_edges=False,
    neg_radius_mm=60.0,
):
    tru_file = f"{tru_prefix}-{entry}.json"
    rec_file = f"{rec_prefix}-{entry}.npz"
    out_file = f"{out_prefix}-{entry}.npz"

    if not os.path.exists(tru_file):
        raise FileNotFoundError(tru_file)
    if not os.path.exists(rec_file):
        raise FileNotFoundError(rec_file)

    # 1) semantic label from TRU JSON
    is_nu = get_isnu_labels(
        truth_file=tru_file,
        rec_file=rec_file,
        max_distance_cm=max_distance_cm,
        z_offset_cm=z_offset_cm,
        tagging_alg=tagging_alg,
        blob_grow_cm=float(blob_grow_cm),
    )

    # 2) beam ν vertex from CellTree
    vx, vy, vz, found = get_beam_nu_vertex_from_celltree(celltree_path, entry)
    vtx = np.array([vx, vy, vz], dtype=np.float32)

    # 3) optional truth-gate
    if found and use_truth_gate:
        frac_q1 = truth_q1_fraction_within_R(tru_file, vtx_cm=vtx, R_cm=vtx_gate_truth_cm)
        if frac_q1 <= 0.0:
            found = False

    # 4) load rec + compute per-hit vtx geometry
    rec = np.load(rec_file, allow_pickle=True)
    out = {k: rec[k] for k in rec.files}

    pts_mm = out["points"][:, :3].astype(np.float32, copy=False)
    pts_cm = pts_mm / 10.0
    n_hits = pts_cm.shape[0]

    if found and n_hits > 0:
        dx = pts_cm[:, 0] - float(vtx[0])
        dy = pts_cm[:, 1] - float(vtx[1])
        dz = pts_cm[:, 2] - float(vtx[2])
        vtx_dist = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)
        vtx_dx = dx.astype(np.float32)
        vtx_dy = dy.astype(np.float32)
        vtx_dz = dz.astype(np.float32)

        is_nu = apply_reco_vtx_gate(is_nu, vtx_dist, vtx_gate_reco_cm)

        origin_label = np.full_like(is_nu, 2, dtype=np.int16)
        origin_label[is_nu == 1] = 0
        origin_label[is_nu == 0] = 1
    else:
        # Policy: if no beam ν vertex, treat whole event as cosmic/other.
        is_nu = np.zeros_like(is_nu, dtype=np.int16)
        origin_label = np.ones_like(is_nu, dtype=np.int16)

        vtx_dist = -np.ones(n_hits, dtype=np.float32)
        vtx_dx = np.zeros(n_hits, dtype=np.float32)
        vtx_dy = np.zeros(n_hits, dtype=np.float32)
        vtx_dz = np.zeros(n_hits, dtype=np.float32)

        vtx = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # 5) write semantic fields
    out["is_nu"] = is_nu.astype(np.int16, copy=False)
    out["origin_label"] = origin_label.astype(np.int16, copy=False)
    out["nu_vtx"] = vtx.astype(np.float32, copy=False)
    out["nu_vtx_found"] = np.array([1 if found else 0], dtype=np.int16)
    out["vtx_dist"] = vtx_dist
    out["vtx_dx"] = vtx_dx
    out["vtx_dy"] = vtx_dy
    out["vtx_dz"] = vtx_dz

    # -------------------------
    # 6) Instance labeling (optional)
    # -------------------------
    if do_instance:
        sim = read_simide_arrays(celltree_path, entry)

        # detect ctpc family in this reco
        fam = detect_ctpc_family(out)  # 'f0' or 'f1'

        plane_cfg = {
            f"{fam}p0": dict(shift=shift_p0),
            f"{fam}p1": dict(shift=shift_p1),
            f"{fam}p2": dict(shift=shift_p2),
        }

        ctpc_by_plane = {}
        tid_by_plane = {}
        conf_by_plane = {}

        for pname, cfg in plane_cfg.items():
            k = f"ctpc_{pname}"
            if k not in out:
                continue
            ctpc = out[k]
            ctpc_by_plane[pname] = ctpc

            tid_ct, conf_ct = match_ctpc_to_simide_trackid(
                ctpc=ctpc,
                sim_channel=sim["channel"],
                sim_tdc=sim["tdc"],
                sim_trackid=sim["trackId"],
                shift=int(cfg["shift"]),
                dtdc_win=int(dtdc_win),
                conf_thr=float(conf_thr),
            )
            tid_by_plane[pname] = tid_ct
            conf_by_plane[pname] = conf_ct

            out[f"truth_tid_{pname}"] = tid_ct.astype(np.int32, copy=False)
            out[f"truth_conf_{pname}"] = conf_ct.astype(np.float32, copy=False)

        # direct point labeling (high-precision seeds)
        tid_points_direct, support_direct, meta = build_truth_tid_points_direct(
            points_mm=out["points"][:, :3],
            ctpc_by_plane=ctpc_by_plane,
            tid_by_plane=tid_by_plane,
            max_dist_mm=float(max_dist_mm),
        )
        
        # -------------------------
        # FIX: remap negative tids on DIRECT point labels too
        # so they don't get dropped later and so debug prints make sense
        # -------------------------
        tid_points_direct = remap_negative_trackids(tid_points_direct.astype(np.int32))
        
        out["truth_tid_points_direct"] = tid_points_direct.astype(np.int32, copy=False)
        out["truth_tid_points_direct_support"] = support_direct.astype(np.int16, copy=False)
        out["truth_tid_points_direct_meta_p2_used"] = np.array([meta.get("p2", {}).get("used", -1)], dtype=np.int32)
        
        # blob propagation (now uses the remapped direct labels)
        tid_points, blob_tid, blob_purity, blob_support = blob_propagate_truth(
            points=out["points"],
            tid_points_direct=tid_points_direct,
            purity_thr=float(purity_thr),
            min_blob_support=int(min_blob_support),
        )
        

        # ------------------------------------------------------------
        # FIX: Remap negative trackIds to positive unique values
        # GEANT4 uses negative trackIds for secondary particles.
        # The converter expects tid >= 0 for valid, tid == -1 for unlabeled.
        # ------------------------------------------------------------
        tid_points = remap_negative_trackids(tid_points.astype(np.int32))
        blob_tid = remap_negative_trackids(blob_tid.astype(np.int32))
        # ------------------------------------------------------------

        # ------------------------------------------------------------
        # NEW: merge "trunk-split" truth tids (same physical track, duplicated IDs)
        # Apply ONLY to labeled points (tid>=0). Keep unlabeled (-1) untouched.
        # ------------------------------------------------------------
        tid_points_merged, merges, comps = merge_trunk_split_tids(
            points_xyz_mm=out["points"][:, :3],
            truth_tid_points=tid_points,
            min_pts=300,
            min_len_mm=800.0,
            angle_max_deg=2.0,
            mednn_max_mm=15.0,
        )

        out["truth_tid_points_merged"] = tid_points_merged.astype(np.int32, copy=False)

        # Use merged tids for everything downstream (edges, converter, etc.)
        tid_points = tid_points_merged

        # recompute blob truth stats from merged point tids (consistent trio)
        blob_id = out["points"][:,4].astype(np.int64, copy=False)
        n_blobs = int(blob_id.max()) + 1 if blob_id.size else 0
        
        blob_tid2 = np.full(n_blobs, -1, dtype=np.int32)
        blob_purity2 = np.zeros(n_blobs, dtype=np.float32)
        blob_support2 = np.zeros(n_blobs, dtype=np.int16)
        
        for b in range(n_blobs):
            m = (blob_id == b)
            tt = tid_points[m]
            tt = tt[tt >= 0]
            if tt.size == 0:
                continue
            u, c = np.unique(tt, return_counts=True)
            k = int(np.argmax(c))
            blob_tid2[b] = np.int32(u[k])
            blob_support2[b] = np.int16(int(c.sum()))
            blob_purity2[b] = np.float32(float(c[k] / c.sum()))
        
        blob_tid, blob_purity, blob_support = blob_tid2, blob_purity2, blob_support2

        out["truth_blob_tid"] = blob_tid.astype(np.int32, copy=False)


        
        out["truth_tid_points"] = tid_points.astype(np.int32, copy=False)
        
        out["truth_blob_purity"] = blob_purity.astype(np.float32, copy=False)
        out["truth_blob_support"] = blob_support.astype(np.int16, copy=False)

        # edge supervision (robust; never fails due to missing negatives)
        write_edge_supervision_to_out(
            out,
            write_edge_sup=bool(write_edge_sup),
            balance_edges=bool(balance_edges),
            neg_radius_mm=float(neg_radius_mm),
            seed=123 + int(entry),
        )

    np.savez(out_file, **out)
    return out_file, n_hits, int(1 if found else 0)


# -----------------------------
# CLI helpers
# -----------------------------
def parse_entries(s):
    s = s.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        a = int(a)
        b = int(b)
        return list(range(a, b + 1))
    return [int(x) for x in s.split(",") if x.strip() != ""]


def pick_celltree_for_rec_prefix(rec_prefix, celltree_apa0, celltree_apa1):
    rp = rec_prefix.lower()
    if "apa0" in rp:
        return celltree_apa0
    if "apa1" in rp:
        return celltree_apa1
    raise ValueError(f"Could not infer APA from --rec-prefix={rec_prefix}. Expected to contain 'apa0' or 'apa1'.")


def main():
    ap = argparse.ArgumentParser()

    # semantic
    ap.add_argument("--tru-prefix", required=True, help="e.g. tru-apa0")
    ap.add_argument("--rec-prefix", required=True, help="e.g. rec-apa0")
    ap.add_argument("--out-prefix", required=True, help="e.g. rec-lab-apa0")
    ap.add_argument("--entries", required=True, help="e.g. 0-11 or 0,1,2")
    ap.add_argument("--max-distance", type=float, default=15.0, help="cm")           # was 5.0
    ap.add_argument("--z-offset-cm", type=float, default=0.0, help="cm")
    ap.add_argument("--tagging-alg", type=str, default="blob", choices=["point", "blob", "cluster"])
    ap.add_argument("--blob-grow-cm", type=float, default=30.0,                       # was 20.0
                help="When tagging_alg=blob, only promote points within this radius (cm) of nu-seed points inside the blob.")
    ap.add_argument("--vtx-gate-reco-cm", type=float, default=500.0, help="cm")       # was 80.0
    ap.add_argument("--use-truth-gate", action="store_true", help="Require truth(q==1) near ν vertex.")
    ap.add_argument("--vtx-gate-truth-cm", type=float, default=50.0, help="cm")
    ap.add_argument("--celltree-apa0", required=True, help="path to celltree_apa0.root")
    ap.add_argument("--celltree-apa1", required=True, help="path to celltree_apa1.root")

    # instance
    ap.add_argument("--no-instance", action="store_true", help="Disable instance labeling.")
    ap.add_argument("--shift-p0", type=int, default=2992)
    ap.add_argument("--shift-p1", type=int, default=2992)
    ap.add_argument("--shift-p2", type=int, default=2992)
    ap.add_argument("--dtdc-win", type=int, default=1)
    ap.add_argument("--conf-thr", type=float, default=0.7)
    ap.add_argument("--purity-thr", type=float, default=0.85)
    ap.add_argument("--min-blob-support", type=int, default=1)
    ap.add_argument("--max-dist-mm", type=float, default=10.0, help="mm")             # was 5.0

    # edge supervision
    ap.add_argument("--write-edge-sup", action="store_true")
    ap.add_argument("--balance-edges", action="store_true")
    ap.add_argument("--neg-radius-mm", type=float, default=60.0)

    args = ap.parse_args()

    celltree = pick_celltree_for_rec_prefix(args.rec_prefix, args.celltree_apa0, args.celltree_apa1)
    do_instance = (not args.no_instance)

    print(f"[INFO] Using celltree: {celltree}")
    print(f"[INFO] semantic: tagging_alg={args.tagging_alg} max_distance_cm={args.max_distance} z_offset_cm={args.z_offset_cm}")
    print(f"[INFO] vtx_gate_reco_cm={args.vtx_gate_reco_cm} use_truth_gate={bool(args.use_truth_gate)} vtx_gate_truth_cm={args.vtx_gate_truth_cm}")
    print(
        f"[INFO] instance: enabled={do_instance} shifts(p0,p1,p2)=({args.shift_p0},{args.shift_p1},{args.shift_p2}) "
        f"DTDC_WIN={args.dtdc_win} conf_thr={args.conf_thr} purity_thr={args.purity_thr} max_dist_mm={args.max_dist_mm} "
        f"edge_sup={bool(args.write_edge_sup)} balance_edges={bool(args.balance_edges)} neg_radius_mm={args.neg_radius_mm}"
    )

    entries = parse_entries(args.entries)

    ok = 0
    fail = 0
    for e in entries:
        print(f"\n[ENTRY {e}] {args.rec_prefix}-{e}.npz -> {args.out_prefix}-{e}.npz")
        try:
            out_file, n_hits, found = process_entry(
                entry=e,
                tru_prefix=args.tru_prefix,
                rec_prefix=args.rec_prefix,
                out_prefix=args.out_prefix,
                celltree_path=celltree,
                max_distance_cm=args.max_distance,
                z_offset_cm=args.z_offset_cm,
                tagging_alg=args.tagging_alg,
                blob_grow_cm=args.blob_grow_cm,
                vtx_gate_reco_cm=args.vtx_gate_reco_cm,
                use_truth_gate=args.use_truth_gate,
                vtx_gate_truth_cm=args.vtx_gate_truth_cm,
                do_instance=do_instance,
                shift_p0=args.shift_p0,
                shift_p1=args.shift_p1,
                shift_p2=args.shift_p2,
                dtdc_win=args.dtdc_win,
                conf_thr=args.conf_thr,
                purity_thr=args.purity_thr,
                min_blob_support=args.min_blob_support,
                max_dist_mm=args.max_dist_mm,
                write_edge_sup=args.write_edge_sup,
                balance_edges=args.balance_edges,
                neg_radius_mm=args.neg_radius_mm,
            )
            print(f"  -> OK hits={n_hits} nu_vtx_found={found} wrote={out_file}")
            ok += 1
        except Exception as ex:
            print(f"  -> FAILED: {ex}")
            fail += 1

    print("\nSummary:")
    print(f"  Total entries: {len(entries)}")
    print(f"  Successful   : {ok}")
    print(f"  Failed       : {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
