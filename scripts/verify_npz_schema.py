#!/usr/bin/env python3
"""
Schema verifier for DL-CLUS NPZ files.

What it checks:
- Keys present, shapes, dtypes
- 'points' columns (is it 3D xyz? or concatenated 2D [U,t,V,t,Y,t]?)
- If a 3D edge list exists: 'ppedges' -> (i, j, weight) with index sanity checks
- Alignment relationships: lengths of is_nu / blobs vs #points
- Per-plane CTPC arrays: basic column structure fingerprints (no assumptions)
- If 'blobs' is per-point IDs vs a per-blob table (structured array or 2D)
- Basic stats/ranges and NaN checks to catch anomalies

Usage:
  python verify_npz_schema.py /path/to/rec-lab-apa1-10.npz
"""

import argparse
import os
import sys
import numpy as np

np.set_printoptions(suppress=True, linewidth=160)

def get_first_present(npz, *keys):
    """
    Return the first array found in the NPZ whose key exists.
    Usage: get_first_present(data, "ctpc_f1p0", "ctpc_p0")
    """
    # np.load returns an NpzFile with .files listing the keys
    available = set(getattr(npz, "files", []))
    for k in keys:
        if k in available:
            return npz[k]
    return None


def short_head(arr, n=5):
    try:
        return arr[:n]
    except Exception:
        return "<unprintable>"

def describe_array(name, arr):
    info = []
    info.append(f"- {name}: shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)}")
    # NaN / Inf checks for numeric arrays
    if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.number):
        isnan = np.isnan(arr).sum()
        isinf = np.isinf(arr).sum()
        info.append(f"  NaNs={int(isnan)}, Infs={int(isinf)}")
    # Structured arrays: list fields
    if isinstance(arr, np.ndarray) and arr.dtype.names:
        info.append(f"  fields={arr.dtype.names}")
    return "\n".join(info)

def check_points(points):
    report = []
    if points is None:
        report.append("! points: MISSING")
        return "\n".join(report)

    report.append(describe_array("points", points))
    if points.ndim != 2:
        report.append(f"! points is not 2D; ndim={points.ndim}")
        return "\n".join(report)

    N, C = points.shape
    report.append(f"  -> N_points={N}, N_columns={C}")

    # Try to detect format:
    # Case A: 3D xyz (C==3 or >=3 and first three look like xyz)
    # Case B: concatenated 2D per-plane (C==6 for [U,t,V,t,Y,t])
    hints = []
    if C == 3:
        hints.append("looks_like_3D_xyz")
    if C == 6:
        hints.append("looks_like_concat_2D_Ut_Vt_Yt")

    # Sample head for manual confirmation
    sample = short_head(points, 5)
    report.append("  sample_rows:\n" + "\n".join(["    " + str(row) for row in sample]))

    # Basic numeric sanity: monotonic-ish ticks in even columns for 2D hypothesis
    if C == 6:
        # even columns 0,2,4 might be wire-like (often integer-ish), 1,3,5 tick-like (float/integer)
        wire_like = points[:, [0, 2, 4]]
        tick_like = points[:, [1, 3, 5]]
        # Check how many are close to integers in wire-like columns
        near_int = np.isclose(wire_like, np.round(wire_like)).mean()
        report.append(f"  2D-hypothesis probe: fraction_near_integer_in_wires={near_int:.3f}")
    elif C >= 3:
        # quick range check on first three columns interpreted as xyz
        mins = np.nanmin(points[:, :3], axis=0)
        maxs = np.nanmax(points[:, :3], axis=0)
        report.append(f"  xyz-range-probe (first 3 cols): min={mins}, max={maxs}")

    if hints:
        report.append(f"  format_hints={hints}")
    else:
        report.append("  format_hints=[] (needs human confirmation)")

    return "\n".join(report)

def check_edges(ppedges, n_points):
    report = []
    if ppedges is None:
        report.append("! ppedges: MISSING")
        return "\n".join(report)

    report.append(describe_array("ppedges", ppedges))
    if not isinstance(ppedges, np.ndarray):
        return "\n".join(report)

    if ppedges.ndim != 2 or ppedges.shape[1] < 2:
        report.append("! ppedges is not (E,>=2). Expected at least (i,j) columns.")
        return "\n".join(report)

    E = ppedges.shape[0]
    report.append(f"  -> N_edges={E}")

    # Interpret columns 0,1 as endpoints; if a 3rd column exists, treat as weight
    i = ppedges[:, 0].astype(int, copy=False)
    j = ppedges[:, 1].astype(int, copy=False)
    if np.any(i < 0) or np.any(j < 0) or np.any(i >= n_points) or np.any(j >= n_points):
        bad = np.sum((i < 0) | (j < 0) | (i >= n_points) | (j >= n_points))
        report.append(f"! edge index out-of-bounds count={bad} (n_points={n_points})")

    if ppedges.shape[1] >= 3:
        w = ppedges[:, 2]
        # Basic weight checks
        finite = np.isfinite(w).mean()
        ge0 = (w >= 0).mean()
        report.append(f"  edge_weight_probe: finite_frac={finite:.3f}, >=0_frac={ge0:.3f}")

    # Show a small head
    head = short_head(ppedges, 10)
    report.append("  sample_edges (first 10 rows):")
    for row in head:
        report.append(f"    {row}")

    return "\n".join(report)

def check_ctpc(ctpc, name):
    report = []
    if ctpc is None:
        report.append(f"! {name}: MISSING")
        return "\n".join(report)

    report.append(describe_array(name, ctpc))
    if not isinstance(ctpc, np.ndarray):
        return "\n".join(report)

    # Try to fingerprint column roles WITHOUT assuming exact order:
    # We’ll print head rows and some per-column stats to help identify (wire_idx, tick, charge, slice, channel)
    if ctpc.ndim == 2:
        R, C = ctpc.shape
        report.append(f"  -> rows={R}, cols={C}")
        head = short_head(ctpc, 5)
        report.append("  sample_rows:")
        for row in head:
            report.append(f"    {row}")
        # Per-column quick stats
        mins = np.nanmin(ctpc, axis=0)
        maxs = np.nanmax(ctpc, axis=0)
        frac_int_like = np.isclose(ctpc, np.round(ctpc)).mean(axis=0)
        report.append(f"  per_col_min={mins}")
        report.append(f"  per_col_max={maxs}")
        report.append(f"  per_col_frac_near_integer={np.round(frac_int_like,3)}")
    else:
        # Could be structured array; list field names and a head
        if ctpc.dtype.names:
            fields = ctpc.dtype.names
            report.append(f"  structured_fields={fields}")
            # show first row field->value
            if ctpc.shape[0] > 0:
                first = ctpc[0]
                for f in fields:
                    report.append(f"    {f}={first[f]}")
    return "\n".join(report)

def check_blobs_and_labels(blobs, is_nu, n_points):
    report = []
    # blobs
    if blobs is None:
        report.append("! blobs: MISSING")
    else:
        report.append(describe_array("blobs", blobs))
        if isinstance(blobs, np.ndarray):
            if blobs.ndim == 1 and blobs.size == n_points and np.issubdtype(blobs.dtype, np.integer):
                report.append("  -> blobs looks like per-point blob IDs (len == #points)")
                uniq = np.unique(blobs)
                report.append(f"  unique_blob_ids (count={len(uniq)}): {uniq[:10]}{'...' if len(uniq)>10 else ''}")
            elif blobs.ndim == 2:
                report.append("  -> blobs looks like a per-blob table (2D). Showing head:")
                head = short_head(blobs, 5)
                for row in head:
                    report.append(f"    {row}")
            elif blobs.dtype.names:
                report.append("  -> blobs is a structured array. Fields:")
                report.append(f"     {blobs.dtype.names}")
                if blobs.shape[0] > 0:
                    first = blobs[0]
                    for f in blobs.dtype.names:
                        report.append(f"     {f}={first[f]}")
            else:
                report.append("  -> blobs format is not recognized; manual inspection needed.")

    # is_nu
    if is_nu is None:
        report.append("! is_nu: MISSING")
    else:
        report.append(describe_array("is_nu", is_nu))
        if isinstance(is_nu, np.ndarray):
            if is_nu.ndim == 0:
                report.append("  -> is_nu appears to be a per-event scalar.")
            elif is_nu.ndim == 1 and is_nu.size == n_points:
                report.append("  -> is_nu appears to be per-point (len == #points).")
                classes, counts = np.unique(is_nu, return_counts=True)
                report.append(f"  classes={classes}, counts={counts}")
            else:
                report.append("  -> is_nu length does not match #points; may be per-blob or other.")

    return "\n".join(report)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path", help="Path to labeled .npz file")
    args = ap.parse_args()

    if not os.path.exists(args.npz_path):
        print(f"ERROR: file not found: {args.npz_path}", file=sys.stderr)
        sys.exit(2)

    data = np.load(args.npz_path, allow_pickle=True)
    keys = list(data.keys())
    print(f"\n=== File: {os.path.basename(args.npz_path)} ===")
    print(f"Keys: {keys}\n")

    # Load arrays if present
    arr_points = data.get("points", None)
    arr_ppedges = data.get("ppedges", None)
    arr_is_nu = data.get("is_nu", None)
    arr_blobs = data.get("blobs", None)
    arr_ctpc0 = get_first_present(data, "ctpc_f1p0", "ctpc_p0")
    arr_ctpc1 = get_first_present(data, "ctpc_f1p1", "ctpc_p1")
    arr_ctpc2 = get_first_present(data, "ctpc_f1p2", "ctpc_p2")


    # points
    print("## POINTS")
    print(check_points(arr_points))
    n_points = int(arr_points.shape[0]) if isinstance(arr_points, np.ndarray) and arr_points.ndim >= 1 else 0
    print()

    # ppedges
    print("## PPEDGES (3D point graph edges)")
    print(check_edges(arr_ppedges, n_points))
    print()

    # CTPC per plane
    print("## CTPC (per-plane channel-time point clouds)")
    print(check_ctpc(arr_ctpc0, "ctpc_f1p0"))
    print(check_ctpc(arr_ctpc1, "ctpc_f1p1"))
    print(check_ctpc(arr_ctpc2, "ctpc_f1p2"))
    print()

    # Blobs & labels
    print("## BLOBS and LABELS")
    print(check_blobs_and_labels(arr_blobs, arr_is_nu, n_points))
    print()

    # Cross-consistency checks
    print("## CROSS-CHECKS")
    if isinstance(arr_blobs, np.ndarray) and arr_blobs.ndim == 1 and arr_blobs.size == n_points:
        # Per-blob counts derived from per-point IDs
        uniq, counts = np.unique(arr_blobs, return_counts=True)
        print(f"- derived B (from per-point blob IDs): {len(uniq)} blobs; example sizes: {counts[:10]}{'...' if len(counts)>10 else ''}")
    if isinstance(arr_ppedges, np.ndarray) and n_points > 0:
        # Simple connectivity probe
        try:
            deg = np.zeros(n_points, dtype=int)
            i = arr_ppedges[:,0].astype(int, copy=False)
            j = arr_ppedges[:,1].astype(int, copy=False)
            ok = (i>=0)&(j>=0)&(i<n_points)&(j<n_points)
            i, j = i[ok], j[ok]
            np.add.at(deg, i, 1)
            np.add.at(deg, j, 1)
            print(f"- edge_degree_probe: min={deg.min()}, max={deg.max()}, mean={deg.mean():.2f}")
        except Exception as e:
            print(f"! edge_degree_probe failed: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
