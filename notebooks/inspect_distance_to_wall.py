#!/usr/bin/env python3
"""
inspect_distance_to_wall.py

Quick study of how well "distance to nearest wall" (d_wall) and
"distance to top wall" (d_top) separate nu vs cosmic hits.

Usage:
  python inspect_distance_to_wall.py \
    --data-path /path/to/23334072_nug4_vertex.h5 \
    --n-events 500 \
    --max-hits 300000 \
    --out dwall_nu_vs_cosmic.png \
    --out-top dtop_nu_vs_cosmic.png
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--n-events", type=int, default=200)
    p.add_argument("--max-hits", type=int, default=300000)
    p.add_argument(
        "--out",
        default="dwall_nu_vs_cosmic.png",
        help="Output PNG for distance-to-wall histogram",
    )
    p.add_argument(
        "--out-top",
        default="dtop_nu_vs_cosmic.png",
        help="Output PNG for distance-to-top histogram",
    )
    p.add_argument("--nu-class-index", type=int, default=0)
    return p.parse_args()


def summarize_dist(name, arr):
    if len(arr) == 0:
        print(f"{name}: NO HITS")
        return
    mean = float(arr.mean())
    med = float(np.median(arr))
    print(f"{name}: n={len(arr)}, mean={mean:.2f} cm, median={med:.2f} cm")
    for thr in (50, 100, 200):
        frac = float((arr < thr).mean())
        print(f"  frac(d < {thr:3d} cm) = {frac:.4f}")


def main():
    args = parse_args()

    print(f"[Info] Inspecting {args.n_events} events from {args.data_path}")

    with h5py.File(args.data_path, "r") as f:
        dset = f["dataset"]
        keys = list(dset.keys())
        n_events = min(args.n_events, len(keys))
        print(f"[Info] Available events: {len(keys)}; using first {n_events}")

        # First pass: infer detector bounds from sp/pos
        x_min = y_min = z_min = +1e9
        x_max = y_max = z_max = -1e9

        for k in keys[:n_events]:
            rec = dset[k][()]
            pos = np.asarray(rec["sp/pos"])  # [N,3]
            if pos.size == 0:
                continue
            x = pos[:, 0]
            y = pos[:, 1]
            z = pos[:, 2]
            x_min = min(x_min, x.min())
            x_max = max(x_max, x.max())
            y_min = min(y_min, y.min())
            y_max = max(y_max, y.max())
            z_min = min(z_min, z.min())
            z_max = max(z_max, z.max())

        print("\n[Detector bounds inferred from data]")
        print(f"  x: [{x_min:.2f}, {x_max:.2f}]")
        print(f"  y: [{y_min:.2f}, {y_max:.2f}]")
        print(f"  z: [{z_min:.2f}, {z_max:.2f}]")

        # Second pass: collect distances for nu and cosmic
        d_wall_nu = []
        d_wall_cos = []
        d_top_nu = []
        d_top_cos = []

        total_hits_nu = 0
        total_hits_cos = 0

        for k in keys[:n_events]:
            rec = dset[k][()]
            pos = np.asarray(rec["sp/pos"])  # [N,3]
            y_sem = np.asarray(rec["sp/y_semantic"])  # [N]
            if pos.size == 0:
                continue

            # distances to each wall
            x = pos[:, 0]
            y = pos[:, 1]
            z = pos[:, 2]

            dx_min = np.minimum(np.abs(x - x_min), np.abs(x - x_max))
            dy_min = np.minimum(np.abs(y - y_min), np.abs(y - y_max))
            dz_min = np.minimum(np.abs(z - z_min), np.abs(z - z_max))
            d_wall = np.minimum(np.minimum(dx_min, dy_min), dz_min)

            # distance to TOP wall (ceiling) = y_max
            d_top = y_max - y

            mask_nu = (y_sem == args.nu_class_index)
            mask_cos = (y_sem != args.nu_class_index) & (y_sem >= 0)

            dw_nu_evt = d_wall[mask_nu]
            dw_cos_evt = d_wall[mask_cos]
            dt_nu_evt = d_top[mask_nu]
            dt_cos_evt = d_top[mask_cos]

            d_wall_nu.append(dw_nu_evt)
            d_wall_cos.append(dw_cos_evt)
            d_top_nu.append(dt_nu_evt)
            d_top_cos.append(dt_cos_evt)

            total_hits_nu += int(mask_nu.sum())
            total_hits_cos += int(mask_cos.sum())

            # Downsample cosmics if too many
            if total_hits_cos > args.max_hits:
                break

        d_wall_nu = np.concatenate(d_wall_nu) if d_wall_nu else np.zeros(0)
        d_wall_cos = np.concatenate(d_wall_cos) if d_wall_cos else np.zeros(0)
        d_top_nu = np.concatenate(d_top_nu) if d_top_nu else np.zeros(0)
        d_top_cos = np.concatenate(d_top_cos) if d_top_cos else np.zeros(0)

        # Optional downsample cosmics to max_hits
        if len(d_wall_cos) > args.max_hits:
            idx = np.random.choice(len(d_wall_cos), size=args.max_hits, replace=False)
            d_wall_cos = d_wall_cos[idx]
        if len(d_top_cos) > args.max_hits:
            idx = np.random.choice(len(d_top_cos), size=args.max_hits, replace=False)
            d_top_cos = d_top_cos[idx]

        print("\n[Collected distances-to-wall]")
        print(f"  nu hits:     {len(d_wall_nu)}")
        print(f"  cosmic hits: {len(d_wall_cos)}")
        print(f"  d_wall nu:   min={d_wall_nu.min():.2f}, max={d_wall_nu.max():.2f}")
        print(f"  d_wall cosm: min={d_wall_cos.min():.2f}, max={d_wall_cos.max():.2f}")

        print("\n[Collected distances-to-TOP]")
        print(f"  d_top nu:    min={d_top_nu.min():.2f}, max={d_top_nu.max():.2f}")
        print(f"  d_top cosm:  min={d_top_cos.min():.2f}, max={d_top_cos.max():.2f}")

        # --- Quantitative separation numbers ---
        print("\n[Quantitative separation for d_wall]")
        summarize_dist("d_wall (nu)    ", d_wall_nu)
        summarize_dist("d_wall (cosmic)", d_wall_cos)

        print("\n[Quantitative separation for d_top]")
        summarize_dist("d_top (nu)     ", d_top_nu)
        summarize_dist("d_top (cosmic) ", d_top_cos)

        # --- Histograms: d_wall ---
        bins = 80

        plt.figure(figsize=(8, 6), dpi=150)
        plt.hist(
            d_wall_cos,
            bins=bins,
            density=True,
            histtype="step",
            label="cosmic",
        )
        plt.hist(
            d_wall_nu,
            bins=bins,
            density=True,
            histtype="step",
            label="nu",
        )
        plt.xlabel("distance to nearest wall (cm)")
        plt.ylabel("normalized density")
        plt.title("Distance-to-wall for nu vs cosmic hits")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out)
        print(f"\n[Info] Saved d_wall histogram to {args.out}")

        # --- Histograms: d_top ---
        plt.figure(figsize=(8, 6), dpi=150)
        plt.hist(
            d_top_cos,
            bins=bins,
            density=True,
            histtype="step",
            label="cosmic",
        )
        plt.hist(
            d_top_nu,
            bins=bins,
            density=True,
            histtype="step",
            label="nu",
        )
        plt.xlabel("distance to TOP wall (y_max - y) (cm)")
        plt.ylabel("normalized density")
        plt.title("Distance-to-top for nu vs cosmic hits")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_top)
        print(f"[Info] Saved d_top histogram to {args.out_top}")


if __name__ == "__main__":
    main()
