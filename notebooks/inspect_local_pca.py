#!/usr/bin/env python
"""
inspect_local_pca.py

Inspect local PCA / shape features for nu vs cosmic hits in a NuGraph H5 file.

Example:
  python inspect_local_pca.py \
      --data-path 23334072_nug4_vertex.h5 \
      --n-events 200 \
      --max-hits-per-class 200000 \
      --k 12 \
      --out-base local_pca_nu_vs_cosmic
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

from build_local_pca_features import compute_local_pca_features


def parse_args():
    p = argparse.ArgumentParser(
        description="Inspect local PCA / shape features for nu vs cosmic hits."
    )
    p.add_argument(
        "--data-path",
        required=True,
        help="NuGraph H5 file (with dataset/<event>/sp/pos and sp/y_semantic).",
    )
    p.add_argument(
        "--n-events",
        type=int,
        default=200,
        help="Max number of events to scan from the H5.",
    )
    p.add_argument(
        "--max-hits-per-class",
        type=int,
        default=200000,
        help="Cap the number of hits per class (nu/cosmic) for plotting.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=12,
        help="Number of nearest neighbours for local PCA.",
    )
    p.add_argument(
        "--nu-idx",
        type=int,
        default=0,
        help="Semantic index for 'nu' in sp/y_semantic.",
    )
    p.add_argument(
        "--cosmic-idx",
        type=int,
        default=1,
        help="Semantic index for 'cosmic' in sp/y_semantic.",
    )
    p.add_argument(
        "--out-base",
        type=str,
        default="local_pca_nu_vs_cosmic",
        help="Base filename for PNGs (suffixes will be added).",
    )
    return p.parse_args()


def _append_limited(target_list, values, max_len):
    """Append up to (max_len - len(target_list)) values."""
    remaining = max_len - len(target_list)
    if remaining <= 0:
        return
    if values.size <= remaining:
        target_list.extend(values.tolist())
    else:
        # Random subset to avoid ordering bias
        idx = np.random.choice(values.size, size=remaining, replace=False)
        target_list.extend(values[idx].tolist())


def main():
    args = parse_args()

    print(f"[Info] Inspecting local PCA features from {args.data_path}")
    rng = np.random.default_rng(12345)

    with h5py.File(args.data_path, "r") as f:
        ds = f["dataset"]
        keys = list(ds.keys())
        n_avail = len(keys)
        n_use = min(args.n_events, n_avail)
        print(f"[Info] Available events: {n_avail}; using first {n_use}")

        lin_nu, lin_cos = [], []
        sph_nu, sph_cos = [], []
        ty_nu, ty_cos = [], []
        tz_nu, tz_cos = [], []

        for i, k in enumerate(keys[:n_use]):
            rec = ds[k][()]  # scalar compound

            if "sp/pos" not in rec.dtype.names or "sp/y_semantic" not in rec.dtype.names:
                print(f"[Warn] Event {k} missing sp/pos or sp/y_semantic, skipping.")
                continue

            pos = np.asarray(rec["sp/pos"], dtype=np.float32)
            y_sem = np.asarray(rec["sp/y_semantic"], dtype=np.int64)

            # Only labelled hits
            mask = y_sem >= 0
            pos = pos[mask]
            y_sem = y_sem[mask]

            if pos.shape[0] < args.k + 1:
                continue

            feats = compute_local_pca_features(pos, k=args.k)
            linearity = feats["linearity"]
            sphericity = feats["sphericity"]
            ty = feats["ty"]
            tz = feats["tz"]

            nu_mask = y_sem == args.nu_idx
            co_mask = y_sem == args.cosmic_idx

            # Subsample to avoid explosion
            _append_limited(lin_nu, linearity[nu_mask], args.max_hits_per_class)
            _append_limited(lin_cos, linearity[co_mask], args.max_hits_per_class)

            _append_limited(sph_nu, sphericity[nu_mask], args.max_hits_per_class)
            _append_limited(sph_cos, sphericity[co_mask], args.max_hits_per_class)

            _append_limited(ty_nu, ty[nu_mask], args.max_hits_per_class)
            _append_limited(ty_cos, ty[co_mask], args.max_hits_per_class)

            _append_limited(tz_nu, tz[nu_mask], args.max_hits_per_class)
            _append_limited(tz_cos, tz[co_mask], args.max_hits_per_class)

            if (
                len(lin_nu) >= args.max_hits_per_class
                and len(lin_cos) >= args.max_hits_per_class
            ):
                print(f"[Info] Reached max-hits-per-class at event #{i}, stopping early.")
                break

    # Convert to numpy
    lin_nu = np.asarray(lin_nu, dtype=np.float32)
    lin_cos = np.asarray(lin_cos, dtype=np.float32)
    sph_nu = np.asarray(sph_nu, dtype=np.float32)
    sph_cos = np.asarray(sph_cos, dtype=np.float32)
    ty_nu = np.asarray(ty_nu, dtype=np.float32)
    ty_cos = np.asarray(ty_cos, dtype=np.float32)
    tz_nu = np.asarray(tz_nu, dtype=np.float32)
    tz_cos = np.asarray(tz_cos, dtype=np.float32)

    print("\n[Collected hits]")
    print(f"  linearity: nu={len(lin_nu)}, cosmic={len(lin_cos)}")
    print(f"  sphericity: nu={len(sph_nu)}, cosmic={len(sph_cos)}")
    print(f"  ty: nu={len(ty_nu)}, cosmic={len(ty_cos)}")
    print(f"  tz: nu={len(tz_nu)}, cosmic={len(tz_cos)}")

    # ---------- Quantitative stats ----------
    def print_stats(name, a_nu, a_cos, thresholds=None):
        print(f"\n[Quantitative separation for {name}]")
        for label, arr in (("nu", a_nu), ("cosmic", a_cos)):
            if arr.size == 0:
                print(f"{name} ({label}): NO DATA")
                continue
            print(
                f"{name} ({label:6s}): n={arr.size}, "
                f"mean={arr.mean():.4f}, std={arr.std():.4f}, "
                f"median={np.median(arr):.4f}"
            )
            if thresholds:
                for thr in thresholds:
                    frac = float((arr > thr).mean())
                    print(f"  frac({name} > {thr:.2f}) = {frac:.4f}")

    print_stats("linearity", lin_nu, lin_cos, thresholds=[0.8, 0.9, 0.95])
    print_stats("sphericity", sph_nu, sph_cos, thresholds=[0.1, 0.2, 0.3])
    print_stats("ty", ty_nu, ty_cos, thresholds=[0.5, 0.8])
    print_stats("tz", tz_nu, tz_cos, thresholds=[0.5, 0.8])

    # ---------- Histograms ----------
    def plot_hist_2(
        vals_cos,
        vals_nu,
        bins,
        xlabel,
        title,
        filename,
        range=None,
    ):
        plt.figure(figsize=(8, 6), dpi=150)
        plt.hist(
            vals_cos,
            bins=bins,
            range=range,
            histtype="step",
            density=True,
            label="cosmic",
        )
        plt.hist(
            vals_nu,
            bins=bins,
            range=range,
            histtype="step",
            density=True,
            label="nu",
        )
        plt.xlabel(xlabel)
        plt.ylabel("normalized density")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        print(f"[Info] Saved {title} to {filename}")
        plt.close()

    base = args.out_base

    plot_hist_2(
        lin_cos,
        lin_nu,
        bins=60,
        xlabel="local linearity",
        title="Linearity for nu vs cosmic hits",
        filename=f"{base}_linearity.png",
        range=(0.0, 1.0),
    )

    plot_hist_2(
        sph_cos,
        sph_nu,
        bins=60,
        xlabel="local sphericity",
        title="Sphericity for nu vs cosmic hits",
        filename=f"{base}_sphericity.png",
        range=(0.0, 1.0),
    )

    plot_hist_2(
        ty_cos,
        ty_nu,
        bins=60,
        xlabel="local tangent_y",
        title="tangent_y for nu vs cosmic hits",
        filename=f"{base}_ty.png",
        range=(-1.0, 1.0),
    )

    plot_hist_2(
        tz_cos,
        tz_nu,
        bins=60,
        xlabel="local tangent_z",
        title="tangent_z for nu vs cosmic hits",
        filename=f"{base}_tz.png",
        range=(-1.0, 1.0),
    )


if __name__ == "__main__":
    main()
