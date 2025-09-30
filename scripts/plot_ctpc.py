#!/usr/bin/env python3
"""Quick-look plots for CTPC wire-cell payloads stored in WCML NPZ bundles."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Matplotlib defaults tuned for readable scatter plots on dense hit clouds
plt.rcParams.update({
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})

DEFAULT_PLANE_NAMES = {
    "ctpc_f0p0": "u",
    "ctpc_f0p1": "v",
    "ctpc_f0p2": "y",
}


def load_ctpc_arrays(path: Path) -> dict[str, np.ndarray]:
    """Return the CTPC arrays stored in a WCML NPZ file."""
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files if key.startswith("ctpc_")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path,
                        help="Path to a WCML NPZ file or a directory containing NPZ files")
    parser.add_argument("--outdir", type=Path,
                        help="Optional destination image path (defaults to showing the plot)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on the number of NPZ files processed")
    parser.add_argument("--charge-log", action="store_true",
                        help="Colour scale uses log10(charge) when set")
    parser.add_argument("--point-size", type=float, default=6.0,
                        help="Marker size for scatter points [default: %(default)s]")
    return parser.parse_args()


def iter_npz_files(source: Path, limit: int | None) -> Iterable[Path]:
    if source.is_file():
        yield source
        return

    if not source.is_dir():
        raise FileNotFoundError(f"No such file or directory: {source}")

    count = 0
    for path in sorted(source.glob("*.npz")):
        yield path
        count += 1
        if limit is not None and count >= limit:
            break


def combined_normalization(values: list[np.ndarray], log_scale: bool) -> tuple[mcolors.Normalize | None, str]:
    """Construct a shared colour normalisation across planes."""
    if not values:
        return None, "charge"

    merged = np.concatenate(values)
    label = "charge"

    if log_scale:
        positive = merged[merged > 0]
        if not positive.size:
            return None, label
        norm = mcolors.LogNorm(vmin=positive.min(), vmax=positive.max())
        label = "charge (log scale)"
    else:
        norm = mcolors.Normalize(vmin=float(merged.min()), vmax=float(merged.max()))
    return norm, label


def plot_ctpc(npz_path: Path,
              outdir: Path,
              charge_log: bool,
              point_size: float) -> None:
    ctpc_items = load_ctpc_arrays(npz_path)
    if not ctpc_items:
        print(f"[warn] {npz_path.name}: no CTPC arrays found", file=sys.stderr)
        return

    ordered_keys: list[str] = [key for key in DEFAULT_PLANE_NAMES if key in ctpc_items]
    extras = sorted(key for key in ctpc_items if key not in DEFAULT_PLANE_NAMES)
    ordered_keys.extend(extras)

    charge_fields = [ctpc_items[key][:, 2] for key in ordered_keys if ctpc_items[key].size]
    norm, colour_label = combined_normalization(charge_fields, charge_log)
    cmap = plt.colormaps.get_cmap("viridis")

    nplanes = len(ordered_keys)
    fig, axes = plt.subplots(1, nplanes, figsize=(4 * nplanes, 4), sharex=False, sharey=False)
    if nplanes == 1:
        axes = [axes]

    mappables = []
    for ax, key in zip(axes, ordered_keys):
        plane_data = ctpc_items[key]
        display_name = DEFAULT_PLANE_NAMES[key].upper() if key in DEFAULT_PLANE_NAMES else key

        if plane_data.size == 0:
            ax.set_title(f"{display_name} plane")
            ax.set_xlabel("wire")
            ax.set_ylabel("drift")
            ax.text(0.5, 0.5, "no hits", transform=ax.transAxes, ha="center", va="center")
            continue

        # Columns: 0=drift (mm), 1=wire coordinate (mm), 2=charge, 3=sigma(charge), 4=first tick
        drift = plane_data[:, 0]
        wire = plane_data[:, 1]
        charge = plane_data[:, 2]

        colours = charge
        plot_norm = norm
        if charge_log and plot_norm is not None:
            positive_mask = charge > 0
            if positive_mask.any():
                colours = np.where(positive_mask, charge, plot_norm.vmin)
            else:
                plot_norm = None

        scat = ax.scatter(wire, drift, c=colours, s=point_size, cmap=cmap, norm=plot_norm, linewidths=0)
        mappables.append(scat)

        ax.set_title(f"{display_name} plane")
        ax.set_xlabel("wire (mm)")
        ax.set_ylabel("drift (mm)")

    # if mappables:
    #     # Give the colourbar a bit of breathing room so it stays outside the final panel
    #     fig.colorbar(
    #         mappables[0],
    #         ax=axes,
    #         label=colour_label,
    #         shrink=0.9,
    #         pad=0.02,
    #         fraction=0.04,
    #     )

    fig.suptitle(f"{npz_path.stem}: CTPC hits")
    fig.subplots_adjust(top=0.85, right=0.94, wspace=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if not outdir:
        plt.show()
        plt.close(fig)
        return
    else:
        dest_dir = outdir / npz_path.stem
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_path = dest_dir / f"{npz_path.stem}_ctpc.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"[ok] wrote {out_path}")


def main() -> None:
    args = parse_args()
    try:
        files = list(iter_npz_files(args.source, args.limit))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if not files:
        print("No NPZ files found to process.", file=sys.stderr)
        sys.exit(1)

    for path in files:
        plot_ctpc(path, args.outdir, args.charge_log, args.point_size)


if __name__ == "__main__":
    main()
