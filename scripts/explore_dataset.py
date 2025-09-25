#!/usr/bin/env python3
"""Utility to explore NuGraph HDF5 datasets by generating quick-look plots."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Matplotlib defaults tuned for publication-ish plots
plt.rcParams.update({
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


def decode_bytes(values):
    """Decode a sequence of bytes/np.bytes_ objects into Python strings."""
    return [v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v) for v in values]


def load_sample(record) -> dict[str, np.ndarray]:
    """Convert a zero-dim structured array record to a plain dict."""
    return {field: record[field] for field in record.dtype.names}


def semantic_palette(classes: list[str]) -> dict[str, str]:
    """Return a stable colour palette for semantic labels."""
    base = plt.cm.get_cmap("tab10", len(classes))
    palette = {"background": "#b3b3b3"}
    for idx, cls in enumerate(classes):
        palette[cls] = mcolors.to_hex(base(idx))
    return palette


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_semantic_scatter(sample_name: str,
                           planes: list[str],
                           semantic_labels: list[str],
                           palette: dict[str, str],
                           data: dict[str, np.ndarray],
                           outdir: Path) -> None:
    """Scatter plot of hit positions per plane coloured by true semantic label."""
    nplanes = len(planes)
    fig, axes = plt.subplots(1, nplanes, figsize=(4 * nplanes, 4), sharex=False, sharey=False)
    if nplanes == 1:
        axes = [axes]

    label_names = ["background"] + semantic_labels

    for ax, plane in zip(axes, planes):
        pos = data[f"{plane}/pos"]
        y_sem = data[f"{plane}/y_semantic"].astype(int)
        labels = [label_names[val + 1] for val in y_sem]
        colours = [palette[label] for label in labels]
        ax.scatter(pos[:, 0], pos[:, 1], c=colours, s=5, linewidths=0)
        ax.set_title(f"{plane.upper()} plane")
        ax.set_xlabel("proj")
        ax.set_ylabel("drift")

        unique_labels = sorted(set(labels), key=label_names.index)
        handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=palette[label], label=label) for label in unique_labels]
        ax.legend(handles=handles, fontsize="small", loc="upper right")

    fig.suptitle(f"Semantic truth labels – {sample_name}")
    fig.tight_layout()
    fig.savefig(outdir / f"{sample_name}_semantic.png", bbox_inches="tight")
    plt.close(fig)


def plot_instance_scatter(sample_name: str,
                          planes: list[str],
                          data: dict[str, np.ndarray],
                          outdir: Path) -> None:
    """Scatter plot coloured by true instance id with optional vertex markers."""
    nplanes = len(planes)
    fig, axes = plt.subplots(1, nplanes, figsize=(4 * nplanes, 4), sharex=False, sharey=False)
    if nplanes == 1:
        axes = [axes]

    for ax, plane in zip(axes, planes):
        pos = data[f"{plane}/pos"]
        inst = data[f"{plane}/y_instance"].astype(int)
        mask = inst >= 0
        valid_ids = np.unique(inst[mask])
        background_color = "#b3b3b3"

        if len(valid_ids) == 0:
            colours = [background_color] * len(inst)
        else:
            cmap = plt.cm.get_cmap("nipy_spectral", len(valid_ids))
            colour_map = {val: cmap(i) for i, val in enumerate(valid_ids)}
            colours = [colour_map.get(val, background_color) for val in inst]

        ax.scatter(pos[:, 0], pos[:, 1], c=colours, s=5, linewidths=0)

        # Plot ground-truth interaction vertices if present
        vtx_key = f"{plane}/y_vtx"
        if vtx_key in data:
            vtx = data[vtx_key]
            if np.ndim(vtx) == 2 and vtx.size:
                ax.scatter(vtx[:, 0], vtx[:, 1], marker="*", s=120, c="k", label="truth vtx")

        ax.set_title(f"{plane.upper()} plane")
        ax.set_xlabel("proj")
        ax.set_ylabel("drift")

        if len(valid_ids) and len(valid_ids) <= 12:
            handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=colour_map[val], label=str(val)) for val in valid_ids]
            if vtx_key in data and np.ndim(data[vtx_key]) == 2 and data[vtx_key].size:
                handles.append(plt.Line2D([0], [0], marker="*", linestyle="", color="k", label="truth vtx"))
            ax.legend(handles=handles, fontsize="x-small", loc="upper right", title="instance id")

    fig.suptitle(f"Instance ids – {sample_name}")
    fig.tight_layout()
    fig.savefig(outdir / f"{sample_name}_instances.png", bbox_inches="tight")
    plt.close(fig)


def plot_semantic_summary(split_name: str,
                          planes: list[str],
                          semantic_labels: list[str],
                          palette: dict[str, str],
                          per_plane_counts: dict[str, np.ndarray],
                          outdir: Path) -> None:
    """Plot stacked bar chart summarizing semantic counts per plane."""
    label_names = ["background"] + semantic_labels
    indices = np.arange(len(planes))
    width = 0.6

    fig, ax = plt.subplots(figsize=(6, 4))
    bottom = np.zeros(len(planes))

    for label_idx, label in enumerate(label_names):
        counts = np.array([per_plane_counts[plane][label_idx] for plane in planes])
        ax.bar(indices, counts, width, bottom=bottom, label=label, color=palette.get(label, "#999999"))
        bottom += counts

    ax.set_xticks(indices)
    ax.set_xticklabels([plane.upper() for plane in planes])
    ax.set_ylabel("hit count")
    ax.set_title(f"Semantic label distribution – {split_name}")
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(outdir / f"{split_name}_semantic_summary.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path, help="Path to NuGraph HDF5 file")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test",
                        help="Dataset split to inspect")
    parser.add_argument("--limit", type=int, default=3, help="Maximum number of samples to plot")
    parser.add_argument("--outdir", type=Path, default=Path("plots"),
                        help="Directory where figures will be written")
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    with h5py.File(args.data_path, "r") as f:
        planes = decode_bytes(f["planes"][()])
        semantic_labels = decode_bytes(f["semantic_classes"][()])
        palette = semantic_palette(semantic_labels)

        samples = decode_bytes(f["samples"][args.split][()])
        if args.limit:
            samples = samples[: args.limit]
        if not samples:
            raise RuntimeError(f"No samples found for split '{args.split}'.")

        per_plane_counts = {plane: np.zeros(len(semantic_labels) + 1, dtype=int) for plane in planes}

        for sample in samples:
            record = load_sample(f[f"dataset/{sample}"][()])
            sample_outdir = args.outdir / sample
            ensure_outdir(sample_outdir)

            plot_semantic_scatter(sample, planes, semantic_labels, palette, record, sample_outdir)
            plot_instance_scatter(sample, planes, record, sample_outdir)

            # update cumulative counts (shift by +1 to account for background)
            for plane in planes:
                y_sem = record[f"{plane}/y_semantic"].astype(int)
                counts = np.bincount(y_sem + 1, minlength=len(semantic_labels) + 1)
                per_plane_counts[plane] += counts

        plot_semantic_summary(args.split, planes, semantic_labels, palette, per_plane_counts, args.outdir)


if __name__ == "__main__":
    main()
