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
    palette = {"background": "#b3b3b3"}
    if not classes:
        return palette

    base = plt.colormaps.get_cmap("tab10")
    for val, cls in zip(np.linspace(0, 1, len(classes), endpoint=False), classes):
        palette[cls] = mcolors.to_hex(base(val))
    return palette


def infer_semantic_encoding(values: np.ndarray, classes: list[str]) -> tuple[int | None, int]:
    """Infer background code and class offset from encoded semantic labels."""
    if values.size == 0:
        background = 0 if classes else None
        offset = 1 if background == 0 else 0
        return background, offset

    codes = np.unique(values.astype(int))

    background: int | None = None
    negative_codes = codes[codes < 0]
    if negative_codes.size:
        background = int(negative_codes.min())
    elif 0 in codes and codes.size > len(classes):
        background = 0

    positive_codes = codes if background is None else codes[codes != background]

    if background == -1:
        offset = 0
    elif background == 0:
        offset = 1
    else:
        offset = 1 if positive_codes.size and positive_codes.min() == 1 else 0

    return background, offset


def semantic_counts(values: np.ndarray, classes: list[str]) -> np.ndarray:
    """Compute semantic label counts with dynamic encoding inference."""
    background_code, offset = infer_semantic_encoding(values, classes)
    counts = np.zeros(len(classes) + 1, dtype=int)

    if background_code is not None:
        counts[0] = np.count_nonzero(values == background_code)

    for idx in range(len(classes)):
        target = idx + offset
        counts[idx + 1] = np.count_nonzero(values == target)

    return counts


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
    label_order = {label: idx for idx, label in enumerate(label_names)}

    for ax, plane in zip(axes, planes):
        pos = data[f"{plane}/pos"]
        y_sem = data[f"{plane}/y_semantic"].astype(int)
        background_code, offset = infer_semantic_encoding(y_sem, semantic_labels)

        labels = []
        for val in y_sem:
            if background_code is not None and val == background_code:
                label = "background"
            else:
                idx = val - offset
                if 0 <= idx < len(semantic_labels):
                    label = semantic_labels[idx]
                else:
                    label = "unknown"
            labels.append(label)

        colours = [palette.get(label, palette["background"]) for label in labels]
        ax.scatter(pos[:, 0], pos[:, 1], c=colours, s=5, linewidths=0)
        ax.set_title(f"{plane.upper()} plane")
        ax.set_xlabel("proj")
        ax.set_ylabel("drift")

        unique_labels = sorted(set(labels), key=lambda lbl: label_order.get(lbl, len(label_order)))
        handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=palette.get(label, palette["background"]), label=label)
                   for label in unique_labels]
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
            cmap = plt.colormaps.get_cmap("nipy_spectral")
            sample_points = np.linspace(0, 1, len(valid_ids), endpoint=False)
            colour_map = {val: cmap(pt) for val, pt in zip(valid_ids, sample_points)}
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

            # update cumulative counts using inferred encoding
            for plane in planes:
                y_sem = record[f"{plane}/y_semantic"].astype(int)
                per_plane_counts[plane] += semantic_counts(y_sem, semantic_labels)

        plot_semantic_summary(args.split, planes, semantic_labels, palette, per_plane_counts, args.outdir)


if __name__ == "__main__":
    main()
