#!/usr/bin/env python3
"""Visualise blob corner projections onto each CTPC plane."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from pywcml.config import ConversionConfig
from pywcml.geometry import project_corners
from pywcml.io import load_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, help="Path to a WCML NPZ payload")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination image path (defaults to showing the plot)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI when saving (default: %(default)s)",
    )
    return parser.parse_args()


def extract_blob_corners(raw_blobs: np.ndarray) -> List[np.ndarray]:
    corners: List[np.ndarray] = []
    offset = 2
    for row in raw_blobs:
        count = int(row[1])
        data = row[offset : offset + 3 * count]
        corners.append(data.reshape(count, 3))
    return corners


def infer_blob_labels(points: np.ndarray, is_nu: np.ndarray | None, n_expected: int, config: ConversionConfig) -> np.ndarray:
    blob_indices = points[:, 4].astype(int)
    inferred = int(blob_indices.max()) + 1 if blob_indices.size else 0
    n_blobs = max(inferred, n_expected)
    labels = np.full(n_blobs, config.semantic_negative, dtype=np.int64)
    if is_nu is None:
        return labels
    for blob_id in range(labels.size):
        mask = blob_indices == blob_id
        if not mask.any():
            continue
        blob_labels = is_nu[mask]
        labels[blob_id] = (
            config.semantic_positive
            if (blob_labels == config.semantic_positive).any()
            else config.semantic_negative
        )
    return labels


def plot_projections(npz_path: Path, output: Path | None, dpi: int) -> None:
    arrays = load_npz(npz_path)
    config = ConversionConfig()
    corners = extract_blob_corners(arrays.blobs)
    labels = infer_blob_labels(arrays.points, arrays.is_nu, len(corners), config)

    plane_specs = list(config.planes.values())
    fig, axes = plt.subplots(1, len(plane_specs), figsize=(4 * len(plane_specs), 4), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    colours = {config.semantic_negative: "C0", config.semantic_positive: "C1"}

    for ax, spec in zip(axes, plane_specs):
        for blob_id, corner_set in enumerate(corners):
            if not corner_set.size:
                continue
            pitch = project_corners(corner_set, spec.angle_rad)
            ax.plot(
                pitch,
                corner_set[:, 0],
                marker="o",
                linestyle="-",
                linewidth=0.8,
                markersize=3,
                color=colours.get(labels[blob_id], "C2"),
                alpha=0.7,
            )
        ax.set_title(f"Plane {spec.name}")
        ax.set_xlabel("Projected pitch [mm]")
    axes[0].set_ylabel("Drift x [mm]")
    handles = [
        plt.Line2D([0], [0], color="C0", label="background"),
        plt.Line2D([0], [0], color="C1", label="neutrino"),
    ]
    fig.legend(handles=handles, loc="upper right")
    fig.suptitle(f"Blob corner projections: {npz_path.stem}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi)
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    plot_projections(args.npz, args.output, args.dpi)


if __name__ == "__main__":
    main()
