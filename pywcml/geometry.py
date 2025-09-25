"""Geometry and graph-building helpers."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
from matplotlib import tri


@dataclass(frozen=True)
class ProjectionResult:
    projected_y: np.ndarray
    scale: float


def project_corners(corners: np.ndarray, angle_rad: float) -> np.ndarray:
    """Project 3D blob corners into a single plane coordinate."""
    yb = corners[:, 1]
    zb = corners[:, 2]
    return np.cos(angle_rad) * zb - np.sin(angle_rad) * yb


def estimate_unit_scale(projected: np.ndarray, observed: np.ndarray, threshold: float) -> float:
    """Detect large unit mismatches (e.g. cm vs mm) and propose a scale factor."""
    if not projected.size or not observed.size:
        return 1.0
    proj_med = np.median(np.abs(projected)) + 1e-6
    obs_med = np.median(np.abs(observed)) + 1e-6
    ratio = obs_med / proj_med
    if ratio > threshold:
        return ratio
    if ratio < 1.0 / threshold:
        return ratio
    return 1.0


def triangulation_edges(points: np.ndarray) -> np.ndarray:
    """Create undirected edges from a Delaunay triangulation."""
    if points.shape[0] < 2:
        return np.empty((2, 0), dtype=np.int64)
    if np.allclose(points[:, :2], points[0, :2]):
        return np.empty((2, 0), dtype=np.int64)
    triang = tri.Triangulation(points[:, 0], points[:, 1])
    edge_set = set()
    for tri_idx in triang.triangles:
        for i, j in combinations(tri_idx, 2):
            if i == j:
                continue
            edge = (min(i, j), max(i, j))
            edge_set.add(edge)
    if not edge_set:
        return np.empty((2, 0), dtype=np.int64)
    edges = np.array(tuple(edge_set), dtype=np.int64)
    return edges.T


__all__ = ["ProjectionResult", "project_corners", "estimate_unit_scale", "triangulation_edges"]
