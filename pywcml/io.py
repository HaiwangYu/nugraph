# filename: pywcml/io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np


@dataclass
class WCMLArrays:
    blobs: np.ndarray
    points: np.ndarray
    ctpc: Dict[str, np.ndarray]
    is_nu: Optional[np.ndarray]
    ppedges: np.ndarray
    origin_label: Optional[np.ndarray]
    # Optional per-hit neutrino-vertex info
    vtx_dist: Optional[np.ndarray]
    vtx_dx: Optional[np.ndarray]
    vtx_dy: Optional[np.ndarray]
    vtx_dz: Optional[np.ndarray]
    # Optional raw neutrino vertex (single 3D point)
    nu_vtx: Optional[np.ndarray]
    nu_vtx_found: Optional[np.ndarray]

    # ---- NEW: instance labeling payload (blob-level truth) ----
    truth_blob_tid: Optional[np.ndarray]
    truth_blob_purity: Optional[np.ndarray]
    truth_blob_support: Optional[np.ndarray]

    # ---- NEW: optional edge supervision written by labeling script ----
    # NOTE: these are point-level if they come straight from labeling_with_truth.py,
    # but we still load them so converter can optionally consume or ignore them.
    edge_index: Optional[np.ndarray]
    edge_y: Optional[np.ndarray]

    # Keep track of origin path (converter uses this)
    path: Path


def load_npz(path: Path | str) -> WCMLArrays:
    """
    Load a WCML-style NPZ file and return a WCMLArrays bundle.

    This includes:
      - required: blobs, points
      - ctpc_* plane arrays
      - optional: is_nu, origin_label, ppedges
      - optional: vtx_dist, vtx_dx, vtx_dy, vtx_dz
      - optional: nu_vtx, nu_vtx_found
      - optional (NEW): truth_blob_tid/purity/support for instance truth at blob level
      - optional (NEW): edge_index/edge_y (edge supervision)
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    # Required
    blobs = data["blobs"]
    points = data["points"]

    # ctpc_* planes
    ctpc: Dict[str, np.ndarray] = {}
    for key in data.files:
        if key.startswith("ctpc_"):
            ctpc[key] = data[key]

    # Optional truth / labels
    is_nu = data["is_nu"] if "is_nu" in data.files else None
    origin_label = data["origin_label"] if "origin_label" in data.files else None

    # Optional edges (ppedges)
    if "ppedges" in data.files:
        ppedges = data["ppedges"]
    else:
        ppedges = np.empty((0, 3), dtype=np.float32)

    # Optional neutrino-vertex geometry per hit
    vtx_dist = data["vtx_dist"] if "vtx_dist" in data.files else None
    vtx_dx = data["vtx_dx"] if "vtx_dx" in data.files else None
    vtx_dy = data["vtx_dy"] if "vtx_dy" in data.files else None
    vtx_dz = data["vtx_dz"] if "vtx_dz" in data.files else None

    # Optional raw neutrino vertex
    nu_vtx = data["nu_vtx"] if "nu_vtx" in data.files else None
    nu_vtx_found = data["nu_vtx_found"] if "nu_vtx_found" in data.files else None

    # ---- NEW: blob-level truth instance payload ----
    truth_blob_tid = data["truth_blob_tid"] if "truth_blob_tid" in data.files else None
    truth_blob_purity = data["truth_blob_purity"] if "truth_blob_purity" in data.files else None
    truth_blob_support = data["truth_blob_support"] if "truth_blob_support" in data.files else None

    # ---- NEW: edge supervision payload (optional) ----
    edge_index = data["edge_index"] if "edge_index" in data.files else None
    edge_y = data["edge_y"] if "edge_y" in data.files else None

    return WCMLArrays(
        blobs=blobs,
        points=points,
        ctpc=ctpc,
        is_nu=is_nu,
        ppedges=ppedges,
        origin_label=origin_label,
        vtx_dist=vtx_dist,
        vtx_dx=vtx_dx,
        vtx_dy=vtx_dy,
        vtx_dz=vtx_dz,
        nu_vtx=nu_vtx,
        nu_vtx_found=nu_vtx_found,
        truth_blob_tid=truth_blob_tid,
        truth_blob_purity=truth_blob_purity,
        truth_blob_support=truth_blob_support,
        edge_index=edge_index,
        edge_y=edge_y,
        path=path,
    )
