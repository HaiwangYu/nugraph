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
    """
    path = Path(path)
    data = np.load(path)

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

    # Optional edges
    if "ppedges" in data.files:
        ppedges = data["ppedges"]
    else:
        ppedges = np.empty((0, 3), dtype=np.float32)

    # Optional neutrino-vertex geometry per hit
    vtx_dist = data["vtx_dist"] if "vtx_dist" in data.files else None
    vtx_dx   = data["vtx_dx"]   if "vtx_dx"   in data.files else None
    vtx_dy   = data["vtx_dy"]   if "vtx_dy"   in data.files else None
    vtx_dz   = data["vtx_dz"]   if "vtx_dz"   in data.files else None

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
        path=path,
    )
