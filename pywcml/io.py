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
    nu_vtx: Optional[np.ndarray]
    nu_vtx_found: Optional[np.ndarray]
    truth_blob_tid: Optional[np.ndarray]
    truth_blob_purity: Optional[np.ndarray]
    truth_blob_support: Optional[np.ndarray]
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
      - optional: vtx_dist, vtx_dx, vtx_dy, vtx_dz, nu_vtx, nu_vtx_found
      - optional: truth_blob_tid, truth_blob_purity, truth_blob_support
      - optional: edge_index, edge_y
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        blobs = data["blobs"]
        points = data["points"]
        ctpc = {key: data[key] for key in data.files if key.startswith("ctpc_")}

        def optional(key: str) -> Optional[np.ndarray]:
            return data[key] if key in data.files else None

        is_nu = optional("is_nu")
        origin_label = optional("origin_label")
        ppedges = optional("ppedges")
        if ppedges is None:
            ppedges = np.empty((0, 3), dtype=np.float32)

        vtx_dist = optional("vtx_dist")
        vtx_dx = optional("vtx_dx")
        vtx_dy = optional("vtx_dy")
        vtx_dz = optional("vtx_dz")
        nu_vtx = optional("nu_vtx")
        nu_vtx_found = optional("nu_vtx_found")
        truth_blob_tid = optional("truth_blob_tid")
        truth_blob_purity = optional("truth_blob_purity")
        truth_blob_support = optional("truth_blob_support")
        edge_index = optional("edge_index")
        edge_y = optional("edge_y")

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
