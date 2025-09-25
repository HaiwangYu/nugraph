"""I/O helpers for Wire-Cell ML NPZ payloads."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass
class WCMLArrays:
    """Container for all arrays stored in a WCML NPZ file."""
    path: Path
    blobs: np.ndarray
    points: np.ndarray
    is_nu: np.ndarray
    ppedges: np.ndarray
    ctpc: Dict[str, np.ndarray]

    def plane_keys(self) -> tuple[str, ...]:
        return tuple(self.ctpc.keys())


def load_npz(path: Path | str) -> WCMLArrays:
    """Load a WCML NPZ bundle from disk."""
    p = Path(path)
    with np.load(p) as payload:
        arrays = {key: payload[key] for key in payload.files}
    ctpc = {key: arrays[key] for key in arrays if key.startswith("ctpc_")}
    return WCMLArrays(
        path=p,
        blobs=arrays["blobs"],
        points=arrays["points"],
        is_nu=arrays.get("is_nu"),
        ppedges=arrays.get("ppedges"),
        ctpc=ctpc,
    )


__all__ = ["WCMLArrays", "load_npz"]
