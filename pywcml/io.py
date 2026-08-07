"""WCML in-memory array contracts and legacy NPZ compatibility loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np


_DIRECT_FIELDS = {
    "blobs",
    "points",
    "is_nu",
    "ppedges",
    "origin_label",
    "vtx_dist",
    "vtx_dx",
    "vtx_dy",
    "vtx_dz",
    "nu_vtx",
    "nu_vtx_found",
    "truth_blob_tid",
    "truth_blob_purity",
    "truth_blob_support",
    "edge_index",
    "edge_y",
}


@dataclass
class WCMLArrays:
    """All arrays needed by the converter plus lossless labeler diagnostics.

    ``path`` is optional and exists only for the historical filename-driven
    route. Streaming callers construct this object directly and carry explicit
    :class:`pywcml.identity.EventIdentity` instead.
    """

    blobs: np.ndarray
    points: np.ndarray
    ctpc: Dict[str, np.ndarray]
    is_nu: Optional[np.ndarray]
    ppedges: np.ndarray
    origin_label: Optional[np.ndarray]
    vtx_dist: Optional[np.ndarray]
    vtx_dx: Optional[np.ndarray]
    vtx_dy: Optional[np.ndarray]
    vtx_dz: Optional[np.ndarray]
    nu_vtx: Optional[np.ndarray]
    nu_vtx_found: Optional[np.ndarray]
    truth_blob_tid: Optional[np.ndarray]
    truth_blob_purity: Optional[np.ndarray]
    truth_blob_support: Optional[np.ndarray]
    # Labeler-owned point-edge supervision remains loadable for diagnostics
    # and legacy workflows. It never constructs NuGraph topology or targets.
    edge_index: Optional[np.ndarray]
    edge_y: Optional[np.ndarray]
    path: Optional[Path] = None
    # Point truth, per-plane truth, and any source-specific arrays not consumed
    # directly by the converter are retained here for lossless CLI round trips.
    extras: Dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        arrays: Mapping[str, np.ndarray],
        *,
        path: Path | str | None = None,
    ) -> "WCMLArrays":
        """Construct from an already-materialized mapping without serialization."""

        if "blobs" not in arrays or "points" not in arrays:
            raise KeyError("WCML arrays require 'blobs' and 'points'")

        def optional(key: str) -> Optional[np.ndarray]:
            return arrays.get(key)

        ppedges = optional("ppedges")
        if ppedges is None:
            ppedges = np.empty((0, 3), dtype=np.float32)

        ctpc = {key: value for key, value in arrays.items() if key.startswith("ctpc_")}
        extras = {
            key: value
            for key, value in arrays.items()
            if key not in _DIRECT_FIELDS and not key.startswith("ctpc_")
        }
        return cls(
            blobs=arrays["blobs"],
            points=arrays["points"],
            ctpc=ctpc,
            is_nu=optional("is_nu"),
            ppedges=ppedges,
            origin_label=optional("origin_label"),
            vtx_dist=optional("vtx_dist"),
            vtx_dx=optional("vtx_dx"),
            vtx_dy=optional("vtx_dy"),
            vtx_dz=optional("vtx_dz"),
            nu_vtx=optional("nu_vtx"),
            nu_vtx_found=optional("nu_vtx_found"),
            truth_blob_tid=optional("truth_blob_tid"),
            truth_blob_purity=optional("truth_blob_purity"),
            truth_blob_support=optional("truth_blob_support"),
            edge_index=optional("edge_index"),
            edge_y=optional("edge_y"),
            path=Path(path) if path is not None else None,
            extras=extras,
        )

    def to_mapping(self) -> Dict[str, np.ndarray]:
        """Return the historical NPZ payload mapping without writing it."""

        result = dict(self.extras)
        result.update(self.ctpc)
        result["blobs"] = self.blobs
        result["points"] = self.points
        result["ppedges"] = self.ppedges
        for key in _DIRECT_FIELDS - {"blobs", "points", "ppedges"}:
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result

    def __getattr__(self, name: str):
        """Expose retained diagnostic arrays to existing converter helpers."""

        extras = self.__dict__.get("extras", {})
        if name in extras:
            return extras[name]
        raise AttributeError(name)


def load_npz(path: Path | str) -> WCMLArrays:
    """Load a legacy WCML NPZ archive into the common in-memory contract."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    return WCMLArrays.from_mapping(payload, path=path)


__all__ = ["WCMLArrays", "load_npz"]
