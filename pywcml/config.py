# filename: pywcml/config.py
"""Configuration helpers for the pywcml converter."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import radians
from typing import Dict, Iterable


@dataclass(frozen=True)
class PlaneSpec:
    """Describe a detector plane and its projection properties."""
    key: str
    name: str
    angle_rad: float


def default_planes(apa: int | str = 0) -> Dict[str, PlaneSpec]:
    """Return the default plane configuration for the requested SBND APA."""
    apa_label = str(apa).lower()
    if apa_label.startswith("apa"):
        apa_label = apa_label[3:]

    specs: Iterable[tuple[str, str, float]]
    if apa_label == "1":
        specs = (
            ("ctpc_f1p0", "u", radians(-60.0)),
            ("ctpc_f1p1", "v", radians(+60.0)),
            ("ctpc_f1p2", "y", radians(0.0)),
        )
    else:
        specs = (
            ("ctpc_f0p0", "u", radians(+60.0)),
            ("ctpc_f0p1", "v", radians(-60.0)),
            ("ctpc_f0p2", "y", radians(0.0)),
        )

    return {key: PlaneSpec(key=key, name=name, angle_rad=angle) for key, name, angle in specs}


@dataclass
class ConversionConfig:
    """Runtime options for converting NPZ payloads into NuGraph graphs."""

    # ------------------------------------------------------------------
    # Geometry / plane mapping
    # ------------------------------------------------------------------
    planes: Dict[str, PlaneSpec] = field(default_factory=default_planes)
    detect_planes_from_name: bool = True

    # ------------------------------------------------------------------
    # Association tolerances (mm)
    # ------------------------------------------------------------------
    x_tolerance: float = 5.0               # CTPC ↔ blob x matching
    projection_tolerance: float = 0.0      # projected wire tolerance
    pitch_gap_tolerance: float = 6.0       # gap splitting CTPC stripes

    # ------------------------------------------------------------------
    # Semantic labeling conventions
    # ------------------------------------------------------------------
    semantic_positive: int = 1             # value in is_nu treated as ν
    semantic_negative: int = 0             # fallback label (cosmic)
    semantic_classes: tuple[str, ...] = ("nu", "cosmic")

    # ------------------------------------------------------------------
    # Dataset splits
    # ------------------------------------------------------------------
    train_fraction: float = 0.7
    val_fraction: float = 0.15

    # ------------------------------------------------------------------
    # Legacy / compatibility (kept if referenced elsewhere)
    # ------------------------------------------------------------------
    unit_ratio_threshold: float = 3.0

    # ------------------------------------------------------------------
    # Feature / behavior toggles (ALL OFF BY DEFAULT)
    # ------------------------------------------------------------------
    enable_sidecar_features: bool = False
    enable_local_pca: bool = False              # requires enable_sidecar_features
    enable_event_bbox_wall_dists: bool = False

    # --- semantic cleanup fixes ---
    enable_vertex_semantic_fix: bool = False    # vertex-radius repair
    enable_semantic_mp_propagation_fix: bool = False  # MP-graph propagation fix
    mp_prop_seed_frac_nu_min: float = 0.60

    # --- edge behavior ---
    merge_sup_edges_into_mp: bool = False

    # --- diagnostics ---
    write_diagnostics: bool = False

    # ------------------------------------------------------------------
    # Sidecar / PCA parameters (used only if enabled)
    # ------------------------------------------------------------------
    sidecar_k: int = 12

    # ------------------------------------------------------------------
    # Semantic MP-propagation parameters (mm units)
    # ------------------------------------------------------------------
    mp_prop_seed_vtx_radius_mm: float = 100.0    # 10 cm seed radius
    mp_prop_edge_len_max_mm: float = 225.0       # p99 MP edge length
    mp_prop_max_hops: int = 5
    mp_prop_max_vtx_dist_mm: float = 2000.0      # stop beyond 2 m

    def __post_init__(self):
        if self.enable_local_pca and not self.enable_sidecar_features:
            raise ValueError(
                "enable_local_pca=True requires enable_sidecar_features=True"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def plane_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.planes.values())

    def planes_for_sample(self, sample_name: str) -> Dict[str, PlaneSpec]:
        """Return plane configuration, switching to APA1 when requested."""
        if not self.detect_planes_from_name:
            return self.planes

        if "apa1" in sample_name.lower():
            return default_planes(1)

        return self.planes


__all__ = ["PlaneSpec", "ConversionConfig", "default_planes"]
