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
    planes: Dict[str, PlaneSpec] = field(default_factory=default_planes)
    x_tolerance: float = 5.0  # mm distance for associating CTPC points with a blob
    unit_ratio_threshold: float = 3.0  # if |ratio| exceeds this, rescale blob corners
    semantic_positive: int = 1  # value in ``is_nu`` treated as signal
    semantic_negative: int = 0  # fallback label when no signal points are present
    semantic_classes: tuple[str, ...] = ("nu", "cosmic")
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    projection_tolerance: float = 0.0  # mm tolerance between projected and observed wire coordinates
    pitch_gap_tolerance: float = 6.0  # mm gap that breaks contiguous CTPC stripes into distinct nodes
    detect_planes_from_name: bool = True

    def plane_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.planes.values())

    def planes_for_sample(self, sample_name: str) -> Dict[str, PlaneSpec]:
        """Return plane configuration, switching to APA1 when the name requests it."""
        if not self.detect_planes_from_name:
            return self.planes

        if "apa1" in sample_name.lower():
            return default_planes(1)

        return self.planes


__all__ = ["PlaneSpec", "ConversionConfig", "default_planes"]
