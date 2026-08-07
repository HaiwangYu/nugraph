"""Compatibility imports for the legacy point-edge diagnostic builder."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pywcml.edge_supervision import (  # noqa: E402,F401
    build_edge_supervision,
    mine_hard_negatives_radius_mm,
    mine_random_negatives,
)

__all__ = [
    "build_edge_supervision",
    "mine_hard_negatives_radius_mm",
    "mine_random_negatives",
]
