"""Convert WCML NPZ files into NuGraph-compatible HDF5 datasets."""
from __future__ import annotations

from dataclasses import dataclass
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import h5py

from pynuml.data import NuGraphData

from .config import ConversionConfig, PlaneSpec
from .geometry import estimate_unit_scale, project_corners, triangulation_edges
from .io import WCMLArrays, load_npz


@dataclass
class PlaneNodes:
    pos: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    instances: np.ndarray
    to_sp: np.ndarray
    edges: np.ndarray


class WCMLConverter:
    """Convert WCML NPZ archives into NuGraph graphs and packaged HDF5 files."""

    def __init__(self, config: ConversionConfig | None = None):
        self.config = config or ConversionConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def convert(self, npz_path: Path | str) -> tuple[str, NuGraphData]:
        arrays = load_npz(npz_path)
        graph_name = arrays.path.stem
        graph = self._build_graph(graph_name, arrays)
        return graph_name, graph

    def convert_many(self, paths: Sequence[Path | str]) -> Dict[str, NuGraphData]:
        graphs = {}
        for path in paths:
            name, data = self.convert(path)
            graphs[name] = data
        return graphs

    def write_hdf5(self,
                   graphs: Dict[str, NuGraphData],
                   output: Path | str,
                   splits: Dict[str, Sequence[str]] | None = None) -> None:
        """Persist converted graphs to an HDF5 file compatible with ``H5DataModule``."""
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if splits is None:
            sample_names = list(graphs.keys())
            train_end = int(len(sample_names) * self.config.train_fraction)
            val_end = train_end + int(len(sample_names) * self.config.val_fraction)
            splits = {
                "train": sample_names[:train_end],
                "validation": sample_names[train_end:val_end],
                "test": sample_names[val_end:],
            }
            if not splits["train"] and sample_names:
                splits["train"] = [sample_names[0]]
            for key in ("validation", "test"):
                if not splits[key] and sample_names:
                    splits[key] = sample_names[:1]

        with h5py.File(out_path, "w") as f:
            planes = np.array(self.config.plane_names(), dtype=h5py.string_dtype())
            semantics = np.array(self.config.semantic_classes, dtype=h5py.string_dtype())
            f.create_dataset("planes", data=planes)
            f.create_dataset("semantic_classes", data=semantics)
            f.create_dataset("gen", data=np.array([3], dtype=np.int64))

            sample_group = f.create_group("samples")
            for split, names in splits.items():
                names = list(dict.fromkeys(names))  # deduplicate, keep order
                sample_group.create_dataset(split, data=np.array(names, dtype=h5py.string_dtype()))

            datasize_group = f.create_group("datasize")
            datasize_group.create_dataset("train", data=self._graph_sizes(graphs, splits["train"]))

            f.require_group("dataset")
            for name, data in graphs.items():
                data.save(f, f"dataset/{name}")

    # ------------------------------------------------------------------
    # Core conversion logic
    # ------------------------------------------------------------------
    def _build_graph(self, sample_name: str, arrays: WCMLArrays) -> NuGraphData:
        charges, centroids, corners = self._extract_blobs(arrays.blobs)
        semantic = self._label_blobs(arrays.points, arrays.is_nu, len(corners), self.config)

        plane_nodes: Dict[str, PlaneNodes] = {}
        for key, spec in self.config.planes.items():
            ctpc = arrays.ctpc.get(key)
            if ctpc is None:
                plane_nodes[spec.name] = PlaneNodes(
                    pos=np.empty((0, 2), dtype=np.float32),
                    features=np.empty((0, 3), dtype=np.float32),
                    labels=np.empty((0,), dtype=np.int64),
                    instances=np.empty((0,), dtype=np.int64),
                    to_sp=np.empty((0, 2), dtype=np.int64),
                    edges=np.empty((2, 0), dtype=np.int64),
                )
                continue
            plane_nodes[spec.name] = self._build_plane(spec, ctpc, corners, centroids, semantic)

        graph = NuGraphData()
        run, subrun, event = self._infer_event_ids(sample_name)
        graph["metadata"].run = run
        graph["metadata"].subrun = subrun
        graph["metadata"].event = event

        graph["sp"].pos = torch.as_tensor(centroids, dtype=torch.float32)
        graph["sp"].q = torch.as_tensor(charges, dtype=torch.float32)
        graph["sp"].y_semantic = torch.as_tensor(semantic, dtype=torch.long)

        for plane_name in self.config.plane_names():
            nodes = plane_nodes.get(plane_name)
            if nodes is None:
                nodes = PlaneNodes(
                    pos=np.empty((0, 2), dtype=np.float32),
                    features=np.empty((0, 3), dtype=np.float32),
                    labels=np.empty((0,), dtype=np.int64),
                    instances=np.empty((0,), dtype=np.int64),
                    to_sp=np.empty((0, 2), dtype=np.int64),
                    edges=np.empty((2, 0), dtype=np.int64),
                )
            store = graph[plane_name]
            store.pos = torch.as_tensor(nodes.pos, dtype=torch.float32)
            store.x = torch.as_tensor(nodes.features, dtype=torch.float32)
            store.id = torch.arange(nodes.pos.shape[0], dtype=torch.long)
            store.y_semantic = torch.as_tensor(nodes.labels, dtype=torch.long)
            store.y_instance = torch.as_tensor(nodes.instances, dtype=torch.long)

            if nodes.edges.size:
                store_edges = torch.as_tensor(nodes.edges, dtype=torch.long)
            else:
                store_edges = torch.empty((2, 0), dtype=torch.long)
            graph[plane_name, "plane", plane_name].edge_index = store_edges

            if nodes.to_sp.size:
                nexus_edges = torch.as_tensor(nodes.to_sp, dtype=torch.long).t()
            else:
                nexus_edges = torch.empty((2, 0), dtype=torch.long)
            graph[plane_name, "nexus", "sp"].edge_index = nexus_edges

        graph["evt"].num_nodes = 1
        graph["evt"].y = torch.tensor([-1], dtype=torch.long)

        return graph

    def _extract_blobs(self, raw_blobs: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        charges = raw_blobs[:, 0]
        corner_counts = raw_blobs[:, 1].astype(int)
        corners: list[np.ndarray] = []
        centroids: list[np.ndarray] = []
        offset = 2
        for idx, count in enumerate(corner_counts):
            end = offset + 3 * count
            coords = raw_blobs[idx, offset:end].reshape(count, 3)
            corners.append(coords)
            centroids.append(coords.mean(axis=0))
        return charges.astype(np.float32), np.asarray(centroids, dtype=np.float32), corners

    def _label_blobs(self,
                     points: np.ndarray,
                     is_nu: np.ndarray | None,
                     n_expected: int,
                     config: ConversionConfig) -> np.ndarray:
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
            labels[blob_id] = config.semantic_positive if (blob_labels == config.semantic_positive).any() else config.semantic_negative
        return labels

    def _build_plane(self,
                     spec: PlaneSpec,
                     ctpc: np.ndarray,
                     corners: Sequence[np.ndarray],
                     centroid: np.ndarray,
                     semantic: np.ndarray) -> PlaneNodes:
        positions: List[List[float]] = []
        features: List[List[float]] = []
        labels: List[int] = []
        instances: List[int] = []
        to_sp: List[Tuple[int, int]] = []

        for blob_id, corner_set in enumerate(corners):
            mask = np.abs(ctpc[:, 0] - centroid[blob_id, 0]) <= self.config.x_tolerance
            plane_points = ctpc[mask]
            if not plane_points.size:
                continue

            projected = project_corners(corner_set, spec.angle_rad)
            scale = estimate_unit_scale(projected, plane_points[:, 1], self.config.unit_ratio_threshold)
            if not np.isclose(scale, 1.0):
                projected *= scale
                warnings.warn(
                    f"Rescaled blob {blob_id} on plane {spec.name} by factor {scale:.2f} to match CTPC units.",
                    RuntimeWarning,
                )

            delta = abs(projected.mean() - plane_points[:, 1].mean())
            if delta > self.config.projection_tolerance:
                warnings.warn(
                    (
                        f"Blob {blob_id} projection mismatch on plane {spec.name}: "
                        f"|Δ|={delta:.2f} mm exceeds tolerance {self.config.projection_tolerance:.2f} mm"
                    ),
                    RuntimeWarning,
                )

            pos_x = plane_points[:, 0].mean()
            pos_y = plane_points[:, 1].mean()
            positions.append([pos_x, pos_y])

            total_charge = float(plane_points[:, 2].sum())
            mean_charge_err = float(plane_points[:, 3].mean())
            nhits = float(len(plane_points))
            features.append([total_charge, mean_charge_err, nhits])

            labels.append(int(semantic[blob_id]))
            instances.append(0)
            to_sp.append((len(positions) - 1, blob_id))

        if positions:
            pos_array = np.asarray(positions, dtype=np.float32)
            feat_array = np.asarray(features, dtype=np.float32)
            label_array = np.asarray(labels, dtype=np.int64)
            inst_array = np.asarray(instances, dtype=np.int64)
            edges = triangulation_edges(pos_array)
            to_sp_arr = np.asarray(to_sp, dtype=np.int64)
        else:
            pos_array = np.empty((0, 2), dtype=np.float32)
            feat_array = np.empty((0, 3), dtype=np.float32)
            label_array = np.empty((0,), dtype=np.int64)
            inst_array = np.empty((0,), dtype=np.int64)
            edges = np.empty((2, 0), dtype=np.int64)
            to_sp_arr = np.empty((0, 2), dtype=np.int64)

        return PlaneNodes(
            pos=pos_array,
            features=feat_array,
            labels=label_array,
            instances=inst_array,
            to_sp=to_sp_arr,
            edges=edges,
        )

    def _infer_event_ids(self, sample_name: str) -> tuple[int, int, int]:
        parts = sample_name.split('-')
        run = 0
        subrun = 0
        event = 0
        for part in parts:
            if part.startswith("apa"):
                try:
                    subrun = int(part.replace("apa", ""))
                except ValueError:
                    pass
            elif part.isdigit():
                event = int(part)
        return run, subrun, event

    def _graph_sizes(self, graphs: Dict[str, NuGraphData], names: Sequence[str]) -> np.ndarray:
        if not names:
            return np.zeros((0,), dtype=np.int64)
        sizes = []
        for name in names:
            data = graphs[name]
            total = 0
            for store in data.stores:  # type: ignore[attr-defined]
                for value in store.values():
                    if isinstance(value, torch.Tensor):
                        total += value.element_size() * value.nelement()
            sizes.append(total)
        return np.asarray(sizes, dtype=np.int64)


def convert_npz_file(npz_path: Path | str,
                     output: Path | str,
                     config: ConversionConfig | None = None) -> Path:
    converter = WCMLConverter(config)
    name, graph = converter.convert(npz_path)
    converter.write_hdf5({name: graph}, output)
    return Path(output)


def convert_npz_directory(directory: Path | str,
                          output: Path | str,
                          config: ConversionConfig | None = None) -> Path:
    converter = WCMLConverter(config)
    directory = Path(directory)
    paths = sorted(directory.glob("*.npz"))
    graphs = converter.convert_many(paths)
    converter.write_hdf5(graphs, output)
    return Path(output)


__all__ = [
    "WCMLConverter",
    "convert_npz_file",
    "convert_npz_directory",
]
