"""Convert WCML NPZ files into NuGraph-compatible HDF5 datasets."""
from __future__ import annotations

from dataclasses import dataclass
import warnings
from pathlib import Path
import re
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


RUN_SUBRUN_PATTERN = re.compile(r"(?P<run>\d+)_(?P<subrun>\d+)")


class WCMLConverter:
    """Convert WCML NPZ archives into NuGraph graphs and packaged HDF5 files."""

    def __init__(self, config: ConversionConfig | None = None):
        self.config = config or ConversionConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def convert(self, npz_path: Path | str, sample_name: str | None = None) -> tuple[str, NuGraphData]:
        path = Path(npz_path)
        arrays = load_npz(path)
        graph_name = sample_name or self._default_sample_name(path)
        graph = self._build_graph(graph_name, arrays)
        return graph_name, graph

    def convert_many(self, paths: Sequence[Path | str]) -> Dict[str, NuGraphData]:
        graphs = {}
        iterator = paths
        try:
            from tqdm import tqdm
            iterator = tqdm(paths, desc="Converting", unit="file")
        except Exception:
            iterator = paths

        for path in iterator:
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
        encoded_semantic = self._encode_semantic_labels(semantic)

        planes = self.config.planes_for_sample(sample_name)
        plane_nodes: Dict[str, PlaneNodes] = {}
        for key, spec in planes.items():
            ctpc = arrays.ctpc.get(key)
            if ctpc is None:
                plane_nodes[spec.name] = PlaneNodes(
                    pos=np.empty((0, 2), dtype=np.float32),
                    features=np.empty((0, 5), dtype=np.float32),
                    labels=np.empty((0,), dtype=np.int64),
                    instances=np.empty((0,), dtype=np.int64),
                    to_sp=np.empty((0, 2), dtype=np.int64),
                    edges=np.empty((2, 0), dtype=np.int64),
                )
                continue
            plane_nodes[spec.name] = self._build_plane(spec, ctpc, corners, centroids, semantic)

        graph = NuGraphData()
        run, subrun, event = self._infer_event_ids(sample_name, arrays.path)
        graph["metadata"].run = run
        graph["metadata"].subrun = subrun
        graph["metadata"].event = event

        graph["sp"].pos = torch.as_tensor(centroids, dtype=torch.float32)
        graph["sp"].q = torch.as_tensor(charges, dtype=torch.float32)
        graph["sp"].y_semantic = torch.as_tensor(encoded_semantic, dtype=torch.long)

        for plane_name in self.config.plane_names():
            nodes = plane_nodes.get(plane_name)
            if nodes is None:
                nodes = PlaneNodes(
                    pos=np.empty((0, 2), dtype=np.float32),
                    features=np.empty((0, 5), dtype=np.float32),
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

    def _encode_semantic_labels(self, labels: np.ndarray) -> np.ndarray:
        """Map raw semantic values to class indices used downstream."""
        encoded = np.full(labels.shape, -1, dtype=np.int64)
        positive_mask = labels == self.config.semantic_positive
        if positive_mask.any():
            encoded[positive_mask] = 0
        negative_mask = labels == self.config.semantic_negative
        if negative_mask.any():
            encoded[negative_mask] = 1
        return encoded

    def _build_plane(self,
                     spec: PlaneSpec,
                     ctpc: np.ndarray,
                     corners: Sequence[np.ndarray],
                     centroid: np.ndarray,
                     semantic: np.ndarray) -> PlaneNodes:
        if not ctpc.size:
            return PlaneNodes(
                pos=np.empty((0, 2), dtype=np.float32),
                features=np.empty((0, 5), dtype=np.float32),
                labels=np.empty((0,), dtype=np.int64),
                instances=np.empty((0,), dtype=np.int64),
                to_sp=np.empty((0, 2), dtype=np.int64),
                edges=np.empty((2, 0), dtype=np.int64),
            )

        x_tol = self.config.x_tolerance
        pitch_tol = self.config.pitch_gap_tolerance

        order = np.argsort(ctpc[:, 0])
        sorted_points = ctpc[order]
        x_groups: List[np.ndarray] = []
        current: List[np.ndarray] = []
        current_ref = None
        for point in sorted_points:
            x_val = point[0]
            if not current:
                current = [point]
                current_ref = x_val
                continue
            if abs(x_val - current_ref) <= x_tol:
                current.append(point)
                current_ref = (current_ref * (len(current) - 1) + x_val) / len(current)
            else:
                x_groups.append(np.asarray(current))
                current = [point]
                current_ref = x_val
        if current:
            x_groups.append(np.asarray(current))

        positions: List[List[float]] = []
        features: List[List[float]] = []
        pitch_ranges: List[Tuple[float, float]] = []

        def add_node(points_slice: np.ndarray) -> None:
            if not points_slice.size:
                return
            charges = points_slice[:, 2]
            total_charge = float(charges.sum())
            weights = charges if total_charge > 0.0 else np.ones_like(charges)
            wx = float(np.average(points_slice[:, 0], weights=weights))
            wy = float(np.average(points_slice[:, 1], weights=weights))
            mean_charge_err = float(points_slice[:, 3].mean())
            nhits = float(points_slice.shape[0])
            pitch_min = float(points_slice[:, 1].min())
            pitch_max = float(points_slice[:, 1].max())
            positions.append([wx, wy])
            features.append([total_charge, mean_charge_err, nhits, pitch_min, pitch_max])
            pitch_ranges.append((pitch_min, pitch_max))

        for group in x_groups:
            if group.size == 0:
                continue
            group_sorted = group[np.argsort(group[:, 1])]
            start = 0
            for idx in range(1, group_sorted.shape[0]):
                if group_sorted[idx, 1] - group_sorted[idx - 1, 1] > pitch_tol:
                    add_node(group_sorted[start:idx])
                    start = idx
            add_node(group_sorted[start:])

        if not positions:
            return PlaneNodes(
                pos=np.empty((0, 2), dtype=np.float32),
                features=np.empty((0, 5), dtype=np.float32),
                labels=np.empty((0,), dtype=np.int64),
                instances=np.empty((0,), dtype=np.int64),
                to_sp=np.empty((0, 2), dtype=np.int64),
                edges=np.empty((2, 0), dtype=np.int64),
            )

        pos_array = np.asarray(positions, dtype=np.float32)
        feat_array = np.asarray(features, dtype=np.float32)

        to_sp: List[Tuple[int, int]] = []
        node_semantics: List[List[int]] = [[] for _ in range(len(positions))]

        node_x = pos_array[:, 0]
        for blob_id, corner_set in enumerate(corners):
            projected = project_corners(corner_set, spec.angle_rad)
            if not projected.size:
                continue

            candidate_mask = np.abs(node_x - centroid[blob_id, 0]) <= x_tol
            candidate_indices = np.nonzero(candidate_mask)[0]
            if not candidate_indices.size:
                continue

            for node_idx in candidate_indices:
                pitch_min, pitch_max = pitch_ranges[node_idx]
                lower_expand = pitch_min - self.config.projection_tolerance
                upper_expand = pitch_max + self.config.projection_tolerance
                if np.any((projected >= lower_expand) & (projected <= upper_expand)):
                    to_sp.append((node_idx, blob_id))
                    node_semantics[node_idx].append(int(semantic[blob_id]))

        labels = np.full(len(positions), self.config.semantic_negative, dtype=np.int64)
        for node_idx, linked_labels in enumerate(node_semantics):
            if not linked_labels:
                continue
            if any(lbl == self.config.semantic_positive for lbl in linked_labels):
                labels[node_idx] = self.config.semantic_positive
            else:
                labels[node_idx] = linked_labels[0]

        instances = np.zeros(len(positions), dtype=np.int64)
        edges = triangulation_edges(pos_array)
        to_sp_arr = np.asarray(to_sp, dtype=np.int64) if to_sp else np.empty((0, 2), dtype=np.int64)

        encoded_labels = self._encode_semantic_labels(labels)

        return PlaneNodes(
            pos=pos_array,
            features=feat_array,
            labels=encoded_labels,
            instances=instances,
            to_sp=to_sp_arr,
            edges=edges,
        )

    def _infer_event_ids(self, sample_name: str, source_path: Path | None = None) -> tuple[int, int, int]:
        run = 0
        subrun = 0
        event = 0

        if source_path is not None:
            run_subrun = self._extract_run_subrun(source_path)
            if run_subrun is not None:
                run, subrun = run_subrun
            event_id = self._extract_event_id(source_path)
            if event_id is not None:
                event = event_id

        if run == 0 or subrun == 0:
            match = RUN_SUBRUN_PATTERN.search(sample_name)
            if match:
                if run == 0:
                    run = int(match.group("run"))
                if subrun == 0:
                    subrun = int(match.group("subrun"))

        parts = sample_name.split('-')
        for part in parts:
            if subrun == 0 and part.startswith("apa"):
                try:
                    subrun = int(part.replace("apa", ""))
                except ValueError:
                    pass
            elif event == 0 and part.isdigit():
                event = int(part)

        return run, subrun, event

    def _default_sample_name(self, path: Path) -> str:
        run_subrun = self._extract_run_subrun(path)
        base = path.stem
        if run_subrun is not None:
            run, subrun = run_subrun
            return f"{run}_{subrun}_{base}"
        return base

    def _extract_run_subrun(self, path: Path) -> tuple[int, int] | None:
        for parent in path.parents:
            match = RUN_SUBRUN_PATTERN.fullmatch(parent.name)
            if match:
                return int(match.group("run")), int(match.group("subrun"))
        return None

    def _extract_event_id(self, path: Path) -> int | None:
        stem_parts = path.stem.split('-')
        for part in reversed(stem_parts):
            if part.isdigit():
                return int(part)
        return None

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
    paths = sorted(p for p in directory.rglob("*.npz") if p.is_file())
    graphs = converter.convert_many(paths)
    converter.write_hdf5(graphs, output)
    return Path(output)


__all__ = [
    "WCMLConverter",
    "convert_npz_file",
    "convert_npz_directory",
]
