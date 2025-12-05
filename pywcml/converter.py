# filename: pywcml/converter.py
"""Convert WCML NPZ files into NuGraph-compatible HDF5 datasets."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import h5py

from pynuml.data import NuGraphData

from .config import ConversionConfig, PlaneSpec
from .geometry import estimate_unit_scale, project_corners, triangulation_edges
from .io import WCMLArrays, load_npz

# Optional sklearn import for local PCA; if unavailable, we fall back to zeros.
try:
    from sklearn.neighbors import NearestNeighbors
    _HAS_SKLEARN = True
except Exception:
    NearestNeighbors = None  # type: ignore
    _HAS_SKLEARN = False


@dataclass
class PlaneNodes:
    pos: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    instances: np.ndarray
    to_sp: np.ndarray
    edges: np.ndarray


RUN_SUBRUN_PATTERN = re.compile(r"(?P<run>\d+)_(?P<subrun>\d+)")
_WORKER_CONVERTER = None

# Try to avoid shared-memory issues in multiprocessing
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except (AttributeError, RuntimeError):
    pass


def _materialize_graph(graph: NuGraphData) -> NuGraphData:
    """Clone tensors so they no longer reference shared-memory storages."""
    for store in graph.stores:  # type: ignore[attr-defined]
        for key, value in list(store.items()):
            if isinstance(value, torch.Tensor):
                store[key] = value.detach().clone()
    return graph


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

    def convert_many(self, paths: Sequence[Path | str], workers: int | None = None) -> Dict[str, NuGraphData]:
        graphs: Dict[str, NuGraphData] = {}
        path_list = [Path(p) for p in paths]
        if not path_list:
            return graphs

        worker_count = max(1, int(workers or 1))

        if worker_count == 1:
            iterator = self._progress(path_list, total=len(path_list))
            for path in iterator:
                name, data = self.convert(path)
                graphs[name] = data
            return graphs

        str_paths = [str(p) for p in path_list]
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(self.config,),
        ) as executor:
            futures = [executor.submit(_convert_worker, path) for path in str_paths]
            for future in self._progress(as_completed(futures), total=len(futures)):
                name, data = future.result()
                graphs[name] = _materialize_graph(data)

        return graphs

    def write_hdf5(
        self,
        graphs: Dict[str, NuGraphData],
        output: Path | str,
        splits: Dict[str, Sequence[str]] | None = None,
    ) -> None:
        """Persist converted graphs to an HDF5 file compatible with H5DataModule."""
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
        """
        Build one NuGraphData event.

        Key outputs relevant for NuGraph4:
          - graph["sp"].features: (N_blobs, 6) =
              [charge, cluster_id, vtx_dist, vtx_dx, vtx_dy, vtx_dz]
          - graph["sp"].y_semantic: nu/cosmic semantic labels
          - graph["sp"].y_instance: truth cluster IDs (instance labels)
          - per-plane node features:
              base (5) + vertex (4) + sidecar (6) = 15 columns
        """
        charges, centroids, corners, cluster_by_blob = self._extract_blobs(arrays.blobs, arrays.points)
        semantic = self._label_blobs(arrays.points, arrays.is_nu, len(corners), self.config)
        encoded_semantic = self._encode_semantic_labels(semantic)

        # --- Aggregate vertex-distance info per blob (sp node) ---
        n_blobs = len(corners)
        blob_indices = arrays.points[:, 4].astype(np.int64) if arrays.points is not None else None

        def _agg_per_blob(per_point: np.ndarray | None, default_value: float = -1.0) -> np.ndarray:
            """Aggregate a per-point quantity to per-blob via average."""
            out = np.full(n_blobs, default_value, dtype=np.float32)
            if per_point is None or blob_indices is None:
                return out
            if not per_point.size or not arrays.points.size:
                return out
            valid = (blob_indices >= 0) & (blob_indices < n_blobs)
            if not np.any(valid):
                return out
            vals = per_point.astype(np.float32)[valid]
            idx = blob_indices[valid]
            sums = np.bincount(idx, weights=vals, minlength=n_blobs)
            counts = np.bincount(idx, minlength=n_blobs)
            mask = counts > 0
            out[mask] = sums[mask] / counts[mask]
            return out

        vtx_dist_by_blob = _agg_per_blob(getattr(arrays, "vtx_dist", None), default_value=-1.0)
        vtx_dx_by_blob   = _agg_per_blob(getattr(arrays, "vtx_dx",   None), default_value=0.0)
        vtx_dy_by_blob   = _agg_per_blob(getattr(arrays, "vtx_dy",   None), default_value=0.0)
        vtx_dz_by_blob   = _agg_per_blob(getattr(arrays, "vtx_dz",   None), default_value=0.0)

        # --- Sidecar-style features at blob level, from centroid positions ---
        d_wall_by_blob, d_top_by_blob = self._compute_distances_to_walls_from_centroids(centroids)
        (
            linearity_by_blob,
            sphericity_by_blob,
            ty_by_blob,
            tz_by_blob,
        ) = self._compute_local_pca_features_from_centroids(centroids, k=self.config.sidecar_k if hasattr(self.config, "sidecar_k") else 12)

        # --- Plane-level nodes (per-plane clusters) ---
        planes = self.config.planes_for_sample(sample_name)
        plane_nodes: Dict[str, PlaneNodes] = {}
        for key, spec in planes.items():
            ctpc = arrays.ctpc.get(key)
            if ctpc is None:
                plane_nodes[spec.name] = PlaneNodes(
                    pos=np.empty((0, 2), dtype=np.float32),
                    features=np.empty((0, 0), dtype=np.float32),
                    labels=np.empty((0,), dtype=np.int64),
                    instances=np.empty((0,), dtype=np.int64),
                    to_sp=np.empty((0, 2), dtype=np.int64),
                    edges=np.empty((2, 0), dtype=np.int64),
                )
                continue

            plane_nodes[spec.name] = self._build_plane(
                spec,
                ctpc,
                corners,
                centroids,
                semantic,
                cluster_by_blob,
                vtx_dist_by_blob,
                vtx_dx_by_blob,
                vtx_dy_by_blob,
                vtx_dz_by_blob,
                d_wall_by_blob,
                d_top_by_blob,
                linearity_by_blob,
                sphericity_by_blob,
                ty_by_blob,
                tz_by_blob,
            )

        # --- Assemble NuGraphData ---
        graph = NuGraphData()
        run, subrun, event = self._infer_event_ids(sample_name, arrays.path)
        graph["metadata"].run = run
        graph["metadata"].subrun = subrun
        graph["metadata"].event = event

        # --- 3D "sp" (blob) nodes ---
        graph["sp"].pos = torch.as_tensor(centroids, dtype=torch.float32)

        # sp.features: [charge, cluster_id, vtx_dist, vtx_dx, vtx_dy, vtx_dz]
        sp_feat = np.stack(
            [
                charges,            # total charge in blob
                cluster_by_blob,    # instance ID (cluster ID)
                vtx_dist_by_blob,   # mean distance to true ν vertex
                vtx_dx_by_blob,     # mean dx = x_hit - x_vtx
                vtx_dy_by_blob,     # mean dy = y_hit - y_vtx
                vtx_dz_by_blob,     # mean dz = z_hit - z_vtx
            ],
            axis=1,
        )
        graph["sp"].features = torch.as_tensor(sp_feat, dtype=torch.float32)

        graph["sp"].y_semantic = torch.as_tensor(encoded_semantic, dtype=torch.long)
        # Truth-level instance labels at blob (sp) level
        graph["sp"].y_instance = torch.as_tensor(cluster_by_blob.astype(np.int64), dtype=torch.long)

        # --- blob–blob (ppedges) edges ---
        _, blob_edges = self._ppedges_to_blobedges(arrays.ppedges, arrays.points)
        if blob_edges.size:
            graph["sp", "nexus", "sp"].edge_index = torch.as_tensor(blob_edges, dtype=torch.long)
        else:
            graph["sp", "nexus", "sp"].edge_index = torch.empty((2, 0), dtype=torch.long)

        # --- per-plane stores ---
        for plane_name in self.config.plane_names():
            nodes = plane_nodes.get(plane_name)
            if nodes is None:
                nodes = PlaneNodes(
                    pos=np.empty((0, 2), dtype=np.float32),
                    features=np.empty((0, 0), dtype=np.float32),
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
            # Instance labels at plane-node level
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

    # ------------------------------------------------------------------
    # Helper: blob/centroid-level sidecar features
    # ------------------------------------------------------------------
    def _compute_distances_to_walls_from_centroids(
        self,
        centroids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute d_wall, d_top from centroid positions for this event.

        We approximate detector bounds from centroids in this event:
          x_min, x_max = min/max centroid x
          y_min, y_max = min/max centroid y
          z_min, z_max = min/max centroid z

        Returns arrays of shape (N_blobs,) each.
        """
        if centroids.size == 0:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        x = centroids[:, 0]
        y = centroids[:, 1]
        z = centroids[:, 2]

        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        z_min, z_max = float(z.min()), float(z.max())

        dx = np.minimum(x - x_min, x_max - x)
        dy = np.minimum(y - y_min, y_max - y)
        dz = np.minimum(z - z_min, z_max - z)

        d_wall = np.minimum(np.minimum(dx, dy), dz)
        d_top = y_max - y

        return d_wall.astype(np.float32), d_top.astype(np.float32)

    def _compute_local_pca_features_from_centroids(
        self,
        centroids: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Local PCA sidecar features at blob level, using blob centroids as points.

        Returns:
          linearity  : (N,)  (λ1 - λ2) / λ1
          sphericity : (N,)  λ3 / λ1
          ty         : (N,)  y-component of principal direction
          tz         : (N,)  z-component of principal direction
        """
        N = centroids.shape[0]
        if N == 0:
            return (
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        if not _HAS_SKLEARN:
            # Fallback: no sklearn, provide zeros so pipeline still works.
            return (
                np.zeros((N,), dtype=np.float32),
                np.zeros((N,), dtype=np.float32),
                np.zeros((N,), dtype=np.float32),
                np.zeros((N,), dtype=np.float32),
            )

        k_eff = min(k, N)
        nbrs = NearestNeighbors(
            n_neighbors=k_eff,
            algorithm="kd_tree",
        )
        nbrs.fit(centroids)
        _, indices = nbrs.kneighbors(centroids, return_distance=True)  # (N, k_eff)

        linearity = np.zeros(N, dtype=np.float32)
        sphericity = np.zeros(N, dtype=np.float32)
        ty = np.zeros(N, dtype=np.float32)
        tz = np.zeros(N, dtype=np.float32)

        for i in range(N):
            neigh_idx = indices[i]
            pts = centroids[neigh_idx]  # (k_eff, 3)
            if pts.shape[0] < 3:
                continue

            c = pts.mean(axis=0, keepdims=True)
            X = pts - c

            C = X.T @ X / float(X.shape[0])

            vals, vecs = np.linalg.eigh(C)  # vals ascending
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            vecs = vecs[:, order]

            lam1, lam2, lam3 = vals
            denom = lam1 if lam1 > 1e-9 else 1e-9

            linearity[i] = (lam1 - lam2) / denom
            sphericity[i] = lam3 / denom

            v1 = vecs[:, 0]
            ty[i] = float(v1[1])
            tz[i] = float(v1[2])

        return linearity, sphericity, ty, tz

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _extract_blobs(
        self, raw_blobs: np.ndarray, points: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
        """Extract blob charges, centroids, corners and aggregate cluster indices.

        Returns:
            charges: (N,) float32
            centroids: (N,3) float32
            corners: list of (M_i,3) arrays
            cluster_by_blob: (N,) float32 array with cluster_idx or -1
        """
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

        n_blobs = len(centroids)

        # cluster_id is stored on the points as the last column
        cluster_by_blob = np.full(n_blobs, -1.0, dtype=np.float32)
        if points is not None and points.size:
            pairs = np.unique(points[:, -2:], axis=0)
            if len(pairs) != n_blobs:
                print(
                    f"n_blobs is {n_blobs} but unique pairs between blob/cluster_idx is "
                    f"{len(pairs)}. Returning -1 for all blobs"
                )
            else:
                cluster_by_blob = pairs[:, -1].astype(np.float32)

        return (
            charges.astype(np.float32),
            np.asarray(centroids, dtype=np.float32),
            corners,
            cluster_by_blob,
        )

    def _label_blobs(
        self,
        points: np.ndarray,
        is_nu: np.ndarray | None,
        n_expected: int,
        config: ConversionConfig,
    ) -> np.ndarray:
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
            labels[blob_id] = (
                config.semantic_positive
                if (blob_labels == config.semantic_positive).any()
                else config.semantic_negative
            )
        return labels

    def _ppedges_to_blobedges(self, ppedges: np.ndarray, points: np.ndarray):
        """
        Convert point-level edges -> unique blob-level edges.
        Assumes that the input/output edges are undirected.

        Args:
            ppedges: (M, >=2) array of [head_point_idx, tail_point_idx, ...]
            points: (P, >=5) array where points[:,4] is int blob index (>=0) or -1 for none
        """
        heads = ppedges[:, 0].astype(np.int64)
        tails = ppedges[:, 1].astype(np.int64)
        blob_idx = points[:, 4].astype(np.int64)

        bh = blob_idx[heads]
        bt = blob_idx[tails]

        # keep only edges where both endpoints have a blob
        mask = (bh >= 0) & (bt >= 0)
        if not np.any(mask):
            return np.empty((0, 2), dtype=np.int64), np.empty((2, 0), dtype=np.int64)
        bh = bh[mask]
        bt = bt[mask]

        # drop any where the head/tail is the same blob
        keep = bh != bt
        if not np.any(keep):
            return np.empty((0, 2), dtype=np.int64), np.empty((2, 0), dtype=np.int64)
        bh = bh[keep]
        bt = bt[keep]

        a = np.minimum(bh, bt)
        b = np.maximum(bh, bt)
        pairs = np.stack([a, b], axis=1)

        # deduplicate rows
        pairs_unique = np.unique(pairs, axis=0).astype(np.int64)

        # build PyG-style edge_index
        if pairs_unique.size == 0:
            return pairs_unique, np.empty((2, 0), dtype=np.int64)

        src = pairs_unique[:, 0]
        dst = pairs_unique[:, 1]
        edge_index = np.stack([src, dst], axis=0)
        return pairs_unique, edge_index

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

    def _build_plane(
        self,
        spec: PlaneSpec,
        ctpc: np.ndarray,
        corners: Sequence[np.ndarray],
        centroid: np.ndarray,
        semantic: np.ndarray,
        cluster_by_blob: np.ndarray,
        vtx_dist_by_blob: np.ndarray,
        vtx_dx_by_blob: np.ndarray,
        vtx_dy_by_blob: np.ndarray,
        vtx_dz_by_blob: np.ndarray,
        d_wall_by_blob: np.ndarray,
        d_top_by_blob: np.ndarray,
        linearity_by_blob: np.ndarray,
        sphericity_by_blob: np.ndarray,
        ty_by_blob: np.ndarray,
        tz_by_blob: np.ndarray,
    ) -> PlaneNodes:
        """Build 2D plane nodes (per-plane clusters) and their features."""
        if not ctpc.size:
            return PlaneNodes(
                pos=np.empty((0, 2), dtype=np.float32),
                features=np.empty((0, 0), dtype=np.float32),
                labels=np.empty((0,), dtype=np.int64),
                instances=np.empty((0,), dtype=np.int64),
                to_sp=np.empty((0, 2), dtype=np.int64),
                edges=np.empty((2, 0), dtype=np.int64),
            )

        x_tol = self.config.x_tolerance
        pitch_tol = self.config.pitch_gap_tolerance

        # Group by x (wire/time) within tolerance
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
        base_features: List[List[float]] = []
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
            base_features.append([total_charge, mean_charge_err, nhits, pitch_min, pitch_max])
            pitch_ranges.append((pitch_min, pitch_max))

        # Split into contiguous pitch segments within each x-group
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
                features=np.empty((0, 0), dtype=np.float32),
                labels=np.empty((0,), dtype=np.int64),
                instances=np.empty((0,), dtype=np.int64),
                to_sp=np.empty((0, 2), dtype=np.int64),
                edges=np.empty((2, 0), dtype=np.int64),
            )

        pos_array = np.asarray(positions, dtype=np.float32)
        base_feat_array = np.asarray(base_features, dtype=np.float32)

        to_sp: List[Tuple[int, int]] = []
        node_semantics: List[List[int]] = [[] for _ in range(len(positions))]
        node_clusters: List[List[int]] = [[] for _ in range(len(positions))]
        node_blobs: List[List[int]] = [[] for _ in range(len(positions))]

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
                    if 0 <= blob_id < cluster_by_blob.shape[0]:
                        node_clusters[node_idx].append(int(cluster_by_blob[blob_id]))
                    node_blobs[node_idx].append(blob_id)

        labels = np.full(len(positions), self.config.semantic_negative, dtype=np.int64)
        for node_idx, linked_labels in enumerate(node_semantics):
            if not linked_labels:
                continue
            if any(lbl == self.config.semantic_positive for lbl in linked_labels):
                labels[node_idx] = self.config.semantic_positive
            else:
                labels[node_idx] = linked_labels[0]

        # Derive instance labels per node from node_clusters
        instances = np.full(len(positions), -1, dtype=np.int64)
        for node_idx, clusters in enumerate(node_clusters):
            if not clusters:
                continue
            valid = np.asarray([c for c in clusters if c >= 0], dtype=np.int64)
            if valid.size == 0:
                continue
            values, counts = np.unique(valid, return_counts=True)
            instances[node_idx] = int(values[np.argmax(counts)])

        # Aggregate vertex + sidecar features per node from node_blobs
        num_nodes = len(positions)
        vertex_feats = np.zeros((num_nodes, 4), dtype=np.float32)
        sidecar_feats = np.zeros((num_nodes, 6), dtype=np.float32)

        for node_idx, blobs in enumerate(node_blobs):
            if not blobs:
                continue
            b = np.asarray(blobs, dtype=np.int64)
            vertex_feats[node_idx, 0] = float(vtx_dist_by_blob[b].mean())
            vertex_feats[node_idx, 1] = float(vtx_dx_by_blob[b].mean())
            vertex_feats[node_idx, 2] = float(vtx_dy_by_blob[b].mean())
            vertex_feats[node_idx, 3] = float(vtx_dz_by_blob[b].mean())

            sidecar_feats[node_idx, 0] = float(d_wall_by_blob[b].mean())
            sidecar_feats[node_idx, 1] = float(d_top_by_blob[b].mean())
            sidecar_feats[node_idx, 2] = float(linearity_by_blob[b].mean())
            sidecar_feats[node_idx, 3] = float(sphericity_by_blob[b].mean())
            sidecar_feats[node_idx, 4] = float(ty_by_blob[b].mean())
            sidecar_feats[node_idx, 5] = float(tz_by_blob[b].mean())

        # Final per-plane feature matrix: base(5) + vertex(4) + sidecar(6) = 15
        feat_array = np.concatenate([base_feat_array, vertex_feats, sidecar_feats], axis=1)

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

    def _infer_event_ids(
        self, sample_name: str, source_path: Path | None = None
    ) -> tuple[int, int, int]:
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

        parts = sample_name.split("-")
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
        stem_parts = path.stem.split("-")
        for part in reversed(stem_parts):
            if part.isdigit():
                return int(part)
        return None

    def _progress(self, iterable: Iterable, total: int | None = None) -> Iterable:
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:
            return iterable
        return tqdm(iterable, total=total, desc="Converting", unit="file")

    def _graph_sizes(
        self, graphs: Dict[str, NuGraphData], names: Sequence[str]
    ) -> np.ndarray:
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


def _init_worker(config: ConversionConfig) -> None:
    global _WORKER_CONVERTER
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except (AttributeError, RuntimeError):
        pass
    _WORKER_CONVERTER = WCMLConverter(config)


def _convert_worker(path: str) -> tuple[str, NuGraphData]:
    if _WORKER_CONVERTER is None:
        raise RuntimeError("Worker converter not initialized")
    return _WORKER_CONVERTER.convert(Path(path))


def convert_npz_file(
    npz_path: Path | str,
    output: Path | str,
    config: ConversionConfig | None = None,
) -> Path:
    converter = WCMLConverter(config)
    name, graph = converter.convert(npz_path)
    converter.write_hdf5({name: graph}, output)
    return Path(output)


def convert_npz_directory(
    directory: Path | str,
    output: Path | str,
    config: ConversionConfig | None = None,
    workers: int | None = None,
) -> Path:
    converter = WCMLConverter(config)
    directory = Path(directory)
    paths = sorted(p for p in directory.rglob("*.npz") if p.is_file())
    graphs = converter.convert_many(paths, workers=workers)
    converter.write_hdf5(graphs, output)
    return Path(output)


__all__ = [
    "WCMLConverter",
    "convert_npz_file",
    "convert_npz_directory",
]
