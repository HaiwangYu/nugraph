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
from .geometry import project_corners, triangulation_edges
from .identity import EventIdentity
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
    """Convert WCML arrays into NuGraph graphs and packaged HDF5 files."""

    def __init__(self, config: ConversionConfig | None = None):
        self.config = config or ConversionConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def convert(self, npz_path: Path | str, sample_name: str | None = None) -> tuple[str, NuGraphData]:
        """Compatibility loader for the historical NPZ-driven route."""

        path = Path(npz_path)
        arrays = load_npz(path)
        graph_name = sample_name or self._default_sample_name(path)
        graph = self._build_graph(graph_name, arrays)
        return graph_name, graph

    def convert_arrays(
        self,
        arrays: WCMLArrays,
        identity: EventIdentity,
        apa: int,
        sample_name: str | None = None,
    ) -> NuGraphData:
        """Convert an in-memory APA payload using explicit physical identity.

        Unlike :meth:`convert`, this supported streaming API never consults a
        filename or ``WCMLArrays.path`` for provenance. The APA selects the
        correct SBND plane family while run/subrun/event are copied directly
        from ``identity``.
        """

        if isinstance(apa, bool) or int(apa) not in (0, 1):
            raise ValueError(f"APA must be 0 or 1, got {apa!r}")
        apa = int(apa)
        graph_name = sample_name or identity.sample_name(apa)
        return self._build_graph(
            graph_name,
            arrays,
            identity=identity,
            apa=apa,
        )

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
    def _build_graph(
        self,
        sample_name: str,
        arrays: WCMLArrays,
        *,
        identity: EventIdentity | None = None,
        apa: int | None = None,
    ) -> NuGraphData:
        """
        Build one NuGraphData event.

        sp.features: (N_blobs, 6) =
            [charge, reco_cluster_id, vtx_dist, vtx_dx, vtx_dy, vtx_dz]

        sp.y_instance: truth instance id per blob, using truth_blob_tid when available.
        """
        charges, centroids, corners = self._extract_blobs(arrays.blobs)

        # semantic labels per blob from point-level is_nu
        semantic, frac_nu_hits = self._label_blobs(arrays.points, arrays.is_nu, len(corners), self.config)
        semantic_truth_values = (
            np.asarray(arrays.is_nu).reshape(-1)
            if arrays.is_nu is not None
            else np.empty((0,), dtype=np.int64)
        )
        has_semantic_truth = (
            arrays.points is not None
            and semantic_truth_values.shape[0] == arrays.points.shape[0]
            and np.any(semantic_truth_values != -2)
        )

        # reco cluster id per blob (mode of points[:,5] within each blob)
        reco_cluster_by_blob = self._reco_cluster_by_blob(arrays.points, n_blobs=centroids.shape[0])

        # truth instance id per blob; missing truth remains unlabelled
        truth_instance_by_blob = self._truth_instance_by_blob(arrays, n_blobs=centroids.shape[0])

        # --- Vertex-distance info per blob: DO NOT recompute in converter ---
        # Labeling already computed per-HIT vtx_dist/dx/dy/dz in *cm*.
        # Here we only aggregate HIT->BLOB and convert cm->mm (x10) to match reco mm convention in H5.
        n_blobs = int(centroids.shape[0])

        vtx_dx_by_blob = np.zeros(n_blobs, dtype=np.float32)
        vtx_dy_by_blob = np.zeros(n_blobs, dtype=np.float32)
        vtx_dz_by_blob = np.zeros(n_blobs, dtype=np.float32)
        vtx_dist_by_blob = np.full(n_blobs, -1.0, dtype=np.float32)

        
        nu_found = (getattr(arrays, "nu_vtx_found", None) is not None) and (int(np.atleast_1d(arrays.nu_vtx_found)[0]) == 1)

        # We only trust labeling-derived vtx_* if nu vertex was found and the arrays exist
        hit_vdist = getattr(arrays, "vtx_dist", None)
        hit_vdx   = getattr(arrays, "vtx_dx", None)
        hit_vdy   = getattr(arrays, "vtx_dy", None)
        hit_vdz   = getattr(arrays, "vtx_dz", None)
        
        if nu_found and hit_vdist is not None and hit_vdx is not None and hit_vdy is not None and hit_vdz is not None:
            points = getattr(arrays, "points", None)
            if points is not None and points.size:
                hit_blob = points[:, 4].astype(np.int64, copy=False)

                hit_vdist_cm = np.asarray(arrays.vtx_dist, dtype=np.float32).reshape(-1)
                hit_vdx_cm   = np.asarray(arrays.vtx_dx,   dtype=np.float32).reshape(-1)
                hit_vdy_cm   = np.asarray(arrays.vtx_dy,   dtype=np.float32).reshape(-1)
                hit_vdz_cm   = np.asarray(arrays.vtx_dz,   dtype=np.float32).reshape(-1)

                # Guard: lengths must match number of hits/points
                n_points = int(points.shape[0])
                if hit_vdist_cm.shape[0] == n_points and hit_vdx_cm.shape[0] == n_points and hit_vdy_cm.shape[0] == n_points and hit_vdz_cm.shape[0] == n_points:
                    for b in range(n_blobs):
                        m = (hit_blob == b)
                        if not np.any(m):
                            continue
                        # mean over hits in blob (labeling-defined quantities)
                        vtx_dist_by_blob[b] = float(np.mean(hit_vdist_cm[m]))
                        vtx_dx_by_blob[b]   = float(np.mean(hit_vdx_cm[m]))
                        vtx_dy_by_blob[b]   = float(np.mean(hit_vdy_cm[m]))
                        vtx_dz_by_blob[b]   = float(np.mean(hit_vdz_cm[m]))

                    # cm -> mm (match reco centroids in mm and what your H5 sanity check expects)
                    vtx_dist_by_blob *= 10.0
                    vtx_dx_by_blob   *= 10.0
                    vtx_dy_by_blob   *= 10.0
                    vtx_dz_by_blob   *= 10.0

        raw_vtx_dist_by_blob = vtx_dist_by_blob.copy()
        
        # Optional vertex-based cleaning (OFF by default)
        VERTEX_LABEL_RADIUS = 50.0  # mm (5 cm)
        
        if self.config.enable_vertex_semantic_fix and nu_found and vtx_dist_by_blob.size > 0:
            close = (vtx_dist_by_blob >= 0.0) & (vtx_dist_by_blob < VERTEX_LABEL_RADIUS)
            has_nu_hits = frac_nu_hits > 0.0
            false_negative = (semantic == self.config.semantic_negative) & close & has_nu_hits
            if false_negative.any():
                semantic[false_negative] = self.config.semantic_positive


        # --- Optional sidecar-style features at blob level (OFF by default) ---
        n_blobs = int(centroids.shape[0])
        
        if self.config.enable_sidecar_features:
            # "wall/top" distances: only if explicitly enabled (note: bbox-based unless you later replace with detector geometry)
            if self.config.enable_event_bbox_wall_dists:
                d_wall_by_blob, d_top_by_blob = self._compute_distances_to_walls_from_centroids(centroids)
            else:
                d_wall_by_blob = np.zeros((n_blobs,), dtype=np.float32)
                d_top_by_blob  = np.zeros((n_blobs,), dtype=np.float32)
        
            # local PCA: only if explicitly enabled (and sidecar is enabled)
            if self.config.enable_local_pca:
                linearity_by_blob, sphericity_by_blob, ty_by_blob, tz_by_blob = self._compute_local_pca_features_from_centroids(
                    centroids,
                    k=getattr(self.config, "sidecar_k", 12),
                )
            else:
                linearity_by_blob  = np.zeros((n_blobs,), dtype=np.float32)
                sphericity_by_blob = np.zeros((n_blobs,), dtype=np.float32)
                ty_by_blob         = np.zeros((n_blobs,), dtype=np.float32)
                tz_by_blob         = np.zeros((n_blobs,), dtype=np.float32)
        else:
            # sidecar completely disabled
            d_wall_by_blob     = np.zeros((n_blobs,), dtype=np.float32)
            d_top_by_blob      = np.zeros((n_blobs,), dtype=np.float32)
            linearity_by_blob  = np.zeros((n_blobs,), dtype=np.float32)
            sphericity_by_blob = np.zeros((n_blobs,), dtype=np.float32)
            ty_by_blob         = np.zeros((n_blobs,), dtype=np.float32)
            tz_by_blob         = np.zeros((n_blobs,), dtype=np.float32)
        

        # --- Plane-level nodes (per-plane clusters) ---
        plane_selector = sample_name if apa is None else f"apa{apa}"
        planes = self.config.planes_for_sample(plane_selector)
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
                semantic,                # cleaned semantic per blob
                reco_cluster_by_blob,    # reco cluster per blob (used only for sp.features col1)
                truth_instance_by_blob,  # truth instance per blob
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
        if identity is None:
            run, subrun, event = self._infer_event_ids(sample_name, arrays.path)
        else:
            run, subrun, event = identity.run, identity.subrun, identity.event
        graph["metadata"].run = run
        graph["metadata"].subrun = subrun
        graph["metadata"].event = event

        # --- 3D "sp" nodes (BLOB LEVEL) ---
        graph["sp"].pos = torch.as_tensor(centroids, dtype=torch.float32)

        # sp.features: [charge, reco_cluster_id, vtx_dist, vtx_dx, vtx_dy, vtx_dz]
        encoded_semantic = self._encode_semantic_labels(semantic)
        
        # ---------------- FIX #1: Ghost blobs must NOT carry vertex features ----------------
        ghost = (encoded_semantic == -1)
        if np.any(ghost):
            vtx_dist_by_blob[ghost] = -1.0
            vtx_dx_by_blob[ghost]   = 0.0
            vtx_dy_by_blob[ghost]   = 0.0
            vtx_dz_by_blob[ghost]   = 0.0
        # -------------------------------------------------------------------------------
        
        # sp.features: [charge, reco_cluster_id, vtx_dist, vtx_dx, vtx_dy, vtx_dz]
        sp_feat = np.stack(
            [
                charges,
                reco_cluster_by_blob.astype(np.float32),
                vtx_dist_by_blob,
                vtx_dx_by_blob,
                vtx_dy_by_blob,
                vtx_dz_by_blob,
            ],
            axis=1,
        )
        graph["sp"].features = torch.as_tensor(sp_feat, dtype=torch.float32)

        # -------------------------------------------------------------------------------
        
        # (assignment deferred until after optional MP propagation)

        
        # Truth-level instance labels at blob (sp) level
        graph["sp"].y_instance = torch.as_tensor(truth_instance_by_blob.astype(np.int64), dtype=torch.long)
        
        # ---------------- FIX #2: Ghost blobs must NOT have instance labels ----------------
        # Ghost SPs (y_semantic == -1) should always have y_instance == -1
        ghost_mask_tensor = torch.from_numpy(ghost)  # reuse ghost mask from above
        if ghost_mask_tensor.any():
            y_inst_fixed = graph["sp"].y_instance.clone()
            y_inst_fixed[ghost_mask_tensor] = -1
            graph["sp"].y_instance = y_inst_fixed
        # -----------------------------------------------------------------------------------
        
        # Keep raw vtx dist for debug
        graph["sp"].raw_vtx_dist = torch.as_tensor(raw_vtx_dist_by_blob, dtype=torch.float32)
        

        # ==================== INSERTION START ====================
        # --- Debug / Diagnostics: Purity, Support, and Mode (OFF by default) ---
        if self.config.write_diagnostics:
            # 1. Blob Purity & Support (Directly from NPZ)
            pur = self._maybe_get_1d(arrays, "truth_blob_purity", n=n_blobs, dtype=np.float32)
            sup = self._maybe_get_1d(arrays, "truth_blob_support", n=n_blobs, dtype=np.int64)
        
            if pur is not None:
                graph["sp"].truth_blob_purity = torch.as_tensor(pur, dtype=torch.float32)
            if sup is not None:
                graph["sp"].truth_blob_support = torch.as_tensor(sup, dtype=torch.long)
        
            # 2. Truth TID Mode (Aggregated from points -> blobs)
            points = getattr(arrays, "points", None)
            if points is not None and points.size > 0:
                n_points = int(points.shape[0])
        
                tid_pts = self._maybe_get_1d(arrays, "truth_tid_points_direct", n=n_points, dtype=np.int64)
                if tid_pts is None:
                    tid_pts = self._maybe_get_1d(arrays, "truth_tid_points", n=n_points, dtype=np.int64)
        
                if tid_pts is not None:
                    hit_blob = points[:, 4].astype(np.int64, copy=False)
        
                    tid_mode = np.full(n_blobs, -1, dtype=np.int64)
                    tid_mode_frac = np.zeros(n_blobs, dtype=np.float32)
        
                    for b in range(n_blobs):
                        m = (hit_blob == b)
                        if not np.any(m):
                            continue
        
                        vals = tid_pts[m]
                        vals = vals[vals >= 0]
                        if vals.size == 0:
                            continue
        
                        u, c = np.unique(vals, return_counts=True)
                        j = int(np.argmax(c))
                        tid_mode[b] = int(u[j])
                        tid_mode_frac[b] = float(c[j]) / float(vals.size)
        
                    graph["sp"].truth_tid_mode = torch.as_tensor(tid_mode, dtype=torch.long)
                    graph["sp"].truth_tid_mode_frac = torch.as_tensor(tid_mode_frac, dtype=torch.float32)
        # ==================== INSERTION END ====================

        # ------------------------------------------------------
        # Blob-level edges:
        #   - message passing: derived from reconstruction ppedges
        #   - supervision candidates: reconstruction-only union of mapped
        #     ppedges, geometry kNN/radius, and reco-cluster neighbors
        #   - supervision targets: assigned from blob-level truth afterward
        # ------------------------------------------------------
        
        mp_edge_index = self._get_blob_mp_edges(arrays=arrays, n_blobs=int(centroids.shape[0]))
        sup_edge_index, edge_y, edge_labelable = self._get_blob_sup_edges_and_labels(
            arrays=arrays,
            n_blobs=int(centroids.shape[0]),
            truth_instance_by_blob=truth_instance_by_blob,
            semantic=semantic,
            vtx_dist_by_blob=vtx_dist_by_blob,
            centroids=centroids,
            reco_cluster_by_blob=reco_cluster_by_blob,
        )



        # ==================== PRUNING FIX START ====================
        # PRUNE MP EDGES TOUCHING GHOSTS
        mp_edge_index = np.asarray(mp_edge_index, dtype=np.int64)
        if mp_edge_index.ndim != 2 or mp_edge_index.shape[0] != 2:
            mp_edge_index = np.empty((2, 0), dtype=np.int64)
        elif has_semantic_truth and mp_edge_index.shape[1] > 0:
            ghost_mask = (encoded_semantic == -1)
            Nsp = int(ghost_mask.shape[0])
        
            src = mp_edge_index[0]
            dst = mp_edge_index[1]
        
            in_range = (src >= 0) & (dst >= 0) & (src < Nsp) & (dst < Nsp)
            keep = in_range & (~ghost_mask[src]) & (~ghost_mask[dst])
        
            mp_edge_index = mp_edge_index[:, keep]
        # ==================== PRUNING FIX END ====================
                
                        
        if self.config.merge_sup_edges_into_mp:
            if sup_edge_index is not None and np.asarray(sup_edge_index).size:
                mp_edge_index = np.concatenate([mp_edge_index, np.asarray(sup_edge_index, dtype=np.int64)], axis=1)
                mp_edge_index = self._unique_undirected_edges(mp_edge_index, int(centroids.shape[0]))
                if mp_edge_index.ndim != 2 or mp_edge_index.shape[0] != 2:
                    mp_edge_index = np.empty((2, 0), dtype=np.int64)
        
        if self.config.enable_semantic_mp_propagation_fix:
            encoded_semantic = self._semantic_propagate_on_mp(
                encoded_semantic=encoded_semantic,
                raw_vtx_dist_mm=raw_vtx_dist_by_blob,
                mp_edge_index=mp_edge_index,
                sp_pos_mm=centroids,
        
                # seeds: must be ν AND close to vtx AND have enough ν-hit support
                frac_nu_hits=frac_nu_hits,
                seed_frac_nu_min=getattr(self.config, "mp_prop_seed_frac_nu_min", 0.60),
        
                seed_vtx_radius_mm=getattr(self.config, "mp_prop_seed_vtx_radius_mm", 100.0),
        
                # propagation constraints
                edge_len_max_mm=getattr(self.config, "mp_prop_edge_len_max_mm", 225.0),
                max_hops=getattr(self.config, "mp_prop_max_hops", 5),
                max_vtx_dist_mm=getattr(self.config, "mp_prop_max_vtx_dist_mm", 2000.0),
            )
        
                                
                        

        # Final semantic labels (after optional MP propagation)
        graph["sp"].y_semantic = torch.as_tensor(encoded_semantic, dtype=torch.long)
        
        # Message passing graph
        graph["sp", "nexus", "sp"].edge_index = torch.as_tensor(mp_edge_index, dtype=torch.long)

        
        # Supervision edges + labels (aligned)
        # IMPORTANT: NuGraphData.save collapses true-empty tensors to scalars in HDF5.
        # So we must NEVER write true-empty supervision tensors. If Esup==0, inject
        # a dummy self-edge (0->0) that is marked unlabelable.
        _sei = np.asarray(sup_edge_index) if sup_edge_index is not None else np.empty((2, 0), dtype=np.int64)
        if _sei.ndim != 2 or _sei.shape[0] != 2:
            _sei = np.empty((2, 0), dtype=np.int64)

        _E = int(_sei.shape[1])
        _Nsp = int(centroids.shape[0])

        if _E == 0:
            # If there are no blobs, do NOT create 0->0. In practice, your dataset
            # probably has Nsp>0 always, but guard it to be safe.
            if _Nsp == 0:
                graph["sp"].edge_label_index = torch.empty((2, 0), dtype=torch.long)
                graph["sp"].edge_y = torch.empty((0,), dtype=torch.long)
                graph["sp"].edge_labelable = torch.empty((0,), dtype=torch.long)
            else:
                graph["sp"].edge_label_index = torch.tensor([[0], [0]], dtype=torch.long)  # (2,1)
                graph["sp"].edge_y = torch.tensor([0], dtype=torch.long)                   # (1,)
                graph["sp"].edge_labelable = torch.tensor([0], dtype=torch.long)           # (1,) 0=unlabelable
        else:
            _ey = np.asarray(edge_y, dtype=np.int64).reshape(-1)
            _el = np.asarray(edge_labelable, dtype=np.int64).reshape(-1)
        
            assert _ey.shape[0] == _E, f"[{sample_name}] edge_y len {_ey.shape[0]} != E {_E}"
            assert _el.shape[0] == _E, f"[{sample_name}] edge_labelable len {_el.shape[0]} != E {_E}"
        
            graph["sp"].edge_label_index = torch.as_tensor(_sei.astype(np.int64, copy=False), dtype=torch.long)
            graph["sp"].edge_y = torch.as_tensor(_ey, dtype=torch.long)
            graph["sp"].edge_labelable = torch.as_tensor(_el, dtype=torch.long)
        


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
            store.y_instance = torch.as_tensor(nodes.instances, dtype=torch.long)

            plane_edge_index = (
                torch.as_tensor(nodes.edges, dtype=torch.long) if nodes.edges.size else torch.empty((2, 0), dtype=torch.long)
            )
            graph[plane_name, "plane", plane_name].edge_index = plane_edge_index

            nexus_edges = (
                torch.as_tensor(nodes.to_sp, dtype=torch.long).t() if nodes.to_sp.size else torch.empty((2, 0), dtype=torch.long)
            )
            graph[plane_name, "nexus", "sp"].edge_index = nexus_edges

        graph["evt"].num_nodes = 1

        if (encoded_semantic == 0).any():
            graph["evt"].y = torch.tensor([1], dtype=torch.long)
        elif (encoded_semantic >= 0).any():
            graph["evt"].y = torch.tensor([0], dtype=torch.long)
        else:
            graph["evt"].y = torch.tensor([-1], dtype=torch.long)


        return graph

    # ------------------------------------------------------------------
    # Truth / reco helpers
    # ------------------------------------------------------------------
    def _reco_cluster_by_blob(self, points: np.ndarray, n_blobs: int) -> np.ndarray:
        """
        points[:,4] = blob_id, points[:,5] = reco cluster id (0..K)
        Return per-blob mode of points[:,5]. If a blob has no points, set -1.
        """
        if points is None or points.size == 0:
            return np.full((n_blobs,), -1, dtype=np.int64)

        blob = points[:, 4].astype(np.int64)
        clus = points[:, 5].astype(np.int64)

        out = np.full((n_blobs,), -1, dtype=np.int64)
        for b in range(n_blobs):
            m = (blob == b)
            if not np.any(m):
                continue
            vals, cnt = np.unique(clus[m], return_counts=True)
            out[b] = int(vals[np.argmax(cnt)])
        return out

    def _truth_instance_by_blob(self, arrays: WCMLArrays, n_blobs: int) -> np.ndarray:
        """Return blob-level truth instance IDs, or -1 when truth is unavailable."""
        tid = getattr(arrays, "truth_blob_tid", None)
        if tid is not None:
            tid = np.asarray(tid).reshape(-1)
            if tid.shape[0] == n_blobs:
                return tid.astype(np.int64)
        return np.full((n_blobs,), -1, dtype=np.int64)


    def _get_blob_mp_edges(self, arrays: WCMLArrays, n_blobs: int) -> np.ndarray:
        """
        Message passing edges: always use ppedges -> blob edges if available.
        Falls back to empty if not present.
        """
        points = getattr(arrays, "points", None)
        ppedges = getattr(arrays, "ppedges", None)
        if points is None or points.size == 0 or ppedges is None or np.asarray(ppedges).size == 0:
            return np.empty((2, 0), dtype=np.int64)
    
        _, ei_blob = self._ppedges_to_blobedges(np.asarray(ppedges), points)
        # ensure in-range / unique undirected
        ei_blob = self._unique_undirected_edges(ei_blob, n_blobs)
        return ei_blob

    @staticmethod
    def _empty_edge_index() -> np.ndarray:
        return np.empty((2, 0), dtype=np.int64)

    def _normalize_edge_index(self, edge_index) -> np.ndarray:
        """Normalize an edge array to int64 shape ``(2, E)``."""
        if edge_index is None:
            return self._empty_edge_index()

        normalized = np.asarray(edge_index)
        if normalized.size == 0 or normalized.ndim != 2:
            return self._empty_edge_index()
        if normalized.shape[0] == 2:
            return normalized.astype(np.int64, copy=False)
        if normalized.shape[1] == 2:
            return normalized.T.astype(np.int64, copy=False)
        return self._empty_edge_index()

    def _point_edges_to_blob_edges(
        self,
        point_edge_index,
        point_to_blob: np.ndarray,
        n_blobs: int,
    ) -> np.ndarray:
        """Map reconstruction point edges to unique blob-level edges."""
        normalized = self._normalize_edge_index(point_edge_index)
        if normalized.shape[1] == 0:
            return self._empty_edge_index()

        point_to_blob = np.asarray(point_to_blob, dtype=np.int64).reshape(-1)
        n_points = int(point_to_blob.shape[0])
        source_points = normalized[0]
        target_points = normalized[1]
        valid_points = (
            (source_points >= 0)
            & (target_points >= 0)
            & (source_points < n_points)
            & (target_points < n_points)
        )
        if not np.any(valid_points):
            return self._empty_edge_index()

        source_blobs = point_to_blob[source_points[valid_points]]
        target_blobs = point_to_blob[target_points[valid_points]]
        valid_blobs = (
            (source_blobs >= 0)
            & (target_blobs >= 0)
            & (source_blobs < n_blobs)
            & (target_blobs < n_blobs)
            & (source_blobs != target_blobs)
        )
        if not np.any(valid_blobs):
            return self._empty_edge_index()

        blob_edges = np.stack([source_blobs[valid_blobs], target_blobs[valid_blobs]], axis=0)
        return self._unique_undirected_edges(blob_edges, n_blobs)

    def _blob_knn_radius_edges(
        self,
        centroids: np.ndarray | None,
        k: int = 8,
        radius_mm: float = 80.0,
    ) -> np.ndarray:
        """Build geometry-only kNN and radius candidates from blob centroids."""
        if centroids is None:
            return self._empty_edge_index()

        all_centroids = np.asarray(centroids, dtype=np.float32)
        if all_centroids.ndim != 2 or all_centroids.shape[0] <= 1:
            return self._empty_edge_index()

        n_blobs = int(all_centroids.shape[0])
        finite_mask = np.all(np.isfinite(all_centroids), axis=1)
        if np.count_nonzero(finite_mask) <= 1:
            return self._empty_edge_index()

        blob_indices = np.nonzero(finite_mask)[0].astype(np.int64)
        finite_centroids = all_centroids[finite_mask]
        n_finite = int(finite_centroids.shape[0])
        edges: list[tuple[int, int]] = []

        if _HAS_SKLEARN:
            n_neighbors = max(1, min(int(k) + 1, n_finite))
            nearest = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree")
            nearest.fit(finite_centroids)
            neighbor_indices = nearest.kneighbors(finite_centroids, return_distance=False)
            for local_source, local_targets in enumerate(neighbor_indices):
                source_blob = int(blob_indices[local_source])
                for local_target in np.atleast_1d(local_targets):
                    target_blob = int(blob_indices[int(local_target)])
                    if source_blob != target_blob:
                        edges.append((source_blob, target_blob))

            if radius_mm > 0.0:
                radius_neighbors = NearestNeighbors(radius=float(radius_mm), algorithm="kd_tree")
                radius_neighbors.fit(finite_centroids)
                neighbor_indices = radius_neighbors.radius_neighbors(
                    finite_centroids,
                    return_distance=False,
                )
                for local_source, local_targets in enumerate(neighbor_indices):
                    source_blob = int(blob_indices[local_source])
                    for local_target in np.atleast_1d(local_targets):
                        target_blob = int(blob_indices[int(local_target)])
                        if source_blob != target_blob:
                            edges.append((source_blob, target_blob))
        else:
            n_neighbors = max(1, min(int(k), n_finite - 1))
            radius_squared = float(radius_mm) ** 2 if radius_mm > 0.0 else -1.0
            for local_source in range(n_finite):
                displacement = finite_centroids - finite_centroids[local_source]
                distance_squared = np.einsum("ij,ij->i", displacement, displacement)
                distance_squared[local_source] = np.inf
                nearest_indices = np.argpartition(
                    distance_squared,
                    kth=n_neighbors - 1,
                )[:n_neighbors]
                source_blob = int(blob_indices[local_source])
                for local_target in nearest_indices:
                    edges.append((source_blob, int(blob_indices[int(local_target)])))
                if radius_squared > 0.0:
                    for local_target in np.nonzero(distance_squared <= radius_squared)[0]:
                        edges.append((source_blob, int(blob_indices[int(local_target)])))

        if not edges:
            return self._empty_edge_index()
        return self._unique_undirected_edges(np.asarray(edges, dtype=np.int64).T, n_blobs)

    def _cluster_neighbor_edges(
        self,
        reco_cluster_by_blob: np.ndarray | None,
        centroids: np.ndarray | None,
        k_per_cluster: int = 4,
    ) -> np.ndarray:
        """Connect nearby blobs within each reconstruction-defined cluster."""
        if reco_cluster_by_blob is None or centroids is None:
            return self._empty_edge_index()

        cluster_ids = np.asarray(reco_cluster_by_blob, dtype=np.int64).reshape(-1)
        all_centroids = np.asarray(centroids, dtype=np.float32)
        if all_centroids.ndim != 2 or all_centroids.shape[0] != cluster_ids.shape[0]:
            return self._empty_edge_index()

        n_blobs = int(cluster_ids.shape[0])
        edges: list[tuple[int, int]] = []
        for cluster_id in np.unique(cluster_ids):
            if cluster_id < 0:
                continue

            blob_indices = np.nonzero(cluster_ids == cluster_id)[0].astype(np.int64)
            finite_mask = np.all(np.isfinite(all_centroids[blob_indices]), axis=1)
            blob_indices = blob_indices[finite_mask]
            cluster_centroids = all_centroids[blob_indices]
            n_cluster = int(blob_indices.shape[0])
            if n_cluster <= 1:
                continue

            if _HAS_SKLEARN:
                n_neighbors = max(1, min(int(k_per_cluster) + 1, n_cluster))
                nearest = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree")
                nearest.fit(cluster_centroids)
                neighbor_indices = nearest.kneighbors(cluster_centroids, return_distance=False)
                for local_source, local_targets in enumerate(neighbor_indices):
                    source_blob = int(blob_indices[local_source])
                    for local_target in np.atleast_1d(local_targets):
                        target_blob = int(blob_indices[int(local_target)])
                        if source_blob != target_blob:
                            edges.append((source_blob, target_blob))
            else:
                n_neighbors = max(1, min(int(k_per_cluster), n_cluster - 1))
                for local_source in range(n_cluster):
                    displacement = cluster_centroids - cluster_centroids[local_source]
                    distance_squared = np.einsum("ij,ij->i", displacement, displacement)
                    distance_squared[local_source] = np.inf
                    nearest_indices = np.argpartition(
                        distance_squared,
                        kth=n_neighbors - 1,
                    )[:n_neighbors]
                    source_blob = int(blob_indices[local_source])
                    for local_target in nearest_indices:
                        edges.append((source_blob, int(blob_indices[int(local_target)])))

        if not edges:
            return self._empty_edge_index()
        return self._unique_undirected_edges(np.asarray(edges, dtype=np.int64).T, n_blobs)
    
    def _get_blob_sup_edges_and_labels(
            self,
            arrays: WCMLArrays,
            n_blobs: int,
            truth_instance_by_blob: np.ndarray,
            semantic: np.ndarray,
            vtx_dist_by_blob: np.ndarray,
            centroids: np.ndarray | None = None,
            reco_cluster_by_blob: np.ndarray | None = None,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Build blob candidates from reconstruction/geometry, then label with truth.

            Candidate topology is exactly the union of mapped ``ppedges``,
            geometry kNN/radius edges (k=8, radius=80 mm), and reconstruction
            cluster-neighbor edges (k=4).  Labeler-owned ``arrays.edge_index``
            and ``arrays.edge_y`` are intentionally ignored: they may depend on
            truth and are retained in ``WCMLArrays`` only for legacy diagnostics.
            """
            if n_blobs <= 0:
                return self._empty_edge_index(), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

            points = getattr(arrays, "points", None)
            candidate_edges: list[np.ndarray] = []
            source_counts: dict[str, int] = {}
            if points is not None:
                points = np.asarray(points)
                if points.ndim == 2 and points.shape[1] >= 5 and points.shape[0] > 0:
                    ppedges = getattr(arrays, "ppedges", None)
                    if ppedges is not None and np.asarray(ppedges).size > 0:
                        try:
                            _, mapped_ppedges = self._ppedges_to_blobedges(np.asarray(ppedges), points)
                            mapped_ppedges = self._unique_undirected_edges(mapped_ppedges, n_blobs)
                        except (IndexError, TypeError, ValueError) as error:
                            if self.config.write_diagnostics:
                                print(f"[WARN] failed to map ppedges to blob edges: {error}", flush=True)
                            mapped_ppedges = self._empty_edge_index()
                        candidate_edges.append(mapped_ppedges)
                        source_counts["ppedges"] = int(mapped_ppedges.shape[1])

            geometry_edges = self._blob_knn_radius_edges(centroids, k=8, radius_mm=80.0)
            candidate_edges.append(geometry_edges)
            source_counts["geom_knn_radius"] = int(geometry_edges.shape[1])

            cluster_edges = self._cluster_neighbor_edges(
                reco_cluster_by_blob,
                centroids,
                k_per_cluster=4,
            )
            candidate_edges.append(cluster_edges)
            source_counts["reco_cluster"] = int(cluster_edges.shape[1])

            nonempty_edges = [
                self._normalize_edge_index(edge_index)
                for edge_index in candidate_edges
                if edge_index is not None and np.asarray(edge_index).size > 0
            ]
            if nonempty_edges:
                edge_index = self._unique_undirected_edges(
                    np.concatenate(nonempty_edges, axis=1),
                    n_blobs,
                )
            else:
                edge_index = self._empty_edge_index()

            edge_index, edge_y, edge_labelable = self._label_blob_edges(
                edge_index,
                truth_instance_by_blob,
                semantic,
                vtx_dist_by_blob,
            )
            if self.config.write_diagnostics:
                labelable = np.asarray(edge_labelable, dtype=np.int64).reshape(-1)
                labels = np.asarray(edge_y, dtype=np.int64).reshape(-1)
                print(
                    "[sp-candidates] "
                    f"n_blobs={n_blobs} sources={source_counts} "
                    f"union_edges={edge_index.shape[1]} "
                    f"labelable={int(labelable.sum()) if labelable.size else 0} "
                    f"pos={int(((labels == 1) & (labelable == 1)).sum()) if labels.size else 0} "
                    f"neg={int(((labels == 0) & (labelable == 1)).sum()) if labels.size else 0}",
                    flush=True,
                )
            return edge_index, edge_y, edge_labelable
    
    def _unique_undirected_edges(self, edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
        """
        Take edge_index (2,E) and return unique undirected edges with src<dst, in range [0,n_nodes).
        """
        ei = np.asarray(edge_index, dtype=np.int64)
        if ei.ndim != 2 or ei.shape[0] != 2 or ei.shape[1] == 0:
            return np.empty((2, 0), dtype=np.int64)

        src = ei[0]
        dst = ei[1]
        m = (src >= 0) & (dst >= 0) & (src < n_nodes) & (dst < n_nodes) & (src != dst)
        if not np.any(m):
            return np.empty((2, 0), dtype=np.int64)

        a = np.minimum(src[m], dst[m])
        b = np.maximum(src[m], dst[m])
        pairs = np.stack([a, b], axis=1)
        pairs_u = np.unique(pairs, axis=0).astype(np.int64)
        return np.stack([pairs_u[:, 0], pairs_u[:, 1]], axis=0)
        
    def _label_blob_edges(
            self,
            edge_index: np.ndarray,
            truth_instance_by_blob: np.ndarray,
            semantic: np.ndarray,
            vtx_dist_by_blob: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Blob-level edge supervision.
            
            Simplified Logic:
            labelable = 1  iff both endpoints have a valid truth instance id (tid != -1).
            edge_y    = 1  iff labelable and both endpoints share the same tid.
            
            We do NOT filter by 'kept' instances anymore, because that destroys 
            hard negatives at the boundaries of instances.
            """
            ei = np.asarray(edge_index, dtype=np.int64)
            if ei.ndim != 2 or ei.shape[0] != 2:
                return (
                    np.empty((2, 0), dtype=np.int64),
                    np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=np.int64),
                )
            
            E = int(ei.shape[1])
            if E == 0:
                return ei, np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
            
            tid = np.asarray(truth_instance_by_blob, dtype=np.int64).reshape(-1)
            
            src = ei[0]
            dst = ei[1]
            t0 = tid[src]
            t1 = tid[dst]
            
            # ------------------------------------------------------------------
            # FIX #2: Ghost blobs must NOT participate in instance supervision.
            # Your ghost convention is semantic == -2 (encodes to y_semantic == -1).
            # ------------------------------------------------------------------
            sem = np.asarray(semantic, dtype=np.int64).reshape(-1)
            src_sem = sem[src]
            dst_sem = sem[dst]
            src_ok = (src_sem != -2)
            dst_ok = (dst_sem != -2)
            
            labelable = ((t0 >= 0) & (t1 >= 0) & src_ok & dst_ok).astype(np.int64)
            
            y = np.zeros((E,), dtype=np.int64)
            y[(labelable == 1) & (t0 == t1)] = 1
            
            return ei, y, labelable
            


    def _ppedges_to_blobedges(self, ppedges: np.ndarray, points: np.ndarray):
        """
        Convert point-point edges to blob-blob undirected unique edges.
        Returns:
          pairs_unique: (E,2) [src_blob, dst_blob] with src<dst
          edge_index:   (2,E)
        """
        heads = ppedges[:, 0].astype(np.int64)
        tails = ppedges[:, 1].astype(np.int64)
        blob_idx = points[:, 4].astype(np.int64)

        bh = blob_idx[heads]
        bt = blob_idx[tails]

        mask = (bh >= 0) & (bt >= 0)
        if not np.any(mask):
            return np.empty((0, 2), dtype=np.int64), np.empty((2, 0), dtype=np.int64)
        bh = bh[mask]
        bt = bt[mask]

        keep = bh != bt
        if not np.any(keep):
            return np.empty((0, 2), dtype=np.int64), np.empty((2, 0), dtype=np.int64)
        bh = bh[keep]
        bt = bt[keep]

        a = np.minimum(bh, bt)
        b = np.maximum(bh, bt)
        pairs = np.stack([a, b], axis=1)
        pairs_unique = np.unique(pairs, axis=0).astype(np.int64)

        if pairs_unique.size == 0:
            return pairs_unique, np.empty((2, 0), dtype=np.int64)

        src = pairs_unique[:, 0]
        dst = pairs_unique[:, 1]
        edge_index = np.stack([src, dst], axis=0)
        return pairs_unique, edge_index

    # ------------------------------------------------------------------
    # Plane building
    # ------------------------------------------------------------------
    def _build_plane(
        self,
        spec: PlaneSpec,
        ctpc: np.ndarray,
        corners: Sequence[np.ndarray],
        centroid: np.ndarray,
        semantic: np.ndarray,
        reco_cluster_by_blob: np.ndarray,
        truth_instance_by_blob: np.ndarray,
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
        node_truth_instances: List[List[int]] = [[] for _ in range(len(positions))]
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
                    node_truth_instances[node_idx].append(int(truth_instance_by_blob[blob_id]))
                    node_blobs[node_idx].append(blob_id)

        labels = np.full(len(positions), -2, dtype=np.int64)
        for node_idx, linked_labels in enumerate(node_semantics):
            if not linked_labels:
                continue
            if any(lbl == self.config.semantic_positive for lbl in linked_labels):
                labels[node_idx] = self.config.semantic_positive
            elif any(lbl == self.config.semantic_negative for lbl in linked_labels):
                labels[node_idx] = self.config.semantic_negative

        # Plane-node instance = mode of linked blob truth instances (ignore -1)
        instances = np.full(len(positions), -1, dtype=np.int64)
        for node_idx, insts in enumerate(node_truth_instances):
            if not insts:
                continue
            valid = np.asarray([c for c in insts if c != -1], dtype=np.int64)
            if valid.size == 0:
                continue
            values, counts = np.unique(valid, return_counts=True)
            instances[node_idx] = int(values[np.argmax(counts)])

        # Aggregate vertex + sidecar features per node from linked blobs
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
    
    def _semantic_propagate_on_mp(
        self,
        *,
        encoded_semantic: np.ndarray,     # (Nsp,) {0=nu,1=cosmic,-1=ghost}
        raw_vtx_dist_mm: np.ndarray,      # (Nsp,) mm, -1 if unknown
        mp_edge_index: np.ndarray,        # (2,E) undirected-ish, blob indices
        sp_pos_mm: np.ndarray | None = None,  # (Nsp,3) needed if you want edge length gating
    
        # seed-quality control
        frac_nu_hits: np.ndarray | None = None,   # (Nsp,) fraction of ν hits in blob
        seed_frac_nu_min: float = 0.60,
    
        seed_vtx_radius_mm: float = 100.0,
        edge_len_max_mm: float = 225.0,
        max_hops: int = 5,
        max_vtx_dist_mm: float = 2000.0,
    ) -> np.ndarray:
        """
        Expand ν labels from conservative seeds along the local MP graph.
        Only flips cosmic->nu; never touches ghosts (-1).
        """
    
        y = encoded_semantic.astype(np.int64, copy=True)
        N = int(y.shape[0])
        if N == 0:
            return y
    
        ei = np.asarray(mp_edge_index, dtype=np.int64) if mp_edge_index is not None else None
        if ei is None or ei.ndim != 2 or ei.shape[0] != 2 or ei.shape[1] == 0:
            return y
    
        dist = np.asarray(raw_vtx_dist_mm, dtype=np.float32).reshape(-1)
        finite = np.isfinite(dist) & (dist >= 0.0)
    
        # seeds: ν, near vtx, and optionally high frac_nu_hits
        seeds = (y == 0) & finite & (dist <= float(seed_vtx_radius_mm))
        if frac_nu_hits is not None:
            fn = np.asarray(frac_nu_hits, dtype=np.float32).reshape(-1)
            if fn.shape[0] == N:
                seeds = seeds & (fn >= float(seed_frac_nu_min))
    
        seed_idx = np.nonzero(seeds)[0]
        if seed_idx.size == 0:
            return y
    
        # adjacency (optionally prune by edge length)
        src = ei[0].astype(np.int64, copy=False)
        dst = ei[1].astype(np.int64, copy=False)
    
        in_range = (src >= 0) & (dst >= 0) & (src < N) & (dst < N) & (src != dst)
        if not np.any(in_range):
            return y
        src = src[in_range]
        dst = dst[in_range]
    
        if sp_pos_mm is not None and edge_len_max_mm is not None and edge_len_max_mm > 0:
            pos = np.asarray(sp_pos_mm, dtype=np.float32)
            d = pos[src] - pos[dst]
            elen = np.sqrt((d * d).sum(axis=1))
            keep = (elen <= float(edge_len_max_mm))
            src = src[keep]
            dst = dst[keep]
    
        adj = [[] for _ in range(N)]
        for a, b in zip(src.tolist(), dst.tolist()):
            adj[a].append(b)
            adj[b].append(a)
    
        visited = np.zeros(N, dtype=np.uint8)
        frontier = seed_idx.tolist()
        for s in frontier:
            visited[s] = 1
    
        for _ in range(int(max_hops)):
            if not frontier:
                break
            new_frontier = []
            for u in frontier:
                for v in adj[u]:
                    if visited[v]:
                        continue
                    if y[v] == -1:  # never traverse into ghosts
                        continue
                    if max_vtx_dist_mm is not None and max_vtx_dist_mm > 0:
                        dv = float(dist[v])
                        # Allow unknown vtx distance (dv < 0) so distal/low-charge ν can be reached.
                        # Constrain traversal only when dv is known (>=0).
                        if dv >= 0.0 and dv > float(max_vtx_dist_mm):
                            continue
                    visited[v] = 1
                    new_frontier.append(v)
            frontier = new_frontier
    
        # only flip cosmic->nu among reached nodes
        flip = (y == 1) & (visited == 1)
        y[flip] = 0
        return y

        

    # ------------------------------------------------------------------
    # Data extraction helpers
    # ------------------------------------------------------------------
    def _maybe_get_1d(self, arrays, key, n=None, dtype=None):
        x = getattr(arrays, key, None)
        if x is None:
            return None
        x = np.asarray(x)
        if dtype is not None:
            x = x.astype(dtype, copy=False)
        x = x.reshape(-1)
        if n is not None and x.shape[0] != n:
            return None
        return x

    
    # ------------------------------------------------------------------
    # Blob extraction, semantic labeling, sidecar features
    # ------------------------------------------------------------------
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

    def _label_blobs(
        self,
        points: np.ndarray,
        is_nu: np.ndarray | None,
        n_expected: int,
        config: ConversionConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        if points is None or points.size == 0:
            return np.full((n_expected,), -2, dtype=np.int64), np.zeros((n_expected,), dtype=np.float32)

        blob_indices = points[:, 4].astype(int)
        inferred = int(blob_indices.max()) + 1 if blob_indices.size else 0
        n_blobs = max(inferred, n_expected)

        labels = np.full(n_blobs, -2, dtype=np.int64)
        frac_nu_hits = np.zeros(n_blobs, dtype=np.float32)

        if is_nu is None:
            return labels, frac_nu_hits

        is_nu = np.asarray(is_nu).reshape(-1)
        if is_nu.shape[0] != points.shape[0]:
            return labels, frac_nu_hits

        for blob_id in range(labels.size):
            mask = blob_indices == blob_id
            if not mask.any():
                continue

            blob_labels = is_nu[mask]

            # --------------------------------------------
            # NEW: mask "ghost/unmatched" blobs
            # Unmatched truth convention: is_nu == -2
            # If ALL points are -2, we mark blob as unknown
            # so it encodes to y_semantic = -1 (ignored by loss).
            # --------------------------------------------
            known = (blob_labels != -2)
            if not np.any(known):
                labels[blob_id] = -2          # will encode to -1
                frac_nu_hits[blob_id] = 0.0
                continue

            blob_labels = blob_labels[known]
            nu_mask = (blob_labels == config.semantic_positive)
            frac_nu_hits[blob_id] = float(nu_mask.sum()) / float(blob_labels.size)
            labels[blob_id] = config.semantic_positive if nu_mask.any() else config.semantic_negative


        return labels, frac_nu_hits

    def _encode_semantic_labels(self, labels: np.ndarray) -> np.ndarray:
        encoded = np.full(labels.shape, -1, dtype=np.int64)
        positive_mask = labels == self.config.semantic_positive
        if positive_mask.any():
            encoded[positive_mask] = 0
        negative_mask = labels == self.config.semantic_negative
        if negative_mask.any():
            encoded[negative_mask] = 1
        return encoded

    def _compute_distances_to_walls_from_centroids(self, centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if centroids.size == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

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
        self, centroids: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        N = centroids.shape[0]
        if N == 0:
            z = np.zeros((0,), dtype=np.float32)
            return z, z, z, z
        if not _HAS_SKLEARN:
            z = np.zeros((N,), dtype=np.float32)
            return z, z, z, z

        k_eff = min(k, N)
        nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="kd_tree")
        nbrs.fit(centroids)
        _, indices = nbrs.kneighbors(centroids, return_distance=True)

        linearity = np.zeros(N, dtype=np.float32)
        sphericity = np.zeros(N, dtype=np.float32)
        ty = np.zeros(N, dtype=np.float32)
        tz = np.zeros(N, dtype=np.float32)

        for i in range(N):
            pts = centroids[indices[i]]
            if pts.shape[0] < 3:
                continue
            c = pts.mean(axis=0, keepdims=True)
            X = pts - c
            C = X.T @ X / float(X.shape[0])

            vals, vecs = np.linalg.eigh(C)
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
    # Event id parsing
    # ------------------------------------------------------------------
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


def convert_npz_file(npz_path: Path | str, output: Path | str, config: ConversionConfig | None = None) -> Path:
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


__all__ = ["WCMLConverter", "convert_npz_file", "convert_npz_directory"]
