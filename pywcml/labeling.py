"""Scientific SBND truth labeling over in-memory event arrays.

This module is the single scientific implementation used by both the streaming
API and the historical filename-driven CLI. Units, operation order, thresholds
and dtype conversions intentionally follow the validated labeler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np
from numpy.linalg import svd
from sklearn.neighbors import NearestNeighbors

from .edge_supervision import build_edge_supervision as _build_edge_supervision_fallback
from .identity import EventIdentity
from .io import WCMLArrays


@dataclass(frozen=True)
class RecoArrays:
    """Wire-Cell reconstruction payload for one APA, held entirely in memory."""

    blobs: np.ndarray
    points: np.ndarray
    ppedges: np.ndarray
    ctpc: Mapping[str, np.ndarray]
    extras: Mapping[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, arrays: Mapping[str, np.ndarray]) -> "RecoArrays":
        if "blobs" not in arrays or "points" not in arrays:
            raise KeyError("reconstruction arrays require 'blobs' and 'points'")
        ctpc = {key: value for key, value in arrays.items() if key.startswith("ctpc_")}
        extras = {
            key: value
            for key, value in arrays.items()
            if key not in {"blobs", "points", "ppedges"} and not key.startswith("ctpc_")
        }
        return cls(
            blobs=arrays["blobs"],
            points=arrays["points"],
            ppedges=arrays.get("ppedges", np.empty((0, 3), dtype=np.float32)),
            ctpc=ctpc,
            extras=extras,
        )

    def to_mapping(self) -> Dict[str, np.ndarray]:
        result = dict(self.extras)
        result.update(self.ctpc)
        result.update(blobs=self.blobs, points=self.points, ppedges=self.ppedges)
        return result


@dataclass(frozen=True)
class SemanticTruth:
    """Truth-deposition coordinates in cm and neutrino-origin flag ``q``."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    q: np.ndarray

    @classmethod
    def from_mapping(cls, arrays: Mapping[str, object]) -> "SemanticTruth":
        return cls(
            x=np.asarray(arrays["x"]),
            y=np.asarray(arrays["y"]),
            z=np.asarray(arrays["z"]),
            q=np.asarray(arrays["q"]),
        )


@dataclass(frozen=True)
class SimIDETruth:
    """Minimal SimIDE instance-matching payload for one physical event."""

    channel: np.ndarray
    tdc: np.ndarray
    track_id: np.ndarray
    x_cm: Optional[np.ndarray] = None
    y_cm: Optional[np.ndarray] = None
    z_cm: Optional[np.ndarray] = None


@dataclass(frozen=True)
class NeutrinoVertex:
    """Beam-neutrino vertex in cm."""

    x: float
    y: float
    z: float
    found: bool

    @property
    def xyz(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.z], dtype=np.float32)


@dataclass(frozen=True)
class LabelingConfig:
    """Validated labeling controls; production edge supervision defaults off."""

    max_distance_cm: float = 15.0
    z_offset_cm: float = 0.0
    tagging_alg: str = "blob"
    blob_grow_cm: float = 30.0
    vtx_gate_reco_cm: float = 500.0
    use_truth_gate: bool = False
    vtx_gate_truth_cm: float = 50.0
    do_instance: bool = True
    shift_p0: int = 2992
    shift_p1: int = 2992
    shift_p2: int = 2992
    dtdc_win: int = 1
    conf_thr: float = 0.7
    purity_thr: float = 0.85
    min_blob_support: int = 1
    max_dist_mm: float = 10.0
    write_edge_sup: bool = False
    balance_edges: bool = False
    neg_radius_mm: float = 60.0

    def __post_init__(self) -> None:
        if self.tagging_alg not in {"point", "blob", "cluster"}:
            raise ValueError("tagging_alg must be point, blob, or cluster")
        if self.balance_edges and not self.write_edge_sup:
            raise ValueError("balance_edges requires write_edge_sup")


def _as_intlike(array: np.ndarray) -> np.ndarray:
    return np.asarray(array).astype(np.int64, copy=False)


def _as_float(array: np.ndarray) -> np.ndarray:
    return np.asarray(array).astype(np.float32, copy=False)


def remap_negative_trackids(tid: np.ndarray) -> np.ndarray:
    """Canonicalize negative secondary IDs while preserving -1 as unlabeled."""

    tid = np.asarray(tid).astype(np.int64, copy=True)
    secondary = tid < -1
    tid[secondary] = -tid[secondary]
    return tid.astype(np.int32, copy=False)


def merge_trunk_split_tids(
    points_xyz_mm: np.ndarray,
    truth_tid_points: np.ndarray,
    *,
    min_pts: int = 300,
    min_len_mm: float = 800.0,
    angle_max_deg: float = 2.0,
    mednn_max_mm: float = 15.0,
):
    """Merge the validated collinear, spatially-overlapping truth-ID trunks."""

    points = points_xyz_mm.astype(float)
    tid = truth_tid_points.astype(int)
    labeled = tid >= 0
    labeled_points = points[labeled]
    labeled_tid = tid[labeled]
    tids, counts = np.unique(labeled_tid, return_counts=True)
    keep = [int(value) for value, count in zip(tids, counts) if count >= min_pts]
    if len(keep) < 2:
        return tid.copy(), [], []

    def pca_axis_len(cloud):
        centered = cloud - cloud.mean(0, keepdims=True)
        _, _, vectors = svd(centered, full_matrices=False)
        axis = vectors[0]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        projection = centered @ axis
        return axis, float(projection.max() - projection.min())

    axes, lengths, clouds = {}, {}, {}
    for value in keep:
        cloud = labeled_points[labeled_tid == value]
        axis, length = pca_axis_len(cloud)
        axes[value], lengths[value], clouds[value] = axis, length, cloud

    def angle_degrees(first, second):
        cosine = abs(float(np.dot(first, second)))
        cosine = max(-1.0, min(1.0, cosine))
        return float(np.degrees(np.arccos(cosine)))

    parent = list(range(len(keep)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    merges = []
    for first_index, first_tid in enumerate(keep):
        for second_index in range(first_index + 1, len(keep)):
            second_tid = keep[second_index]
            if min(lengths[first_tid], lengths[second_tid]) < min_len_mm:
                continue
            angle = angle_degrees(axes[first_tid], axes[second_tid])
            if angle > angle_max_deg:
                continue
            first_cloud, second_cloud = clouds[first_tid], clouds[second_tid]
            larger, smaller = (
                (first_cloud, second_cloud)
                if len(first_cloud) >= len(second_cloud)
                else (second_cloud, first_cloud)
            )
            distance = (
                NearestNeighbors(n_neighbors=1)
                .fit(larger)
                .kneighbors(smaller, return_distance=True)[0]
                .ravel()
            )
            median_distance = float(np.median(distance))
            if median_distance < mednn_max_mm:
                union(first_index, second_index)
                merges.append((first_tid, second_tid, angle, median_distance))

    component_map: Dict[int, list[int]] = {}
    for index, value in enumerate(keep):
        component_map.setdefault(find(index), []).append(value)
    components = [sorted(values) for values in component_map.values() if len(values) > 1]
    representatives = {}
    for values in component_map.values():
        representative = min(values)
        for value in values:
            representatives[value] = representative
    merged = tid.copy()
    for value, representative in representatives.items():
        merged[merged == value] = representative
    return merged, merges, components


def semantic_labels(
    reco: RecoArrays,
    truth: SemanticTruth,
    *,
    max_distance_cm: float,
    z_offset_cm: float,
    tagging_alg: str,
    blob_grow_cm: float,
) -> np.ndarray:
    """Apply the validated nearest-truth semantic labeling algorithm."""

    points = reco.points
    x = points[:, 0] / 10.0
    y = points[:, 1] / 10.0
    z = points[:, 2] / 10.0 + float(z_offset_cm)
    blob_index = points[:, 4].astype(np.int64, copy=False)
    cluster_index = points[:, 5].astype(np.int64, copy=False)
    reco_xyz = np.column_stack([x, y, z]).astype(np.float32, copy=False)
    truth_xyz = np.column_stack([truth.x, truth.y, truth.z]).astype(np.float32, copy=False)
    truth_q = np.asarray(truth.q).astype(np.int16, copy=False)

    nearest = NearestNeighbors(n_neighbors=1)
    nearest.fit(truth_xyz)
    distances, indices = nearest.kneighbors(reco_xyz)
    distances = distances.reshape(-1)
    indices = indices.reshape(-1)
    is_nu = np.full(len(reco_xyz), -2, dtype=np.int16)
    close = distances <= float(max_distance_cm)
    is_nu[close] = truth_q[indices[close]]

    if tagging_alg == "blob":
        radius = float(blob_grow_cm)
        seeds_source = is_nu.copy()
        for blob in np.unique(blob_index):
            in_blob = blob_index == blob
            if not np.any(in_blob):
                continue
            seeds = in_blob & (seeds_source == 1)
            if not np.any(seeds):
                continue
            nearest_seed = NearestNeighbors(n_neighbors=1)
            nearest_seed.fit(reco_xyz[seeds])
            distance, _ = nearest_seed.kneighbors(reco_xyz[in_blob])
            grow = distance.reshape(-1) <= radius
            is_nu[np.where(in_blob)[0][grow]] = 1
    elif tagging_alg == "cluster":
        for cluster in np.unique(cluster_index):
            in_cluster = cluster_index == cluster
            if np.any(is_nu[in_cluster] == 1):
                is_nu[in_cluster] = 1
    elif tagging_alg != "point":
        raise ValueError(f"Unknown tagging_alg={tagging_alg} (expected point/blob/cluster)")
    return is_nu


def truth_q1_fraction_within_radius(
    truth: SemanticTruth,
    vertex_cm: np.ndarray,
    radius_cm: float,
) -> float:
    q = np.asarray(truth.q).astype(int)
    if np.sum(q == 1) == 0:
        return 0.0
    xyz = np.column_stack([truth.x, truth.y, truth.z]).astype(float)
    distance = np.linalg.norm(xyz[q == 1] - np.asarray(vertex_cm, dtype=float)[None, :], axis=1)
    return float(np.mean(distance <= float(radius_cm)))


def apply_reco_vtx_gate(is_nu: np.ndarray, distance_cm: np.ndarray, gate_cm: float) -> np.ndarray:
    outside = (is_nu == 1) & (distance_cm > float(gate_cm))
    if np.any(outside):
        is_nu = is_nu.copy()
        is_nu[outside] = 0
    return is_nu


def ctpc_col_candidates(ctpc: np.ndarray) -> tuple[int, int]:
    """Retain the validated heuristic for CTPC channel/tick columns."""

    n_columns = ctpc.shape[1]
    columns = list(range(n_columns))

    def intlike_fraction(column):
        values = ctpc[:, column]
        if not np.issubdtype(values.dtype, np.number):
            return 0.0
        return float(np.mean(np.abs(values - np.round(values)) < 1e-3))

    tick_best, tick_score = None, -1
    for column in columns:
        values = ctpc[:, column]
        if not np.issubdtype(values.dtype, np.number):
            continue
        minimum, maximum = float(np.min(values)), float(np.max(values))
        if minimum < -1e-3 or maximum > 50000:
            continue
        score = intlike_fraction(column) + 0.02 * (column / max(1, n_columns - 1))
        if score > tick_score:
            tick_score, tick_best = score, column
    if tick_best is None:
        tick_best = min(6, n_columns - 1)

    channel_best, channel_score = None, -1
    for column in columns:
        values = ctpc[:, column]
        if not np.issubdtype(values.dtype, np.number):
            continue
        _, maximum = float(np.min(values)), float(np.max(values))
        if maximum < 50 or maximum > 20000:
            continue
        preference = 0.15 if column in (4, 5) else 0.0
        score = intlike_fraction(column) + preference
        if score > channel_score:
            channel_score, channel_best = score, column
    if channel_best is None:
        channel_best = min(4, n_columns - 1)
    return channel_best, tick_best


def match_ctpc_to_simide_trackid(
    ctpc: np.ndarray,
    sim_channel: np.ndarray,
    sim_tdc: np.ndarray,
    sim_trackid: np.ndarray,
    shift: int,
    dtdc_win: int = 1,
    conf_thr: float = 0.7,
    chan_col: Optional[int] = None,
    tick_col: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if chan_col is None or tick_col is None:
        detected_channel, detected_tick = ctpc_col_candidates(ctpc)
        chan_col = detected_channel if chan_col is None else chan_col
        tick_col = detected_tick if tick_col is None else tick_col

    ctpc_channel = np.round(ctpc[:, chan_col]).astype(np.int64, copy=False)
    ctpc_tick = np.round(ctpc[:, tick_col]).astype(np.int64, copy=False)
    target_tdc = ctpc_tick + int(shift)

    from collections import defaultdict

    channel_to_indices = defaultdict(list)
    for index, channel in enumerate(sim_channel):
        channel_to_indices[int(channel)].append(index)

    tid = np.full(ctpc.shape[0], -1, dtype=np.int32)
    confidence = np.zeros(ctpc.shape[0], dtype=np.float32)
    window, threshold = int(dtdc_win), float(conf_thr)
    for index in range(ctpc.shape[0]):
        candidates = []
        indices = channel_to_indices.get(int(ctpc_channel[index]), None)
        if not indices:
            continue
        target = int(target_tdc[index])
        for sim_index in indices:
            delta = int(sim_tdc[sim_index]) - target
            if -window <= delta <= window:
                candidates.append(int(sim_trackid[sim_index]))
        if not candidates:
            continue
        if len(candidates) == 1:
            tid[index], confidence[index] = np.int32(candidates[0]), 1.0
            continue
        candidates = np.asarray(candidates, dtype=np.int64)
        values, counts = np.unique(candidates, return_counts=True)
        best = int(np.argmax(counts))
        best_confidence = float(counts[best] / np.sum(counts))
        if best_confidence >= threshold:
            tid[index] = np.int32(int(values[best]))
            confidence[index] = np.float32(best_confidence)
    return tid, confidence


def detect_ctpc_family(arrays: Mapping[str, np.ndarray]) -> str:
    has_f0 = any(f"ctpc_f0p{plane}" in arrays for plane in (0, 1, 2))
    has_f1 = any(f"ctpc_f1p{plane}" in arrays for plane in (0, 1, 2))
    if has_f0 and has_f1:
        raise RuntimeError("Reco contains BOTH ctpc_f0p* and ctpc_f1p*; ambiguous.")
    if not has_f0 and not has_f1:
        raise RuntimeError("Reco contains neither ctpc_f0p* nor ctpc_f1p*; cannot instance-label.")
    return "f0" if has_f0 else "f1"


def build_truth_tid_points_direct(
    points_mm: np.ndarray,
    ctpc_by_plane: Mapping[str, np.ndarray],
    tid_by_plane: Mapping[str, np.ndarray],
    max_dist_mm: float = 5.0,
):
    """Use the validated p2 XZ match to seed direct point truth IDs."""

    n_points = points_mm.shape[0]
    tid_points = np.full(n_points, -1, dtype=np.int32)
    support = np.zeros(n_points, dtype=np.int16)
    metadata = {}
    point_xz = points_mm[:, :3].astype(np.float32, copy=False)[:, [0, 2]]
    p2_name = next((name for name in ("f0p2", "f1p2") if name in ctpc_by_plane), None)
    if p2_name is None:
        metadata["p2"] = {"used": 0, "pair": "xz", "median_mm": None, "within_frac": 0.0, "reason": "no_p2"}
        return tid_points, support, metadata

    ctpc = ctpc_by_plane[p2_name]
    tid_ctpc = tid_by_plane[p2_name]
    labeled = tid_ctpc != -1
    if not np.any(labeled):
        metadata["p2"] = {"used": 0, "pair": "xz", "median_mm": None, "within_frac": 0.0, "reason": "no_labeled_ctpc"}
        return tid_points, support, metadata
    selected_ctpc = ctpc[labeled]
    selected_tid = tid_ctpc[labeled].astype(np.int32, copy=False)
    if selected_ctpc.shape[1] < 2:
        metadata["p2"] = {"used": 0, "pair": "xz", "median_mm": None, "within_frac": 0.0, "reason": "ctpc_cols<2"}
        return tid_points, support, metadata

    ctpc_xz = selected_ctpc[:, [0, 1]].astype(np.float32, copy=False)
    nearest = NearestNeighbors(n_neighbors=1)
    nearest.fit(point_xz)
    distance, point_index = nearest.kneighbors(ctpc_xz)
    distance = distance.reshape(-1)
    point_index = point_index.reshape(-1).astype(np.int64, copy=False)
    close = distance <= float(max_dist_mm)
    metadata["p2"] = {
        "used": int(np.sum(close)),
        "pair": "xz",
        "median_mm": float(np.median(distance)) if distance.size else None,
        "within_frac": float(np.mean(close)) if distance.size else 0.0,
        "p2_name": p2_name,
    }
    if not np.any(close):
        return tid_points, support, metadata
    for point, value in zip(point_index[close], selected_tid[close]):
        point, value = int(point), int(value)
        if tid_points[point] == -1:
            tid_points[point], support[point] = np.int32(value), np.int16(1)
        elif int(tid_points[point]) == value:
            support[point] = np.int16(int(support[point]) + 1)
    return tid_points, support, metadata


def blob_propagate_truth(
    points: np.ndarray,
    tid_points_direct: np.ndarray,
    purity_thr: float = 0.85,
    min_blob_support: int = 1,
):
    blob_id = points[:, 4].astype(np.int64, copy=False)
    n_blobs = int(blob_id.max()) + 1 if blob_id.size else 0
    blob_tid = np.full(n_blobs, -1, dtype=np.int32)
    blob_purity = np.zeros(n_blobs, dtype=np.float32)
    blob_support = np.zeros(n_blobs, dtype=np.int16)
    for blob in range(n_blobs):
        in_blob = blob_id == blob
        if not np.any(in_blob):
            continue
        tids = tid_points_direct[in_blob]
        tids = tids[tids != -1]
        if tids.size < int(min_blob_support):
            continue
        values, counts = np.unique(tids, return_counts=True)
        best = int(np.argmax(counts))
        support = int(np.sum(counts))
        purity = float(counts[best] / support) if support > 0 else 0.0
        blob_support[blob] = np.int16(support)
        blob_purity[blob] = np.float32(purity)
        if purity >= float(purity_thr):
            blob_tid[blob] = np.int32(int(values[best]))
    tid_points = np.full(points.shape[0], -1, dtype=np.int32)
    for blob in range(n_blobs):
        if blob_tid[blob] != -1:
            tid_points[blob_id == blob] = blob_tid[blob]
    return tid_points, blob_tid, blob_purity, blob_support


def write_edge_supervision_to_out(
    out: Dict[str, np.ndarray],
    write_edge_sup: bool = False,
    balance_edges: bool = False,
    neg_radius_mm: float = 60.0,
    seed: int = 123,
) -> None:
    """Retain the historical optional point-edge diagnostic output."""

    if not write_edge_sup:
        return
    for required in ("ppedges", "truth_tid_points", "points"):
        if required not in out:
            raise RuntimeError(f"write_edge_sup requested but '{required}' not found.")
    ppedges = out["ppedges"]
    if ppedges.ndim != 2 or ppedges.shape[1] < 2:
        raise RuntimeError(f"Unexpected ppedges shape: {ppedges.shape}")
    xyz_mm = out["points"][:, :3]
    if xyz_mm.ndim != 2 or xyz_mm.shape[1] != 3:
        raise RuntimeError(f"Unexpected points[:, :3] shape: {xyz_mm.shape}")
    tid = out["truth_tid_points"].astype(np.int64, copy=False)
    if tid.shape[0] != xyz_mm.shape[0]:
        raise RuntimeError("truth_tid_points length != n_points")

    source = ppedges[:, 0].astype(np.int64, copy=False)
    target = ppedges[:, 1].astype(np.int64, copy=False)
    labelable = (tid >= 0)
    labelable_edges = labelable[source] & labelable[target]
    source, target = source[labelable_edges], target[labelable_edges]
    if source.size == 0:
        out["edge_index"] = np.empty((2, 0), dtype=np.int64)
        out["edge_y"] = np.empty((0,), dtype=np.int8)
        return
    labels = (tid[source] == tid[target]).astype(np.int8, copy=False)
    if not balance_edges:
        out["edge_index"] = np.stack([source, target], axis=0).astype(np.int64, copy=False)
        out["edge_y"] = labels.astype(np.int8, copy=False)
        return

    positive_mask = labels == 1
    if not np.any(positive_mask):
        out["edge_index"] = np.stack([source, target], axis=0).astype(np.int64, copy=False)
        out["edge_y"] = labels.astype(np.int8, copy=False)
        return
    positive_pairs = np.stack([source[positive_mask], target[positive_mask]], axis=1).astype(np.int64, copy=False)

    try:
        from scipy.spatial import cKDTree
    except Exception:
        cKDTree = None
    if cKDTree is not None:
        rng = np.random.default_rng(int(seed))
        positive_set = set()
        for first, second in positive_pairs.tolist():
            if first == second:
                continue
            if first > second:
                first, second = second, first
            positive_set.add((int(first), int(second)))
        for value in np.unique(tid[tid >= 0]).tolist():
            indices = np.where(tid == value)[0]
            count = int(indices.size)
            if count < 400:
                continue
            if count > 3000:
                indices = rng.choice(indices, size=3000, replace=False)
                count = int(indices.size)
            if count < 2:
                continue
            tree = cKDTree(xyz_mm.astype(np.float32, copy=False)[indices])
            k = min(3, count)
            distances, neighbors = tree.query(xyz_mm.astype(np.float32, copy=False)[indices], k=k, workers=-1)
            for local_index in range(count):
                first = int(indices[local_index])
                for neighbor in range(1, k):
                    if float(distances[local_index, neighbor]) > 20.0:
                        continue
                    second = int(indices[int(neighbors[local_index, neighbor])])
                    if first == second:
                        continue
                    positive_set.add((first, second) if first < second else (second, first))
        positive_pairs = (
            np.asarray(list(positive_set), dtype=np.int64)
            if positive_set
            else np.empty((0, 2), dtype=np.int64)
        )
    if positive_pairs.shape[0] == 0:
        out["edge_index"] = np.stack([source, target], axis=0).astype(np.int64, copy=False)
        out["edge_y"] = labels.astype(np.int8, copy=False)
        return
    edge_index, edge_y = _build_edge_supervision_fallback(
        xyz_mm=xyz_mm.astype(np.float32, copy=False),
        tid_points=tid,
        pos_pairs=positive_pairs,
        balance_edges=True,
        neg_radius_mm=float(neg_radius_mm),
        seed=int(seed),
    )
    out["edge_index"] = edge_index.astype(np.int64, copy=False)
    out["edge_y"] = edge_y.astype(np.int8, copy=False)


def label_event(
    reco: RecoArrays,
    semantic_truth: SemanticTruth,
    simide_truth: Optional[SimIDETruth],
    neutrino_vertex: NeutrinoVertex,
    identity: EventIdentity,
    apa: int,
    config: LabelingConfig = LabelingConfig(),
) -> WCMLArrays:
    """Label one APA view of one physical event without filesystem I/O."""

    if apa not in (0, 1):
        raise ValueError(f"APA must be 0 or 1, got {apa!r}")
    out = reco.to_mapping()
    is_nu = semantic_labels(
        reco,
        semantic_truth,
        max_distance_cm=config.max_distance_cm,
        z_offset_cm=config.z_offset_cm,
        tagging_alg=config.tagging_alg,
        blob_grow_cm=float(config.blob_grow_cm),
    )
    vertex = neutrino_vertex.xyz
    found = bool(neutrino_vertex.found)
    if found and config.use_truth_gate:
        fraction = truth_q1_fraction_within_radius(
            semantic_truth,
            vertex,
            config.vtx_gate_truth_cm,
        )
        if fraction <= 0.0:
            found = False

    points_mm = out["points"][:, :3].astype(np.float32, copy=False)
    points_cm = points_mm / 10.0
    n_hits = points_cm.shape[0]
    if found and n_hits > 0:
        dx = points_cm[:, 0] - float(vertex[0])
        dy = points_cm[:, 1] - float(vertex[1])
        dz = points_cm[:, 2] - float(vertex[2])
        vertex_distance = np.sqrt(dx * dx + dy * dy + dz * dz).astype(np.float32)
        vertex_dx = dx.astype(np.float32)
        vertex_dy = dy.astype(np.float32)
        vertex_dz = dz.astype(np.float32)
        is_nu = apply_reco_vtx_gate(is_nu, vertex_distance, config.vtx_gate_reco_cm)
        origin_label = np.full_like(is_nu, 2, dtype=np.int16)
        origin_label[is_nu == 1] = 0
        origin_label[is_nu == 0] = 1
    else:
        is_nu = np.zeros_like(is_nu, dtype=np.int16)
        origin_label = np.ones_like(is_nu, dtype=np.int16)
        vertex_distance = -np.ones(n_hits, dtype=np.float32)
        vertex_dx = np.zeros(n_hits, dtype=np.float32)
        vertex_dy = np.zeros(n_hits, dtype=np.float32)
        vertex_dz = np.zeros(n_hits, dtype=np.float32)
        vertex = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)

    out["is_nu"] = is_nu.astype(np.int16, copy=False)
    out["origin_label"] = origin_label.astype(np.int16, copy=False)
    out["nu_vtx"] = vertex.astype(np.float32, copy=False)
    out["nu_vtx_found"] = np.asarray([1 if found else 0], dtype=np.int16)
    out["vtx_dist"] = vertex_distance
    out["vtx_dx"] = vertex_dx
    out["vtx_dy"] = vertex_dy
    out["vtx_dz"] = vertex_dz

    if config.do_instance:
        if simide_truth is None:
            raise ValueError("simide_truth is required when do_instance=True")
        family = detect_ctpc_family(out)
        expected_family = f"f{apa}"
        if family != expected_family:
            raise ValueError(f"APA {apa} does not match reconstruction CTPC family {family}")
        plane_config = {
            f"{family}p0": config.shift_p0,
            f"{family}p1": config.shift_p1,
            f"{family}p2": config.shift_p2,
        }
        ctpc_by_plane, tid_by_plane, confidence_by_plane = {}, {}, {}
        for plane, shift in plane_config.items():
            key = f"ctpc_{plane}"
            if key not in out:
                continue
            ctpc = out[key]
            ctpc_by_plane[plane] = ctpc
            tid_ctpc, confidence_ctpc = match_ctpc_to_simide_trackid(
                ctpc=ctpc,
                sim_channel=simide_truth.channel,
                sim_tdc=simide_truth.tdc,
                sim_trackid=simide_truth.track_id,
                shift=int(shift),
                dtdc_win=int(config.dtdc_win),
                conf_thr=float(config.conf_thr),
            )
            tid_by_plane[plane] = tid_ctpc
            confidence_by_plane[plane] = confidence_ctpc
            out[f"truth_tid_{plane}"] = tid_ctpc.astype(np.int32, copy=False)
            out[f"truth_conf_{plane}"] = confidence_ctpc.astype(np.float32, copy=False)

        direct_tid, direct_support, metadata = build_truth_tid_points_direct(
            points_mm=out["points"][:, :3],
            ctpc_by_plane=ctpc_by_plane,
            tid_by_plane=tid_by_plane,
            max_dist_mm=float(config.max_dist_mm),
        )
        out["truth_tid_points_direct_raw"] = direct_tid.astype(np.int32, copy=False)
        direct_tid = remap_negative_trackids(direct_tid.astype(np.int32))
        out["truth_tid_points_direct"] = direct_tid.astype(np.int32, copy=False)
        out["truth_tid_points_direct_support"] = direct_support.astype(np.int16, copy=False)
        out["truth_tid_points_direct_meta_p2_used"] = np.asarray(
            [metadata.get("p2", {}).get("used", -1)], dtype=np.int32
        )

        tid_points, blob_tid, blob_purity, blob_support = blob_propagate_truth(
            points=out["points"],
            tid_points_direct=direct_tid,
            purity_thr=float(config.purity_thr),
            min_blob_support=int(config.min_blob_support),
        )
        tid_points = remap_negative_trackids(tid_points.astype(np.int32))
        blob_tid = remap_negative_trackids(blob_tid.astype(np.int32))
        tid_points_merged, _merges, _components = merge_trunk_split_tids(
            points_xyz_mm=out["points"][:, :3],
            truth_tid_points=tid_points,
            min_pts=300,
            min_len_mm=800.0,
            angle_max_deg=2.0,
            mednn_max_mm=15.0,
        )
        out["truth_tid_points_merged"] = tid_points_merged.astype(np.int32, copy=False)
        tid_points = tid_points_merged

        blob_id = out["points"][:, 4].astype(np.int64, copy=False)
        n_blobs = int(blob_id.max()) + 1 if blob_id.size else 0
        blob_tid = np.full(n_blobs, -1, dtype=np.int32)
        blob_purity = np.zeros(n_blobs, dtype=np.float32)
        blob_support = np.zeros(n_blobs, dtype=np.int16)
        for blob in range(n_blobs):
            values = tid_points[blob_id == blob]
            values = values[values >= 0]
            if values.size == 0:
                continue
            unique, counts = np.unique(values, return_counts=True)
            best = int(np.argmax(counts))
            blob_tid[blob] = np.int32(unique[best])
            blob_support[blob] = np.int16(int(counts.sum()))
            blob_purity[blob] = np.float32(float(counts[best] / counts.sum()))
        out["truth_blob_tid"] = blob_tid.astype(np.int32, copy=False)
        out["truth_tid_points"] = tid_points.astype(np.int32, copy=False)
        out["truth_blob_purity"] = blob_purity.astype(np.float32, copy=False)
        out["truth_blob_support"] = blob_support.astype(np.int16, copy=False)
        write_edge_supervision_to_out(
            out,
            write_edge_sup=bool(config.write_edge_sup),
            balance_edges=bool(config.balance_edges),
            neg_radius_mm=float(config.neg_radius_mm),
            seed=123 + int(identity.source_index),
        )
    return WCMLArrays.from_mapping(out)


__all__ = [
    "EventIdentity",
    "LabelingConfig",
    "NeutrinoVertex",
    "RecoArrays",
    "SemanticTruth",
    "SimIDETruth",
    "apply_reco_vtx_gate",
    "blob_propagate_truth",
    "build_truth_tid_points_direct",
    "ctpc_col_candidates",
    "detect_ctpc_family",
    "label_event",
    "match_ctpc_to_simide_trackid",
    "merge_trunk_split_tids",
    "remap_negative_trackids",
    "semantic_labels",
    "truth_q1_fraction_within_radius",
    "write_edge_supervision_to_out",
]
