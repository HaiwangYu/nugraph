from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.testing as npt

from pywcml.config import ConversionConfig
from pywcml.converter import WCMLConverter
from pywcml.io import WCMLArrays


def _synthetic_arrays(*, with_truth: bool = True) -> WCMLArrays:
    """Build an event with an adversarial edge outside the reco candidate union."""
    n_blobs = 12
    xyz = np.column_stack(
        [
            np.arange(n_blobs, dtype=np.float32) * 1000.0,
            np.zeros(n_blobs, dtype=np.float32),
            np.zeros(n_blobs, dtype=np.float32),
        ]
    )

    # One point and one one-corner blob per SP node.  Unique reco cluster IDs
    # make the cluster-neighbor contribution empty; the long 0--11 edge is also
    # outside k=8/radius=80 mm geometry and absent from ppedges.
    points = np.zeros((n_blobs, 6), dtype=np.float32)
    points[:, :3] = xyz
    points[:, 4] = np.arange(n_blobs)
    points[:, 5] = np.arange(n_blobs)

    blobs = np.zeros((n_blobs, 5), dtype=np.float32)
    blobs[:, 0] = 1.0
    blobs[:, 1] = 1.0
    blobs[:, 2:5] = xyz

    ppedges = np.array([[0, 1, 0], [5, 6, 0]], dtype=np.int64)
    truth_blob_tid = np.arange(n_blobs, dtype=np.int64) % 3 if with_truth else None
    is_nu = np.zeros(n_blobs, dtype=np.int16) if with_truth else None

    return WCMLArrays(
        blobs=blobs,
        points=points,
        ctpc={},
        is_nu=is_nu,
        ppedges=ppedges,
        origin_label=None,
        vtx_dist=None,
        vtx_dx=None,
        vtx_dy=None,
        vtx_dz=None,
        nu_vtx=None,
        nu_vtx_found=None,
        truth_blob_tid=truth_blob_tid,
        truth_blob_purity=None,
        truth_blob_support=None,
        edge_index=np.array([[0], [11]], dtype=np.int64),
        edge_y=np.array([1], dtype=np.int8),
        path=Path("rec-lab-apa0-0.npz"),
    )


def _candidate_result(converter: WCMLConverter, arrays: WCMLArrays):
    _, centroids, _ = converter._extract_blobs(arrays.blobs)
    reco_cluster = converter._reco_cluster_by_blob(arrays.points, len(centroids))
    truth_instance = converter._truth_instance_by_blob(arrays, len(centroids))
    semantic, _ = converter._label_blobs(
        arrays.points,
        arrays.is_nu,
        len(centroids),
        converter.config,
    )
    return converter._get_blob_sup_edges_and_labels(
        arrays=arrays,
        n_blobs=len(centroids),
        truth_instance_by_blob=truth_instance,
        semantic=semantic,
        vtx_dist_by_blob=np.zeros(len(centroids), dtype=np.float32),
        centroids=centroids,
        reco_cluster_by_blob=reco_cluster,
    )


def test_input_edge_invariance() -> None:
    """Labeler edge_index/edge_y must not affect topology or HDF5 targets."""
    converter = WCMLConverter()
    arrays = _synthetic_arrays()
    without_labeler_edges = replace(arrays, edge_index=None, edge_y=None)
    adversarial_labeler_edges = replace(
        arrays,
        edge_index=np.array([[0, 1, 2], [11, 10, 9]], dtype=np.int64),
        edge_y=np.array([1, 0, 1], dtype=np.int8),
    )

    expected = _candidate_result(converter, without_labeler_edges)
    actual = _candidate_result(converter, adversarial_labeler_edges)
    for expected_array, actual_array in zip(expected, actual):
        npt.assert_array_equal(actual_array, expected_array)


def test_truth_topology_invariance() -> None:
    """Truth may change targets and masks, never the candidate edge_index."""
    converter = WCMLConverter()
    arrays = _synthetic_arrays()
    all_same_truth = replace(arrays, truth_blob_tid=np.zeros(12, dtype=np.int64))
    partly_missing_truth = replace(
        arrays,
        truth_blob_tid=np.array([0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1]),
    )

    edge_a, y_a, labelable_a = _candidate_result(converter, all_same_truth)
    edge_b, y_b, labelable_b = _candidate_result(converter, partly_missing_truth)
    npt.assert_array_equal(edge_a, edge_b)
    assert not np.array_equal(y_a, y_b)
    assert not np.array_equal(labelable_a, labelable_b)


def test_exact_reconstruction_only_union() -> None:
    """Supervision candidates equal ppedges U geometry U reco-cluster edges."""
    converter = WCMLConverter()
    arrays = _synthetic_arrays()
    _, centroids, _ = converter._extract_blobs(arrays.blobs)
    reco_cluster = converter._reco_cluster_by_blob(arrays.points, len(centroids))

    _, ppedges = converter._ppedges_to_blobedges(arrays.ppedges, arrays.points)
    ppedges = converter._unique_undirected_edges(ppedges, len(centroids))
    geometry = converter._blob_knn_radius_edges(centroids, k=8, radius_mm=80.0)
    clusters = converter._cluster_neighbor_edges(reco_cluster, centroids, k_per_cluster=4)
    expected = converter._unique_undirected_edges(
        np.concatenate([ppedges, geometry, clusters], axis=1),
        len(centroids),
    )

    actual, _, _ = _candidate_result(converter, arrays)
    npt.assert_array_equal(actual, expected)
    assert not np.any(np.all(actual.T == np.array([0, 11]), axis=1))


def test_unlabeled_npz_conversion_regression(tmp_path: Path) -> None:
    """Generic truth-free NPZ input remains convertible and unlabelable."""
    arrays = _synthetic_arrays(with_truth=False)
    npz = tmp_path / "rec-apa0-0.npz"
    np.savez(
        npz,
        blobs=arrays.blobs,
        points=arrays.points,
        ppedges=arrays.ppedges,
    )

    _, graph = WCMLConverter().convert(npz)
    assert graph["sp"].edge_label_index.shape[1] > 0
    assert np.all(graph["sp"].edge_y.numpy() == 0)
    assert np.all(graph["sp"].edge_labelable.numpy() == 0)


def test_message_passing_independent_of_labeler_edges_and_truth() -> None:
    """The SP message-passing topology depends only on reconstruction ppedges."""
    converter = WCMLConverter(ConversionConfig(merge_sup_edges_into_mp=False))
    arrays = _synthetic_arrays()
    changed = replace(
        arrays,
        edge_index=np.array([[0, 2, 3], [11, 10, 9]], dtype=np.int64),
        edge_y=np.array([0, 1, 0], dtype=np.int8),
        truth_blob_tid=np.full(12, -1, dtype=np.int64),
    )

    expected_graph = converter._build_graph("rec-lab-apa0-0", arrays)
    actual_graph = converter._build_graph("rec-lab-apa0-0", changed)
    expected = expected_graph["sp", "nexus", "sp"].edge_index.numpy()
    actual = actual_graph["sp", "nexus", "sp"].edge_index.numpy()
    _, mapped_ppedges = converter._ppedges_to_blobedges(arrays.ppedges, arrays.points)
    mapped_ppedges = converter._unique_undirected_edges(mapped_ppedges, 12)

    npt.assert_array_equal(actual, expected)
    npt.assert_array_equal(actual, mapped_ppedges)
