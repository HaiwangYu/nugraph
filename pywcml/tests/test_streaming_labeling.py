from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import numpy.testing as npt
import pytest

from pywcml.identity import EventIdentity
from pywcml.labeling import (
    LabelingConfig,
    NeutrinoVertex,
    RecoArrays,
    SemanticTruth,
    SimIDETruth,
    label_event,
)


def _event_inputs():
    points = np.zeros((4, 6), dtype=np.float32)
    points[:, 0] = np.asarray([0.0, 10.0, 100.0, 110.0], dtype=np.float32)
    points[:, 4] = np.asarray([0, 0, 1, 1], dtype=np.float32)
    points[:, 5] = np.asarray([0, 0, 1, 1], dtype=np.float32)
    blobs = np.zeros((2, 5), dtype=np.float32)
    blobs[:, :2] = 1.0
    blobs[:, 2] = np.asarray([5.0, 105.0], dtype=np.float32)

    ctpc = np.zeros((4, 7), dtype=np.float32)
    ctpc[:, 0] = points[:, 0]
    ctpc[:, 1] = points[:, 2]
    ctpc[:, 4] = np.asarray([100, 100, 101, 101])
    ctpc[:, 6] = np.asarray([10, 11, 12, 13])
    reco = RecoArrays(
        blobs=blobs,
        points=points,
        ppedges=np.asarray([[0, 1, 0], [2, 3, 0]], dtype=np.int64),
        ctpc={"ctpc_f0p2": ctpc},
    )
    semantic = SemanticTruth(
        x=points[:, 0] / 10.0,
        y=np.zeros(4, dtype=np.float32),
        z=np.zeros(4, dtype=np.float32),
        q=np.asarray([1, 1, 0, 0], dtype=np.int16),
    )
    simide = SimIDETruth(
        channel=np.asarray([100, 100, 101, 101], dtype=np.int64),
        tdc=np.asarray([3002, 3003, 3004, 3005], dtype=np.int64),
        track_id=np.asarray([-7, -7, 9, 9], dtype=np.int64),
    )
    identity = EventIdentity(
        campaign_id="unit-test",
        shard_id=4,
        source_index=2,
        run=12,
        subrun=4,
        event=3,
        random_seed=12345,
    )
    return reco, semantic, simide, identity


def test_label_event_preserves_validated_science_and_dtypes() -> None:
    reco, semantic, simide, identity = _event_inputs()
    result = label_event(
        reco,
        semantic,
        simide,
        NeutrinoVertex(0.0, 0.0, 0.0, True),
        identity,
        0,
        LabelingConfig(max_distance_cm=0.1, blob_grow_cm=0.0),
    )
    arrays = result.to_mapping()

    npt.assert_array_equal(result.is_nu, np.asarray([1, 1, 0, 0], dtype=np.int16))
    npt.assert_array_equal(result.origin_label, np.asarray([0, 0, 1, 1], dtype=np.int16))
    npt.assert_array_equal(arrays["truth_tid_points_direct_raw"], [-7, -7, 9, 9])
    npt.assert_array_equal(arrays["truth_tid_points_direct"], [7, 7, 9, 9])
    npt.assert_array_equal(arrays["truth_tid_points"], [7, 7, 9, 9])
    npt.assert_array_equal(result.truth_blob_tid, [7, 9])
    npt.assert_array_equal(result.truth_blob_support, [2, 2])
    npt.assert_allclose(result.truth_blob_purity, [1.0, 1.0], rtol=0, atol=0)
    npt.assert_allclose(result.vtx_dist, [0.0, 1.0, 10.0, 11.0], rtol=0, atol=0)
    assert result.is_nu.dtype == np.int16
    assert arrays["truth_tid_points"].dtype == np.int32
    assert result.truth_blob_purity.dtype == np.float32
    assert result.truth_blob_support.dtype == np.int16
    assert result.path is None
    assert result.edge_index is None and result.edge_y is None


def test_event_identity_is_immutable_and_split_is_apa_independent() -> None:
    *_, identity = _event_inputs()
    assert identity.sample_name(0) != identity.sample_name(1)
    assert identity.physical_key == ("unit-test", 12, 4, 3)
    assert identity.split() in {"train", "validation", "test"}
    with pytest.raises(FrozenInstanceError):
        identity.event = 9  # type: ignore[misc]


def test_label_event_rejects_mismatched_apa_family() -> None:
    reco, semantic, simide, identity = _event_inputs()
    with pytest.raises(ValueError, match="does not match"):
        label_event(
            reco,
            semantic,
            simide,
            NeutrinoVertex(0.0, 0.0, 0.0, True),
            identity,
            1,
        )
