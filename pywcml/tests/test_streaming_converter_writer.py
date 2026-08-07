from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import numpy.testing as npt
import pytest
import torch

from nugraph.data import NuGraphDataModule, NuGraphDataset

from pywcml.config import default_planes
from pywcml.converter import WCMLConverter
from pywcml.geometry import project_corners
from pywcml.h5writer import StreamingH5Writer
from pywcml.identity import EventIdentity
from pywcml.io import WCMLArrays


def _arrays(apa: int) -> WCMLArrays:
    n_blobs = 12
    xyz = np.column_stack(
        [
            np.arange(n_blobs, dtype=np.float32) * 100.0,
            np.asarray([index % 3 for index in range(n_blobs)], dtype=np.float32) * 20.0,
            np.asarray([index // 3 for index in range(n_blobs)], dtype=np.float32) * 15.0,
        ]
    )
    points = np.zeros((n_blobs, 6), dtype=np.float32)
    points[:, :3] = xyz
    points[:, 4] = np.arange(n_blobs)
    points[:, 5] = np.arange(n_blobs) // 3
    blobs = np.zeros((n_blobs, 5), dtype=np.float32)
    blobs[:, :2] = 1.0
    blobs[:, 2:5] = xyz
    ctpc = {}
    for plane_name, spec in default_planes(apa).items():
        plane = np.zeros((n_blobs, 7), dtype=np.float32)
        plane[:, 0] = xyz[:, 0]
        plane[:, 1] = np.asarray(
            [project_corners(point[None, :], spec.angle_rad)[0] for point in xyz],
            dtype=np.float32,
        )
        plane[:, 2] = 1.0
        plane[:, 3] = 0.1
        ctpc[plane_name] = plane
    return WCMLArrays(
        blobs=blobs,
        points=points,
        ctpc=ctpc,
        is_nu=np.asarray([1, 1, 0, 0] * 3, dtype=np.int16),
        ppedges=np.asarray([[index, index + 1, 0] for index in range(n_blobs - 1)], dtype=np.int64),
        origin_label=np.asarray([0, 0, 1, 1] * 3, dtype=np.int16),
        vtx_dist=np.arange(n_blobs, dtype=np.float32),
        vtx_dx=np.zeros(n_blobs, dtype=np.float32),
        vtx_dy=np.zeros(n_blobs, dtype=np.float32),
        vtx_dz=np.zeros(n_blobs, dtype=np.float32),
        nu_vtx=np.zeros(3, dtype=np.float32),
        nu_vtx_found=np.ones(1, dtype=np.int16),
        truth_blob_tid=np.arange(n_blobs, dtype=np.int32) // 3,
        truth_blob_purity=np.ones(n_blobs, dtype=np.float32),
        truth_blob_support=np.ones(n_blobs, dtype=np.int16),
        edge_index=np.asarray([[0], [11]], dtype=np.int64),
        edge_y=np.ones(1, dtype=np.int8),
        path=None,
    )


def _identity() -> EventIdentity:
    identity = EventIdentity(
        campaign_id="sophia-smoke",
        shard_id=171396,
        source_index=0,
        run=1,
        subrun=171396,
        event=0,
        random_seed=123,
    )
    assert identity.split() == "train"
    return identity


def _assert_graph_equal(first, second) -> None:
    assert len(first.stores) == len(second.stores)
    for first_store, second_store in zip(first.stores, second.stores):
        assert first_store._key == second_store._key
        assert set(first_store.keys()) == set(second_store.keys())
        for key in first_store.keys():
            first_value, second_value = first_store[key], second_store[key]
            if isinstance(first_value, torch.Tensor):
                assert torch.equal(first_value, second_value), (first_store._key, key)
            else:
                assert first_value == second_value


def _save_npz(path: Path, arrays: WCMLArrays) -> None:
    np.savez(path, **arrays.to_mapping())


def test_convert_arrays_matches_legacy_loader_and_overrides_path(tmp_path: Path) -> None:
    converter = WCMLConverter()
    identity = _identity()
    for apa in (0, 1):
        arrays = _arrays(apa)
        path = tmp_path / f"rec-lab-apa{apa}-0.npz"
        _save_npz(path, arrays)
        sample_name = identity.sample_name(apa)
        legacy_name, legacy_graph = converter.convert(path, sample_name=sample_name)
        streaming_graph = converter.convert_arrays(arrays, identity, apa)
        assert legacy_name == sample_name
        _assert_graph_equal(legacy_graph, streaming_graph)
        assert streaming_graph["metadata"].run == identity.run
        assert streaming_graph["metadata"].subrun == identity.subrun
        assert streaming_graph["metadata"].event == identity.event


def test_incremental_writer_matches_schema_and_existing_readers(tmp_path: Path) -> None:
    converter = WCMLConverter()
    identity = _identity()
    graphs = {apa: converter.convert_arrays(_arrays(apa), identity, apa) for apa in (0, 1)}
    names = [identity.sample_name(apa) for apa in (0, 1)]
    output = tmp_path / "streamed.h5"
    writer = StreamingH5Writer(output, converter.config)
    written = writer.append_event(identity, graphs[0], graphs[1])
    assert list(written) == names
    assert writer.partial.exists()
    assert not output.exists()
    writer.finalize()
    assert output.exists()
    assert not Path(f"{output}.partial").exists()

    reference = tmp_path / "reference.h5"
    converter.write_hdf5(
        {names[0]: graphs[0], names[1]: graphs[1]},
        reference,
        splits={"train": names, "validation": [], "test": []},
    )
    _assert_hdf5_equal(reference, output)

    for name in names:
        loaded = NuGraphDataset(str(output), [name]).get(0)
        assert loaded["sp"].num_nodes == graphs[0]["sp"].num_nodes
    data_module = NuGraphDataModule(
        data_path=str(output),
        model=None,
        batch_size=1,
        num_workers=0,
        shuffle="random",
    )
    assert len(data_module.train_dataset) == 2
    assert len(data_module.val_dataset) == 0
    assert len(data_module.test_dataset) == 0


def test_paired_event_write_rolls_back_both_apas_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    converter = WCMLConverter()
    identity = _identity()
    graphs = {apa: converter.convert_arrays(_arrays(apa), identity, apa) for apa in (0, 1)}
    graph_type = type(graphs[0])
    original_save = graph_type.save

    def fail_after_apa1_save(graph, h5file, path) -> None:
        original_save(graph, h5file, path)
        if "apa1" in path:
            raise RuntimeError("injected APA1 write failure")

    monkeypatch.setattr(graph_type, "save", fail_after_apa1_save)
    writer = StreamingH5Writer(tmp_path / "failed.h5", converter.config)
    with pytest.raises(RuntimeError, match="injected APA1"):
        writer.append_event(identity, graphs[0], graphs[1])
    assert list(writer._file["dataset"].keys()) == []
    assert writer._sample_names == {"train": [], "validation": [], "test": []}
    assert writer._written == set()
    writer.close()


def _assert_hdf5_equal(first_path: Path, second_path: Path) -> None:
    with h5py.File(first_path, "r") as first, h5py.File(second_path, "r") as second:
        first_objects, second_objects = {}, {}

        def collect_first(name, obj) -> None:
            first_objects[name] = type(obj)

        def collect_second(name, obj) -> None:
            second_objects[name] = type(obj)

        first.visititems(collect_first)
        second.visititems(collect_second)
        assert first_objects == second_objects
        for name, object_type in first_objects.items():
            if object_type is not h5py.Dataset:
                continue
            first_dataset, second_dataset = first[name], second[name]
            assert first_dataset.shape == second_dataset.shape
            assert first_dataset.dtype == second_dataset.dtype
            first_value, second_value = first_dataset[()], second_dataset[()]
            if first_dataset.dtype.metadata and "vlen" in first_dataset.dtype.metadata:
                npt.assert_array_equal(first_dataset.asstr()[()], second_dataset.asstr()[()])
            else:
                npt.assert_array_equal(first_value, second_value)
