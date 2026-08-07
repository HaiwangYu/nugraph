"""Incremental, failure-safe writer for one final NuGraph HDF5 job output."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

import h5py
import numpy as np
import torch

from pynuml.data import NuGraphData

from .config import ConversionConfig
from .identity import EventIdentity


class StreamingH5Writer:
    """Append paired APA graphs directly to the final HDF5 under construction.

    The only file created before successful completion is ``<output>.partial``.
    A successful :meth:`finalize` validates that file and atomically renames it
    to ``output``. No per-event/chunk HDF5 files or merge step are used.
    """

    def __init__(
        self,
        output: Path | str,
        config: Optional[ConversionConfig] = None,
    ) -> None:
        self.output = Path(output)
        self.partial = Path(f"{self.output}.partial")
        self.config = config or ConversionConfig()
        if self.output.exists():
            raise FileExistsError(f"final output already exists: {self.output}")
        if self.partial.exists():
            raise FileExistsError(f"partial output already exists: {self.partial}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.partial, "x")
        self._file.create_dataset(
            "planes",
            data=np.asarray(self.config.plane_names(), dtype=h5py.string_dtype()),
        )
        self._file.create_dataset(
            "semantic_classes",
            data=np.asarray(self.config.semantic_classes, dtype=h5py.string_dtype()),
        )
        self._file.create_dataset("gen", data=np.asarray([3], dtype=np.int64))
        self._file.create_group("samples")
        self._file.create_group("datasize")
        self._file.create_group("dataset")
        self._sample_names = {"train": [], "validation": [], "test": []}
        self._train_sizes = []
        self._written = set()
        self._finalized = False

    def append_event(
        self,
        identity: EventIdentity,
        apa0: NuGraphData,
        apa1: NuGraphData,
        *,
        sample_names: Optional[Mapping[int, str]] = None,
    ) -> tuple[str, str]:
        """Append APA0 and APA1 as one physical-event transaction."""

        self._require_open()
        graphs = {0: apa0, 1: apa1}
        names = {
            apa: (
                sample_names[apa]
                if sample_names is not None and apa in sample_names
                else identity.sample_name(apa)
            )
            for apa in (0, 1)
        }
        if names[0] == names[1]:
            raise ValueError("APA0 and APA1 sample names must be distinct")
        for apa, graph in graphs.items():
            self._validate_graph_identity(graph, identity, apa)
            if names[apa] in self._written or f"dataset/{names[apa]}" in self._file:
                raise ValueError(f"duplicate sample name: {names[apa]}")

        paths = [f"dataset/{names[apa]}" for apa in (0, 1)]
        try:
            for apa, path in zip((0, 1), paths):
                graphs[apa].save(self._file, path)
        except Exception:
            # NuGraphData.save may have created its compound dataset before an
            # exception is raised, so inspect both intended transaction paths
            # rather than only the calls that returned successfully.
            for path in paths:
                if path in self._file:
                    del self._file[path]
            self._file.flush()
            raise

        split = identity.split(
            train_fraction=self.config.train_fraction,
            val_fraction=self.config.val_fraction,
        )
        for apa in (0, 1):
            name = names[apa]
            self._sample_names[split].append(name)
            self._written.add(name)
            if split == "train":
                self._train_sizes.append(_graph_size_bytes(graphs[apa]))
        self._file.flush()
        return names[0], names[1]

    def finalize(self) -> Path:
        """Finish schema metadata, validate, and atomically publish the file."""

        self._require_open()
        samples = self._file["samples"]
        string_dtype = h5py.string_dtype()
        for split in ("train", "validation", "test"):
            samples.create_dataset(
                split,
                data=np.asarray(self._sample_names[split], dtype=string_dtype),
            )
        self._file["datasize"].create_dataset(
            "train",
            data=np.asarray(self._train_sizes, dtype=np.int64),
        )
        self._file.flush()
        self._file.close()
        self._file = None

        _validate_hdf5(self.partial, expected_samples=self._written)
        with open(self.partial, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(self.partial, self.output)
        directory_fd = os.open(self.output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self._finalized = True
        return self.output

    def close(self) -> None:
        """Close without publishing; the failed/incomplete ``.partial`` remains."""

        if self._file is not None:
            self._file.close()
            self._file = None

    def _require_open(self) -> None:
        if self._finalized:
            raise RuntimeError("writer is already finalized")
        if self._file is None:
            raise RuntimeError("writer is closed")

    @staticmethod
    def _validate_graph_identity(graph: NuGraphData, identity: EventIdentity, apa: int) -> None:
        metadata = graph["metadata"]
        actual = (int(metadata.run), int(metadata.subrun), int(metadata.event))
        expected = (identity.run, identity.subrun, identity.event)
        if actual != expected:
            raise ValueError(
                f"APA{apa} graph metadata {actual} does not match physical event {expected}"
            )

    def __enter__(self) -> "StreamingH5Writer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is None and not self._finalized:
            self.finalize()
        else:
            self.close()


def _graph_size_bytes(graph: NuGraphData) -> int:
    total = 0
    for store in graph.stores:  # type: ignore[attr-defined]
        for value in store.values():
            if isinstance(value, torch.Tensor):
                total += value.element_size() * value.nelement()
    return total


def _validate_hdf5(path: Path, expected_samples: set[str]) -> None:
    required_top = {"planes", "semantic_classes", "gen", "samples", "datasize", "dataset"}
    required_fields = {
        "metadata/run",
        "metadata/subrun",
        "metadata/event",
        "sp/pos",
        "sp/features",
        "sp/y_semantic",
        "sp/y_instance",
        "sp/raw_vtx_dist",
        "sp/edge_label_index",
        "sp/edge_y",
        "sp/edge_labelable",
        "sp_nexus_sp/edge_index",
        "evt/y",
    }
    with h5py.File(path, "r") as h5file:
        missing_top = required_top - set(h5file.keys())
        if missing_top:
            raise RuntimeError(f"incomplete HDF5 schema; missing {sorted(missing_top)}")
        if set(h5file["samples"].keys()) != {"train", "validation", "test"}:
            raise RuntimeError("incomplete HDF5 sample-split schema")
        if "train" not in h5file["datasize"]:
            raise RuntimeError("missing datasize/train")
        dataset_names = set(h5file["dataset"].keys())
        if dataset_names != expected_samples:
            raise RuntimeError(
                f"dataset/sample mismatch: expected {len(expected_samples)}, got {len(dataset_names)}"
            )
        split_names = []
        for split in ("train", "validation", "test"):
            split_names.extend(h5file[f"samples/{split}"].asstr()[()].tolist())
        if set(split_names) != expected_samples or len(split_names) != len(expected_samples):
            raise RuntimeError("sample splits do not contain every graph exactly once")
        if len(h5file["datasize/train"]) != len(h5file["samples/train"]):
            raise RuntimeError("datasize/train is not aligned with samples/train")
        for name in expected_samples:
            fields = set(h5file[f"dataset/{name}"].dtype.names or ())
            missing_fields = required_fields - fields
            if missing_fields:
                raise RuntimeError(f"sample {name} is missing fields {sorted(missing_fields)}")


__all__ = ["StreamingH5Writer"]
