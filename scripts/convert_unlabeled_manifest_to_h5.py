#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

# Direct script execution places scripts/, not the repository root, on
# sys.path. Add only this checkout's root so the integrated pywcml is used.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pywcml.config import ConversionConfig
from pywcml.converter import WCMLConverter


class InferenceOnlyConverter(WCMLConverter):
    """Build normal reconstruction graphs without fabricating truth labels."""

    def _truth_instance_by_blob(self, arrays, n_blobs, fallback):
        return np.full(int(n_blobs), -1, dtype=np.int64)

    def _label_blob_edges(
        self,
        edge_index,
        truth_instance_by_blob,
        semantic,
        vtx_dist_by_blob,
    ):
        edge_index = np.asarray(edge_index, dtype=np.int64)

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            edge_index = np.empty((2, 0), dtype=np.int64)

        n_edges = int(edge_index.shape[1])
        return (
            edge_index,
            np.zeros(n_edges, dtype=np.int64),
            np.zeros(n_edges, dtype=np.int64),
        )

    def convert_unlabeled(
        self,
        npz_path,
        sample_name,
        run,
        subrun,
        event,
    ):
        name, graph = super().convert(npz_path, sample_name=sample_name)

        # The base converter temporarily treats missing semantic truth as cosmic
        # so reconstruction nodes and message-passing edges are not pruned.
        # Remove those temporary labels before writing the inference dataset.
        for store_name in ("sp",) + tuple(self.config.plane_names()):
            store = graph[store_name]

            if hasattr(store, "y_semantic"):
                store.y_semantic = torch.full_like(store.y_semantic, -1)

            if hasattr(store, "y_instance"):
                store.y_instance = torch.full_like(store.y_instance, -1)

            if hasattr(store, "pid"):
                store.pid = torch.full_like(store.pid, -1)

        sp = graph["sp"]

        if hasattr(sp, "edge_y"):
            sp.edge_y = torch.zeros_like(sp.edge_y)

        if hasattr(sp, "edge_labelable"):
            sp.edge_labelable = torch.zeros_like(sp.edge_labelable)

        graph["evt"].y = torch.tensor([-1], dtype=torch.long)

        graph["metadata"].run = int(run)
        graph["metadata"].subrun = int(subrun)
        graph["metadata"].event = int(event)

        return name, graph


def read_process_map(path):
    result = {}

    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        fields = line.split()
        if len(fields) < 6:
            continue

        try:
            process = int(fields[0])
            run = int(fields[1])
            subrun = int(fields[2])
            event = int(fields[3])
            index = int(fields[-1])
        except ValueError:
            continue

        result[process] = {
            "run": run,
            "subrun": subrun,
            "event": event,
            "index": index,
        }

    return result


def identify_sample(path, process_map):
    path_string = str(path)

    apa_match = re.search(r"rec-apa([01])-(\d+)\.npz$", path.name)
    if not apa_match:
        raise RuntimeError("Cannot determine APA/index from %s" % path)

    apa = int(apa_match.group(1))
    npz_index = int(apa_match.group(2))

    retry_match = re.search(
        r"Data_NCSideband_exact_RSE(\d+)_(\d+)_(\d+)_v2",
        path_string,
    )

    if retry_match:
        run = int(retry_match.group(1))
        subrun = int(retry_match.group(2))
        event = int(retry_match.group(3))
    else:
        process_match = re.search(r"/29354246_(\d+)/", path_string)
        if not process_match:
            raise RuntimeError("Cannot map staged path to RSE: %s" % path)

        process = int(process_match.group(1))
        if process not in process_map:
            raise RuntimeError("Process %d missing from process map" % process)

        record = process_map[process]
        run = record["run"]
        subrun = record["subrun"]
        event = record["event"]

        if npz_index != record["index"]:
            raise RuntimeError(
                "NPZ index mismatch for process %d: path=%d map=%d"
                % (process, npz_index, record["index"])
            )

    sample_name = "data_r%d_s%d_e%d_apa%d" % (
        run,
        subrun,
        event,
        apa,
    )

    return {
        "sample_name": sample_name,
        "path": str(path),
        "run": run,
        "subrun": subrun,
        "event": event,
        "apa": apa,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--process-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = [
        Path(line.strip())
        for line in Path(args.manifest).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    if args.limit is not None:
        paths = paths[:args.limit]

    if not paths:
        raise RuntimeError("Input manifest is empty")

    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("Missing staged files:\n" + "\n".join(missing))

    process_map = read_process_map(args.process_map)
    records = [identify_sample(path, process_map) for path in paths]

    names = [record["sample_name"] for record in records]
    if len(names) != len(set(names)):
        raise RuntimeError("Duplicate HDF5 sample names: %s" % names)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.overwrite:
        raise RuntimeError(
            "Output already exists: %s\nUse --overwrite only intentionally."
            % output
        )

    temporary = Path(str(output) + ".partial")
    if temporary.exists():
        temporary.unlink()

    config = ConversionConfig(write_diagnostics=args.diagnostics)
    converter = InferenceOnlyConverter(config)

    graphs = {}

    for index, record in enumerate(records, 1):
        name, graph = converter.convert_unlabeled(
            record["path"],
            sample_name=record["sample_name"],
            run=record["run"],
            subrun=record["subrun"],
            event=record["event"],
        )

        candidate_edges = int(graph["sp"].edge_label_index.shape[1])
        message_edges = int(
            graph["sp", "nexus", "sp"].edge_index.shape[1]
        )

        print(
            "[%d/%d] %s blobs=%d candidate_edges=%d message_edges=%d"
            % (
                index,
                len(records),
                name,
                int(graph["sp"].pos.shape[0]),
                candidate_edges,
                message_edges,
            ),
            flush=True,
        )

        graphs[name] = graph

    first_name = names[0]

    # All data samples belong to test. One duplicated split entry keeps the
    # existing DataModule's fit/setup initialization satisfied.
    splits = {
        "train": [first_name],
        "validation": [first_name],
        "val": [first_name],
        "test": names,
    }

    converter.write_hdf5(graphs, temporary, splits=splits)

    with h5py.File(temporary, "a") as output_file:
        output_file.attrs["inference_only"] = 1
        output_file.attrs["has_semantic_truth"] = 0
        output_file.attrs["has_instance_truth"] = 0
        output_file.attrs["source"] = "SBND data NC sideband"

        provenance = output_file.create_group("provenance")
        string_type = h5py.string_dtype(encoding="utf-8")

        provenance.create_dataset(
            "sample_name",
            data=np.asarray(names, dtype=object),
            dtype=string_type,
        )
        provenance.create_dataset(
            "source_path",
            data=np.asarray(
                [record["path"] for record in records],
                dtype=object,
            ),
            dtype=string_type,
        )

        for key in ("run", "subrun", "event", "apa"):
            provenance.create_dataset(
                key,
                data=np.asarray(
                    [record[key] for record in records],
                    dtype=np.int64,
                ),
            )

    if output.exists():
        output.unlink()
    temporary.replace(output)

    print("Wrote:", output)
    print("Samples:", len(names))
    print("Unique RSEs:", len(set(
        (r["run"], r["subrun"], r["event"]) for r in records
    )))


if __name__ == "__main__":
    main()
