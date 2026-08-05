#!/usr/bin/env python
"""Export NuGraph predictions as an upload-ready BEE event set."""

import argparse
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

import h5py
import numpy as np
import torch


DEFAULT_DATA = (
    "/lus/eagle/projects/neutrinoGPU/abhat/sbnd/clustering/"
    "converted_data_NCSideband_exact_19events_geom_edges_inference.h5"
)
DEFAULT_CHECKPOINT = (
    "/lus/eagle/projects/neutrinoGPU/abhat/sbnd/clustering/nugraph/notebooks/log/"
    "N4_nw0_bs_4_lr3e4_nuhits_0_bf0p1_tf1p0_if4_hf256_nf64_intf32_nit10_"
    "shuffle_random_ledg0p03_epw1p0_lemb0p3_lcoh0_"
    "converted_labeled_samples_merged_350k_sophia/checkpoints/best-joint-f1.ckpt"
)
DEFAULT_OUTPUT = "bee_data_nc_sideband_best_joint_edge0p3_nuthr0p612"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run NuGraph inference and package semantic and instance predictions "
            "in the JSON/ZIP layout accepted by BEE."
        )
    )
    parser.add_argument("--data-path", default=DEFAULT_DATA)
    parser.add_argument("--ckpt", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("train", "val", "validation", "test"), default="test")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--zip-path", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--in-features", type=int, default=None)
    parser.add_argument("--nu-thr", type=float, default=0.612)
    parser.add_argument("--edge-thr", type=float, default=0.30)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--skip-events", type=int, default=0)
    parser.add_argument("--limit-events", type=int, default=None)
    parser.add_argument("--geom", default="sbnd")
    parser.add_argument("--coordinate-precision", type=int, default=3)
    parser.add_argument("--omit-charge", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message):
    print(message, flush=True)


def decode_text(value):
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def parse_apa_from_sample_name(sample_name):
    match = re.search(r"_apa(\d+)$", sample_name)
    return int(match.group(1)) if match else None


def load_checkpoint_hyperparameters(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("hyper_parameters", {})


def read_split_metadata(path, split):
    split_name = "val" if split == "validation" else split
    with h5py.File(path, "r") as data:
        split_path = f"samples/{split_name}"
        if split_path not in data:
            raise KeyError(f"HDF5 does not contain {split_path}")

        sample_names = [decode_text(value) for value in data[split_path][:]]
        metadata = OrderedDict()
        for sample_name in sample_names:
            record = data["dataset"][sample_name][()]
            positions = np.asarray(record["sp/pos"], dtype=np.float64)
            metadata[sample_name] = {
                "run": int(record["metadata/run"]),
                "subrun": int(record["metadata/subrun"]),
                "event": int(record["metadata/event"]),
                "n_sp": int(positions.shape[0]),
                "position_sum": positions.sum(axis=0),
                "position_min": positions.min(axis=0),
                "position_max": positions.max(axis=0),
            }
            if "metadata/apa" in record.dtype.names:
                metadata[sample_name]["apa"] = int(record["metadata/apa"])
            else:
                metadata[sample_name]["apa"] = parse_apa_from_sample_name(sample_name)

        inference_only = bool(int(data.attrs.get("inference_only", 0)))
        source = decode_text(data.attrs.get("source", ""))

    return sample_names, metadata, inference_only, source


def select_physical_events(sample_names, metadata, skip_events, limit_events):
    ordered_rses = []
    seen = set()
    for sample_name in sample_names:
        item = metadata[sample_name]
        rse = (item["run"], item["subrun"], item["event"])
        if rse not in seen:
            seen.add(rse)
            ordered_rses.append(rse)

    start = max(0, int(skip_events))
    stop = None if limit_events is None else start + max(0, int(limit_events))
    return ordered_rses[start:stop]


def identify_sample(positions, metadata, used_samples):
    positions = np.asarray(positions, dtype=np.float64)
    signature = {
        "n_sp": int(positions.shape[0]),
        "position_sum": positions.sum(axis=0),
        "position_min": positions.min(axis=0),
        "position_max": positions.max(axis=0),
    }
    matches = []
    for sample_name, item in metadata.items():
        if sample_name in used_samples or item["n_sp"] != signature["n_sp"]:
            continue
        if all(
            np.allclose(signature[key], item[key], rtol=1e-5, atol=1e-2)
            for key in ("position_sum", "position_min", "position_max")
        ):
            matches.append(sample_name)
    if len(matches) != 1:
        raise RuntimeError(
            "Could not uniquely match a DataLoader batch to an HDF5 sample; "
            f"found {len(matches)} candidates"
        )
    return matches[0]


def run_model_inference(model, batch, inference_only):
    if not inference_only:
        return model(batch, stage="test")

    supervision_key = ("sp", "supervision", "sp")
    if supervision_key not in batch.edge_types:
        raise RuntimeError("Inference HDF5 is missing the SP supervision candidate-edge store")

    sp = batch["sp"]
    edge_store = batch[supervision_key]
    original = {
        "y_semantic": getattr(sp, "y_semantic", None),
        "y_instance": getattr(sp, "y_instance", None),
        "pid": getattr(sp, "pid", None),
        "edge_labelable": edge_store.edge_labelable,
        "edge_y": edge_store.edge_y,
    }

    try:
        for name in ("y_semantic", "y_instance", "pid"):
            if original[name] is not None:
                setattr(sp, name, torch.zeros_like(original[name]))
        edge_store.edge_labelable = torch.ones_like(original["edge_labelable"])
        edge_store.edge_y = torch.zeros_like(original["edge_y"])
        return model(batch, stage=None)
    finally:
        for name in ("y_semantic", "y_instance", "pid"):
            if original[name] is not None:
                setattr(sp, name, original[name])
        edge_store.edge_labelable = original["edge_labelable"]
        edge_store.edge_y = original["edge_y"]


def project_hit_semantic_to_sp(batch, nu_threshold):
    hit = batch["hit"]
    sp = batch["sp"]
    hit_probabilities = hit.x_semantic.detach().cpu().numpy()
    hit_predictions = 1 - (hit_probabilities[:, 0] >= float(nu_threshold)).astype(np.int64)
    nexus_key = ("hit", "nexus", "sp")
    if nexus_key not in batch.edge_types:
        return np.full(sp.num_nodes, -1, dtype=np.int64)

    nexus_edges = batch[nexus_key].edge_index.detach().cpu().numpy()
    hit_batch = hit.batch.detach().cpu().numpy()
    sp_batch = sp.batch.detach().cpu().numpy()
    predictions_by_sp = defaultdict(list)

    for hit_index, sp_index in nexus_edges.T.tolist():
        if hit_batch[hit_index] == sp_batch[sp_index]:
            predictions_by_sp[sp_index].append(int(hit_predictions[hit_index]))

    predictions = np.full(sp.num_nodes, -1, dtype=np.int64)
    for sp_index, values in predictions_by_sp.items():
        predictions[sp_index] = Counter(values).most_common(1)[0][0]
    return predictions


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.size = [1] * size

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, first, second):
        first_root = self.find(int(first))
        second_root = self.find(int(second))
        if first_root == second_root:
            return
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.size[first_root] += self.size[second_root]


def cluster_predicted_edges(edge_index, edge_score, batch_sp, threshold, min_cluster_size):
    num_sp = int(batch_sp.shape[0])
    union_find = UnionFind(num_sp)
    edge_index = np.asarray(edge_index, dtype=np.int64)
    edge_score = np.asarray(edge_score, dtype=np.float32).reshape(-1)

    if edge_index.ndim == 2 and edge_index.shape[0] == 2:
        keep = edge_score >= float(threshold)
        for source, target in edge_index[:, keep].T.tolist():
            if (
                0 <= source < num_sp
                and 0 <= target < num_sp
                and batch_sp[source] == batch_sp[target]
            ):
                union_find.union(source, target)

    roots = np.asarray([union_find.find(index) for index in range(num_sp)], dtype=np.int64)
    counts = Counter(roots.tolist())
    predictions = np.empty_like(roots)
    component_ids = {}
    next_id = 0

    for index, root in enumerate(roots.tolist()):
        if counts[root] < int(min_cluster_size):
            predictions[index] = next_id
            next_id += 1
        else:
            if root not in component_ids:
                component_ids[root] = next_id
                next_id += 1
            predictions[index] = component_ids[root]
    return predictions


def remap_local_clusters(labels):
    labels = np.asarray(labels, dtype=np.int64)
    output = np.full(labels.shape, -1, dtype=np.int64)
    for new_id, old_id in enumerate(np.unique(labels[labels >= 0]).tolist()):
        output[labels == old_id] = new_id
    return output


def remap_clusters_by_size(labels):
    labels = np.asarray(labels, dtype=np.int64)
    output = np.zeros(labels.shape, dtype=np.int64)
    valid = labels >= 0
    if not np.any(valid):
        return output

    unique, counts = np.unique(labels[valid], return_counts=True)
    order = sorted(zip(unique.tolist(), counts.tolist()), key=lambda item: (-item[1], item[0]))
    for bee_id, (old_id, _) in enumerate(order, start=1):
        output[labels == old_id] = bee_id
    return output


def new_event_accumulator(run, subrun, event):
    return {
        "run": run,
        "subrun": subrun,
        "event": event,
        "positions": [],
        "charge": [],
        "semantic": [],
        "instances": [],
        "reco_clusters": [],
        "sample_names": [],
        "apas": [],
        "instance_offset": 0,
        "reco_offset": 0,
    }


def append_sample(event, sample_name, apa, positions, charge, semantic, instances, reco_clusters):
    instances = remap_local_clusters(instances)
    valid_instances = instances >= 0
    if np.any(valid_instances):
        instances[valid_instances] += event["instance_offset"]
        event["instance_offset"] = int(instances[valid_instances].max()) + 1

    reco_clusters = remap_local_clusters(reco_clusters)
    valid_reco = reco_clusters >= 0
    if np.any(valid_reco):
        reco_clusters[valid_reco] += event["reco_offset"]
        event["reco_offset"] = int(reco_clusters[valid_reco].max()) + 1

    event["positions"].append(np.asarray(positions, dtype=np.float32))
    event["charge"].append(np.asarray(charge, dtype=np.float32))
    event["semantic"].append(np.asarray(semantic, dtype=np.int64))
    event["instances"].append(instances)
    event["reco_clusters"].append(reco_clusters)
    event["sample_names"].append(sample_name)
    event["apas"].append(apa)


def finish_event(event):
    return {
        "run": event["run"],
        "subrun": event["subrun"],
        "event": event["event"],
        "positions": np.concatenate(event["positions"], axis=0),
        "charge": np.concatenate(event["charge"], axis=0),
        "semantic": np.concatenate(event["semantic"], axis=0),
        "instances": remap_clusters_by_size(np.concatenate(event["instances"], axis=0)),
        "reco_clusters": remap_clusters_by_size(
            np.concatenate(event["reco_clusters"], axis=0)
        ),
        "sample_names": event["sample_names"],
        "apas": event["apas"],
    }


def rounded_list(values, precision):
    return np.round(
        np.asarray(values, dtype=np.float64), decimals=int(precision)
    ).tolist()


def make_bee_payload(event, mask, cluster_ids, args):
    positions_cm = event["positions"][mask] / 10.0
    payload = {
        "runNo": event["run"],
        "subRunNo": event["subrun"],
        "eventNo": event["event"],
        "x": rounded_list(positions_cm[:, 0], args.coordinate_precision),
        "y": rounded_list(positions_cm[:, 1], args.coordinate_precision),
        "z": rounded_list(positions_cm[:, 2], args.coordinate_precision),
        "cluster_id": np.asarray(cluster_ids[mask], dtype=np.int64).tolist(),
        "geom": args.geom,
        "type": "wire-cell",
    }
    if not args.omit_charge:
        charge = np.nan_to_num(event["charge"][mask], nan=0.0, posinf=0.0, neginf=0.0)
        payload["q"] = np.rint(np.clip(charge, 0.0, None)).astype(np.int64).tolist()

    if positions_cm.shape[0]:
        minima = positions_cm.min(axis=0)
        maxima = positions_cm.max(axis=0)
        payload["bounding_box"] = rounded_list(
            [minima[0], maxima[0], minima[1], maxima[1], minima[2], maxima[2]],
            args.coordinate_precision,
        )
    return payload


def write_json(path, payload):
    with path.open("w") as output:
        json.dump(payload, output, separators=(",", ":"), allow_nan=False)


def validate_payload(payload, path):
    lengths = {key: len(payload[key]) for key in ("x", "y", "z", "cluster_id")}
    if "q" in payload:
        lengths["q"] = len(payload["q"])
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Array-length mismatch in {path}: {lengths}")
    for coordinate in ("x", "y", "z"):
        if not np.isfinite(np.asarray(payload[coordinate], dtype=np.float64)).all():
            raise ValueError(f"Non-finite {coordinate} coordinate in {path}")


def write_bee_tree(events, output_dir, args):
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True)
    event_manifest = []

    for bee_index, event in enumerate(events):
        event_dir = data_dir / str(bee_index)
        event_dir.mkdir()
        all_points = np.ones(event["positions"].shape[0], dtype=bool)
        semantic_ids = np.where(event["semantic"] == 0, 1, np.where(event["semantic"] == 1, 2, 3))
        algorithms = OrderedDict(
            (
                ("NuGraphSemantic", (all_points, semantic_ids)),
                ("NuGraphInstances", (all_points, event["instances"])),
                ("WireCellRecoClusters", (all_points, event["reco_clusters"])),
            )
        )

        for algorithm, (mask, cluster_ids) in algorithms.items():
            payload = make_bee_payload(event, mask, cluster_ids, args)
            json_path = event_dir / f"{bee_index}-{algorithm}.json"
            validate_payload(payload, json_path)
            write_json(json_path, payload)

        semantic_counts = Counter(event["semantic"].tolist())
        event_manifest.append(
            {
                "bee_index": bee_index,
                "run": event["run"],
                "subrun": event["subrun"],
                "event": event["event"],
                "samples": event["sample_names"],
                "apas": event["apas"],
                "n_spacepoints": int(event["positions"].shape[0]),
                "n_nu_pred": int(semantic_counts.get(0, 0)),
                "n_cosmic_pred": int(semantic_counts.get(1, 0)),
                "n_unmapped": int(semantic_counts.get(-1, 0)),
                "n_predicted_instances": int(np.unique(event["instances"]).size),
                "n_reco_clusters": int(np.unique(event["reco_clusters"]).size),
            }
        )
    return event_manifest


def create_zip(data_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(data_dir.rglob("*.json")):
            archive.write(path, path.relative_to(data_dir.parent).as_posix())


def validate_zip(zip_path, event_count):
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if not names or any(not name.startswith("data/") for name in names):
            raise ValueError("BEE ZIP must contain only paths rooted at data/")
        for event_index in range(event_count):
            prefix = f"data/{event_index}/{event_index}-"
            if not any(name.startswith(prefix) and name.endswith(".json") for name in names):
                raise ValueError(f"BEE ZIP is missing event directory {event_index}")


def main():
    args = parse_args()
    args.split = "val" if args.split == "validation" else args.split
    data_path = Path(args.data_path).resolve()
    checkpoint_path = Path(args.ckpt).resolve()
    output_dir = Path(args.output_dir).resolve()
    zip_path = (
        Path(args.zip_path).resolve()
        if args.zip_path
        else output_dir.with_suffix(".zip")
    )

    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_dir}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    if zip_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"ZIP exists: {zip_path}; pass --overwrite to replace it")
        zip_path.unlink()

    sample_names, metadata, inference_only, source = read_split_metadata(data_path, args.split)
    selected_rses = select_physical_events(
        sample_names,
        metadata,
        args.skip_events,
        args.limit_events,
    )
    if not selected_rses:
        raise ValueError("No physical events selected")
    selected_set = set(selected_rses)

    hyperparameters = load_checkpoint_hyperparameters(checkpoint_path)
    in_features = args.in_features or hyperparameters.get("in_features") or 4
    log(f"[info] Data: {data_path}")
    log(f"[info] Checkpoint: {checkpoint_path}")
    log(f"[info] Split: {args.split}")
    log(f"[info] Selected physical events: {len(selected_rses)}")
    log(f"[info] Truth-free inference: {inference_only}")
    log(f"[info] Thresholds: nu={args.nu_thr}, edge={args.edge_thr}")
    log(f"[info] Device: {args.device}")
    if args.geom == "sbnd":
        log(
            "[note] The JSON records geom='sbnd'. Current upstream BEE falls back to the "
            "MicroBooNE detector outline for unknown geometries, but the point coordinates "
            "remain valid SBND global coordinates in cm."
        )

    import nugraph as ng

    model_class = ng.models.NuGraph4
    data_module = ng.data.NuGraphDataModule(
        model=model_class,
        data_path=str(data_path),
        in_features=int(in_features),
    )
    data_module.batch_size = 1
    data_module.num_workers = args.num_workers
    if hasattr(data_module, "shuffle"):
        data_module.shuffle = "none"
    data_module.setup("fit")
    data_module.setup("test")

    if args.split == "train":
        loader = data_module.train_dataloader()
    elif args.split == "val":
        loader = data_module.val_dataloader()
    else:
        loader = data_module.test_dataloader()

    model = model_class.load_from_checkpoint(str(checkpoint_path), map_location="cpu")
    model.eval().to(args.device)
    accumulators = OrderedDict(
        (rse, new_event_accumulator(*rse)) for rse in selected_rses
    )
    processed_samples = []
    seen_loader_samples = set()

    from tqdm import tqdm

    for batch_index, batch in enumerate(tqdm(loader, desc="NuGraph -> BEE")):
        positions = batch["sp"].pos.detach().cpu().numpy()
        sample_name = identify_sample(positions, metadata, seen_loader_samples)
        seen_loader_samples.add(sample_name)
        item = metadata[sample_name]
        rse = (item["run"], item["subrun"], item["event"])
        if rse not in selected_set:
            continue

        batch = batch.to(args.device)
        with torch.inference_mode():
            run_model_inference(model, batch, inference_only)

        sp = batch["sp"]
        positions = sp.pos.detach().cpu().numpy()
        if not hasattr(sp, "features") or sp.features.shape[1] < 2:
            raise RuntimeError("SP features must contain charge and reco-cluster ID columns")

        semantic = project_hit_semantic_to_sp(batch, args.nu_thr)
        edge_logits = sp.edge_logits.float().reshape(-1)
        edge_scores = torch.sigmoid(edge_logits).detach().cpu().numpy()
        instances = cluster_predicted_edges(
            sp.edge_index.detach().cpu().numpy(),
            edge_scores,
            sp.batch.detach().cpu().numpy(),
            args.edge_thr,
            args.min_cluster_size,
        )
        features = sp.features.detach().cpu().numpy()
        append_sample(
            accumulators[rse],
            sample_name,
            item["apa"],
            positions,
            features[:, 0],
            semantic,
            instances,
            np.rint(features[:, 1]).astype(np.int64),
        )
        processed_samples.append(sample_name)

        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    missing_rses = [rse for rse, event in accumulators.items() if not event["positions"]]
    if missing_rses:
        raise RuntimeError(f"No samples were processed for selected events: {missing_rses}")

    events = [finish_event(accumulators[rse]) for rse in selected_rses]
    output_dir.mkdir(parents=True)
    event_manifest = write_bee_tree(events, output_dir, args)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_h5": str(data_path),
        "source_description": source,
        "checkpoint": str(checkpoint_path),
        "checkpoint_hyperparameters": {
            key: hyperparameters.get(key)
            for key in (
                "in_features",
                "lambda_edge",
                "lambda_embed",
                "edge_pos_weight",
                "num_iters",
            )
            if key in hyperparameters
        },
        "split": args.split,
        "truth_available": not inference_only,
        "nu_threshold": args.nu_thr,
        "edge_threshold": args.edge_thr,
        "min_cluster_size": args.min_cluster_size,
        "coordinate_input_units": "mm",
        "coordinate_bee_units": "cm",
        "geom": args.geom,
        "algorithms": {
            "NuGraphSemantic": "cluster_id 1=nu, 2=cosmic, 3=unmapped",
            "NuGraphInstances": "NuGraph same-instance connected components",
            "WireCellRecoClusters": "input Wire-Cell reconstructed cluster identity",
        },
        "apa_handling": (
            "APA records sharing run/subrun/event are merged. Instance and reco-cluster "
            "IDs are offset before merging so IDs cannot collide across APAs."
        ),
        "events": event_manifest,
    }
    write_json(output_dir / "bee_export_manifest.json", manifest)
    create_zip(output_dir / "data", zip_path)
    validate_zip(zip_path, len(events))

    log(f"[done] BEE events: {len(events)}")
    log(f"[done] HDF5 samples used: {len(processed_samples)}")
    log(f"[done] Directory: {output_dir}")
    log(f"[done] Upload ZIP: {zip_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
