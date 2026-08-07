#!/usr/bin/env python3
"""Legacy NPZ/JSON/ROOT wrapper for the shared in-memory SBND labeler."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

# Keep direct execution from a source checkout functional without requiring an
# editable pywcml install in the legacy SL7 labeling environment.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import uproot
except Exception:
    uproot = None

from pywcml.labeling import (  # noqa: E402
    EventIdentity,
    LabelingConfig,
    NeutrinoVertex,
    RecoArrays,
    SemanticTruth,
    SimIDETruth,
    apply_reco_vtx_gate,
    blob_propagate_truth,
    build_truth_tid_points_direct,
    ctpc_col_candidates,
    detect_ctpc_family,
    label_event,
    match_ctpc_to_simide_trackid,
    merge_trunk_split_tids,
    remap_negative_trackids,
    semantic_labels,
    truth_q1_fraction_within_radius,
    write_edge_supervision_to_out,
)


def _open_celltree(tree_path):
    if uproot is None:
        raise RuntimeError("uproot is not available; install it or load it in your env.")
    if not os.path.exists(tree_path):
        raise FileNotFoundError(tree_path)
    root_file = uproot.open(tree_path)
    if "Event/Sim" not in root_file:
        keys = list(root_file.keys())
        raise KeyError(f"Could not find 'Event/Sim' in {tree_path}. Keys include: {keys[:20]}")
    return root_file["Event/Sim"]


def _read_one(tree, entry, branches):
    arrays = tree.arrays(branches, entry_start=entry, entry_stop=entry + 1, library="np")
    result = {}
    for key, value in arrays.items():
        result[key] = value[0] if isinstance(value, np.ndarray) and value.shape[0] == 1 else value
    return result


def get_beam_nu_vertex_from_celltree(celltree_path, entry):
    """Return the historical ``(x, y, z, found)`` tuple in cm."""

    tree = _open_celltree(celltree_path)
    branches = set(tree.keys())
    if "mc_nu_pos" in branches:
        value = _read_one(tree, entry, ["mc_nu_pos"])["mc_nu_pos"]
        try:
            value = np.asarray(value, dtype=float).reshape(-1)
            if value.size >= 3:
                return float(value[0]), float(value[1]), float(value[2]), True
        except Exception:
            pass
    candidates = ("mc_nu_pos_x", "mc_nu_pos_y", "mc_nu_pos_z")
    if all(candidate in branches for candidate in candidates):
        values = _read_one(tree, entry, list(candidates))
        try:
            return tuple(float(values[candidate]) for candidate in candidates) + (True,)
        except Exception:
            pass
    return 0.0, 0.0, 0.0, False


def load_simide_truth(celltree_path, entry) -> SimIDETruth:
    tree = _open_celltree(celltree_path)
    branches = [
        "simide_size",
        "simide_channelIdY",
        "simide_tdc",
        "simide_x",
        "simide_y",
        "simide_z",
        "simide_trackId",
    ]
    available = set(tree.keys())
    missing = [branch for branch in branches if branch not in available]
    if missing:
        raise KeyError(f"Missing branches in celltree: {missing}")
    arrays = tree.arrays(branches, entry_start=entry, entry_stop=entry + 1, library="np")
    values = {branch: arrays[branch][0] for branch in branches}
    size = int(values["simide_size"])
    for branch in branches[1:]:
        if len(values[branch]) != size:
            raise RuntimeError(f"SimIDE length mismatch for {branch}: got {len(values[branch])}, expected {size}")
    return SimIDETruth(
        channel=np.asarray(values["simide_channelIdY"]).astype(np.int64, copy=False),
        tdc=np.asarray(values["simide_tdc"]).astype(np.int64, copy=False),
        track_id=np.asarray(values["simide_trackId"]).astype(np.int64, copy=False),
        x_cm=np.asarray(values["simide_x"]).astype(np.float32, copy=False),
        y_cm=np.asarray(values["simide_y"]).astype(np.float32, copy=False),
        z_cm=np.asarray(values["simide_z"]).astype(np.float32, copy=False),
    )


def read_simide_arrays(celltree_path, entry):
    """Compatibility view retaining the historical dictionary keys."""

    truth = load_simide_truth(celltree_path, entry)
    return {
        "channel": truth.channel,
        "tdc": truth.tdc,
        "x_cm": truth.x_cm,
        "y_cm": truth.y_cm,
        "z_cm": truth.z_cm,
        "trackId": truth.track_id,
    }


def _load_reco(path: str | Path) -> RecoArrays:
    with np.load(path, allow_pickle=True) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return RecoArrays.from_mapping(arrays)


def _load_semantic_truth(path: str | Path) -> SemanticTruth:
    with open(path, "r", encoding="utf-8") as stream:
        return SemanticTruth.from_mapping(json.load(stream))


def _legacy_identity(celltree_path, entry) -> EventIdentity:
    run = subrun = 0
    event = int(entry)
    tree = _open_celltree(celltree_path)
    available = set(tree.keys())
    branches = [name for name in ("runNo", "subRunNo", "eventNo") if name in available]
    if branches:
        values = _read_one(tree, entry, branches)
        run = int(values.get("runNo", run))
        subrun = int(values.get("subRunNo", subrun))
        event = int(values.get("eventNo", event))
    return EventIdentity(
        campaign_id="legacy-file-cli",
        shard_id=0,
        source_index=int(entry),
        run=max(0, run),
        subrun=max(0, subrun),
        event=max(0, event),
        random_seed=123 + int(entry),
    )


def get_isnu_labels(
    truth_file,
    rec_file,
    max_distance_cm=5.0,
    z_offset_cm=0.0,
    tagging_alg="blob",
    blob_grow_cm=20.0,
):
    """Compatibility loader for the historical helper function."""

    return semantic_labels(
        _load_reco(rec_file),
        _load_semantic_truth(truth_file),
        max_distance_cm=max_distance_cm,
        z_offset_cm=z_offset_cm,
        tagging_alg=tagging_alg,
        blob_grow_cm=blob_grow_cm,
    )


def truth_q1_fraction_within_R(tru_file, vtx_cm, R_cm):
    return truth_q1_fraction_within_radius(_load_semantic_truth(tru_file), vtx_cm, R_cm)


def process_entry(
    entry,
    tru_prefix,
    rec_prefix,
    out_prefix,
    celltree_path,
    max_distance_cm=5.0,
    z_offset_cm=0.0,
    tagging_alg="blob",
    blob_grow_cm=20.0,
    vtx_gate_reco_cm=80.0,
    use_truth_gate=False,
    vtx_gate_truth_cm=50.0,
    do_instance=True,
    shift_p0=2992,
    shift_p1=2992,
    shift_p2=2992,
    dtdc_win=1,
    conf_thr=0.7,
    purity_thr=0.85,
    min_blob_support=1,
    max_dist_mm=5.0,
    write_edge_sup=False,
    balance_edges=False,
    neg_radius_mm=60.0,
):
    """Load historical files, invoke ``label_event``, and write legacy NPZ."""

    truth_file = f"{tru_prefix}-{entry}.json"
    reco_file = f"{rec_prefix}-{entry}.npz"
    output_file = f"{out_prefix}-{entry}.npz"
    if not os.path.exists(truth_file):
        raise FileNotFoundError(truth_file)
    if not os.path.exists(reco_file):
        raise FileNotFoundError(reco_file)

    reco = _load_reco(reco_file)
    semantic_truth = _load_semantic_truth(truth_file)
    vertex_values = get_beam_nu_vertex_from_celltree(celltree_path, entry)
    vertex = NeutrinoVertex(*vertex_values)
    simide = load_simide_truth(celltree_path, entry) if do_instance else None
    identity = _legacy_identity(celltree_path, entry)
    apa = 0 if detect_ctpc_family(reco.to_mapping()) == "f0" else 1
    config = LabelingConfig(
        max_distance_cm=max_distance_cm,
        z_offset_cm=z_offset_cm,
        tagging_alg=tagging_alg,
        blob_grow_cm=blob_grow_cm,
        vtx_gate_reco_cm=vtx_gate_reco_cm,
        use_truth_gate=use_truth_gate,
        vtx_gate_truth_cm=vtx_gate_truth_cm,
        do_instance=do_instance,
        shift_p0=shift_p0,
        shift_p1=shift_p1,
        shift_p2=shift_p2,
        dtdc_win=dtdc_win,
        conf_thr=conf_thr,
        purity_thr=purity_thr,
        min_blob_support=min_blob_support,
        max_dist_mm=max_dist_mm,
        write_edge_sup=write_edge_sup,
        balance_edges=balance_edges,
        neg_radius_mm=neg_radius_mm,
    )
    labeled = label_event(
        reco,
        semantic_truth,
        simide,
        vertex,
        identity,
        apa,
        config,
    )
    np.savez(output_file, **labeled.to_mapping())
    return output_file, int(reco.points.shape[0]), int(labeled.nu_vtx_found[0])


def parse_entries(value):
    value = value.strip()
    if "-" in value:
        first, last = value.split("-", 1)
        return list(range(int(first), int(last) + 1))
    return [int(item) for item in value.split(",") if item.strip()]


def pick_celltree_for_rec_prefix(rec_prefix, celltree_apa0, celltree_apa1):
    rec_prefix = rec_prefix.lower()
    if "apa0" in rec_prefix:
        return celltree_apa0
    if "apa1" in rec_prefix:
        return celltree_apa1
    raise ValueError(f"Could not infer APA from --rec-prefix={rec_prefix}. Expected to contain 'apa0' or 'apa1'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tru-prefix", required=True, help="e.g. tru-apa0")
    parser.add_argument("--rec-prefix", required=True, help="e.g. rec-apa0")
    parser.add_argument("--out-prefix", required=True, help="e.g. rec-lab-apa0")
    parser.add_argument("--entries", required=True, help="e.g. 0-11 or 0,1,2")
    parser.add_argument("--max-distance", type=float, default=15.0, help="cm")
    parser.add_argument("--z-offset-cm", type=float, default=0.0, help="cm")
    parser.add_argument("--tagging-alg", default="blob", choices=["point", "blob", "cluster"])
    parser.add_argument("--blob-grow-cm", type=float, default=30.0)
    parser.add_argument("--vtx-gate-reco-cm", type=float, default=500.0, help="cm")
    parser.add_argument("--use-truth-gate", action="store_true")
    parser.add_argument("--vtx-gate-truth-cm", type=float, default=50.0, help="cm")
    parser.add_argument("--celltree-apa0", required=True)
    parser.add_argument("--celltree-apa1", required=True)
    parser.add_argument("--no-instance", action="store_true")
    parser.add_argument("--shift-p0", type=int, default=2992)
    parser.add_argument("--shift-p1", type=int, default=2992)
    parser.add_argument("--shift-p2", type=int, default=2992)
    parser.add_argument("--dtdc-win", type=int, default=1)
    parser.add_argument("--conf-thr", type=float, default=0.7)
    parser.add_argument("--purity-thr", type=float, default=0.85)
    parser.add_argument("--min-blob-support", type=int, default=1)
    parser.add_argument("--max-dist-mm", type=float, default=10.0, help="mm")
    parser.add_argument("--write-edge-sup", action="store_true")
    parser.add_argument("--balance-edges", action="store_true")
    parser.add_argument("--neg-radius-mm", type=float, default=60.0)
    args = parser.parse_args()

    celltree = pick_celltree_for_rec_prefix(args.rec_prefix, args.celltree_apa0, args.celltree_apa1)
    do_instance = not args.no_instance
    print(f"[INFO] Using celltree: {celltree}")
    print(f"[INFO] semantic: tagging_alg={args.tagging_alg} max_distance_cm={args.max_distance} z_offset_cm={args.z_offset_cm}")
    print(f"[INFO] vtx_gate_reco_cm={args.vtx_gate_reco_cm} use_truth_gate={bool(args.use_truth_gate)} vtx_gate_truth_cm={args.vtx_gate_truth_cm}")
    print(
        f"[INFO] instance: enabled={do_instance} shifts(p0,p1,p2)=({args.shift_p0},{args.shift_p1},{args.shift_p2}) "
        f"DTDC_WIN={args.dtdc_win} conf_thr={args.conf_thr} purity_thr={args.purity_thr} max_dist_mm={args.max_dist_mm} "
        f"edge_sup={bool(args.write_edge_sup)} balance_edges={bool(args.balance_edges)} neg_radius_mm={args.neg_radius_mm}"
    )

    entries = parse_entries(args.entries)
    successful = failed = 0
    for entry in entries:
        print(f"\n[ENTRY {entry}] {args.rec_prefix}-{entry}.npz -> {args.out_prefix}-{entry}.npz")
        try:
            output_file, n_hits, found = process_entry(
                entry=entry,
                tru_prefix=args.tru_prefix,
                rec_prefix=args.rec_prefix,
                out_prefix=args.out_prefix,
                celltree_path=celltree,
                max_distance_cm=args.max_distance,
                z_offset_cm=args.z_offset_cm,
                tagging_alg=args.tagging_alg,
                blob_grow_cm=args.blob_grow_cm,
                vtx_gate_reco_cm=args.vtx_gate_reco_cm,
                use_truth_gate=args.use_truth_gate,
                vtx_gate_truth_cm=args.vtx_gate_truth_cm,
                do_instance=do_instance,
                shift_p0=args.shift_p0,
                shift_p1=args.shift_p1,
                shift_p2=args.shift_p2,
                dtdc_win=args.dtdc_win,
                conf_thr=args.conf_thr,
                purity_thr=args.purity_thr,
                min_blob_support=args.min_blob_support,
                max_dist_mm=args.max_dist_mm,
                write_edge_sup=args.write_edge_sup,
                balance_edges=args.balance_edges,
                neg_radius_mm=args.neg_radius_mm,
            )
            print(f"  -> OK hits={n_hits} nu_vtx_found={found} wrote={output_file}")
            successful += 1
        except Exception as error:
            print(f"  -> FAILED: {error}")
            failed += 1
    print("\nSummary:")
    print(f"  Total entries: {len(entries)}")
    print(f"  Successful   : {successful}")
    print(f"  Failed       : {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
