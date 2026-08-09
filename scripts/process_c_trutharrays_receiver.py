#!/usr/bin/env python3
"""Real Python-3.11 Process-C receiver for paired SBND truth inputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipc-fd", required=True, type=int)
    parser.add_argument("--nugraph-root", required=True, type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--shard-id", required=True, type=int)
    parser.add_argument("--source-index", required=True, type=int)
    parser.add_argument("--random-seed", required=True, type=int)
    parser.add_argument("--reference-semantic-apa0", type=Path)
    parser.add_argument("--reference-semantic-apa1", type=Path)
    return parser.parse_args()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()


def _git_branch(root: Path) -> str:
    return _git(root, "symbolic-ref", "--short", "HEAD")


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _loaded_wirecell_libraries() -> list[str]:
    result: set[str] = set()
    with open("/proc/self/maps", encoding="utf-8") as maps:
        for line in maps:
            marker = line.find("/")
            if marker < 0:
                continue
            path = line[marker:].strip()
            if "/libWireCell" in path and ".so" in path:
                result.add(path)
    return sorted(result)


def _first_difference(expected: bytes, actual: bytes) -> str:
    for offset, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return f"byte={offset} C++={left} Python={right}"
    return f"size C++={len(expected)} Python={len(actual)}"


def _logical_bytes(decoded) -> int:
    arrays = (
        decoded.semantic_truth.x,
        decoded.semantic_truth.y,
        decoded.semantic_truth.z,
        decoded.semantic_truth.q,
        decoded.simide_truth.channel,
        decoded.simide_truth.tdc,
        decoded.simide_truth.track_id,
    )
    return sum(array.nbytes for array in arrays) + 3 * np.dtype("<f4").itemsize + 1


def _reference_semantic(path: Path, expected_identity) -> object:
    from pywcml.labeling import SemanticTruth

    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    actual_identity = (int(raw["runNo"]), int(raw["subRunNo"]), int(raw["eventNo"]))
    if actual_identity != expected_identity:
        raise ValueError(f"legacy semantic reference identity differs: {actual_identity}")
    return SemanticTruth.from_mapping(raw)


def _require_array_exact(name: str, expected: np.ndarray, actual: np.ndarray) -> None:
    if expected.dtype != actual.dtype:
        raise ValueError(f"{name} dtype differs: {expected.dtype} != {actual.dtype}")
    if expected.shape != actual.shape:
        raise ValueError(f"{name} shape differs: {expected.shape} != {actual.shape}")
    if expected.tobytes(order="C") != actual.tobytes(order="C"):
        raise ValueError(f"{name} values/order/raw bytes differ")


def _require_semantic_reference_exact(apa: int, reference, actual) -> None:
    for field in ("x", "y", "z", "q"):
        _require_array_exact(
            f"APA{apa} SemanticTruth.{field}",
            getattr(reference, field),
            getattr(actual, field),
        )


def main() -> int:
    args = _arguments()
    root = args.nugraph_root.resolve()
    if not (root / "pywcml" / "labeling.py").is_file():
        raise ValueError("NuGraph root does not contain the accepted pywcml API")
    sys.path.insert(0, str(root))

    from pywcml.identity import EventIdentity
    from pywcml.labeling import NeutrinoVertex, SemanticTruth, SimIDETruth
    from pywcml.recoarrays_ipc import (
        END_OF_STREAM,
        ERROR,
        IPCError,
        IPCIdentity,
        decode_sbndipc_message,
        encode_ack,
        encode_error_message,
        read_sbndipc_message,
    )
    from pywcml.trutharrays_ipc import (
        TRUTH_ARRAYS,
        TruthArraysLimits,
        TruthArraysTransaction,
        decode_truth_arrays_message,
        encode_truth_arrays_message,
    )

    channel = socket.socket(fileno=args.ipc_fd)
    limits = TruthArraysLimits()
    transaction = TruthArraysTransaction(
        args.campaign_id, args.shard_id, args.source_index, args.random_seed
    )
    decoded_by_apa = {}
    encoded_by_apa: dict[int, bytes] = {}
    decode_ns: dict[int, int] = {}
    identity = IPCIdentity(0, 0, 0)
    try:
        while True:
            encoded = read_sbndipc_message(channel, limits.max_message_payload_bytes)
            if encoded is None:
                raise IPCError("Process A closed before TruthArrays end-of-stream")
            message = decode_sbndipc_message(encoded, limits.max_message_payload_bytes)
            identity = message.identity
            if message.kind == END_OF_STREAM:
                if len(message.payload):
                    raise IPCError("end-of-stream payload is not empty")
                pair = transaction.finish()
                physical = (pair.identity.run, pair.identity.subrun, pair.identity.event)
                if physical != (identity.run, identity.subrun, identity.event):
                    raise IPCError("end-of-stream event identity mismatch")
                break
            if message.kind == ERROR:
                raise IPCError("Process A sent an error message")
            if message.kind != TRUTH_ARRAYS:
                raise IPCError("wrong message type in Process-C TruthArrays transaction")
            begin = time.perf_counter_ns()
            decoded = decode_truth_arrays_message(encoded, limits=limits)
            decode_ns[decoded.apa] = time.perf_counter_ns() - begin
            transaction.add(decoded)
            decoded_by_apa[decoded.apa] = decoded
            encoded_by_apa[decoded.apa] = encoded

        if not isinstance(pair.identity, EventIdentity):
            raise IPCError("Process C did not construct the existing EventIdentity")
        references = (args.reference_semantic_apa0, args.reference_semantic_apa1)
        for apa in (0, 1):
            decoded = decoded_by_apa[apa]
            if not isinstance(decoded.semantic_truth, SemanticTruth):
                raise IPCError("Process C did not construct the existing SemanticTruth")
            if not isinstance(decoded.simide_truth, SimIDETruth):
                raise IPCError("Process C did not construct the existing SimIDETruth")
            if not isinstance(decoded.neutrino_vertex, NeutrinoVertex):
                raise IPCError("Process C did not construct the existing NeutrinoVertex")

            reference_path = references[apa]
            if reference_path is not None:
                reference = _reference_semantic(reference_path, (identity.run, identity.subrun, identity.event))
                _require_semantic_reference_exact(apa, reference, decoded.semantic_truth)
            begin = time.perf_counter_ns()
            canonical = encode_truth_arrays_message(
                decoded.identity,
                apa,
                decoded.semantic_truth,
                decoded.simide_truth,
                decoded.neutrino_vertex,
            )
            reencode_ns = time.perf_counter_ns() - begin
            if canonical != encoded_by_apa[apa]:
                raise IPCError(
                    f"APA{apa} canonical round-trip differs: "
                    + _first_difference(encoded_by_apa[apa], canonical)
                )
            channel.sendall(canonical)
            q_values, q_counts = np.unique(decoded.semantic_truth.q, return_counts=True)
            q_distribution = ",".join(
                f"{int(value)}:{int(count)}" for value, count in zip(q_values, q_counts)
            )
            print(
                f"PROCESS_C_TRUTH apa={apa} semantic={decoded.semantic_truth.q.size} "
                f"simide={decoded.simide_truth.channel.size} q={q_distribution} "
                f"logical_bytes={_logical_bytes(decoded)} wire_bytes={len(canonical)} "
                f"semantic_reference_exact={str(reference_path is not None).lower()} "
                f"canonical_round_trip=true decode_ms={decode_ns[apa] / 1e6:.6f} "
                f"reencode_ms={reencode_ns / 1e6:.6f}",
                flush=True,
            )

        libraries = _loaded_wirecell_libraries()
        if libraries:
            raise IPCError("Process C loaded Wire-Cell libraries: " + ",".join(libraries))
        branch = _git_branch(root)
        commit = _git(root, "rev-parse", "HEAD")
        print(
            "PROCESS_C_TRUTH_ENV"
            f" python={sys.executable}"
            f" python_version={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            f" numpy={np.__version__}"
            f" torch={_version('torch')}"
            f" pyg={_version('torch-geometric')}"
            f" nugraph={root} branch={branch} commit={commit}"
            " wirecell_libraries=0 label_event_calls=0",
            flush=True,
        )
        print(
            f"PROCESS_C_TRUTH_TRANSACTION PASS event={pair.identity.run}/"
            f"{pair.identity.subrun}/{pair.identity.event} apa0=true apa1=true "
            "event_identity_existing_class=true",
            flush=True,
        )
        channel.sendall(encode_ack(identity))
        channel.close()
        return 0
    except Exception as error:
        try:
            channel.sendall(encode_error_message(identity, str(error)))
        except Exception:
            pass
        channel.close()
        print(f"PROCESS_C_TRUTH FAIL: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
