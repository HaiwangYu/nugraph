#!/usr/bin/env python3
"""Real Python-3.11 Process-C receiver for the RecoArrays IPC milestone."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np

from pywcml.recoarrays_ipc import (
    END_OF_STREAM,
    ERROR,
    IPCError,
    IPCIdentity,
    RECO_ARRAYS_APA0,
    RECO_ARRAYS_APA1,
    RecoArraysLimits,
    RecoArraysTransaction,
    decode_reco_arrays_message,
    decode_sbndipc_message,
    encode_ack,
    encode_error_message,
    encode_reco_arrays_message,
    read_sbndipc_message,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipc-fd", required=True, type=int)
    parser.add_argument("--nugraph-root", required=True, type=Path)
    return parser.parse_args()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()


def _git_branch(root: Path) -> str:
    """Return the branch using syntax supported by the SL7 Git client."""

    return _git(root, "symbolic-ref", "--short", "HEAD")


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
    common = min(len(expected), len(actual))
    for offset in range(common):
        if expected[offset] != actual[offset]:
            return f"byte={offset} C++={expected[offset]} Python={actual[offset]}"
    return f"size C++={len(expected)} Python={len(actual)}"


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> int:
    args = _arguments()
    root = args.nugraph_root.resolve()
    if not (root / "pywcml" / "labeling.py").is_file():
        raise IPCError("NuGraph root does not contain the accepted pywcml API")

    channel = socket.socket(fileno=args.ipc_fd)
    limits = RecoArraysLimits()
    transaction = RecoArraysTransaction()
    decoded_by_apa = {}
    encoded_by_apa: dict[int, bytes] = {}
    decode_ns: dict[int, int] = {}
    identity = IPCIdentity(0, 0, 0)
    try:
        while True:
            encoded = read_sbndipc_message(channel, limits.max_message_payload_bytes)
            if encoded is None:
                raise IPCError("Process B closed before RecoArrays end-of-stream")
            message = decode_sbndipc_message(encoded, limits.max_message_payload_bytes)
            identity = message.identity
            if message.kind == END_OF_STREAM:
                if len(message.payload):
                    raise IPCError("end-of-stream payload is not empty")
                pair = transaction.finish()
                if pair.identity != message.identity:
                    raise IPCError("end-of-stream event identity mismatch")
                break
            if message.kind == ERROR:
                raise IPCError("Process B sent an error message")
            if message.kind not in (RECO_ARRAYS_APA0, RECO_ARRAYS_APA1):
                raise IPCError("wrong message type in Process-C RecoArrays transaction")
            begin = time.perf_counter_ns()
            decoded = decode_reco_arrays_message(encoded, limits=limits)
            decode_ns[decoded.apa] = time.perf_counter_ns() - begin
            transaction.add(decoded)
            decoded_by_apa[decoded.apa] = decoded
            encoded_by_apa[decoded.apa] = encoded

        for apa in (0, 1):
            decoded = decoded_by_apa[apa]
            begin = time.perf_counter_ns()
            canonical = encode_reco_arrays_message(decoded.identity, apa, decoded.reco)
            reencode_ns = time.perf_counter_ns() - begin
            if canonical != encoded_by_apa[apa]:
                raise IPCError(
                    f"APA{apa} canonical round-trip differs: "
                    + _first_difference(encoded_by_apa[apa], canonical)
                )
            channel.sendall(canonical)
            print(
                f"PROCESS_C_RECOARRAYS apa={apa} arrays={len(decoded.array_order)} "
                f"exact_raw_bytes=true canonical_round_trip=true "
                f"wire_bytes={len(canonical)} decode_ms={decode_ns[apa] / 1e6:.6f} "
                f"reencode_ms={reencode_ns / 1e6:.6f}",
                flush=True,
            )

        libraries = _loaded_wirecell_libraries()
        if libraries:
            raise IPCError("Process C loaded custom Wire-Cell libraries: " + ",".join(libraries))
        branch = _git_branch(root)
        commit = _git(root, "rev-parse", "HEAD")
        print(
            "PROCESS_C_ENV"
            f" python={sys.executable}"
            f" python_version={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            f" numpy={np.__version__}"
            f" torch={_version('torch')}"
            f" pyg={_version('torch-geometric')}"
            f" nugraph={root} branch={branch} commit={commit}"
            " wirecell_libraries=0",
            flush=True,
        )
        print(
            f"PROCESS_C_RECOARRAYS_TRANSACTION PASS event={pair.identity.run}/"
            f"{pair.identity.subrun}/{pair.identity.event} apa0=true apa1=true",
            flush=True,
        )
        channel.sendall(encode_ack(pair.identity))
        channel.close()
        return 0
    except Exception as error:
        try:
            channel.sendall(encode_error_message(identity, str(error)))
        except Exception:
            pass
        channel.close()
        print(f"PROCESS_C_RECOARRAYS FAIL: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
