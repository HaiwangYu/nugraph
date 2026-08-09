"""Explicit SBNDIPC codec for Process-A labeler-compatible truth inputs.

This module constructs the existing :mod:`pywcml.labeling` truth classes
directly.  It contains no truth matching, labeling, reconstruction, or file
I/O.  The fixed schema carries only fields read by ``label_event()``.
"""

from __future__ import annotations

from dataclasses import dataclass
import socket
import struct
from typing import Final

import numpy as np

from .identity import EventIdentity
from .labeling import NeutrinoVertex, SemanticTruth, SimIDETruth
from .recoarrays_ipc import (
    END_OF_STREAM,
    ERROR,
    IPCError,
    IPCIdentity,
    decode_sbndipc_message,
    read_sbndipc_message,
)


MAGIC: Final = b"SBNDIPC\0"
PROTOCOL_VERSION: Final = 1
TRUTH_ARRAYS: Final = 2
TRUTH_ARRAYS_SCHEMA_VERSION: Final = 1

FIELD_SEMANTIC_X: Final = 1
FIELD_SEMANTIC_Y: Final = 2
FIELD_SEMANTIC_Z: Final = 3
FIELD_SEMANTIC_Q: Final = 4
FIELD_SIMIDE_CHANNEL: Final = 5
FIELD_SIMIDE_TDC: Final = 6
FIELD_SIMIDE_TRACK_ID: Final = 7
FIELD_VERTEX_XYZ: Final = 8
FIELD_VERTEX_FOUND: Final = 9

TYPE_FLOAT64_LE: Final = 1
TYPE_INT64_LE: Final = 2
TYPE_FLOAT32_LE: Final = 3
TYPE_UINT8: Final = 4

_ENVELOPE = struct.Struct("<8sHHQIII")
_PAYLOAD_PREFIX = struct.Struct("<IHHI")
_FIELD_META = struct.Struct("<HHIQQ")
_FIELD_SPECS: Final = (
    (FIELD_SEMANTIC_X, TYPE_FLOAT64_LE, np.dtype("<f8"), "semantic_x"),
    (FIELD_SEMANTIC_Y, TYPE_FLOAT64_LE, np.dtype("<f8"), "semantic_y"),
    (FIELD_SEMANTIC_Z, TYPE_FLOAT64_LE, np.dtype("<f8"), "semantic_z"),
    (FIELD_SEMANTIC_Q, TYPE_INT64_LE, np.dtype("<i8"), "semantic_q"),
    (FIELD_SIMIDE_CHANNEL, TYPE_INT64_LE, np.dtype("<i8"), "simide_channel"),
    (FIELD_SIMIDE_TDC, TYPE_INT64_LE, np.dtype("<i8"), "simide_tdc"),
    (FIELD_SIMIDE_TRACK_ID, TYPE_INT64_LE, np.dtype("<i8"), "simide_track_id"),
    (FIELD_VERTEX_XYZ, TYPE_FLOAT32_LE, np.dtype("<f4"), "vertex_xyz"),
    (FIELD_VERTEX_FOUND, TYPE_UINT8, np.dtype("u1"), "vertex_found"),
)


@dataclass(frozen=True, slots=True)
class TruthArraysLimits:
    """Hard limits applied before any owning NumPy allocation."""

    max_message_payload_bytes: int = 256 * 1024 * 1024
    max_fields: int = 9
    max_array_elements: int = 20_000_000
    max_total_elements: int = 40_000_000
    max_array_bytes: int = 160 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class DecodedTruthArrays:
    identity: IPCIdentity
    apa: int
    semantic_truth: SemanticTruth
    simide_truth: SimIDETruth
    neutrino_vertex: NeutrinoVertex


@dataclass(frozen=True, slots=True)
class TruthInputs:
    """The three existing truth objects passed to ``label_event()`` for one APA."""

    semantic_truth: SemanticTruth
    simide_truth: SimIDETruth
    neutrino_vertex: NeutrinoVertex


@dataclass(frozen=True, slots=True)
class TruthArraysPair:
    identity: EventIdentity
    apa0: TruthInputs
    apa1: TruthInputs


def _validate_apa(apa: int) -> int:
    if isinstance(apa, bool) or not isinstance(apa, int) or apa not in (0, 1):
        raise IPCError("TruthArrays APA must be zero or one")
    return apa


def _validate_origin(q: np.ndarray) -> None:
    if not np.all((q == -1) | (q == 0) | (q == 1)):
        raise IPCError("invalid semantic origin: expected only -1, 0, or 1")


def _validate_array(name: str, array: np.ndarray, dtype: np.dtype) -> None:
    if not isinstance(array, np.ndarray):
        raise IPCError(f"TruthArrays field {name} is not a NumPy array")
    if array.ndim != 1:
        raise IPCError(f"TruthArrays field {name} is not rank one")
    if array.dtype != dtype:
        raise IPCError(f"TruthArrays field {name} has the wrong little-endian dtype")
    if not array.flags.c_contiguous:
        raise IPCError(f"TruthArrays field {name} is not C-contiguous")


def _encoding_fields(
    semantic_truth: SemanticTruth,
    simide_truth: SimIDETruth,
    neutrino_vertex: NeutrinoVertex,
) -> tuple[np.ndarray, ...]:
    fields = (
        semantic_truth.x,
        semantic_truth.y,
        semantic_truth.z,
        semantic_truth.q,
        simide_truth.channel,
        simide_truth.tdc,
        simide_truth.track_id,
        np.asarray(neutrino_vertex.xyz, dtype="<f4"),
        np.asarray([1 if neutrino_vertex.found else 0], dtype="u1"),
    )
    for (_, _, dtype, name), array in zip(_FIELD_SPECS, fields):
        _validate_array(name, array, dtype)
    semantic_lengths = {array.size for array in fields[:4]}
    if len(semantic_lengths) != 1:
        raise IPCError("TruthArrays semantic field lengths differ")
    simide_lengths = {array.size for array in fields[4:7]}
    if len(simide_lengths) != 1:
        raise IPCError("TruthArrays SimIDE field lengths differ")
    _validate_origin(fields[3])
    return fields


def encode_truth_arrays_message(
    identity: IPCIdentity,
    apa: int,
    semantic_truth: SemanticTruth,
    simide_truth: SimIDETruth,
    neutrino_vertex: NeutrinoVertex,
) -> bytes:
    """Return one canonical message for existing labeler truth structures."""

    apa = _validate_apa(apa)
    fields = _encoding_fields(semantic_truth, simide_truth, neutrino_vertex)
    payload = bytearray(_PAYLOAD_PREFIX.pack(TRUTH_ARRAYS_SCHEMA_VERSION, apa, 0, len(fields)))
    for (field_id, element_type, _dtype, _name), array in zip(_FIELD_SPECS, fields):
        payload += _FIELD_META.pack(field_id, element_type, 0, array.size, array.nbytes)
        payload += array.tobytes(order="C")
    return _ENVELOPE.pack(
        MAGIC,
        PROTOCOL_VERSION,
        TRUTH_ARRAYS,
        len(payload),
        identity.run,
        identity.subrun,
        identity.event,
    ) + payload


class _Cursor:
    def __init__(self, data: memoryview):
        self._data = data
        self._position = 0

    @property
    def remaining(self) -> int:
        return len(self._data) - self._position

    def take(self, size: int, what: str) -> memoryview:
        if size < 0 or size > self.remaining:
            raise IPCError(f"truncated {what}")
        begin = self._position
        self._position += size
        return self._data[begin:self._position]

    def unpack(self, parser: struct.Struct, what: str):
        return parser.unpack(self.take(parser.size, what))

    def require_end(self, what: str) -> None:
        if self.remaining:
            raise IPCError(f"trailing bytes after {what}")


def decode_truth_arrays_message(
    data: bytes | bytearray | memoryview,
    *,
    expected_apa: int | None = None,
    limits: TruthArraysLimits = TruthArraysLimits(),
) -> DecodedTruthArrays:
    """Decode one message directly into the existing Python truth classes."""

    message = decode_sbndipc_message(data, limits.max_message_payload_bytes)
    if message.kind != TRUTH_ARRAYS:
        raise IPCError("wrong IPC message type for TruthArrays")
    cursor = _Cursor(message.payload)
    schema, apa, reserved, field_count = cursor.unpack(_PAYLOAD_PREFIX, "TruthArrays prefix")
    if schema != TRUTH_ARRAYS_SCHEMA_VERSION:
        raise IPCError("unsupported TruthArrays schema version")
    _validate_apa(apa)
    if expected_apa is not None and apa != _validate_apa(expected_apa):
        raise IPCError("TruthArrays APA differs from expected APA")
    if reserved != 0:
        raise IPCError("TruthArrays reserved field is nonzero")
    if field_count > limits.max_fields or field_count != len(_FIELD_SPECS):
        raise IPCError("TruthArrays field count is not nine")

    arrays: dict[str, np.ndarray] = {}
    total_elements = 0
    for expected_id, expected_type, dtype, name in _FIELD_SPECS:
        field_id, element_type, field_reserved, count, byte_count = cursor.unpack(
            _FIELD_META, f"TruthArrays metadata for {name}"
        )
        if field_id != expected_id:
            raise IPCError(f"TruthArrays field identifier/order differs at {name}")
        if element_type != expected_type:
            raise IPCError(f"TruthArrays element type differs for {name}")
        if field_reserved != 0:
            raise IPCError(f"TruthArrays reserved field is nonzero for {name}")
        if count > limits.max_array_elements:
            raise IPCError(f"TruthArrays element count exceeds decoder limit for {name}")
        total_elements += count
        if total_elements > limits.max_total_elements:
            raise IPCError("total TruthArrays elements exceed decoder limit")
        expected_bytes = count * dtype.itemsize
        if byte_count != expected_bytes:
            raise IPCError(f"TruthArrays raw byte count disagrees with count for {name}")
        if byte_count > limits.max_array_bytes:
            raise IPCError(f"TruthArrays raw byte count exceeds decoder limit for {name}")
        raw = cursor.take(byte_count, f"TruthArrays raw data for {name}")
        array = np.frombuffer(raw, dtype=dtype, count=count).copy(order="C")
        if array.tobytes(order="C") != raw.tobytes():
            raise IPCError(f"TruthArrays raw bytes changed during NumPy construction for {name}")
        arrays[name] = array
    cursor.require_end("TruthArrays payload")

    semantic_lengths = {arrays[name].size for name in ("semantic_x", "semantic_y", "semantic_z", "semantic_q")}
    if len(semantic_lengths) != 1:
        raise IPCError("TruthArrays semantic field lengths differ")
    simide_lengths = {arrays[name].size for name in ("simide_channel", "simide_tdc", "simide_track_id")}
    if len(simide_lengths) != 1:
        raise IPCError("TruthArrays SimIDE field lengths differ")
    if arrays["vertex_xyz"].size != 3:
        raise IPCError("TruthArrays vertex XYZ count is not three")
    if arrays["vertex_found"].size != 1 or arrays["vertex_found"][0] not in (0, 1):
        raise IPCError("TruthArrays vertex found field is invalid")
    _validate_origin(arrays["semantic_q"])

    semantic = SemanticTruth(
        arrays["semantic_x"],
        arrays["semantic_y"],
        arrays["semantic_z"],
        arrays["semantic_q"],
    )
    simide = SimIDETruth(
        channel=arrays["simide_channel"],
        tdc=arrays["simide_tdc"],
        track_id=arrays["simide_track_id"],
    )
    xyz = arrays["vertex_xyz"]
    vertex = NeutrinoVertex(float(xyz[0]), float(xyz[1]), float(xyz[2]), bool(arrays["vertex_found"][0]))
    return DecodedTruthArrays(message.identity, apa, semantic, simide, vertex)


class TruthArraysTransaction:
    """Withhold truth until exactly one matching payload from each APA exists."""

    def __init__(self, campaign_id: str, shard_id: int, source_index: int, random_seed: int):
        self._provenance = campaign_id, shard_id, source_index, random_seed
        self._identity: IPCIdentity | None = None
        self._lanes: dict[int, TruthInputs] = {}

    def add(self, decoded: DecodedTruthArrays) -> None:
        if decoded.apa in self._lanes:
            raise IPCError(f"duplicate APA{decoded.apa} TruthArrays")
        if self._identity is None:
            self._identity = decoded.identity
        elif decoded.identity != self._identity:
            raise IPCError("TruthArrays event identity mismatch between APAs")
        self._lanes[decoded.apa] = TruthInputs(
            decoded.semantic_truth, decoded.simide_truth, decoded.neutrino_vertex
        )

    def finish(self) -> TruthArraysPair:
        missing = [apa for apa in (0, 1) if apa not in self._lanes]
        if missing:
            raise IPCError("missing " + ", ".join(f"APA{apa}" for apa in missing) + " TruthArrays")
        assert self._identity is not None
        campaign_id, shard_id, source_index, random_seed = self._provenance
        identity = EventIdentity(
            campaign_id=campaign_id,
            shard_id=shard_id,
            source_index=source_index,
            run=self._identity.run,
            subrun=self._identity.subrun,
            event=self._identity.event,
            random_seed=random_seed,
        )
        return TruthArraysPair(identity, self._lanes[0], self._lanes[1])


def receive_truth_arrays_pair(
    channel: socket.socket,
    campaign_id: str,
    shard_id: int,
    source_index: int,
    random_seed: int,
    limits: TruthArraysLimits = TruthArraysLimits(),
) -> TruthArraysPair:
    """Receive a complete two-APA transaction over an anonymous socket."""

    transaction = TruthArraysTransaction(campaign_id, shard_id, source_index, random_seed)
    while True:
        encoded = read_sbndipc_message(channel, limits.max_message_payload_bytes)
        if encoded is None:
            raise IPCError("peer closed before TruthArrays end-of-stream")
        message = decode_sbndipc_message(encoded, limits.max_message_payload_bytes)
        if message.kind == END_OF_STREAM:
            if len(message.payload):
                raise IPCError("end-of-stream payload is not empty")
            pair = transaction.finish()
            physical = (pair.identity.run, pair.identity.subrun, pair.identity.event)
            if physical != (message.identity.run, message.identity.subrun, message.identity.event):
                raise IPCError("end-of-stream event identity mismatch")
            return pair
        if message.kind == ERROR:
            raise IPCError("Process A sent an error message")
        if message.kind != TRUTH_ARRAYS:
            raise IPCError("wrong message type in TruthArrays transaction")
        transaction.add(decode_truth_arrays_message(encoded, limits=limits))


__all__ = [
    "DecodedTruthArrays",
    "FIELD_SEMANTIC_Q",
    "FIELD_SEMANTIC_X",
    "FIELD_SEMANTIC_Y",
    "FIELD_SEMANTIC_Z",
    "FIELD_SIMIDE_CHANNEL",
    "FIELD_SIMIDE_TDC",
    "FIELD_SIMIDE_TRACK_ID",
    "FIELD_VERTEX_FOUND",
    "FIELD_VERTEX_XYZ",
    "TRUTH_ARRAYS",
    "TRUTH_ARRAYS_SCHEMA_VERSION",
    "TYPE_FLOAT32_LE",
    "TYPE_FLOAT64_LE",
    "TYPE_INT64_LE",
    "TYPE_UINT8",
    "TruthArraysLimits",
    "TruthArraysPair",
    "TruthArraysTransaction",
    "TruthInputs",
    "decode_truth_arrays_message",
    "encode_truth_arrays_message",
    "receive_truth_arrays_pair",
]
