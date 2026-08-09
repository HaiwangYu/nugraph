"""Independent Python SBNDIPC codec for paired Process-B RecoArrays.

The wire format is explicit little-endian data and constructs the accepted
``pywcml.labeling.RecoArrays`` object directly.  It intentionally contains no
labeling or NuGraph conversion calls and performs no filesystem I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
import socket
import struct
from typing import Final

import numpy as np

from .labeling import RecoArrays


MAGIC: Final = b"SBNDIPC\0"
PROTOCOL_VERSION: Final = 1
RECO_ARRAYS_SCHEMA_VERSION: Final = 1
RECO_ARRAYS_APA0: Final = 3
RECO_ARRAYS_APA1: Final = 4
ACK: Final = 5
ERROR: Final = 6
END_OF_STREAM: Final = 7
DTYPE_FLOAT32_LE: Final = 1

_ENVELOPE = struct.Struct("<8sHHQIII")
_PAYLOAD_PREFIX = struct.Struct("<IHHI")
_ARRAY_META = struct.Struct("<IQQHHQ")
_U32 = struct.Struct("<I")
_KNOWN_TYPES = frozenset({1, 2, RECO_ARRAYS_APA0, RECO_ARRAYS_APA1, ACK, ERROR, END_OF_STREAM, 0x8001, 0x8002})


class IPCError(ValueError):
    """Malformed or inconsistent SBNDIPC input."""


@dataclass(frozen=True, slots=True)
class IPCIdentity:
    run: int
    subrun: int
    event: int

    def __post_init__(self) -> None:
        for name in ("run", "subrun", "event"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
                raise IPCError(f"{name} must be a uint32")


@dataclass(frozen=True, slots=True)
class RecoArraysLimits:
    max_message_payload_bytes: int = 256 * 1024 * 1024
    max_arrays: int = 6
    max_name_bytes: int = 64
    max_rank: int = 2
    max_dimension: int = 50_000_000
    max_array_elements: int = 50_000_000
    max_total_elements: int = 60_000_000
    max_array_bytes: int = 200 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class DecodedRecoArrays:
    identity: IPCIdentity
    apa: int
    reco: RecoArrays
    array_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DecodedMessage:
    kind: int
    identity: IPCIdentity
    payload: memoryview


@dataclass(frozen=True, slots=True)
class RecoArraysPair:
    identity: IPCIdentity
    apa0: RecoArrays
    apa1: RecoArrays


def _validate_apa(apa: int) -> int:
    if isinstance(apa, bool) or not isinstance(apa, int) or apa not in (0, 1):
        raise IPCError("RecoArrays APA must be zero or one")
    return apa


def _message_type(apa: int) -> int:
    return RECO_ARRAYS_APA0 + _validate_apa(apa)


def _expected_names(apa: int) -> tuple[str, ...]:
    family = f"ctpc_f{_validate_apa(apa)}"
    return (f"{family}p0", f"{family}p1", f"{family}p2", "blobs", "points", "ppedges")


def _encode_envelope(kind: int, identity: IPCIdentity, payload: bytes) -> bytes:
    if kind not in _KNOWN_TYPES:
        raise IPCError("unknown IPC message type")
    return _ENVELOPE.pack(
        MAGIC,
        PROTOCOL_VERSION,
        kind,
        len(payload),
        identity.run,
        identity.subrun,
        identity.event,
    ) + payload


def decode_sbndipc_message(
    data: bytes | bytearray | memoryview,
    max_payload_bytes: int = 256 * 1024 * 1024,
) -> DecodedMessage:
    """Validate one complete SBNDIPC envelope and return its borrowed payload."""

    view = memoryview(data).cast("B")
    if len(view) < _ENVELOPE.size:
        raise IPCError("truncated IPC envelope")
    magic, version, kind, payload_bytes, run, subrun, event = _ENVELOPE.unpack_from(view)
    if magic != MAGIC:
        raise IPCError("invalid IPC magic")
    if version != PROTOCOL_VERSION:
        raise IPCError("unsupported IPC protocol version")
    if kind not in _KNOWN_TYPES:
        raise IPCError("unknown IPC message type")
    if payload_bytes > max_payload_bytes:
        raise IPCError("declared IPC payload exceeds decoder limit")
    expected_bytes = _ENVELOPE.size + payload_bytes
    if len(view) < expected_bytes:
        raise IPCError("truncated IPC payload")
    if len(view) > expected_bytes:
        raise IPCError("trailing bytes after IPC payload")
    return DecodedMessage(kind, IPCIdentity(run, subrun, event), view[_ENVELOPE.size:])


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

    def string(self, limit: int, what: str) -> str:
        (size,) = self.unpack(_U32, f"{what} length")
        if size > limit:
            raise IPCError(f"{what} length exceeds decoder limit")
        try:
            return self.take(size, what).tobytes().decode("ascii")
        except UnicodeDecodeError as error:
            raise IPCError(f"{what} is not ASCII") from error

    def require_end(self, what: str) -> None:
        if self.remaining:
            raise IPCError(f"trailing bytes after {what}")


def _validate_encoding_array(name: str, array: np.ndarray) -> None:
    if not isinstance(array, np.ndarray):
        raise IPCError(f"RecoArrays value {name} is not a NumPy array")
    if array.ndim != 2:
        raise IPCError(f"RecoArrays rank is not two for {name}")
    if array.dtype != np.dtype("<f4"):
        raise IPCError(f"RecoArrays dtype is not little-endian float32 for {name}")
    if not array.flags.c_contiguous:
        raise IPCError(f"RecoArrays array is not C-contiguous for {name}")


def encode_reco_arrays_message(identity: IPCIdentity, apa: int, reco: RecoArrays) -> bytes:
    """Return one canonical SBNDIPC RecoArrays message."""

    names = _expected_names(apa)
    mapping = reco.to_mapping()
    if set(mapping) != set(names) or len(mapping) != len(names):
        raise IPCError("RecoArrays names do not match the six-array APA contract")
    payload = bytearray(_PAYLOAD_PREFIX.pack(RECO_ARRAYS_SCHEMA_VERSION, apa, 0, len(names)))
    for name in names:
        array = mapping[name]
        _validate_encoding_array(name, array)
        encoded_name = name.encode("ascii")
        payload += _U32.pack(len(encoded_name))
        payload += encoded_name
        payload += _ARRAY_META.pack(
            2,
            array.shape[0],
            array.shape[1],
            DTYPE_FLOAT32_LE,
            0,
            array.nbytes,
        )
        payload += array.tobytes(order="C")
    return _encode_envelope(_message_type(apa), identity, bytes(payload))


def decode_reco_arrays_message(
    data: bytes | bytearray | memoryview,
    *,
    expected_apa: int | None = None,
    limits: RecoArraysLimits = RecoArraysLimits(),
) -> DecodedRecoArrays:
    """Decode one message into the existing owning Python ``RecoArrays`` API."""

    message = decode_sbndipc_message(data, limits.max_message_payload_bytes)
    if message.kind not in (RECO_ARRAYS_APA0, RECO_ARRAYS_APA1):
        raise IPCError("wrong IPC message type for RecoArrays")
    type_apa = message.kind - RECO_ARRAYS_APA0
    if expected_apa is not None and type_apa != _validate_apa(expected_apa):
        raise IPCError("RecoArrays APA message type differs from expected APA")

    cursor = _Cursor(message.payload)
    schema, payload_apa, reserved, array_count = cursor.unpack(_PAYLOAD_PREFIX, "RecoArrays prefix")
    if schema != RECO_ARRAYS_SCHEMA_VERSION:
        raise IPCError("unsupported RecoArrays schema version")
    if payload_apa != type_apa:
        raise IPCError("RecoArrays APA payload disagrees with message type")
    if reserved != 0:
        raise IPCError("RecoArrays reserved field is nonzero")
    if array_count > limits.max_arrays or array_count != 6:
        raise IPCError("RecoArrays array count is not six")

    names = _expected_names(type_apa)
    arrays: dict[str, np.ndarray] = {}
    total_elements = 0
    for expected_name in names:
        name = cursor.string(limits.max_name_bytes, "RecoArrays array name")
        if name != expected_name or name in arrays:
            raise IPCError(f"RecoArrays array name/order differs at {expected_name}")
        rank, rows, columns, dtype, array_reserved, byte_count = cursor.unpack(
            _ARRAY_META, f"RecoArrays metadata for {name}"
        )
        if rank > limits.max_rank or rank != 2:
            raise IPCError(f"RecoArrays rank is not two for {name}")
        if rows > limits.max_dimension or columns > limits.max_dimension:
            raise IPCError(f"RecoArrays dimension exceeds decoder limit for {name}")
        elements = rows * columns
        if elements > limits.max_array_elements:
            raise IPCError(f"RecoArrays element count exceeds decoder limit for {name}")
        total_elements += elements
        if total_elements > limits.max_total_elements:
            raise IPCError("total RecoArrays elements exceed decoder limit")
        if dtype != DTYPE_FLOAT32_LE:
            raise IPCError(f"RecoArrays dtype is not little-endian float32 for {name}")
        if array_reserved != 0:
            raise IPCError(f"RecoArrays reserved array field is nonzero for {name}")
        expected_bytes = elements * np.dtype("<f4").itemsize
        if byte_count != expected_bytes:
            raise IPCError(f"RecoArrays raw byte count disagrees with shape for {name}")
        if byte_count > limits.max_array_bytes:
            raise IPCError(f"RecoArrays raw byte count exceeds decoder limit for {name}")
        raw = cursor.take(byte_count, f"RecoArrays float32 data for {name}")
        # One direct byte copy gives the array independent lifetime.  No value
        # conversion, Python list, normalization or sorting is involved.
        array = np.frombuffer(raw, dtype="<f4", count=elements).reshape((rows, columns), order="C").copy(order="C")
        if array.tobytes(order="C") != raw.tobytes():
            raise IPCError(f"RecoArrays raw bytes changed during NumPy construction for {name}")
        arrays[name] = array

    cursor.require_end("RecoArrays payload")
    reco = RecoArrays.from_mapping(arrays)
    return DecodedRecoArrays(message.identity, type_apa, reco, names)


def encode_end_of_stream(identity: IPCIdentity) -> bytes:
    return _encode_envelope(END_OF_STREAM, identity, b"")


def encode_ack(identity: IPCIdentity) -> bytes:
    return _encode_envelope(ACK, identity, b"")


def encode_error_message(identity: IPCIdentity, error: str) -> bytes:
    encoded = error.encode("utf-8")
    if len(encoded) > 4096:
        encoded = encoded[:4096]
    return _encode_envelope(ERROR, identity, _U32.pack(len(encoded)) + encoded)


class RecoArraysTransaction:
    """Withhold an event until one matching message from each APA is present."""

    def __init__(self) -> None:
        self._identity: IPCIdentity | None = None
        self._lanes: dict[int, RecoArrays] = {}

    def add(self, decoded: DecodedRecoArrays) -> None:
        if decoded.apa in self._lanes:
            raise IPCError(f"duplicate APA{decoded.apa} RecoArrays")
        if self._identity is None:
            self._identity = decoded.identity
        elif decoded.identity != self._identity:
            raise IPCError("RecoArrays event identity mismatch between APAs")
        self._lanes[decoded.apa] = decoded.reco

    def finish(self) -> RecoArraysPair:
        missing = [apa for apa in (0, 1) if apa not in self._lanes]
        if missing:
            raise IPCError("missing " + ", ".join(f"APA{apa}" for apa in missing) + " RecoArrays")
        assert self._identity is not None
        return RecoArraysPair(self._identity, self._lanes[0], self._lanes[1])


def _read_exact(channel: socket.socket, size: int, *, allow_clean_eof: bool = False) -> bytes | None:
    data = bytearray(size)
    view = memoryview(data)
    received = 0
    while received != size:
        try:
            count = channel.recv_into(view[received:])
        except InterruptedError:
            continue
        if count == 0:
            if received == 0 and allow_clean_eof:
                return None
            raise IPCError("peer closed during IPC message")
        received += count
    return bytes(data)


def read_sbndipc_message(channel: socket.socket, max_payload_bytes: int) -> bytes | None:
    header = _read_exact(channel, _ENVELOPE.size, allow_clean_eof=True)
    if header is None:
        return None
    magic, version, kind, payload_bytes, run, subrun, event = _ENVELOPE.unpack(header)
    if magic != MAGIC:
        raise IPCError("invalid IPC magic")
    if version != PROTOCOL_VERSION:
        raise IPCError("unsupported IPC protocol version")
    if kind not in _KNOWN_TYPES:
        raise IPCError("unknown IPC message type")
    IPCIdentity(run, subrun, event)
    if payload_bytes > max_payload_bytes:
        raise IPCError("declared IPC payload exceeds decoder limit")
    payload = _read_exact(channel, payload_bytes)
    assert payload is not None
    return header + payload


def receive_reco_arrays_pair(
    channel: socket.socket,
    limits: RecoArraysLimits = RecoArraysLimits(),
) -> RecoArraysPair:
    transaction = RecoArraysTransaction()
    while True:
        encoded = read_sbndipc_message(channel, limits.max_message_payload_bytes)
        if encoded is None:
            raise IPCError("peer closed before RecoArrays end-of-stream")
        message = decode_sbndipc_message(encoded, limits.max_message_payload_bytes)
        if message.kind == END_OF_STREAM:
            if len(message.payload):
                raise IPCError("end-of-stream payload is not empty")
            pair = transaction.finish()
            if message.identity != pair.identity:
                raise IPCError("end-of-stream event identity mismatch")
            return pair
        if message.kind not in (RECO_ARRAYS_APA0, RECO_ARRAYS_APA1):
            raise IPCError("wrong message type in RecoArrays transaction")
        transaction.add(decode_reco_arrays_message(encoded, limits=limits))


__all__ = [
    "DecodedRecoArrays",
    "DecodedMessage",
    "IPCError",
    "IPCIdentity",
    "RecoArraysLimits",
    "RecoArraysPair",
    "RecoArraysTransaction",
    "decode_reco_arrays_message",
    "decode_sbndipc_message",
    "encode_ack",
    "encode_end_of_stream",
    "encode_error_message",
    "encode_reco_arrays_message",
    "read_sbndipc_message",
    "receive_reco_arrays_pair",
]
