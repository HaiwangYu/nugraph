"""Tests for pywcml.integrated_transaction using in-process socket pairs."""

from __future__ import annotations

import socket
import struct
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from pywcml.labeling import NeutrinoVertex, RecoArrays, SemanticTruth, SimIDETruth
from pywcml.recoarrays_ipc import (
    ERROR,
    IPCError,
    IPCIdentity,
    encode_end_of_stream,
    encode_error_message,
    encode_reco_arrays_message,
)
from pywcml.trutharrays_ipc import encode_truth_arrays_message
from pywcml.integrated_transaction import (
    CombinedLimits,
    IntegratedResult,
    _receive_combined_pair,
    run_integrated_event,
)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

IDENTITY = IPCIdentity(1, 2236, 1)
CAMPAIGN_ID = "test"
SHARD_ID = 0
SOURCE_INDEX = 0
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Minimal array helpers
# ---------------------------------------------------------------------------

def _minimal_reco_arrays(apa: int) -> RecoArrays:
    """Minimal valid 6-array RecoArrays with zeros, dtype <f4."""
    return RecoArrays.from_mapping(
        {
            f"ctpc_f{apa}p0": np.zeros((3, 7), dtype="<f4"),
            f"ctpc_f{apa}p1": np.zeros((3, 7), dtype="<f4"),
            f"ctpc_f{apa}p2": np.zeros((3, 7), dtype="<f4"),
            "blobs": np.zeros((2, 38), dtype="<f4"),
            "points": np.zeros((3, 6), dtype="<f4"),
            "ppedges": np.zeros((1, 3), dtype="<f4"),
        }
    )


def _minimal_semantic(n: int = 3) -> SemanticTruth:
    """Minimal SemanticTruth with zeros."""
    return SemanticTruth(
        x=np.zeros(n, dtype="<f8"),
        y=np.zeros(n, dtype="<f8"),
        z=np.zeros(n, dtype="<f8"),
        q=np.zeros(n, dtype="<i8"),
    )


def _minimal_simide(n: int = 3) -> SimIDETruth:
    """Minimal SimIDETruth with zeros."""
    return SimIDETruth(
        channel=np.zeros(n, dtype="<i8"),
        tdc=np.zeros(n, dtype="<i8"),
        track_id=np.zeros(n, dtype="<i8"),
    )


def _minimal_vertex() -> NeutrinoVertex:
    """Minimal NeutrinoVertex at origin, not found."""
    return NeutrinoVertex(x=0.0, y=0.0, z=0.0, found=False)


def _make_reco_message(identity: IPCIdentity, apa: int) -> bytes:
    """Encode a minimal RecoArrays message."""
    return encode_reco_arrays_message(identity, apa, _minimal_reco_arrays(apa))


def _make_truth_message(identity: IPCIdentity, apa: int) -> bytes:
    """Encode a minimal TruthArrays message."""
    return encode_truth_arrays_message(
        identity,
        apa,
        _minimal_semantic(),
        _minimal_simide(),
        _minimal_vertex(),
    )


def _send_stream(messages: list[bytes]) -> socket.socket:
    """Create a socketpair, send all messages to the write end, return the read end."""
    reader, writer_sock = socket.socketpair()
    for msg in messages:
        writer_sock.sendall(msg)
    writer_sock.close()
    return reader


def _make_valid_stream() -> socket.socket:
    """Return a read-end socket loaded with a complete, valid combined event stream."""
    return _send_stream([
        _make_reco_message(IDENTITY, 0),
        _make_reco_message(IDENTITY, 1),
        _make_truth_message(IDENTITY, 0),
        _make_truth_message(IDENTITY, 1),
        encode_end_of_stream(IDENTITY),
    ])


# ---------------------------------------------------------------------------
# Raw truth message builder (bypasses encoder q-validation for test 11)
# ---------------------------------------------------------------------------

_ENVELOPE = struct.Struct("<8sHHQIII")
_PAYLOAD_PREFIX = struct.Struct("<IHHI")
_FIELD_META = struct.Struct("<HHIQQ")
_MAGIC = b"SBNDIPC\0"


def _raw_truth_message_with_bad_q(identity: IPCIdentity, apa: int, q: np.ndarray) -> bytes:
    """Build a raw TruthArrays message with arbitrary q, bypassing encoder validation.

    The nine fields are laid out in the same order as encode_truth_arrays_message,
    but q is accepted verbatim even if its values would fail _validate_origin.
    The decoder will still validate q on receipt.
    """
    n = int(q.size)
    field_data = [
        # (field_id, element_type, array)
        (1, 1, np.zeros(n, dtype="<f8")),   # FIELD_SEMANTIC_X,  TYPE_FLOAT64_LE
        (2, 1, np.zeros(n, dtype="<f8")),   # FIELD_SEMANTIC_Y
        (3, 1, np.zeros(n, dtype="<f8")),   # FIELD_SEMANTIC_Z
        (4, 2, q),                           # FIELD_SEMANTIC_Q,  TYPE_INT64_LE
        (5, 2, np.zeros(n, dtype="<i8")),   # FIELD_SIMIDE_CHANNEL
        (6, 2, np.zeros(n, dtype="<i8")),   # FIELD_SIMIDE_TDC
        (7, 2, np.zeros(n, dtype="<i8")),   # FIELD_SIMIDE_TRACK_ID
        (8, 3, np.zeros(3, dtype="<f4")),   # FIELD_VERTEX_XYZ,  TYPE_FLOAT32_LE
        (9, 4, np.array([0], dtype="u1")),  # FIELD_VERTEX_FOUND, TYPE_UINT8
    ]
    payload = bytearray(_PAYLOAD_PREFIX.pack(1, apa, 0, len(field_data)))
    for field_id, element_type, array in field_data:
        payload += _FIELD_META.pack(field_id, element_type, 0, array.size, array.nbytes)
        payload += array.tobytes(order="C")
    return _ENVELOPE.pack(
        _MAGIC,
        1,       # PROTOCOL_VERSION
        2,       # TRUTH_ARRAYS
        len(payload),
        identity.run,
        identity.subrun,
        identity.event,
    ) + payload


# ---------------------------------------------------------------------------
# Tests: _receive_combined_pair
# ---------------------------------------------------------------------------

class TestReceiveCombinedPair(unittest.TestCase):

    def _call(self, channel: socket.socket):
        return _receive_combined_pair(
            channel,
            CAMPAIGN_ID,
            SHARD_ID,
            SOURCE_INDEX,
            RANDOM_SEED,
            CombinedLimits(),
        )

    # 1. Reco-first ordering works
    def test_valid_reco_first_receive(self) -> None:
        channel = _send_stream([
            _make_reco_message(IDENTITY, 0),
            _make_reco_message(IDENTITY, 1),
            _make_truth_message(IDENTITY, 0),
            _make_truth_message(IDENTITY, 1),
            encode_end_of_stream(IDENTITY),
        ])
        try:
            reco_pair, truth_pair = self._call(channel)
            self.assertEqual(reco_pair.identity, IDENTITY)
            self.assertEqual(
                (truth_pair.identity.run, truth_pair.identity.subrun, truth_pair.identity.event),
                (IDENTITY.run, IDENTITY.subrun, IDENTITY.event),
            )
        finally:
            channel.close()

    # 2. Truth-first ordering works
    def test_valid_truth_first_receive(self) -> None:
        channel = _send_stream([
            _make_truth_message(IDENTITY, 0),
            _make_truth_message(IDENTITY, 1),
            _make_reco_message(IDENTITY, 0),
            _make_reco_message(IDENTITY, 1),
            encode_end_of_stream(IDENTITY),
        ])
        try:
            reco_pair, truth_pair = self._call(channel)
            self.assertEqual(reco_pair.identity, IDENTITY)
            self.assertEqual(
                (truth_pair.identity.run, truth_pair.identity.subrun, truth_pair.identity.event),
                (IDENTITY.run, IDENTITY.subrun, IDENTITY.event),
            )
        finally:
            channel.close()

    # 3. Duplicate reco APA0 raises
    def test_duplicate_reco_apa0(self) -> None:
        channel = _send_stream([
            _make_reco_message(IDENTITY, 0),
            _make_reco_message(IDENTITY, 0),  # duplicate
            _make_reco_message(IDENTITY, 1),
            _make_truth_message(IDENTITY, 0),
            _make_truth_message(IDENTITY, 1),
            encode_end_of_stream(IDENTITY),
        ])
        try:
            with self.assertRaisesRegex(IPCError, "duplicate"):
                self._call(channel)
        finally:
            channel.close()

    # 4. Duplicate reco APA1 raises
    def test_duplicate_reco_apa1(self) -> None:
        channel = _send_stream([
            _make_reco_message(IDENTITY, 0),
            _make_reco_message(IDENTITY, 1),
            _make_reco_message(IDENTITY, 1),  # duplicate
            _make_truth_message(IDENTITY, 0),
            _make_truth_message(IDENTITY, 1),
            encode_end_of_stream(IDENTITY),
        ])
        try:
            with self.assertRaisesRegex(IPCError, "duplicate"):
                self._call(channel)
        finally:
            channel.close()

    # 5. Duplicate truth APA0 raises
    def test_duplicate_truth_apa0(self) -> None:
        channel = _send_stream([
            _make_truth_message(IDENTITY, 0),
            _make_truth_message(IDENTITY, 0),  # duplicate
            _make_reco_message(IDENTITY, 0),
            _make_reco_message(IDENTITY, 1),
            _make_truth_message(IDENTITY, 1),
            encode_end_of_stream(IDENTITY),
        ])
        try:
            with self.assertRaisesRegex(IPCError, "duplicate"):
                self._call(channel)
        finally:
            channel.close()

    # 6. Duplicate truth APA1 raises
    def test_duplicate_truth_apa1(self) -> None:
        channel = _send_stream([
            _make_reco_message(IDENTITY, 0),
            _make_reco_message(IDENTITY, 1),
            _make_truth_message(IDENTITY, 0),
            _make_truth_message(IDENTITY, 1),
            _make_truth_message(IDENTITY, 1),  # duplicate
            encode_end_of_stream(IDENTITY),
        ])
        try:
            with self.assertRaisesRegex(IPCError, "duplicate"):
                self._call(channel)
        finally:
            channel.close()

    # 7. Missing reco APA1 raises after EOS
    def test_missing_reco_apa1(self) -> None:
        channel = _send_stream([
            _make_reco_message(IDENTITY, 0),
            _make_truth_message(IDENTITY, 0),
            _make_truth_message(IDENTITY, 1),
            encode_end_of_stream(IDENTITY),
        ])
        try:
            with self.assertRaisesRegex(IPCError, "missing"):
                self._call(channel)
        finally:
            channel.close()

    # 8. Missing truth APA0 raises after EOS
    def test_missing_truth_apa0(self) -> None:
        channel = _send_stream([
            _make_reco_message(IDENTITY, 0),
            _make_reco_message(IDENTITY, 1),
            _make_truth_message(IDENTITY, 1),
            encode_end_of_stream(IDENTITY),
        ])
        try:
            with self.assertRaisesRegex(IPCError, "missing"):
                self._call(channel)
        finally:
            channel.close()

    # 9. Reco/truth physical identity mismatch raises
    def test_mismatched_reco_truth_identity(self) -> None:
        other_identity = IPCIdentity(1, 2236, 2)
        channel = _send_stream([
            _make_reco_message(IDENTITY, 0),
            _make_reco_message(IDENTITY, 1),
            _make_truth_message(other_identity, 0),
            _make_truth_message(other_identity, 1),
            encode_end_of_stream(IDENTITY),
        ])
        try:
            with self.assertRaisesRegex(IPCError, "mismatch"):
                self._call(channel)
        finally:
            channel.close()

    # 10. EOS identity mismatch raises
    def test_mismatched_eos_identity(self) -> None:
        eos_wrong = encode_end_of_stream(IPCIdentity(1, 2236, 99))
        channel = _send_stream([
            _make_reco_message(IDENTITY, 0),
            _make_reco_message(IDENTITY, 1),
            _make_truth_message(IDENTITY, 0),
            _make_truth_message(IDENTITY, 1),
            eos_wrong,
        ])
        try:
            with self.assertRaisesRegex(IPCError, "mismatch"):
                self._call(channel)
        finally:
            channel.close()

    # 11. Truth message with invalid q (=1000) raises IPCError on decode
    def test_malformed_q_fails(self) -> None:
        bad_q = np.array([1000, 0, 0], dtype="<i8")
        bad_truth0 = _raw_truth_message_with_bad_q(IDENTITY, 0, bad_q)
        channel = _send_stream([
            _make_reco_message(IDENTITY, 0),
            _make_reco_message(IDENTITY, 1),
            bad_truth0,
            _make_truth_message(IDENTITY, 1),
            encode_end_of_stream(IDENTITY),
        ])
        try:
            with self.assertRaisesRegex(IPCError, "origin"):
                self._call(channel)
        finally:
            channel.close()

    # 15. Error message received → IPCError about error
    def test_error_message_rejected(self) -> None:
        channel = _send_stream([
            encode_error_message(IDENTITY, "upstream failure"),
        ])
        try:
            with self.assertRaisesRegex(IPCError, "error"):
                self._call(channel)
        finally:
            channel.close()


# ---------------------------------------------------------------------------
# Tests: run_integrated_event (with mocked converter and writer)
# ---------------------------------------------------------------------------

class TestRunIntegratedEvent(unittest.TestCase):

    def _run(self, channel: socket.socket, *, converter, writer):
        return run_integrated_event(
            channel,
            campaign_id=CAMPAIGN_ID,
            shard_id=SHARD_ID,
            source_index=SOURCE_INDEX,
            random_seed=RANDOM_SEED,
            converter=converter,
            writer=writer,
        )

    # 12. label_event raises → RuntimeError propagates, append_event not called
    def test_label_event_exception_no_ack(self) -> None:
        channel = _make_valid_stream()
        converter = MagicMock()
        writer = MagicMock()
        try:
            with patch(
                "pywcml.integrated_transaction.label_event",
                side_effect=RuntimeError("label_fail"),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    self._run(channel, converter=converter, writer=writer)
            self.assertIn("label_fail", str(ctx.exception))
            writer.append_event.assert_not_called()
        finally:
            channel.close()

    # 13. converter.convert_arrays raises → RuntimeError propagates, append_event not called
    def test_converter_exception_no_ack(self) -> None:
        channel = _make_valid_stream()
        converter = MagicMock()
        writer = MagicMock()
        converter.convert_arrays.side_effect = RuntimeError("conv_fail")
        try:
            with patch(
                "pywcml.integrated_transaction.label_event",
                return_value=MagicMock(),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    self._run(channel, converter=converter, writer=writer)
            self.assertIn("conv_fail", str(ctx.exception))
            writer.append_event.assert_not_called()
        finally:
            channel.close()

    # 14. writer.append_event raises → RuntimeError propagates, finalize not called
    def test_hdf5_append_exception_no_finalize(self) -> None:
        channel = _make_valid_stream()
        converter = MagicMock()
        writer = MagicMock()
        writer.append_event.side_effect = RuntimeError("hdf5_fail")
        try:
            with patch(
                "pywcml.integrated_transaction.label_event",
                return_value=MagicMock(),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    self._run(channel, converter=converter, writer=writer)
            self.assertIn("hdf5_fail", str(ctx.exception))
            writer.finalize.assert_not_called()
        finally:
            channel.close()


if __name__ == "__main__":
    unittest.main()
