from __future__ import annotations

import socket
import struct
import unittest

import numpy as np
import numpy.testing as npt

from pywcml.labeling import NeutrinoVertex, SemanticTruth, SimIDETruth
from pywcml.recoarrays_ipc import IPCError, IPCIdentity, encode_end_of_stream
from pywcml.trutharrays_ipc import (
    FIELD_SEMANTIC_Q,
    FIELD_SEMANTIC_X,
    FIELD_SEMANTIC_Y,
    FIELD_SEMANTIC_Z,
    FIELD_SIMIDE_CHANNEL,
    FIELD_SIMIDE_TDC,
    FIELD_SIMIDE_TRACK_ID,
    FIELD_VERTEX_FOUND,
    FIELD_VERTEX_XYZ,
    TYPE_FLOAT32_LE,
    TYPE_FLOAT64_LE,
    TYPE_INT64_LE,
    TYPE_UINT8,
    TruthArraysLimits,
    TruthArraysTransaction,
    decode_truth_arrays_message,
    encode_truth_arrays_message,
    receive_truth_arrays_pair,
)


_ENVELOPE = struct.Struct("<8sHHQIII")
_PAYLOAD_PREFIX = struct.Struct("<IHHI")
_FIELD_META = struct.Struct("<HHIQQ")
_MAGIC = b"SBNDIPC\0"


def _identity(event: int = 1) -> IPCIdentity:
    return IPCIdentity(run=1, subrun=2236, event=event)


def _inputs(apa: int = 0) -> tuple[SemanticTruth, SimIDETruth, NeutrinoVertex]:
    semantic = SemanticTruth(
        x=np.asarray([1.2 + apa, -2.2], dtype="<f8"),
        y=np.asarray([3.3, 4.4 + apa], dtype="<f8"),
        z=np.asarray([5.5, 6.6], dtype="<f8"),
        q=np.asarray([0, 1], dtype="<i8"),
    )
    simide = SimIDETruth(
        channel=np.asarray([100, 101, 102], dtype="<i8"),
        tdc=np.asarray([2992, 2993, 2994], dtype="<i8"),
        track_id=np.asarray([42, -7, 9], dtype="<i8"),
    )
    vertex = NeutrinoVertex(-70.06083679199219, -110.98822021484375, 407.0470275878906, True)
    return semantic, simide, vertex


def _fields(apa: int = 0):
    semantic, simide, vertex = _inputs(apa)
    return [
        (FIELD_SEMANTIC_X, TYPE_FLOAT64_LE, semantic.x),
        (FIELD_SEMANTIC_Y, TYPE_FLOAT64_LE, semantic.y),
        (FIELD_SEMANTIC_Z, TYPE_FLOAT64_LE, semantic.z),
        (FIELD_SEMANTIC_Q, TYPE_INT64_LE, semantic.q),
        (FIELD_SIMIDE_CHANNEL, TYPE_INT64_LE, simide.channel),
        (FIELD_SIMIDE_TDC, TYPE_INT64_LE, simide.tdc),
        (FIELD_SIMIDE_TRACK_ID, TYPE_INT64_LE, simide.track_id),
        (FIELD_VERTEX_XYZ, TYPE_FLOAT32_LE, np.asarray(vertex.xyz, dtype="<f4")),
        (FIELD_VERTEX_FOUND, TYPE_UINT8, np.asarray([vertex.found], dtype="u1")),
    ]


def _wire(
    apa: int,
    *,
    identity: IPCIdentity | None = None,
    magic: bytes = _MAGIC,
    protocol: int = 1,
    message_type: int = 2,
    schema: int = 1,
    payload_apa: int | None = None,
    fields=None,
    field_count: int | None = None,
    trailing_payload: bytes = b"",
) -> bytes:
    identity = identity or _identity()
    fields = list(_fields(apa) if fields is None else fields)
    payload = bytearray(
        _PAYLOAD_PREFIX.pack(
            schema,
            apa if payload_apa is None else payload_apa,
            0,
            len(fields) if field_count is None else field_count,
        )
    )
    for field in fields:
        field_id, element_type, array = field[:3]
        raw = array.tobytes(order="C") if len(field) < 5 else field[4]
        count = array.size if len(field) < 4 else field[3]
        payload += _FIELD_META.pack(field_id, element_type, 0, count, len(raw))
        payload += raw
    payload += trailing_payload
    return _ENVELOPE.pack(
        magic,
        protocol,
        message_type,
        len(payload),
        identity.run,
        identity.subrun,
        identity.event,
    ) + payload


class TruthArraysIPCTest(unittest.TestCase):
    def test_decode_constructs_existing_truth_classes_and_round_trips(self) -> None:
        for apa in (0, 1):
            semantic, simide, vertex = _inputs(apa)
            encoded = encode_truth_arrays_message(_identity(), apa, semantic, simide, vertex)
            decoded = decode_truth_arrays_message(encoded, expected_apa=apa)
            self.assertIsInstance(decoded.semantic_truth, SemanticTruth)
            self.assertIsInstance(decoded.simide_truth, SimIDETruth)
            self.assertIsInstance(decoded.neutrino_vertex, NeutrinoVertex)
            self.assertEqual(decoded.identity, _identity())
            self.assertEqual(decoded.apa, apa)
            self.assertEqual(
                encode_truth_arrays_message(
                    decoded.identity,
                    decoded.apa,
                    decoded.semantic_truth,
                    decoded.simide_truth,
                    decoded.neutrino_vertex,
                ),
                encoded,
            )

            for actual, expected, dtype in (
                (decoded.semantic_truth.x, semantic.x, "<f8"),
                (decoded.semantic_truth.y, semantic.y, "<f8"),
                (decoded.semantic_truth.z, semantic.z, "<f8"),
                (decoded.semantic_truth.q, semantic.q, "<i8"),
                (decoded.simide_truth.channel, simide.channel, "<i8"),
                (decoded.simide_truth.tdc, simide.tdc, "<i8"),
                (decoded.simide_truth.track_id, simide.track_id, "<i8"),
            ):
                self.assertEqual(actual.dtype, np.dtype(dtype))
                self.assertTrue(actual.flags.c_contiguous)
                self.assertTrue(actual.flags.owndata)
                npt.assert_array_equal(actual, expected)
                self.assertEqual(actual.tobytes(), expected.tobytes())
            self.assertIsNone(decoded.simide_truth.x_cm)
            self.assertIsNone(decoded.simide_truth.y_cm)
            self.assertIsNone(decoded.simide_truth.z_cm)
            npt.assert_array_equal(decoded.neutrino_vertex.xyz, vertex.xyz)
            self.assertEqual(decoded.neutrino_vertex.found, vertex.found)

    def test_encoder_and_decoder_reject_charge_like_origins(self) -> None:
        for bad_q in (1000, 1001):
            semantic, simide, vertex = _inputs()
            semantic = SemanticTruth(semantic.x, semantic.y, semantic.z, np.asarray([0, bad_q], dtype="<i8"))
            with self.subTest(side="encoder", q=bad_q), self.assertRaisesRegex(IPCError, "origin"):
                encode_truth_arrays_message(_identity(), 0, semantic, simide, vertex)

            fields = _fields()
            fields[3] = (FIELD_SEMANTIC_Q, TYPE_INT64_LE, np.asarray([0, bad_q], dtype="<i8"))
            with self.subTest(side="decoder", q=bad_q), self.assertRaisesRegex(IPCError, "origin"):
                decode_truth_arrays_message(_wire(0, fields=fields))

    def test_envelope_schema_apa_and_field_contract_are_strict(self) -> None:
        fields = _fields()
        cases = [
            (_wire(0, magic=b"BADMAGIC"), "magic"),
            (_wire(0, protocol=99), "protocol"),
            (_wire(0, message_type=3), "message type"),
            (_wire(0, schema=99), "schema"),
            (_wire(0, payload_apa=2), "APA"),
            (_wire(0, field_count=8), "field count"),
            (_wire(0, fields=[fields[0], fields[0], *fields[2:]]), "field identifier/order"),
            (_wire(0, fields=[(FIELD_SEMANTIC_X, TYPE_FLOAT32_LE, fields[0][2]), *fields[1:]]), "element type"),
            (_wire(0, fields=[(FIELD_SEMANTIC_X, TYPE_FLOAT64_LE, fields[0][2], 2, b"\0"), *fields[1:]]), "byte count"),
        ]
        for encoded, match in cases:
            with self.subTest(match=match), self.assertRaisesRegex(IPCError, match):
                decode_truth_arrays_message(encoded, expected_apa=0)

    def test_length_allocation_truncation_and_trailing_guards(self) -> None:
        fields = _fields()
        unequal_semantic = list(fields)
        unequal_semantic[1] = (FIELD_SEMANTIC_Y, TYPE_FLOAT64_LE, np.asarray([3.3], dtype="<f8"))
        with self.assertRaisesRegex(IPCError, "semantic field lengths"):
            decode_truth_arrays_message(_wire(0, fields=unequal_semantic))

        unequal_simide = list(fields)
        unequal_simide[5] = (FIELD_SIMIDE_TDC, TYPE_INT64_LE, np.asarray([2992], dtype="<i8"))
        with self.assertRaisesRegex(IPCError, "SimIDE field lengths"):
            decode_truth_arrays_message(_wire(0, fields=unequal_simide))

        absurd = list(fields)
        absurd[0] = (FIELD_SEMANTIC_X, TYPE_FLOAT64_LE, np.empty(0, dtype="<f8"), 2**63, b"")
        with self.assertRaisesRegex(IPCError, "limit"):
            decode_truth_arrays_message(_wire(0, fields=absurd), limits=TruthArraysLimits())

        good = _wire(0)
        with self.assertRaisesRegex(IPCError, "truncated"):
            decode_truth_arrays_message(good[:-1])
        with self.assertRaisesRegex(IPCError, "trailing"):
            decode_truth_arrays_message(good + b"\0")
        with self.assertRaisesRegex(IPCError, "trailing"):
            decode_truth_arrays_message(_wire(0, trailing_payload=b"\0"))

    def test_pair_transaction_rejects_duplicate_mismatch_and_missing(self) -> None:
        apa0 = decode_truth_arrays_message(_wire(0))
        apa1 = decode_truth_arrays_message(_wire(1))

        duplicate = TruthArraysTransaction("golden", 0, 0, 123)
        duplicate.add(apa0)
        with self.assertRaisesRegex(IPCError, "duplicate APA0"):
            duplicate.add(apa0)

        mismatch = TruthArraysTransaction("golden", 0, 0, 123)
        mismatch.add(apa0)
        with self.assertRaisesRegex(IPCError, "identity"):
            mismatch.add(decode_truth_arrays_message(_wire(1, identity=_identity(2))))

        missing = TruthArraysTransaction("golden", 0, 0, 123)
        missing.add(apa0)
        with self.assertRaisesRegex(IPCError, "missing APA1"):
            missing.finish()

        complete = TruthArraysTransaction("golden", 0, 0, 123)
        complete.add(apa0)
        complete.add(apa1)
        pair = complete.finish()
        self.assertEqual((pair.identity.run, pair.identity.subrun, pair.identity.event), (1, 2236, 1))
        self.assertEqual(pair.identity.source_index, 0)
        self.assertIsInstance(pair.apa0.semantic_truth, SemanticTruth)
        self.assertIsInstance(pair.apa1.simide_truth, SimIDETruth)

    def test_socket_receiver_requires_both_apas_before_eos(self) -> None:
        left, right = socket.socketpair()
        try:
            for apa in (0, 1):
                left.sendall(encode_truth_arrays_message(_identity(), apa, *_inputs(apa)))
            left.sendall(encode_end_of_stream(_identity()))
            pair = receive_truth_arrays_pair(right, "golden", 0, 0, 123)
            self.assertEqual(pair.identity.event, 1)
        finally:
            left.close()
            right.close()

        left, right = socket.socketpair()
        try:
            left.sendall(encode_truth_arrays_message(_identity(), 0, *_inputs(0)))
            left.sendall(encode_end_of_stream(_identity()))
            with self.assertRaisesRegex(IPCError, "missing APA1"):
                receive_truth_arrays_pair(right, "golden", 0, 0, 123)
        finally:
            left.close()
            right.close()


if __name__ == "__main__":
    unittest.main()
