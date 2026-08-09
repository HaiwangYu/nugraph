from __future__ import annotations

import socket
import struct
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt

from pywcml.labeling import RecoArrays
from pywcml.recoarrays_ipc import (
    IPCError,
    IPCIdentity,
    RecoArraysTransaction,
    decode_reco_arrays_message,
    encode_end_of_stream,
    encode_reco_arrays_message,
    receive_reco_arrays_pair,
)
from scripts.process_c_recoarrays_receiver import _git_branch


_ENVELOPE = struct.Struct("<8sHHQIII")
_PAYLOAD_PREFIX = struct.Struct("<IHHI")
_ARRAY_PREFIX = struct.Struct("<I")
_ARRAY_META = struct.Struct("<IQQHHQ")
_MAGIC = b"SBNDIPC\0"


def _identity(event: int = 1) -> IPCIdentity:
    return IPCIdentity(run=1, subrun=2236, event=event)


def _reco(apa: int) -> RecoArrays:
    def values(begin: int) -> np.ndarray:
        return np.asarray([[begin, begin + 1], [begin + 2, begin + 3]], dtype="<f4")

    return RecoArrays.from_mapping(
        {
            f"ctpc_f{apa}p0": values(0),
            f"ctpc_f{apa}p1": values(4),
            f"ctpc_f{apa}p2": values(8),
            "blobs": values(12),
            "points": values(16),
            "ppedges": values(20),
        }
    )


def _specs(apa: int):
    mapping = _reco(apa).to_mapping()
    names = [f"ctpc_f{apa}p0", f"ctpc_f{apa}p1", f"ctpc_f{apa}p2", "blobs", "points", "ppedges"]
    return [(name, 2, mapping[name].shape, 1, mapping[name].nbytes, mapping[name].tobytes()) for name in names]


def _wire(
    apa: int,
    *,
    identity: IPCIdentity | None = None,
    message_type: int | None = None,
    schema: int = 1,
    specs=None,
    array_count: int | None = None,
) -> bytes:
    identity = identity or _identity()
    specs = list(_specs(apa) if specs is None else specs)
    payload = bytearray(_PAYLOAD_PREFIX.pack(schema, apa, 0, len(specs) if array_count is None else array_count))
    for name, rank, shape, dtype, nbytes, raw in specs:
        encoded_name = name.encode("ascii")
        payload += _ARRAY_PREFIX.pack(len(encoded_name))
        payload += encoded_name
        rows = shape[0] if len(shape) > 0 else 0
        columns = shape[1] if len(shape) > 1 else 0
        payload += _ARRAY_META.pack(rank, rows, columns, dtype, 0, nbytes)
        payload += raw
    kind = (3 + apa) if message_type is None else message_type
    return _ENVELOPE.pack(
        _MAGIC,
        1,
        kind,
        len(payload),
        identity.run,
        identity.subrun,
        identity.event,
    ) + payload


class RecoArraysIPCTest(unittest.TestCase):
    def test_process_c_provenance_uses_legacy_compatible_git(self):
        root = Path(__file__).resolve().parents[2]
        self.assertTrue(_git_branch(root))

    def test_decode_constructs_existing_recoarrays_and_round_trips_canonically(self) -> None:
        for apa in (0, 1):
            encoded = encode_reco_arrays_message(_identity(), apa, _reco(apa))
            decoded = decode_reco_arrays_message(encoded, expected_apa=apa)
            self.assertIsInstance(decoded.reco, RecoArrays)
            self.assertEqual(decoded.identity, _identity())
            self.assertEqual(decoded.apa, apa)
            self.assertEqual(
                encode_reco_arrays_message(decoded.identity, decoded.apa, decoded.reco), encoded
            )

            expected = _reco(apa).to_mapping()
            actual = decoded.reco.to_mapping()
            self.assertEqual(
                decoded.array_order,
                (
                    f"ctpc_f{apa}p0",
                    f"ctpc_f{apa}p1",
                    f"ctpc_f{apa}p2",
                    "blobs",
                    "points",
                    "ppedges",
                ),
            )
            for name in decoded.array_order:
                self.assertEqual(actual[name].dtype, np.dtype("<f4"))
                self.assertTrue(actual[name].flags.c_contiguous)
                self.assertTrue(actual[name].flags.owndata)
                npt.assert_array_equal(actual[name], expected[name])
                self.assertEqual(actual[name].tobytes(order="C"), expected[name].tobytes(order="C"))

    def test_malformed_payloads_are_rejected(self) -> None:
        cases = [
            (_wire(0, schema=99), "schema"),
            (_wire(0, message_type=4), "APA"),
            (_wire(0, array_count=5), "array count"),
            (_wire(0, specs=_specs(0)[:-1]), "array count"),
            (_wire(0, specs=[_specs(0)[0], _specs(0)[0], *_specs(0)[2:]]), "name"),
            (_wire(0, specs=[(*_specs(0)[0][:1], 1, (4,), *_specs(0)[0][3:]), *_specs(0)[1:]]), "rank"),
            (_wire(0, specs=[(*_specs(0)[0][:2], (2**63, 2), *_specs(0)[0][3:]), *_specs(0)[1:]]), "dimension"),
            (_wire(0, specs=[(*_specs(0)[0][:3], 99, *_specs(0)[0][4:]), *_specs(0)[1:]]), "dtype"),
            (_wire(0, specs=[(*_specs(0)[0][:4], 7, _specs(0)[0][5]), *_specs(0)[1:]]), "byte count"),
        ]
        for encoded, match in cases:
            with self.subTest(match=match), self.assertRaisesRegex(IPCError, match):
                decode_reco_arrays_message(encoded, expected_apa=0)

    def test_truncated_trailing_and_absurd_inputs_are_rejected(self) -> None:
        good = _wire(0)
        with self.assertRaisesRegex(IPCError, "truncated"):
            decode_reco_arrays_message(good[:-1], expected_apa=0)
        with self.assertRaisesRegex(IPCError, "trailing"):
            decode_reco_arrays_message(good + b"\0", expected_apa=0)

        specs = _specs(0)
        absurd = [(specs[0][0], 2, (50_000_001, 2), 1, 400_000_008, b""), *specs[1:]]
        with self.assertRaisesRegex(IPCError, "limit"):
            decode_reco_arrays_message(_wire(0, specs=absurd), expected_apa=0)

    def test_pair_transaction_rejects_duplicate_mismatch_and_missing(self) -> None:
        apa0 = decode_reco_arrays_message(_wire(0), expected_apa=0)
        apa1 = decode_reco_arrays_message(_wire(1), expected_apa=1)

        duplicate = RecoArraysTransaction()
        duplicate.add(apa0)
        with self.assertRaisesRegex(IPCError, "duplicate APA0"):
            duplicate.add(apa0)

        mismatch = RecoArraysTransaction()
        mismatch.add(apa0)
        other_event = decode_reco_arrays_message(_wire(1, identity=_identity(2)), expected_apa=1)
        with self.assertRaisesRegex(IPCError, "identity"):
            mismatch.add(other_event)

        missing = RecoArraysTransaction()
        missing.add(apa0)
        with self.assertRaisesRegex(IPCError, "missing APA1"):
            missing.finish()

        complete = RecoArraysTransaction()
        complete.add(apa0)
        complete.add(apa1)
        pair = complete.finish()
        self.assertEqual(pair.identity, _identity())
        self.assertIsInstance(pair.apa0, RecoArrays)
        self.assertIsInstance(pair.apa1, RecoArrays)

    def test_socket_receiver_requires_both_apas_before_eos(self) -> None:
        left, right = socket.socketpair()
        try:
            left.sendall(encode_reco_arrays_message(_identity(), 0, _reco(0)))
            left.sendall(encode_reco_arrays_message(_identity(), 1, _reco(1)))
            left.sendall(encode_end_of_stream(_identity()))
            pair = receive_reco_arrays_pair(right)
            self.assertEqual(pair.identity, _identity())
        finally:
            left.close()
            right.close()

        left, right = socket.socketpair()
        try:
            left.sendall(encode_reco_arrays_message(_identity(), 0, _reco(0)))
            left.sendall(encode_end_of_stream(_identity()))
            with self.assertRaisesRegex(IPCError, "missing APA1"):
                receive_reco_arrays_pair(right)
        finally:
            left.close()
            right.close()


if __name__ == "__main__":
    unittest.main()
