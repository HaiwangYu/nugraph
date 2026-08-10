"""Tests for the _execute_transaction helper in scripts/process_c_integrated.py.

Specifically covers Correction 3: StreamingH5Writer.finalize() raises →
no ACK is emitted, the exception propagates, the event is not committed.
"""

from __future__ import annotations

import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Make scripts/ importable from tests/
_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import process_c_integrated as pci  # noqa: E402


class TestExecuteTransactionFinalizeFailure(unittest.TestCase):
    """_execute_transaction: finalize failure path."""

    # -----------------------------------------------------------------------
    # 15. finalize raises → exception propagates, writer.close() called,
    #     no bytes emitted on the channel (no ACK).
    # -----------------------------------------------------------------------
    def test_finalize_exception_raises_and_no_ack(self) -> None:
        """When finalize raises, no ACK is emitted and the exception propagates."""
        a_side, c_side = socket.socketpair()

        mock_writer = MagicMock()
        mock_writer.finalize.side_effect = RuntimeError("disk full during finalize")
        mock_converter = MagicMock()

        mock_result = MagicMock()
        mock_result.identity.run = 1
        mock_result.identity.subrun = 2236
        mock_result.identity.event = 1
        mock_result.sample_name_apa0 = "1_2236_rec-lab-apa0-1"
        mock_result.sample_name_apa1 = "1_2236_rec-lab-apa1-1"

        try:
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                pci._execute_transaction(
                    c_side,
                    mock_converter,
                    mock_writer,
                    campaign_id="test",
                    shard_id=0,
                    source_index=0,
                    random_seed=42,
                    _run_fn=lambda *args, **kwargs: mock_result,
                )

            # writer.close() must be called as part of exception cleanup
            mock_writer.close.assert_called_once()

            # The channel must carry zero bytes from _execute_transaction —
            # no ACK was emitted before the exception.
            a_side.setblocking(False)
            try:
                data = a_side.recv(4096)
                self.assertEqual(
                    b"",
                    data,
                    "Expected empty channel: no ACK may be emitted after finalize failure",
                )
            except BlockingIOError:
                pass  # Expected: nothing to read

        finally:
            c_side.close()
            a_side.close()

    # -----------------------------------------------------------------------
    # 16. finalize succeeds → returns result, finalize called, close NOT called.
    # -----------------------------------------------------------------------
    def test_finalize_success_returns_result(self) -> None:
        """When finalize succeeds, _execute_transaction returns the result."""
        a_side, c_side = socket.socketpair()

        mock_writer = MagicMock()
        mock_converter = MagicMock()

        mock_result = MagicMock()
        mock_result.identity.run = 1
        mock_result.identity.subrun = 2236
        mock_result.identity.event = 1
        mock_result.sample_name_apa0 = "1_2236_rec-lab-apa0-1"
        mock_result.sample_name_apa1 = "1_2236_rec-lab-apa1-1"

        try:
            result = pci._execute_transaction(
                c_side,
                mock_converter,
                mock_writer,
                campaign_id="test",
                shard_id=0,
                source_index=0,
                random_seed=42,
                _run_fn=lambda *args, **kwargs: mock_result,
            )
            self.assertIs(result, mock_result)
            mock_writer.finalize.assert_called_once()
            mock_writer.close.assert_not_called()

        finally:
            c_side.close()
            a_side.close()

    # -----------------------------------------------------------------------
    # 17. run_integrated_event raises → finalize NOT called, writer closed.
    # -----------------------------------------------------------------------
    def test_run_event_exception_no_finalize(self) -> None:
        """When run_integrated_event raises, finalize is not called."""
        a_side, c_side = socket.socketpair()

        mock_writer = MagicMock()
        mock_converter = MagicMock()

        def failing_run(*args, **kwargs):
            raise RuntimeError("receive_failed")

        try:
            with self.assertRaisesRegex(RuntimeError, "receive_failed"):
                pci._execute_transaction(
                    c_side,
                    mock_converter,
                    mock_writer,
                    campaign_id="test",
                    shard_id=0,
                    source_index=0,
                    random_seed=42,
                    _run_fn=failing_run,
                )

            mock_writer.finalize.assert_not_called()
            mock_writer.close.assert_called_once()

        finally:
            c_side.close()
            a_side.close()


if __name__ == "__main__":
    unittest.main()
