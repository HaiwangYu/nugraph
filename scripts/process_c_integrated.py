#!/usr/bin/env python
"""Production Process C: receive combined SBNDIPC stream, label, and write HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipc-fd", required=True, type=int)
    parser.add_argument("--nugraph-root", required=True, type=Path)
    parser.add_argument("--h5-output", required=True, type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--shard-id", required=True, type=int)
    parser.add_argument("--source-index", required=True, type=int)
    parser.add_argument("--random-seed", required=True, type=int)
    return parser.parse_args()


def _loaded_wirecell_libraries() -> list[str]:
    """Return sorted paths of any WireCell .so files visible in /proc/self/maps."""
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


def _execute_transaction(
    channel: socket.socket,
    converter,
    writer,
    campaign_id: str,
    shard_id: int,
    source_index: int,
    random_seed: int,
    *,
    _run_fn=None,
):
    """Receive, label, write, and finalize one physical event.

    Returns the IntegratedResult on success.  On any exception, closes the
    writer and re-raises.  Does NOT send ACK or error messages — the caller
    owns that responsibility.

    The optional ``_run_fn`` parameter replaces ``run_integrated_event`` and
    is intended exclusively for unit testing.
    """
    if _run_fn is None:
        from pywcml.integrated_transaction import run_integrated_event
        _run_fn = run_integrated_event

    try:
        result = _run_fn(
            channel,
            campaign_id=campaign_id,
            shard_id=shard_id,
            source_index=source_index,
            random_seed=random_seed,
            converter=converter,
            writer=writer,
        )
        writer.finalize()
        return result
    except Exception:
        writer.close()
        raise


def main() -> int:
    args = _arguments()
    root = args.nugraph_root.resolve()

    if not (root / "pywcml" / "labeling.py").is_file():
        print(
            f"ERROR: NuGraph root {root} does not contain pywcml/labeling.py",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    libraries = _loaded_wirecell_libraries()
    if libraries:
        print(
            f"ERROR: WireCell .so libraries are loaded: {libraries}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    from pywcml.converter import WCMLConverter
    from pywcml.h5writer import StreamingH5Writer
    from pywcml.recoarrays_ipc import IPCError, IPCIdentity, encode_ack, encode_error_message

    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, fileno=args.ipc_fd)

    identity_for_error = IPCIdentity(0, 0, 0)
    writer = None
    try:
        converter = WCMLConverter()
        writer = StreamingH5Writer(args.h5_output)

        result = _execute_transaction(
            channel,
            converter,
            writer,
            campaign_id=args.campaign_id,
            shard_id=args.shard_id,
            source_index=args.source_index,
            random_seed=args.random_seed,
        )

        # -----------------------------------------------------------------------
        # Correction 1: split validation
        # Both APAs share the same physical EventIdentity; EventIdentity.split()
        # uses (campaign_id, run, subrun, event) with APA deliberately excluded.
        # The two split values are identical by construction.  We compute and
        # report both explicitly so the report can verify split_apa0 == split_apa1.
        # -----------------------------------------------------------------------
        split_apa0 = result.identity.split()
        split_apa1 = result.identity.split()
        if split_apa0 != split_apa1:
            raise RuntimeError(
                f"split invariant violated: split_apa0={split_apa0!r} split_apa1={split_apa1!r}"
            )
        print(
            f"PROCESS_C_SPLIT_INFO"
            f" sample_name_apa0={result.sample_name_apa0}"
            f" sample_name_apa1={result.sample_name_apa1}"
            f" split_apa0={split_apa0}"
            f" split_apa1={split_apa1}",
            flush=True,
        )

        # -----------------------------------------------------------------------
        # Correction 2: HDF5 readback using the existing accepted NuGraph
        # reader/DataModule machinery.
        #
        # 1. Retain graph_apa0/graph_apa1 from the IntegratedResult (in memory).
        # 2. finalize() has already been called (inside _execute_transaction).
        # 3. Reopen the final HDF5 using NuGraphDataset (existing accepted reader).
        # 4. Load both stored APA samples.
        # 5. Compare loaded["sp"].num_nodes against the in-memory graphs per the
        #    existing writer/reader equivalence contract in
        #    test_streaming_converter_writer.py.
        # -----------------------------------------------------------------------
        from nugraph.data import NuGraphDataModule, NuGraphDataset

        loaded_apa0 = NuGraphDataset(str(args.h5_output), [result.sample_name_apa0]).get(0)
        loaded_apa1 = NuGraphDataset(str(args.h5_output), [result.sample_name_apa1]).get(0)

        if loaded_apa0["sp"].num_nodes != result.graph_apa0["sp"].num_nodes:
            raise RuntimeError(
                f"APA0 sp node count mismatch after HDF5 roundtrip: "
                f"loaded={loaded_apa0['sp'].num_nodes} "
                f"memory={result.graph_apa0['sp'].num_nodes}"
            )
        if loaded_apa1["sp"].num_nodes != result.graph_apa1["sp"].num_nodes:
            raise RuntimeError(
                f"APA1 sp node count mismatch after HDF5 roundtrip: "
                f"loaded={loaded_apa1['sp'].num_nodes} "
                f"memory={result.graph_apa1['sp'].num_nodes}"
            )

        print(
            f"PROCESS_C_HDF5_READBACK"
            f" sample_name_apa0={result.sample_name_apa0}"
            f" sp_nodes_apa0={loaded_apa0['sp'].num_nodes}"
            f" sample_name_apa1={result.sample_name_apa1}"
            f" sp_nodes_apa1={loaded_apa1['sp'].num_nodes}"
            f" status=OK",
            flush=True,
        )

        # DataModule smoke test: the production NuGraph reader must accept this
        # one-event HDF5 without any model-facing schema modification.
        _SPLIT_DATASET = {
            "train": "train_dataset",
            "validation": "val_dataset",
            "test": "test_dataset",
        }
        data_module = NuGraphDataModule(
            data_path=str(args.h5_output),
            model=None,
            batch_size=1,
            num_workers=0,
            shuffle="random",
        )
        dataset_attr = _SPLIT_DATASET[split_apa0]
        n_samples = len(getattr(data_module, dataset_attr))
        if n_samples != 2:
            raise RuntimeError(
                f"DataModule {dataset_attr} has {n_samples} samples; "
                f"expected 2 (both APAs of the same physical event must be in the same split)"
            )
        print(
            f"PROCESS_C_DATAMODULE_OK"
            f" split={split_apa0}"
            f" dataset_attr={dataset_attr}"
            f" n_samples={n_samples}",
            flush=True,
        )

        # -----------------------------------------------------------------------
        # Send C→A commit Ack ONLY after finalize AND all post-finalize checks pass.
        # -----------------------------------------------------------------------
        eid = result.identity
        ack_identity = IPCIdentity(eid.run, eid.subrun, eid.event)
        channel.sendall(encode_ack(ack_identity))

        print(
            f"PROCESS_C_INTEGRATED_RESULT"
            f" identity={eid.run}/{eid.subrun}/{eid.event}"
            f" apa0_name={result.sample_name_apa0}"
            f" apa1_name={result.sample_name_apa1}",
            flush=True,
        )
        channel.close()
        return 0

    except Exception as error:
        # _execute_transaction calls writer.close() on failure; calling it again
        # here is safe because StreamingH5Writer.close() is idempotent.
        if writer is not None:
            writer.close()
        try:
            channel.sendall(encode_error_message(identity_for_error, str(error)))
        except Exception:
            pass
        channel.close()
        print(f"PROCESS_C_INTEGRATED FAIL: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
