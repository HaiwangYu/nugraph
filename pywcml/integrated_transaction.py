"""Combined SBNDIPC transaction receiver for the integrated Process-C pipeline.

Receives a mixed-type stream of RecoArrays and TruthArrays messages for one
physical event, validates both pairs agree on run/subrun/event identity, runs
label_event and convert_arrays for each APA, and appends the results to the
streaming HDF5 writer.  The caller is responsible for calling
writer.finalize() and sending the Ack, in that order, only if this function
returns without exception.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import socket
from typing import TYPE_CHECKING

from pynuml.data import NuGraphData

from .identity import EventIdentity
from .io import WCMLArrays
from .labeling import LabelingConfig, label_event
from .recoarrays_ipc import (
    END_OF_STREAM,
    ERROR,
    RECO_ARRAYS_APA0,
    RECO_ARRAYS_APA1,
    IPCError,
    IPCIdentity,
    RecoArraysLimits,
    RecoArraysPair,
    RecoArraysTransaction,
    decode_reco_arrays_message,
    decode_sbndipc_message,
    read_sbndipc_message,
)
from .trutharrays_ipc import (
    TRUTH_ARRAYS,
    TruthArraysLimits,
    TruthArraysPair,
    TruthArraysTransaction,
    decode_truth_arrays_message,
)

if TYPE_CHECKING:
    from .converter import WCMLConverter
    from .h5writer import StreamingH5Writer


@dataclass(frozen=True, slots=True)
class CombinedLimits:
    """Decode limits for both RecoArrays and TruthArrays in a combined stream."""

    reco: RecoArraysLimits = field(default_factory=RecoArraysLimits)
    truth: TruthArraysLimits = field(default_factory=TruthArraysLimits)


@dataclass(frozen=True, slots=True)
class IntegratedResult:
    """All outputs of a successful integrated event pipeline run."""

    identity: EventIdentity
    reco_pair: RecoArraysPair
    truth_pair: TruthArraysPair
    labeled_apa0: WCMLArrays
    labeled_apa1: WCMLArrays
    graph_apa0: NuGraphData
    graph_apa1: NuGraphData
    sample_name_apa0: str
    sample_name_apa1: str


def _receive_combined_pair(
    channel: socket.socket,
    campaign_id: str,
    shard_id: int,
    source_index: int,
    random_seed: int,
    limits: CombinedLimits,
) -> tuple[RecoArraysPair, TruthArraysPair]:
    """Read a mixed RecoArrays+TruthArrays stream and return validated pairs.

    The dispatch loop accepts messages in any order (reco before truth or vice
    versa).  EndOfStream must be the last message in the stream.  The physical
    run/subrun/event identity must agree across all messages and the EOS
    sentinel.
    """

    reco_tx = RecoArraysTransaction()
    truth_tx = TruthArraysTransaction(campaign_id, shard_id, source_index, random_seed)
    max_payload = max(
        limits.reco.max_message_payload_bytes,
        limits.truth.max_message_payload_bytes,
    )
    eos_identity: IPCIdentity | None = None

    while True:
        encoded = read_sbndipc_message(channel, max_payload)
        if encoded is None:
            raise IPCError("peer closed before end-of-stream in combined transaction")
        message = decode_sbndipc_message(encoded, max_payload)

        if message.kind == END_OF_STREAM:
            if len(message.payload):
                raise IPCError("end-of-stream payload is not empty")
            eos_identity = message.identity
            break
        elif message.kind in (RECO_ARRAYS_APA0, RECO_ARRAYS_APA1):
            decoded_reco = decode_reco_arrays_message(encoded, limits=limits.reco)
            reco_tx.add(decoded_reco)
        elif message.kind == TRUTH_ARRAYS:
            decoded_truth = decode_truth_arrays_message(encoded, limits=limits.truth)
            truth_tx.add(decoded_truth)
        elif message.kind == ERROR:
            raise IPCError("received an error message in combined transaction")
        else:
            raise IPCError(
                f"unexpected message type {message.kind!r} in combined transaction"
            )

    reco_pair = reco_tx.finish()
    truth_pair = truth_tx.finish()

    reco_phys = (reco_pair.identity.run, reco_pair.identity.subrun, reco_pair.identity.event)
    truth_phys = (
        truth_pair.identity.run,
        truth_pair.identity.subrun,
        truth_pair.identity.event,
    )
    assert eos_identity is not None
    eos_phys = (eos_identity.run, eos_identity.subrun, eos_identity.event)

    if reco_phys != truth_phys:
        raise IPCError(
            f"reco and truth physical identity mismatch: reco={reco_phys} truth={truth_phys}"
        )
    if eos_phys != reco_phys:
        raise IPCError(
            f"end-of-stream physical identity mismatch: eos={eos_phys} expected={reco_phys}"
        )

    return reco_pair, truth_pair


def run_integrated_event(
    channel: socket.socket,
    *,
    campaign_id: str,
    shard_id: int,
    source_index: int,
    random_seed: int,
    config: LabelingConfig = LabelingConfig(),
    converter: "WCMLConverter",
    writer: "StreamingH5Writer",
    limits: CombinedLimits = CombinedLimits(),
) -> IntegratedResult:
    """Receive one combined event, label both APAs, convert, and append to HDF5.

    The caller must call ``writer.finalize()`` and then send an Ack, in that
    order, only when this function returns without exception.  This function
    never calls ``finalize()`` or sends the Ack itself.

    Any exception from any step (receive, label_event, convert_arrays,
    append_event) propagates directly to the caller.
    """

    reco_pair, truth_pair = _receive_combined_pair(
        channel, campaign_id, shard_id, source_index, random_seed, limits
    )

    identity: EventIdentity = truth_pair.identity

    labeled_apa0 = label_event(
        reco_pair.apa0,
        truth_pair.apa0.semantic_truth,
        truth_pair.apa0.simide_truth,
        truth_pair.apa0.neutrino_vertex,
        identity,
        0,
        config,
    )
    labeled_apa1 = label_event(
        reco_pair.apa1,
        truth_pair.apa1.semantic_truth,
        truth_pair.apa1.simide_truth,
        truth_pair.apa1.neutrino_vertex,
        identity,
        1,
        config,
    )

    graph_apa0 = converter.convert_arrays(labeled_apa0, identity, 0)
    graph_apa1 = converter.convert_arrays(labeled_apa1, identity, 1)

    sample_name_apa0, sample_name_apa1 = writer.append_event(identity, graph_apa0, graph_apa1)

    return IntegratedResult(
        identity=identity,
        reco_pair=reco_pair,
        truth_pair=truth_pair,
        labeled_apa0=labeled_apa0,
        labeled_apa1=labeled_apa1,
        graph_apa0=graph_apa0,
        graph_apa1=graph_apa1,
        sample_name_apa0=sample_name_apa0,
        sample_name_apa1=sample_name_apa1,
    )


__all__ = [
    "CombinedLimits",
    "IntegratedResult",
    "run_integrated_event",
]
