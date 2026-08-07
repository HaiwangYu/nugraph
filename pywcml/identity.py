"""Immutable physical-event identity and deterministic dataset assignment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib


_SPLIT_BUCKETS = 10_000


@dataclass(frozen=True)
class EventIdentity:
    """Campaign-wide identity and provenance for one physical event.

    APA is deliberately not part of this object: APA0 and APA1 are two graph
    views of the same physical event and must carry identical run/subrun/event
    values, random seed, and dataset split.
    """

    campaign_id: str
    shard_id: int
    source_index: int
    run: int
    subrun: int
    event: int
    random_seed: int

    def __post_init__(self) -> None:
        if not self.campaign_id:
            raise ValueError("campaign_id must be non-empty")
        for name in ("shard_id", "source_index", "run", "subrun", "event", "random_seed"):
            value = getattr(self, name)
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def physical_key(self) -> tuple[str, int, int, int]:
        """Stable key used for physical-event-level partitioning."""

        return self.campaign_id, self.run, self.subrun, self.event

    def sample_name(self, apa: int) -> str:
        """Return the existing WCML-style deterministic HDF5 sample name."""

        apa = _validate_apa(apa)
        return f"{self.run}_{self.subrun}_rec-lab-apa{apa}-{self.event}"

    def split(self, train_fraction: float = 0.70, val_fraction: float = 0.15) -> str:
        """Assign this physical event deterministically to train/validation/test."""

        if not 0.0 <= train_fraction <= 1.0:
            raise ValueError("train_fraction must be in [0, 1]")
        if not 0.0 <= val_fraction <= 1.0:
            raise ValueError("val_fraction must be in [0, 1]")
        if train_fraction + val_fraction > 1.0:
            raise ValueError("train_fraction + val_fraction must be <= 1")

        payload = "\0".join(map(str, self.physical_key)).encode("utf-8")
        bucket = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % _SPLIT_BUCKETS
        train_end = int(train_fraction * _SPLIT_BUCKETS)
        val_end = train_end + int(val_fraction * _SPLIT_BUCKETS)
        if bucket < train_end:
            return "train"
        if bucket < val_end:
            return "validation"
        return "test"


def _validate_apa(apa: int) -> int:
    if isinstance(apa, bool) or int(apa) not in (0, 1):
        raise ValueError(f"APA must be 0 or 1, got {apa!r}")
    return int(apa)


__all__ = ["EventIdentity"]
