# filename: nugraph/data/data_module.py
"""NuGraph data module"""
from argparse import ArgumentParser
import warnings

import os
import sys
import h5py
import tqdm
import numpy as np  # used for filtering masks & weights

import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from pytorch_lightning import LightningDataModule

from ..data import NuGraphDataset, BalanceSampler

DEFAULT_DATA = ("/exp/sbnd/app/users/yuhw/nugraph/test/NG2-paper.gnn.keep1.h5")


class NuGraphDataModule(LightningDataModule):
    """PyTorch Lightning data module for neutrino graph data."""
    def __init__(
        self,
        data_path: str = "auto",
        model: type[torch.nn.Module] = None,
        batch_size: int = 64,
        num_workers: int = 5,
        shuffle: str = "random",
        balance_frac: float = 0.1,
        min_nu_hits: int = 0,
        train_fraction: float = 1.0,
        in_features: int = 4,   # final leakage-safe hit.x dim after Transform
    ):
        super().__init__()

        warnings.filterwarnings("ignore", ".*does not have many workers.*")

        if data_path == "auto":
            data_path = DEFAULT_DATA

        self.train_fraction = float(train_fraction)
        self.filename = os.path.expandvars(data_path)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.in_features = int(in_features)  # <-- NEW

        # Dataloader knobs (make them overridable by train.py)
        self.pin_memory = False
        self.persistent_workers = True
        self.prefetch_factor = 2
        self.timeout = 0
        self.drop_last = True

        if shuffle not in ("random", "balance", "weighted"):
            raise ValueError('shuffle must be "random", "balance", or "weighted".')
        self.shuffle = shuffle
        self.balance_frac = float(balance_frac)
        self.min_nu_hits = int(min_nu_hits) if min_nu_hits is not None else 0

        self.train_event_weights = None

        with h5py.File(self.filename, "r") as f:
            # metadata
            try:
                self.planes = f["planes"].asstr()[()].tolist()
                self.semantic_classes = f["semantic_classes"].asstr()[()].tolist()
            except KeyError as e:
                raise RuntimeError(
                    "Metadata not found in file: 'planes' and 'semantic_classes' are required."
                ) from e

            # graph structure generation
            try:
                self.gen = f["gen"][()].item()
            except KeyError:
                self.gen = 1

            # optional event labels
            if "event_classes" in f:
                self.event_classes = f["event_classes"].asstr()[()].tolist()
            else:
                self.event_classes = None

            # sample splits
            try:
                train_samples = f["samples/train"].asstr()[()]
                if "samples/validation" in f:
                    val_samples = f["samples/validation"].asstr()[()]
                elif "samples/val" in f:
                    print("[Data] Using samples/val as validation split.")
                    val_samples = f["samples/val"].asstr()[()]
                else:
                    raise KeyError("samples/validation")
                test_samples = f["samples/test"].asstr()[()]
            except KeyError as e:
                raise RuntimeError(
                    'Sample splits not found in file! Call "generate_samples" to create them.'
                ) from e

            # datasize/train (for BalanceSampler)
            try:
                if "datasize/train" in f:
                    self.train_datasize = f["datasize/train"][()]
                elif "datasize" in f and isinstance(f["datasize"], h5py.Dataset):
                    print("[Data] datasize is split counts; using unit train sizes for random/weighted sampling.")
                    self.train_datasize = np.ones(len(train_samples), dtype=np.int64)
                else:
                    raise KeyError("datasize/train")
            except KeyError as e:
                raise RuntimeError(
                    'Data size array not found in file! Call "generate_samples" to create it.'
                ) from e

            # --- train_fraction sub-sampling (before min_nu_hits filtering) ---
            if 0.0 < self.train_fraction < 1.0:
                rng = np.random.default_rng(1337)
                train_samples_np = np.asarray(train_samples)
                n_total = len(train_samples_np)
                n_keep = max(1, int(round(self.train_fraction * n_total)))
                idx = rng.choice(n_total, size=n_keep, replace=False)
                idx.sort()

                print(f"[Data] train_fraction={self.train_fraction:.2f}: train {n_total} -> {n_keep}")
                train_samples = train_samples_np[idx]

                if len(self.train_datasize) == n_total:
                    self.train_datasize = self.train_datasize[idx]
                else:
                    print("[Data] Warning: datasize/train length mismatch; BalanceSampler may be off.")

            # ---- helpers for min_nu_hits + weighted sampling ----
            try:
                nu_index = int(self.semantic_classes.index("nu"))
            except Exception:
                nu_index = 0

            def count_nu_hits(rec: np.void, nu_idx: int) -> int:
                total = 0
                for pl in ("u", "v", "y"):
                    key = f"{pl}/y_semantic"
                    if key in rec.dtype.names:
                        arr = np.asarray(rec[key])
                        total += int((arr == nu_idx).sum())
                return total

            # ---- min_nu_hits filter on all splits ----
            if self.min_nu_hits > 0:

                def keep_mask_for(keys_array):
                    mask = np.zeros(len(keys_array), dtype=bool)
                    for i, name in enumerate(keys_array):
                        rec = f["dataset"][name][()]
                        mask[i] = (count_nu_hits(rec, nu_index) >= self.min_nu_hits)
                    return mask

                # train
                train_samples_np = np.asarray(train_samples)
                train_mask = keep_mask_for(train_samples_np)
                before, after = len(train_samples_np), int(train_mask.sum())
                print(f"[Data] min_nu_hits={self.min_nu_hits}: train {before} -> {after}")
                train_samples = train_samples_np[train_mask]

                if len(self.train_datasize) == before:
                    self.train_datasize = self.train_datasize[train_mask]
                else:
                    print("[Data] Warning: datasize/train length mismatch after filtering.")

                # val
                val_samples_np = np.asarray(val_samples)
                val_mask = keep_mask_for(val_samples_np)
                before, after = len(val_samples_np), int(val_mask.sum())
                print(f"[Data] min_nu_hits={self.min_nu_hits}: validation {before} -> {after}")
                val_samples = val_samples_np[val_mask]

                # test
                test_samples_np = np.asarray(test_samples)
                test_mask = keep_mask_for(test_samples_np)
                before, after = len(test_samples_np), int(test_mask.sum())
                print(f"[Data] min_nu_hits={self.min_nu_hits}: test {before} -> {after}")
                test_samples = test_samples_np[test_mask]

            # ---- weighted sampling weights for TRAIN ----
            if self.shuffle == "weighted":
                train_samples_np = np.asarray(train_samples)
                if len(train_samples_np) > 0:
                    nu_hits = np.empty(len(train_samples_np), dtype=np.int64)
                    for i, name in enumerate(train_samples_np):
                        rec = f["dataset"][name][()]
                        nu_hits[i] = count_nu_hits(rec, nu_index)

                    alpha = 1.0
                    med = max(1.0, float(np.median(nu_hits)))

                    # Oversample ν-rich events (larger nu_hits -> larger weight)
                    w = (np.maximum(nu_hits, 1) / med) ** (alpha)

                    # Keep weights sane
                    w = np.clip(w, 0.5, 5.0)

                    # Normalize to mean=1 (nice for sampler stability)
                    w = w * (len(w) / w.sum())

                    self.train_event_weights = torch.as_tensor(w, dtype=torch.double)
                else:
                    self.train_event_weights = None

        # model is a class; Transform must be driven by explicit in_features
        transform = model.transform(self.planes, in_features=self.in_features) if model else None

        self.train_dataset = NuGraphDataset(self.filename, train_samples, transform)
        self.val_dataset = NuGraphDataset(self.filename, val_samples, transform)
        self.test_dataset = NuGraphDataset(self.filename, test_samples, transform)

    def train_dataloader(self) -> DataLoader:
        train_len = len(self.train_dataset)
        # =========================
        # DEBUG: epoch size sanity
        # =========================
        print("\n================ TRAIN DATALOADER DEBUG ================")
        print("train_dataset len        =", train_len)
        print("batch_size               =", self.batch_size)
        print("shuffle mode             =", self.shuffle)
        print("min_nu_hits              =", self.min_nu_hits)
        print("train_fraction           =", self.train_fraction)
        print("train_event_weights is None?", self.train_event_weights is None)
        if self.train_event_weights is not None:
            print("train_event_weights len  =", len(self.train_event_weights))
            print("train_event_weights mean =", float(self.train_event_weights.mean()))
        print("========================================================\n")
        drop_last = bool(self.drop_last) and (train_len >= self.batch_size)

        sampler = None
        shuffle = True

        if self.shuffle == "balance" and drop_last:
            shuffle = False
            sampler = BalanceSampler.BalanceSampler(
                datasize=self.train_datasize,
                batch_size=self.batch_size,
                balance_frac=self.balance_frac,
            )

        elif self.shuffle == "weighted" and self.train_event_weights is not None:
            shuffle = False
            sampler = WeightedRandomSampler(
                weights=self.train_event_weights,
                num_samples=train_len,
                replacement=True,
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=sampler,
            drop_last=drop_last,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            persistent_workers=(self.persistent_workers and self.num_workers > 0),
            prefetch_factor=(self.prefetch_factor if self.num_workers > 0 else None),
            timeout=self.timeout,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=(self.persistent_workers and self.num_workers > 0),
            prefetch_factor=(self.prefetch_factor if self.num_workers > 0 else None),
            timeout=self.timeout,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=(self.persistent_workers and self.num_workers > 0),
            prefetch_factor=(self.prefetch_factor if self.num_workers > 0 else None),
            timeout=self.timeout,
        )

    @staticmethod
    def add_data_args(parser: ArgumentParser) -> ArgumentParser:
        data = parser.add_argument_group("data", "Data module configuration")
        data.add_argument("--data-path", type=str, default="auto",
                          help="Location of input data file")
        data.add_argument("--batch-size", type=int, default=64,
                          help="Size of each batch of graphs")
        data.add_argument("--num-workers", type=int, default=5,
                          help="Number of data loader worker processes")
        data.add_argument("--limit_train_batches", type=int, default=None,
                          help="Max number of training batches to be used")
        data.add_argument("--limit_val_batches", type=int, default=None,
                          help="Max number of validation batches to be used")
        data.add_argument("--shuffle", type=str, default="balance",
                          help="Dataset shuffling scheme: random | balance | weighted")
        data.add_argument("--balance-frac", type=float, default=0.1,
                          help="Fraction of dataset to use for workload balancing")
        data.add_argument("--min-nu-hits", type=int, default=0,
                          help="Require at least this many ν hits per event across U/V/Y; 0 disables.")
        return parser
