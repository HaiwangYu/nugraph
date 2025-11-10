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
from torch.utils.data import WeightedRandomSampler  # NEW: for ν-hit weighted sampling
from pytorch_lightning import LightningDataModule

from ..data import NuGraphDataset, BalanceSampler

DEFAULT_DATA = ("/exp/sbnd/app/users/yuhw/nugraph/test/NG2-paper.gnn.keep1.h5")

class NuGraphDataModule(LightningDataModule):
    """PyTorch Lightning data module for neutrino graph data."""
    def __init__(self,
                 data_path: str = "auto",
                 model: type[torch.nn.Module] = None,
                 batch_size: int = 64,
                 num_workers: int = 5,
                 shuffle: str = 'random',
                 balance_frac: float = 0.1,
                 min_nu_hits: int = 0  # neutrino-hit cut (0 = no cut)
                 ):
        super().__init__()

        # for this HDF5 dataloader, worker processes slow things down
        # so we silence PyTorch Lightning's warnings
        warnings.filterwarnings("ignore", ".*does not have many workers.*")

        if data_path == "auto":
            data_path = DEFAULT_DATA
        self.filename = os.path.expandvars(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers

        # allow "weighted" in addition to your existing choices
        if shuffle not in ("random", "balance", "weighted"):
            print('shuffle argument must be "random", "balance", or "weighted".')
            sys.exit()
        self.shuffle = shuffle
        self.balance_frac = balance_frac
        self.min_nu_hits = int(min_nu_hits) if min_nu_hits is not None else 0

        # will hold per-train-sample weights when using weighted sampling
        self.train_event_weights = None  # NEW

        with h5py.File(self.filename) as f:

            # load metadata
            try:
                # pylint: disable=no-member
                self.planes = f['planes'].asstr()[()].tolist()
                self.semantic_classes = f['semantic_classes'].asstr()[()].tolist()
            except KeyError:
                print(("Metadata not found in file! "
                       "\"planes\" and \"semantic_classes\" are required."))
                sys.exit()

            # get graph structure generation
            try:
                # pylint: disable=no-member
                self.gen = f["gen"][()].item()
            except KeyError:
                self.gen = 1

            # load optional event labels
            if 'event_classes' in f:
                # pylint: disable=no-member
                self.event_classes = f['event_classes'].asstr()[()].tolist()
            else:
                self.event_classes = None

            # load sample splits
            try:
                # pylint: disable=no-member
                train_samples = f['samples/train'].asstr()[()]
                val_samples   = f['samples/validation'].asstr()[()]
                test_samples  = f['samples/test'].asstr()[()]
            except KeyError:
                print(("Sample splits not found in file! "
                       "Call \"generate_samples\" to create them."))
                sys.exit()

            # load data sizes
            try:
                self.train_datasize = f['datasize/train'][()]
            except KeyError:
                print(("Data size array not found in file! "
                       "Call \"generate_samples\" to create it."))
                sys.exit()

            # -------- neutrino-hit helpers (used by filtering and weighting) --------
            try:
                nu_index = int(self.semantic_classes.index('nu'))
            except Exception:
                nu_index = 0

            def count_nu_hits(rec: np.void, nu_idx: int = 0) -> int:
                total = 0
                for pl in ("u", "v", "y"):
                    key = f"{pl}/y_semantic"
                    if key in rec.dtype.names:
                        arr = rec[key]
                        if not isinstance(arr, np.ndarray):
                            arr = np.asarray(arr)
                        total += int((arr == nu_idx).sum())
                return total

            # -------- apply min_nu_hits cut to all splits (if requested) --------
            if self.min_nu_hits > 0:

                def keep_mask_for(keys_array):
                    """Return boolean mask of which keys pass the min_nu_hits cut."""
                    mask = np.zeros(len(keys_array), dtype=bool)
                    for i, name in enumerate(keys_array):
                        rec = f['dataset'][name][()]  # scalar compound
                        mask[i] = (count_nu_hits(rec, nu_index) >= self.min_nu_hits)
                    return mask

                # Train split
                train_samples_np = np.asarray(train_samples)
                train_mask = keep_mask_for(train_samples_np)
                if not train_mask.any():
                    print(f"[Data] min_nu_hits={self.min_nu_hits}: train 0 kept (all filtered).")
                before, after = len(train_samples_np), int(train_mask.sum())
                if after != before:
                    print(f"[Data] min_nu_hits={self.min_nu_hits}: train {before} -> {after}")
                train_samples = train_samples_np[train_mask]

                # Keep train_datasize aligned with filtered train_samples
                if len(self.train_datasize) == before:
                    self.train_datasize = self.train_datasize[train_mask]
                else:
                    print("[Data] Warning: datasize/train length mismatch after filtering; "
                          "BalanceSampler may not be used effectively for this run.")

                # Validation split
                val_samples_np = np.asarray(val_samples)
                val_mask = keep_mask_for(val_samples_np)
                before, after = len(val_samples_np), int(val_mask.sum())
                if after != before:
                    print(f"[Data] min_nu_hits={self.min_nu_hits}: validation {before} -> {after}")
                val_samples = val_samples_np[val_mask]

                # Test split
                test_samples_np = np.asarray(test_samples)
                test_mask = keep_mask_for(test_samples_np)
                before, after = len(test_samples_np), int(test_mask.sum())
                if after != before:
                    print(f"[Data] min_nu_hits={self.min_nu_hits}: test {before} -> {after}")
                test_samples = test_samples_np[test_mask]

            # -------- build ν-hit-based sampling weights for TRAIN (shuffle=weighted) --------
            if self.shuffle == "weighted":
                train_samples_np = np.asarray(train_samples)
                if len(train_samples_np) > 0:
                    nu_hits = np.empty(len(train_samples_np), dtype=np.int64)
                    for i, name in enumerate(train_samples_np):
                        rec = f['dataset'][name][()]  # scalar compound
                        nu_hits[i] = count_nu_hits(rec, nu_index)

                    # Up-weight small-ν events smoothly:
                    # w = (nu_hits / median)^(-alpha), clipped to [0.5,5.0], normalized
                    alpha = 1.0
                    med = max(1.0, float(np.median(nu_hits)))
                    w = (np.maximum(nu_hits, 1) / med) ** (-alpha)
                    w = np.clip(w, 0.5, 5.0)
                    w = w * (len(w) / w.sum())  # optional normalization
                    self.train_event_weights = torch.as_tensor(w, dtype=torch.double)
                else:
                    self.train_event_weights = None
            # --------------------------------------------------------------------

        transform = model.transform(self.planes) if model else None

        self.train_dataset = NuGraphDataset(self.filename, train_samples, transform)
        self.val_dataset   = NuGraphDataset(self.filename, val_samples,   transform)
        self.test_dataset  = NuGraphDataset(self.filename, test_samples,  transform)

    @staticmethod
    def generate_samples(data_path: str):
        with h5py.File(data_path) as f:
            samples = list(f['dataset'].keys())
        split = int(0.05 * len(samples))
        splits = [ len(samples)-(2*split), split, split ]
        train, val, test = torch.utils.data.random_split(samples, splits)

        with h5py.File(data_path, "r+") as f:
            for name in [ 'train', 'validation', 'test' ]:
                key = f'samples/{name}'
                if key in f:
                    del f[key]

        with h5py.File(data_path, "r+") as f:
            f.create_dataset("samples/train", data=list(train))
            f.create_dataset("samples/validation", data=list(val))
            f.create_dataset("samples/test", data=list(test))

        with h5py.File(data_path, "r+") as f:
            try:
                planes = f['planes'].asstr()[()].tolist()
            except:
                print('Metadata not found in file! "planes" is required.')
                sys.exit()

        with h5py.File(data_path, "r+") as f:
            if 'datasize/train' in f:
                del f['datasize/train']
        # NOTE: PositionFeatures import not shown here; assumed available in your repo
        transform = PositionFeatures(planes)
        dataset = NuGraphDataset(data_path, train, transform)
        def datasize(data):
            ret = 0
            for store in data.stores:
                for val in store.values():
                    ret += val.element_size() * val.nelement()
            return ret
        dsize = [datasize(data) for data in tqdm.tqdm(dataset)]
        del dataset
        with h5py.File(data_path, "r+") as f:
            f.create_dataset('datasize/train', data=dsize)

    def train_dataloader(self) -> DataLoader:
        train_len = len(self.train_dataset)
        drop_last = train_len >= self.batch_size

        sampler = None
        shuffle = True

        if self.shuffle == 'balance' and drop_last:
            shuffle = False
            sampler = BalanceSampler.BalanceSampler(
                        datasize=self.train_datasize,
                        batch_size=self.batch_size,
                        balance_frac=self.balance_frac)

        elif self.shuffle == 'weighted' and self.train_event_weights is not None:
            # Weighted sampling based on per-event ν-hit counts
            shuffle = False
            sampler = WeightedRandomSampler(
                weights=self.train_event_weights,
                num_samples=train_len,  # draw one epoch worth of samples
                replacement=True
            )

        # else: fallback to standard random shuffle

        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=sampler,
                          drop_last=drop_last,
                          shuffle=shuffle,
                          pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, num_workers=self.num_workers,
                          batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, num_workers=self.num_workers,
                          batch_size=self.batch_size)

    @staticmethod
    def add_data_args(parser: ArgumentParser) -> ArgumentParser:
        data = parser.add_argument_group('data', 'Data module configuration')
        data.add_argument('--data-path', type=str, default="auto",
                          help='Location of input data file')
        data.add_argument('--batch-size', type=int, default=64,
                          help='Size of each batch of graphs')
        data.add_argument('--num-workers', type=int, default=5,
                          help='Number of data loader worker processes')
        data.add_argument('--limit_train_batches', type=int, default=None,
                          help='Max number of training batches to be used')
        data.add_argument('--limit_val_batches', type=int, default=None,
                          help='Max number of validation batches to be used')
        data.add_argument('--shuffle', type=str, default='balance',
                          help='Dataset shuffling scheme to use: random | balance | weighted')
        data.add_argument('--balance-frac', type=float, default=0.1,
                          help='Fraction of dataset to use for workload balancing')
        data.add_argument('--min-nu-hits', type=int, default=0,
                          help='Require at least this many ν hits per event across U/V/Y; 0 disables the cut.')
        return parser
