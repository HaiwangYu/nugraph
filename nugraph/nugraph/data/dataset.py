"""NuGraph dataset"""
from typing import Callable, Optional

import h5py
import numpy as np
import torch
from torch_geometric.data import Dataset

from pynuml.data import NuGraphData


class NuGraphDataset(Dataset):
    """NuGraph dataset

    Args:
        filename: Name of dataset file
        samples: List of graph object dataset names in file
        transform: Transforms to apply to graph objects
    """
    def __init__(self,
                 filename: str,
                 samples: list[str],
                 transform: Optional[Callable] = None):
        super().__init__(transform=transform)
        self.file = h5py.File(filename)
        self.samples = samples

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> NuGraphData:
        """
        Load one event via NuGraphData, then *augment* hit with y_instance
        built from per-plane y_instance arrays in the raw HDF5 record.
        """
        key = f"/dataset/{self.samples[idx]}"
        dset = self.file[key]          # h5py Dataset (scalar compound)
        data = NuGraphData.load(dset)  # usual NuGraphData (with hit, edges, etc.)

        # If transform is attached, PyG will apply it *after* this get(),
        # so we modify 'data' now, before transform() runs.
        if "hit" not in data.node_stores:
            return data

        hit = data["hit"]

        # If the raw record doesn't have per-plane instance labels, just return.
        rec = dset[()]  # numpy.void with fields like 'u/y_instance', 'v/y_instance', ...
        needed = ("u/y_instance", "v/y_instance", "y/y_instance")
        if not all(name in dset.dtype.names for name in needed):
            return data

        # Fetch per-plane instance arrays
        u_inst = np.asarray(rec["u/y_instance"])
        v_inst = np.asarray(rec["v/y_instance"])
        y_inst = np.asarray(rec["y/y_instance"])

        # Allocate final y_instance aligned with hit nodes
        N = hit.pos.size(0)
        y_instance = torch.empty(N, dtype=torch.long)

        # Map per-plane arrays onto hit nodes using the 'plane' attribute:
        # plane == 0 -> U-plane hits, 1 -> V, 2 -> Y (this matches NuGraphData convention)
        plane_mask_u = (hit.plane == 0)
        plane_mask_v = (hit.plane == 1)
        plane_mask_y = (hit.plane == 2)

        # Sanity: lengths should match
        if plane_mask_u.sum().item() != len(u_inst):
            # If something is off, just bail out rather than crashing training
            return data
        if plane_mask_v.sum().item() != len(v_inst):
            return data
        if plane_mask_y.sum().item() != len(y_inst):
            return data

        y_instance[plane_mask_u] = torch.from_numpy(u_inst).long()
        y_instance[plane_mask_v] = torch.from_numpy(v_inst).long()
        y_instance[plane_mask_y] = torch.from_numpy(y_inst).long()

        # Attach to hit store
        hit.y_instance = y_instance

        return data
