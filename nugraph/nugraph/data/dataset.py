# filename: nugraph/data/dataset.py
"""NuGraph dataset"""
from typing import Callable, Optional, List

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

    def __init__(
        self,
        filename: str,
        samples: List[str],
        transform: Optional[Callable] = None,
    ):
        super().__init__(transform=transform)
        # Keep the file open for the lifetime of the dataset
        self.file = h5py.File(filename, "r")
        self.samples = list(samples)

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> NuGraphData:
        """
        Load one event via NuGraphData, then augment the 'hit' store with
        truth-level y_instance and extra per-hit features.
        """
        name = self.samples[idx]
        dset = self.file[f"/dataset/{name}"]    # scalar compound dataset
        rec = dset[()]                          # numpy.void record

        # Base graph: this already builds hit.x, hit.y_semantic, edges, etc.
        data = NuGraphData.load(dset)

        # If there is no hit store (shouldn't happen for your file), just return
        if "hit" not in data.node_stores:
            return data

        hit = data["hit"]

        # ------------------------------------------------------------------
        # 1) Build truth y_instance from per-plane y_instance arrays
        # ------------------------------------------------------------------
        needed_fields = ("u/y_instance", "v/y_instance", "y/y_instance")
        if all(field in rec.dtype.names for field in needed_fields):

            u_inst = np.asarray(rec["u/y_instance"])
            v_inst = np.asarray(rec["v/y_instance"])
            y_inst = np.asarray(rec["y/y_instance"])

            N = hit.pos.size(0)
            y_instance = torch.full((N,), -1, dtype=torch.long)

            if not hasattr(hit, "plane"):
                # If plane metadata is missing, skip instance labeling
                return data

            plane = hit.plane
            mask_u = (plane == 0)
            mask_v = (plane == 1)
            mask_y = (plane == 2)

            # Sanity checks: number of hits per plane must match the per-plane arrays
            if mask_u.sum().item() == len(u_inst) and \
               mask_v.sum().item() == len(v_inst) and \
               mask_y.sum().item() == len(y_inst):

                y_instance[mask_u] = torch.from_numpy(u_inst).long()
                y_instance[mask_v] = torch.from_numpy(v_inst).long()
                y_instance[mask_y] = torch.from_numpy(y_inst).long()

                # Attach to hit store as truth instance labels
                hit.y_instance = y_instance

        # ------------------------------------------------------------------
        # 2) NEW: append 2 extra per-hit features to hit.x
        # ------------------------------------------------------------------
        # At this point, hit.x is whatever NuGraphData.load built
        # (currently 5 features), and PositionFeatures later will
        # concatenate pos (3) to make 8. We now add *two* more so that
        # after PositionFeatures we end up with 10 features total.
        # ------------------------------------------------------------------
        if hasattr(hit, "x") and hasattr(hit, "plane"):
            N = hit.x.size(0)
            plane = hit.plane

            # allocate extra feature tensor (N hits, 2 new features)
            extra = torch.zeros((N, 2), dtype=hit.x.dtype)

            # TODO: replace these field names with your actual datasets
            # e.g. "u/vtx_r" and "u/vtx_zrel", or whatever you wrote in H5
            # Make sure all 6 names exist in rec.dtype.names.
            try:
                u_f0 = np.asarray(rec["u/vtx_feat0"])  # <-- CHANGE NAME
                u_f1 = np.asarray(rec["u/vtx_feat1"])  # <-- CHANGE NAME
                v_f0 = np.asarray(rec["v/vtx_feat0"])  # <-- CHANGE NAME
                v_f1 = np.asarray(rec["v/vtx_feat1"])  # <-- CHANGE NAME
                y_f0 = np.asarray(rec["y/vtx_feat0"])  # <-- CHANGE NAME
                y_f1 = np.asarray(rec["y/vtx_feat1"])  # <-- CHANGE NAME
            except KeyError:
                # If features aren't present, just return without modifying hit.x
                return data

            mask_u = (plane == 0)
            mask_v = (plane == 1)
            mask_y = (plane == 2)

            # sanity: lengths must match hits-per-plane
            if mask_u.sum().item() != len(u_f0) or mask_u.sum().item() != len(u_f1):
                return data
            if mask_v.sum().item() != len(v_f0) or mask_v.sum().item() != len(v_f1):
                return data
            if mask_y.sum().item() != len(y_f0) or mask_y.sum().item() != len(y_f1):
                return data

            # fill extra features per plane
            extra[mask_u, 0] = torch.from_numpy(u_f0)
            extra[mask_u, 1] = torch.from_numpy(u_f1)
            extra[mask_v, 0] = torch.from_numpy(v_f0)
            extra[mask_v, 1] = torch.from_numpy(v_f1)
            extra[mask_y, 0] = torch.from_numpy(y_f0)
            extra[mask_y, 1] = torch.from_numpy(y_f1)

            # concatenate onto existing hit.x
            hit.x = torch.cat([hit.x, extra], dim=-1)
            # (Before PositionFeatures: 5+2 = 7; after cat(pos, x): 3+7 = 10)

            # optional one-time debug print (comment out later)
            # print("DEBUG hit.x shape after augmentation:", hit.x.shape)

        return data
