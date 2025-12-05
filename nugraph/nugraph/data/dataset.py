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
        Load one event via NuGraphData, then augment the 'hit' store with:
          1) truth-level y_instance (per-hit instance IDs)
          2) extra per-hit semantic features from sidecar HDF5 group
             /sem_features/sp (e.g. d_wall, d_top, linearity, sphericity, ty, tz).
        """
        name = self.samples[idx]
        dset = self.file[f"/dataset/{name}"]    # scalar compound dataset
        rec = dset[()]                          # numpy.void record

        # Base graph: builds hit.x, hit.y_semantic, edges, plane nodes, etc.
        data = NuGraphData.load(dset)

        if "hit" not in data.node_stores:
            return data

        hit = data["hit"]

        # ------------------------------------------------------------------
        # 1) Attach per-hit truth instance labels from u/v/y/y_instance
        # ------------------------------------------------------------------
        needed_fields = ("u/y_instance", "v/y_instance", "y/y_instance")
        if all(field in rec.dtype.names for field in needed_fields):

            u_inst = np.asarray(rec["u/y_instance"])
            v_inst = np.asarray(rec["v/y_instance"])
            y_inst = np.asarray(rec["y/y_instance"])

            if not hasattr(hit, "pos") or not hasattr(hit, "plane"):
                return data

            N = hit.pos.size(0)
            y_instance = torch.full((N,), -1, dtype=torch.long)
            plane = hit.plane

            mask_u = (plane == 0)
            mask_v = (plane == 1)
            mask_y = (plane == 2)

            # Sanity: hits per plane must match lengths of per-plane arrays
            if mask_u.sum().item() == len(u_inst) and \
               mask_v.sum().item() == len(v_inst) and \
               mask_y.sum().item() == len(y_inst):

                y_instance[mask_u] = torch.from_numpy(u_inst).long()
                y_instance[mask_v] = torch.from_numpy(v_inst).long()
                y_instance[mask_y] = torch.from_numpy(y_inst).long()

                hit.y_instance = y_instance  # truth instance IDs per hit

        # ------------------------------------------------------------------
        # 2) Append sidecar semantic features to hit.x
        #
        # Sidecar layout (per event):
        #   /sem_features/sp/<name>  ->  (N_hits, F_sem)
        #
        # We *do not* add any separate vertex features here anymore.
        # Vertex info is already encoded at the sp-node level (sp.features)
        # and also available for building the sidecar.
        # ------------------------------------------------------------------
        if hasattr(hit, "x"):
            N = hit.x.size(0)

            if "sem_features" in self.file:
                sp_group = self.file["sem_features"]
                if "sp" in sp_group and name in sp_group["sp"]:
                    sem_arr = np.asarray(sp_group["sp"][name])  # (N_hits, F_sem)
                    sem = torch.from_numpy(sem_arr).to(hit.x.dtype)

                    # Require exact per-hit alignment
                    if sem.ndim == 2 and sem.shape[0] == N:
                        hit.x = torch.cat([hit.x, sem], dim=-1)

                        # Optional debug once:
                        # print(f"[DEBUG] {name}: hit.x.shape after sidecar = {hit.x.shape}")
                    # else:
                        # print(f"[WARN] sem_features/sp/{name} shape {sem.shape} != (N_hits={N}, F_sem)")

        return data
