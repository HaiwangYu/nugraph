# filename: nugraph/data/dataset.py
"""NuGraph dataset"""
from typing import Callable, Optional, List
import os

import h5py
import numpy as np
import torch
from torch_geometric.data import Dataset

from pynuml.data import NuGraphData


class NuGraphDataset(Dataset):
    """NuGraph dataset (HDF5 opened lazily per worker/process)."""

    def __init__(
        self,
        filename: str,
        samples: List[str],
        transform: Optional[Callable] = None,
    ):
        super().__init__(transform=transform)

        # IMPORTANT: do NOT open h5py.File here (breaks num_workers>0 + spawn)
        self.filename = filename
        self.samples = list(samples)

        # Lazy per-process handle (each worker gets its own)
        self._file = None

    # --- NEW ---
    def _get_file(self) -> h5py.File:
        # Each dataloader worker process will have its own Dataset instance,
        # so caching the handle here is safe (and avoids reopening every item).
        if self._file is None:
            self._file = h5py.File(self.filename, "r")
        return self._file

    # --- NEW ---
    def __getstate__(self):
        # When spawning workers, Dataset is pickled. Drop the open handle.
        d = dict(self.__dict__)
        d["_file"] = None
        return d

    # (optional but nice)
    def __del__(self):
        try:
            if self._file is not None:
                self._file.close()
        except Exception:
            pass

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> NuGraphData:
        """
        Load one event via NuGraphData, then:
          - move sp/edge_label_index (+ labels) into an edge store as edge_index
          - attach per-hit y_instance from u/v/y/y_instance (if present)
          - optionally append /sem_features/sp/<event> only when explicitly enabled
          - enforce y_instance[y_semantic==-1] = -1 (ghost masking, in-memory only)
        """
        f = self._get_file()  # <-- NEW: open lazily per process
        name = self.samples[idx]
        dset = f[f"/dataset/{name}"]
        rec = dset[()]  # numpy.void record

        data = NuGraphData.load(dset)

        # ------------------------------------------------------------------
        # (A) CRITICAL FIX: make supervision edges a proper edge store
        # ------------------------------------------------------------------
        if "sp" in data.node_types and hasattr(data["sp"], "edge_label_index"):
            sup_ei = data["sp"].edge_label_index
            data[("sp", "supervision", "sp")].edge_index = sup_ei.long()

            if hasattr(data["sp"], "edge_y"):
                data[("sp", "supervision", "sp")].edge_y = data["sp"].edge_y
            if hasattr(data["sp"], "edge_labelable"):
                data[("sp", "supervision", "sp")].edge_labelable = data["sp"].edge_labelable

            del data["sp"].edge_label_index
            if hasattr(data["sp"], "edge_y"):
                del data["sp"].edge_y
            if hasattr(data["sp"], "edge_labelable"):
                del data["sp"].edge_labelable

        # ------------------------------------------------------------------
        # If no hit store, nothing else to do
        # ------------------------------------------------------------------
        if "hit" not in data.node_types:
            return data

        hit = data["hit"]

        # ------------------------------------------------------------------
        # (B) Attach per-hit truth instance labels from u/v/y/y_instance
        # ------------------------------------------------------------------
        needed_fields = ("u/y_instance", "v/y_instance", "y/y_instance")
        if all(field in rec.dtype.names for field in needed_fields):
            u_inst = np.asarray(rec["u/y_instance"])
            v_inst = np.asarray(rec["v/y_instance"])
            y_inst = np.asarray(rec["y/y_instance"])

            if hasattr(hit, "pos") and hasattr(hit, "plane"):
                N = hit.pos.size(0)
                y_instance = torch.full((N,), -1, dtype=torch.long)
                plane = hit.plane

                mask_u = (plane == 0)
                mask_v = (plane == 1)
                mask_y = (plane == 2)

                if mask_u.sum().item() == len(u_inst) and \
                   mask_v.sum().item() == len(v_inst) and \
                   mask_y.sum().item() == len(y_inst):

                    y_instance[mask_u] = torch.from_numpy(u_inst).long()
                    y_instance[mask_v] = torch.from_numpy(v_inst).long()
                    y_instance[mask_y] = torch.from_numpy(y_inst).long()
                    hit.y_instance = y_instance

        # ------------------------------------------------------------------
        # (C) Optional sidecar semantic features: /sem_features/sp/<event>
        #
        # Default OFF. These were used in earlier experiments and can introduce
        # truth/geometry leakage if an old sidecar-enriched file is used by
        # accident. Re-enable only for explicit ablation studies.
        # ------------------------------------------------------------------
        enable_sem_sidecar = os.environ.get("NUGRAPH_ENABLE_SEM_SIDECAR", "0").lower()
        if enable_sem_sidecar in ("1", "true", "yes", "y") and hasattr(hit, "x"):
            N = hit.x.size(0)
            if "sem_features" in f:
                sf = f["sem_features"]
                if "sp" in sf and name in sf["sp"]:
                    sem_arr = np.asarray(sf["sp"][name])  # (N_hits, F_sem)
                    sem = torch.from_numpy(sem_arr).to(hit.x.dtype)
                    if sem.ndim == 2 and sem.shape[0] == N:
                        hit.x = torch.cat([hit.x, sem], dim=-1)

        # ------------------------------------------------------------------
        # (D) REQUIRED: Semantic + instance hygiene (in-memory only)
        # ------------------------------------------------------------------
        if hasattr(hit, "y_semantic"):
            hit.y_semantic = hit.y_semantic.long()
            ghost = (hit.y_semantic == -1)

            if hasattr(hit, "y_instance"):
                hit.y_instance = hit.y_instance.long().masked_fill(ghost, -1)
            if hasattr(hit, "pid"):
                hit.pid = hit.pid.long().masked_fill(ghost, -1)

        return data
