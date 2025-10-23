"""NuGraph3 instance decoder"""
from typing import Any, Tuple, Optional

import torch
from torch import nn
from torchmetrics.functional.clustering import adjusted_rand_score
from torch_geometric.data import Batch
from torch_geometric.utils import cumsum, unbatch

from ....util import ObjCondensationLoss
from ..types import Data, N_IT, E_H_IT, N_IP, E_H_IP

# --- Optional cuML (DBSCAN) ---
try:
    from cuml import DBSCAN  # type: ignore
except Exception:  # ModuleNotFoundError etc.
    DBSCAN = None  # type: ignore


class InstanceDecoder(nn.Module):
    """
    NuGraph3 instance decoder module

    Convolve object condensation node embedding into a beta value and a set of
    coordinates for each hit.

    Args:
        hit_features: Number of hit node features
        instance_features: Number of instance features
    """
    def __init__(self, hit_features: int, instance_features: int):
        super().__init__()

        # loss function
        self.loss = ObjCondensationLoss()

        # temperature parameter
        self.temp = nn.Parameter(torch.tensor(0.0))

        # networks
        self.beta_net = nn.Linear(hit_features, 1)
        self.coord_net = nn.Linear(hit_features, instance_features)

        # cuML DBSCAN is required when this decoder is enabled
        if DBSCAN is None:
            raise RuntimeError(
                "cuml (DBSCAN) is required for InstanceDecoder. "
                "Install 'cuml-cu12' or run with --no-instance-head."
            )
        self.dbscan = DBSCAN()

    # pylint: disable=arguments-differ
    def forward(self, data: Data, stage: Optional[str] = None) -> Tuple[torch.Tensor, dict[str, Any]]:
        """
        NuGraph3 instance decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        h = data["hit"]
        device = h.x.device

        # run network and add output to graph object
        h.of = self.beta_net(h.x).squeeze(dim=-1).sigmoid()
        h.ox = self.coord_net(h.x)

        if isinstance(data, Batch):
            # keep PyG bookkeeping in sync
            # pylint: disable=protected-access
            data._slice_dict["hit"]["of"] = h.ptr
            data._slice_dict["hit"]["ox"] = h.ptr
            data._inc_dict["hit"]["of"] = data._inc_dict["hit"]["x"]
            data._inc_dict["hit"]["ox"] = data._inc_dict["hit"]["x"]

        # calculate loss
        loss = self.loss(h.ox, h.of, data.y_i(), h.y_semantic,
                         data[N_IT].num_nodes, data[E_H_IT].edge_index)
        loss *= (-1 * self.temp).exp()
        b, v = loss
        loss = loss.sum() + self.temp

        # metrics
        metrics: dict[str, Any] = {}
        if stage:
            metrics[f"instance/loss-{stage}"] = loss
            metrics[f"instance/bkg-loss-{stage}"] = b
            metrics[f"instance/potential-loss-{stage}"] = v

        # add materialized instances
        mask = torch.ones_like(h.of, dtype=torch.bool, device=device)
        if hasattr(h, "x_filter"):
            mask = mask & (h.x_filter > 0.5)
        if hasattr(h, "x_semantic"):
            # drop background / class 6 if that’s your convention
            mask = mask & (h.x_semantic.argmax(dim=1) != 6)

        if isinstance(data, Batch):
            x_ip_list, e_h_ip_list = [], []

            for ox_g, m_g in zip(unbatch(h.ox, h.batch), unbatch(mask, h.batch)):
                x_g, e_g = self.materialize(ox_g, m_g)
                x_ip_list.append(x_g)
                e_h_ip_list.append(e_g)

            # particle nodes
            if len(x_ip_list) and any(x.numel() for x in x_ip_list):
                data[N_IP].x = torch.cat([x for x in x_ip_list if x.numel()], dim=0)
                # build batch indices
                ip_batch = []
                for i, x in enumerate(x_ip_list):
                    if x.numel():
                        ip_batch.append(torch.full((x.size(0),), i, dtype=torch.long, device=device))
                data[N_IP].batch = torch.cat(ip_batch, dim=0) if ip_batch else torch.empty(0, dtype=torch.long, device=device)
                sizes = torch.tensor([x.size(0) for x in x_ip_list], device=device)
            else:
                feat = h.ox.size(1)
                data[N_IP].x = torch.empty(0, feat, dtype=torch.float, device=device)
                data[N_IP].batch = torch.empty(0, dtype=torch.long, device=device)
                sizes = torch.zeros(data.num_graphs, dtype=torch.long, device=device)

            data[N_IP].ptr = cumsum(sizes)
            # pylint: disable=protected-access
            data._slice_dict[N_IP] = {"x": data[N_IP].ptr}
            data._inc_dict[N_IP] = {"x": torch.zeros(data.num_graphs, dtype=torch.long, device=device)}

            # particle edges (hit -> instance)
            if len(e_h_ip_list) and any(e.numel() for e in e_h_ip_list):
                # increments: add hit/node and particle offsets per-graph
                e_inc = torch.stack((h.ptr[:-1], data[N_IP].ptr[:-1]), dim=1).unsqueeze(2)  # (G, 2, 1)
                data[E_H_IP].edge_index = torch.cat(
                    [e + inc for e, inc in zip(e_h_ip_list, e_inc)], dim=1
                )
                # pylint: disable=protected-access
                data._slice_dict[E_H_IP] = {
                    "edge_index": cumsum(torch.tensor([e.size(1) for e in e_h_ip_list], device=device))
                }
                data._inc_dict[E_H_IP] = {"edge_index": e_inc}
            else:
                data[E_H_IP].edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
                # pylint: disable=protected-access
                data._slice_dict[E_H_IP] = {"edge_index": cumsum(torch.zeros(data.num_graphs, dtype=torch.long, device=device))}
                data._inc_dict[E_H_IP] = {"edge_index": torch.stack((h.ptr[:-1], data[N_IP].ptr[:-1]), dim=1).unsqueeze(2)}

            # adjusted rand score per-graph (ignore negatives)
            rands = []
            for l in data.to_data_list():
                m = l["hit"].y_semantic >= 0
                rands.append(adjusted_rand_score(l.x_i()[m], l.y_i()[m]))
            rand = torch.stack(rands).mean() if rands else torch.tensor(0.0, device=device)

        else:
            data[N_IP].x, data[E_H_IP].edge_index = self.materialize(h.ox, mask)
            rand = adjusted_rand_score(data.x_i(), data.y_i())

        # clamp/validate ARI
        if isinstance(rand, torch.Tensor):
            rand_val = float(rand.detach().cpu().item())
        else:
            rand_val = float(rand)
        if rand_val < -1.0 - 1e-6 or rand_val > 1.0 + 1e-6:
            raise RuntimeError(f"Adjusted Rand Score metric value {rand_val} is outside allowed range!")
        rand = max(min(rand_val, 1.0), -1.0)
        rand = torch.tensor(rand, dtype=torch.float, device=device)

        if stage:
            metrics[f"instance/adjusted-rand-{stage}"] = rand
        if stage == "train":
            metrics["temperature/instance"] = self.temp

        return loss, metrics

    def materialize(self, ox: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Materialize instance embedding.

        Args:
            ox: object condensation embedding tensor (H, F)
            mask: bool mask tensor (H,) for background hit removal

        Returns:
            x_ip: (P, F) particle node features (placeholder: zeros)
            e_h_ip: (2, E) edges from hit indices to particle ids
        """
        device = ox.device
        feat = ox.size(1)

        # no signal hits
        if not mask.any():
            x_ip = torch.empty(0, feat, dtype=torch.float, device=device)
            e_h_ip = torch.empty(2, 0, dtype=torch.long, device=device)
            return x_ip, e_h_ip

        # cuML expects host array (or CuPy) — simplest: NumPy on CPU
        arr = ox[mask].detach().cpu().numpy()
        labels = self.dbscan.fit_predict(arr)  # ndarray on host

        # collect labels back on device; -1 means noise
        i = torch.full((ox.size(0),), -1, dtype=torch.long, device=device)
        i[mask] = torch.as_tensor(labels, dtype=torch.long, device=device)

        # if no clusters (all -1)
        max_lab = int(i.max().item())
        if max_lab < 0:
            x_ip = torch.empty(0, feat, dtype=torch.float, device=device)
            e_h_ip = torch.empty(2, 0, dtype=torch.long, device=device)
            return x_ip, e_h_ip

        # create P particle nodes; placeholder features (zeros)
        x_ip = torch.zeros(max_lab + 1, feat, dtype=torch.float, device=device)

        # edges: hit -> particle_id for all non-noise hits
        hit_idx = torch.nonzero(i >= 0, as_tuple=False).squeeze(1)
        e_h_ip = torch.stack((hit_idx, i[hit_idx]), dim=0).long()

        return x_ip, e_h_ip

    def on_epoch_end(self, logger: "WandbLogger", stage: str, epoch: int) -> None:
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: Wandb logger
            stage: Training stage
            epoch: Training epoch index
        """
        # (no-op here; keep signature in case you want to log custom artifacts)
        return
