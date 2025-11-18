import torch
import torch.nn.functional as F
from torch import nn

from ..nugraph3.nugraph3 import NuGraph3  # as before


class NuGraph4(NuGraph3):
    def __init__(
        self,
        *args,
        edge_hidden_dim: int = 32,
        embed_dim: int = 64,
        lambda_edge: float = 0.1,
        edge_pos_weight: float = 3.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # --- Edge MLP to learn per-edge logits from 7-D features ---
        in_edge_dim = 7  # [dx, dy, dz, dr, plane_src, charge_diff, time_diff]
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, 1),
        )

        # Embedding-based semantic decoder (we can keep this, even if not heavily used yet)
        in_feat = getattr(self, "hit_features", 256)
        out_feat = len(getattr(self, "semantic_classes", []) or [0, 1])

        if hasattr(self, "semantic_decoder"):
            try:
                net = getattr(self.semantic_decoder, "net", None)
                if isinstance(net, nn.Sequential) and isinstance(net[-1], nn.Linear):
                    last_fc = net[-1]
                    in_feat = last_fc.in_features
                    out_feat = last_fc.out_features
            except Exception:
                pass

        self.embedding_decoder = nn.Sequential(
            nn.Linear(in_feat, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.semantic_head = nn.Linear(embed_dim, out_feat)

        # Edge loss hyperparameters
        self.lambda_edge = lambda_edge
        self.edge_pos_weight = edge_pos_weight

    # ------------------------------------------------------------------
    # Helper: build per-edge features from current batch
    # ------------------------------------------------------------------
    def _build_edge_attr(self, h, edge_index):
        """
        Build 7-D edge features per edge:
        [dx, dy, dz, dr, plane_src, charge_diff, time_diff]
        """
        pos = h.pos  # [N, D]
        plane = getattr(h, "plane", None)
        charge = getattr(h, "q", None)
        time = getattr(h, "t", None)

        if plane is None:
            plane = torch.zeros(pos.size(0), device=pos.device, dtype=torch.float)

        src, dst = edge_index[0], edge_index[1]

        d = pos[dst] - pos[src]  # [E, D]
        if d.size(1) == 2:
            pad = torch.zeros(d.size(0), 1, device=d.device, dtype=d.dtype)
            d = torch.cat([d, pad], dim=1)

        dx = d[:, 0:1]
        dy = d[:, 1:2]
        dz = d[:, 2:3]
        dr = torch.linalg.vector_norm(d, ord=2, dim=1, keepdim=True)

        p_src = plane[src].float().unsqueeze(1)

        if charge is not None:
            qc = (charge[dst] - charge[src]).float().unsqueeze(1)
        else:
            qc = torch.zeros_like(dr)

        if time is not None:
            tc = (time[dst] - time[src]).float().unsqueeze(1)
        else:
            tc = torch.zeros_like(dr)

        edge_attr = torch.cat([dx, dy, dz, dr, p_src, qc, tc], dim=1)
        return edge_attr  # [E, 7]

    def _edge_logits_and_labels(self, x, edge_index, batch):
        """
        Example edge labels:
        - positive if both endpoints are nu-hits
        - negative otherwise
        """
        h = batch["hit"]
        y_sem = getattr(h, "y_semantic", None)
        if y_sem is None:
            return None, None

        src, dst = edge_index
        # y=0 -> nu, y=1 -> cosmic (based on your dataset)
        y_src = y_sem[src]
        y_dst = y_sem[dst]
        y_edge = (y_src == 0) & (y_dst == 0)      # both nu
        y_edge = y_edge.float()                  # [E]

        edge_attr = self._build_edge_attr(h, edge_index)
        edge_logit = self.edge_mlp(edge_attr).squeeze(-1)  # [E]

        return edge_logit, y_edge

    # ------------------------------------------------------------------
    # Forward with edge loss augmentation
    # ------------------------------------------------------------------
    def forward(self, batch, stage=None):
        """
        Forward pass with edge loss augmentation:

        1. Call NuGraph3.forward to compute the usual loss + metrics.
        2. For 'train' and 'val', compute an edge loss:
               total_loss = loss_semantic + lambda_edge * loss_edge
        3. For other stages, leave behavior unchanged.
        """
        base = super().forward(batch, stage)

        # If parent returns something unexpected, don't break.
        if not (isinstance(base, tuple) and len(base) == 2):
            return base

        loss, metrics = base

        if stage not in ("train", "val"):
            return loss, metrics

        # Node features after encoder/core
        h = batch["hit"]
        if not hasattr(h, "x"):
            return loss, metrics
        x = h.x

        # Use planar Delaunay edges
        edge_key = ("hit", "delaunay-planar", "hit")
        if not hasattr(batch, "edge_index_dict") or edge_key not in batch.edge_index_dict:
            return loss, metrics
        edge_index = batch[edge_key].edge_index

        edge_logit, y_edge = self._edge_logits_and_labels(x, edge_index, batch)
        if edge_logit is None or y_edge is None or y_edge.numel() == 0:
            return loss, metrics

        pos_w = torch.tensor(self.edge_pos_weight, device=edge_logit.device)
        edge_loss = F.binary_cross_entropy_with_logits(
            edge_logit,
            y_edge,
            pos_weight=pos_w,
        )

        total_loss = loss + self.lambda_edge * edge_loss

        # Extend metrics
        if not isinstance(metrics, dict):
            metrics = {}
        else:
            metrics = dict(metrics)

        metrics["loss/edge"] = edge_loss.detach()
        metrics["loss/total"] = total_loss.detach()

        return total_loss, metrics
