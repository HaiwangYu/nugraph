"""
NuGraph4 model architecture (Incremental refinement of NuGraph3)

Design intent vs NuGraph3:
  - Learned edge features via a small edge MLP
  - Embedding-based semantic decoder: x -> embedding -> logits

For this alpha version:
  - We define these components but KEEP TRAINING BEHAVIOUR IDENTICAL
    to NuGraph3 by delegating forward() to the parent.
  - To avoid DDP complaining about "unused parameters", we freeze the
    new modules' parameters (requires_grad = False), so that DDP does
    not expect them to participate in the loss.
"""

from torch import nn
import torch

from ..nugraph3.nugraph3 import NuGraph3  # reuse encoder/core/Lightning logic


class NuGraph4(NuGraph3):
    def __init__(
        self,
        *args,
        edge_hidden_dim: int = 32,
        embed_dim: int = 64,
        **kwargs,
    ):
        """
        NuGraph4 shares the same constructor as NuGraph3, plus:

          edge_hidden_dim : hidden size for the edge MLP
          embed_dim       : embedding dimension for the node-level decoder
        """
        super().__init__(*args, **kwargs)

        # --- Edge MLP to learn per-edge weights from (dx,dy,dz,dr,plane,charge_diff,time_diff) ---
        in_edge_dim = 7  # [dx, dy, dz, dr, plane_src, charge_diff, time_diff]
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, 1),
        )

        # --- Embedding-based semantic decoder (not wired into losses yet) ---
        # Try to infer input/output dims from NuGraph3's semantic_decoder
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
                # Fall back to defaults if inspection fails
                pass

        self.embedding_decoder = nn.Sequential(
            nn.Linear(in_feat, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        # New semantic head that operates on the embedding z
        self.semantic_head = nn.Linear(embed_dim, out_feat)

        # ------------------------------------------------------------------
        # IMPORTANT FOR NOW:
        # Freeze the new modules so DDP doesn't expect them in the loss.
        # This keeps behaviour numerically identical to NuGraph3.
        # ------------------------------------------------------------------
        for p in self.edge_mlp.parameters():
            p.requires_grad = False
        for p in self.embedding_decoder.parameters():
            p.requires_grad = False
        for p in self.semantic_head.parameters():
            p.requires_grad = False

    # -------------------------------------------------------------------------
    # Helper for building edge_attr on the fly when encode() does not provide it
    # (currently UNUSED in training; kept for future integration)
    # -------------------------------------------------------------------------
    def _build_edge_attr(self, batch, edge_index):
        """
        Build 7-D edge features per edge:

        [dx, dy, dz, dr, plane_src, charge_diff, time_diff]

        For alpha1, only pos + plane are guaranteed to be present.
        charge_diff and time_diff are filled with zeros if missing.
        """
        h = batch["hit"]

        pos = h.pos  # [N, D]
        plane = getattr(h, "plane", None)
        charge = getattr(h, "q", None)
        time = getattr(h, "t", None)

        if plane is None:
            plane = torch.zeros(pos.size(0), device=pos.device, dtype=torch.float)

        src, dst = edge_index[0], edge_index[1]

        # Geometric differences
        d = pos[dst] - pos[src]  # [E, D]
        # Ensure 3 components: if D=2, pad a zero dz
        if d.size(1) == 2:
            pad = torch.zeros(d.size(0), 1, device=d.device, dtype=d.dtype)
            d = torch.cat([d, pad], dim=1)

        dx = d[:, 0:1]
        dy = d[:, 1:2]
        dz = d[:, 2:3]
        dr = torch.linalg.vector_norm(d, ord=2, dim=1, keepdim=True)

        # Plane (source node)
        p_src = plane[src].float().unsqueeze(1)

        # Charge difference (optional)
        if charge is not None:
            qc = (charge[dst] - charge[src]).float().unsqueeze(1)
        else:
            qc = torch.zeros_like(dr)

        # Time difference (optional)
        if time is not None:
            tc = (time[dst] - time[src]).float().unsqueeze(1)
        else:
            tc = torch.zeros_like(dr)

        edge_attr = torch.cat([dx, dy, dz, dr, p_src, qc, tc], dim=1)
        return edge_attr  # [E, 7]

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(self, batch, stage=None):
        """
        Delegate to NuGraph3.forward(batch, stage).

        This preserves the existing training/evaluation API
        (returns (loss, metrics) for training/validation/test)
        while NuGraph4 architectural components are being developed.
        """
        return super().forward(batch, stage)
