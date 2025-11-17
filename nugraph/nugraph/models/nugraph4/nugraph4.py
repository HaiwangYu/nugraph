"""
NuGraph4 model architecture (Incremental refinement of NuGraph3)
"""
from torch import nn
import torch

from ..nugraph3.nugraph3 import NuGraph3  # reuse encoder/core for now

class NuGraph4(NuGraph3):
    def __init__(self, cfg):
        super().__init__(cfg)
        # --- New modules ---
        # Edge MLP to learn per-edge features
        in_edge_dim = 7  # dx,dy,dz,dr,plane,charge_diff,time_diff (adjust later)
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge_dim, cfg.model.edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.model.edge_hidden_dim, 1),
        )

        # Embedding-based decoder head (replaces/extends semantic)
        embed_dim = cfg.model.embed_dim
        self.embedding_decoder = nn.Sequential(
            nn.Linear(self.semantic_head.in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.semantic_head = nn.Linear(embed_dim, self.semantic_head.out_features)

    def forward(self, batch):
        # Reuse NuGraph3 encoder/core logic
        x, edge_index, edge_attr = self.encode(batch)

        # --- Learned edge weighting ---
        if edge_attr is not None:
            w = torch.sigmoid(self.edge_mlp(edge_attr))
            x = self.core(x, edge_index, edge_weight=w)
        else:
            x = self.core(x, edge_index)

        # --- Embedding decoder ---
        z = self.embedding_decoder(x)
        logits = self.semantic_head(z)
        return logits, z
