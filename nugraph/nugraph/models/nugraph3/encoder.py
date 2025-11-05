# filename: nugraph/nugraph/models/nugraph3/encoder.py
"""NuGraph3 encoder (with optional Fourier features on hit positions) — lazy in_features + ckpt pre-load hook"""
from __future__ import annotations
import torch
from torch import nn
from .types import Data
from ...util import InputNorm

class FourierEmbed(nn.Module):
    def __init__(self, num_dims: int = 3, num_freqs: int = 3, scale: float = 1.0):
        super().__init__()
        self.num_dims = num_dims
        self.num_freqs = num_freqs
        self.register_buffer(
            "freqs",
            (scale * 2.0 ** torch.arange(num_freqs, dtype=torch.float32)).view(1, 1, num_freqs),
            persistent=False,
        )

    @property
    def out_dim(self) -> int:
        return self.num_dims * self.num_freqs * 2  # sin+cos

    def forward(self, pos: torch.Tensor | None) -> torch.Tensor:
        if pos is None:
            return torch.empty(0, device=self.freqs.device)
        p = pos[:, :self.num_dims]
        p = torch.where(torch.isfinite(p), p, torch.zeros_like(p))
        mean = p.mean(dim=0, keepdim=True)
        std  = p.std(dim=0, keepdim=True).clamp_min(1e-6)
        p = (p - mean) / std
        angles = p.unsqueeze(-1) * self.freqs
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return emb.view(p.size(0), -1)


class Encoder(nn.Module):
    """
    Args:
        in_features: nominal hit feature dim (used to init InputNorm only)
        hit_features: output hit embedding dim
        nexus_features: nexus embedding dim (zero seeded)
        interaction_features: interaction embedding dim (zero seeded)
        fourier_freqs: number of Fourier frequencies (0 disables)
        fourier_scale: base frequency scale
    """
    def __init__(self,
                 in_features: int,
                 hit_features: int,
                 nexus_features: int,
                 interaction_features: int,
                 fourier_freqs: int = 0,
                 fourier_scale: float = 1.0):
        super().__init__()
        self.use_fourier = fourier_freqs > 0
        self.fourier = FourierEmbed(num_dims=3, num_freqs=fourier_freqs, scale=fourier_scale) if self.use_fourier else None

        # Dim-agnostic normalization over observed features
        self.input_norm = InputNorm(in_features)

        self._hit_in: nn.Module | None = None
        self._hit_in_in_dim: int | None = None
        self._hit_features = hit_features

        self.nexus_features = nexus_features
        self.interaction_features = interaction_features

        # --- Pre-load hook so checkpoints with concrete _hit_in weights can restore strictly ---
        # If a checkpoint contains keys like "...encoder._hit_in.0.weight", build _hit_in
        # with the correct input width BEFORE load_state_dict applies parameters.
        self._register_load_state_dict_pre_hook(self._preload_hook)

    # --------------------
    # Checkpoint pre-hook:
    # --------------------
    def _preload_hook(self, state_dict, prefix, *args):
        """
        Create self._hit_in with the correct input dimension if weights for it
        exist in the incoming checkpoint state_dict.

        Prefix is the module path for this Encoder inside the owning LightningModule,
        so keys look like f"{prefix}_hit_in.0.weight".
        """
        w_key = f"{prefix}_hit_in.0.weight"
        if w_key in state_dict:
            # weight shape is [out_dim, in_dim]; infer in_dim
            in_dim = int(state_dict[w_key].shape[1])
            # state_dict tensors are typically on CPU at load time
            try:
                device = next(iter(state_dict.values())).device
            except StopIteration:
                device = torch.device("cpu")
            self._ensure_hit_in(in_dim, device)

    # --------------------
    # Helpers:
    # --------------------
    def _get_pos(self, hit) -> torch.Tensor | None:
        pos = getattr(hit, "pos", None)
        if pos is None:
            pos = getattr(hit, "xyz", None)
        if pos is None:
            pos = getattr(hit, "coord", None)
        if isinstance(pos, (list, tuple)):
            pos = torch.as_tensor(pos)
        return pos if torch.is_tensor(pos) else None

    def _ensure_hit_in(self, in_dim: int, device: torch.device) -> None:
        if (self._hit_in is None) or (self._hit_in_in_dim != in_dim):
            # (Re)create with the actually observed input dim
            seq = nn.Sequential(
                nn.Linear(in_dim, self._hit_features, bias=True),
                nn.GELU(),
                nn.LayerNorm(self._hit_features),
            )
            self._hit_in = seq.to(device)
            self._hit_in_in_dim = in_dim

    # --------------------
    # Forward:
    # --------------------
    def forward(self, data: Data) -> None:
        # base features
        x_base = self.input_norm(data["hit"].x)
        if x_base.dim() != 2:
            raise RuntimeError(f"hit.x must be 2D [N, F]; got {tuple(x_base.shape)}")
        N, _ = x_base.size(0), x_base.size(1)

        # optional Fourier
        f = None
        if self.use_fourier:
            pos = self._get_pos(data["hit"])
            if pos is not None and pos.dim() == 2 and pos.size(0) == N and pos.size(1) >= 3:
                pos = pos.to(device=x_base.device, dtype=x_base.dtype).contiguous()
                f = self.fourier(pos)  # [N, extra]

        x = torch.cat([x_base, f], dim=1) if isinstance(f, torch.Tensor) and f.numel() > 0 else x_base
        in_dim = x.size(1)

        # lazily (re)build the projection with the observed input width
        self._ensure_hit_in(in_dim, x.device)

        # project
        data["hit"].x = self._hit_in(x)

        # zero seeds for graph-level and nexus-level nodes
        device = data["hit"].x.device
        data["sp"].x  = torch.zeros(data["sp"].num_nodes,  self.nexus_features,       device=device)
        data["evt"].x = torch.zeros(data["evt"].num_nodes, self.interaction_features, device=device)
