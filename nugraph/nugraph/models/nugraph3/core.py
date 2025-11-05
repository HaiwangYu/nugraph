# filename: nugraph/nugraph/models/nugraph3/core.py
"""NuGraph core message-passing engine (with deterministic DropEdge on sp↔sp)."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing
from .types import T, TD, Data


def _dropedge_deterministic(edge_index: torch.Tensor, p: float) -> torch.Tensor:
    """
    Deterministic DropEdge mask based on hashing the edge indices, so all DDP ranks
    keep/drop the same edges given identical edge_index. Never drops all edges.

    Args:
        edge_index: LongTensor [2, E]
        p: drop probability in [0, 1)

    Returns:
        Filtered edge_index with ~ (1 - p) fraction of edges kept.
    """
    if p <= 0.0 or edge_index.numel() == 0:
        return edge_index

    # Vectorized 64-bit mix -> take low 32 bits as a pseudo-uniform
    ei0 = edge_index[0].to(torch.int64)
    ei1 = edge_index[1].to(torch.int64)

    x = ei0 ^ (ei1 * 0x9E3779B97F4A7C15)       # mix
    x ^= (x >> 30)
    x *= 0xBF58476D1CE4E5B9
    x ^= (x >> 27)
    x *= 0x94D049BB133111EB
    x ^= (x >> 31)

    u = (x & ((1 << 32) - 1)).to(torch.float32) / float(1 << 32)  # [0,1)
    keep = u >= p
    if keep.sum() == 0:
        return edge_index  # avoid empty graph
    return edge_index[:, keep]


class NuGraphBlock(MessagePassing):  # pylint: disable=abstract-method
    """
    Standard NuGraph message-passing block.

    Generates attention weights per edge from (x_i, x_j), applies them to x_j,
    aggregates with softmax, then updates target node features via a small MLP.
    """

    def __init__(self, source_features: int, target_features: int, out_features: int):
        super().__init__(aggr="softmax")

        self.edge_net = nn.Sequential(
            nn.Linear(source_features + target_features, 1),
            nn.Sigmoid()
        )

        self.net = nn.Sequential(
            nn.Linear(source_features + target_features, out_features),
            nn.Mish(),
            nn.Linear(out_features, out_features),
            nn.Mish()
        )

    def forward(self, x: T, edge_index: T) -> T:  # pylint: disable=arguments-differ
        return self.propagate(edge_index, x=x)

    def message(self, x_i: T, x_j: T) -> T:  # pylint: disable=arguments-differ
        # Detach attention inputs to stabilize training (as in your baseline)
        att = self.edge_net(torch.cat((x_i, x_j), dim=1).detach())
        return att * x_j

    def update(self, aggr_out: T, x: T) -> T:  # pylint: disable=arguments-differ
        if isinstance(x, tuple):
            _, x = x
        return self.net(torch.cat((aggr_out, x), dim=1))


class NuGraphCore(nn.Module):
    """
    NuGraph core message-passing engine.

    Args:
        hit_features: planar hit feature dim
        nexus_features: nexus feature dim
        interaction_features: interaction feature dim
        use_checkpointing: enable gradient checkpointing for MP blocks
        dropedge_sp: DropEdge probability applied ONLY to ("sp","nexus","sp") edges
    """

    def __init__(
        self,
        hit_features: int,
        nexus_features: int,
        interaction_features: int,
        use_checkpointing: bool = True,
        dropedge_sp: float = 0.0,
    ):
        super().__init__()

        self.use_checkpointing = use_checkpointing
        self.dropedge_sp = float(dropedge_sp)

        # internal planar message-passing
        self.plane_net = NuGraphBlock(hit_features, hit_features, hit_features)

        # internal nexus message-passing
        self.nexus_net = NuGraphBlock(nexus_features, nexus_features, nexus_features)

        # message-passing from planar nodes to nexus nodes
        self.plane_to_nexus = NuGraphBlock(hit_features, nexus_features, nexus_features)

        # message-passing from nexus nodes to interaction nodes
        self.nexus_to_interaction = NuGraphBlock(
            nexus_features, interaction_features, interaction_features
        )

        # message-passing from interaction nodes to nexus nodes
        self.interaction_to_nexus = NuGraphBlock(
            interaction_features, nexus_features, nexus_features
        )

        # message-passing from nexus nodes to planar nodes
        self.nexus_to_plane = NuGraphBlock(nexus_features, hit_features, hit_features)

    def checkpoint(self, net: nn.Module, *args) -> TD:
        if self.use_checkpointing and self.training:
            return checkpoint(net, *args, use_reentrant=False)
        else:
            return net(*args)

    def forward(self, data: Data) -> None:
        # message-passing in hits
        data["hit"].x = self.checkpoint(
            self.plane_net,
            data["hit"].x,
            data["hit", "delaunay-planar", "hit"].edge_index,
        )

        # message-passing from hits to nexus
        data["sp"].x = self.checkpoint(
            self.plane_to_nexus,
            (data["hit"].x, data["sp"].x),
            data["hit", "nexus", "sp"].edge_index,
        )

        # message-passing in nexus with deterministic DropEdge
        sp_sp_ei = data["sp", "nexus", "sp"].edge_index
        if self.training and self.dropedge_sp > 0.0:
            sp_sp_ei = _dropedge_deterministic(sp_sp_ei, self.dropedge_sp)

        data["sp"].x = self.checkpoint(self.nexus_net, data["sp"].x, sp_sp_ei)

        # message-passing from nexus to interaction
        data["evt"].x = self.checkpoint(
            self.nexus_to_interaction,
            (data["sp"].x, data["evt"].x),
            data["sp", "in", "evt"].edge_index,
        )

        # message-passing from interaction to nexus (reverse edge order)
        data["sp"].x = self.checkpoint(
            self.interaction_to_nexus,
            (data["evt"].x, data["sp"].x),
            data["sp", "in", "evt"].edge_index[(1, 0), :],
        )

        # message-passing from nexus to hits (reverse edge order)
        data["hit"].x = self.checkpoint(
            self.nexus_to_plane,
            (data["sp"].x, data["hit"].x),
            data["hit", "nexus", "sp"].edge_index[(1, 0), :],
        )
