"""Position features transform"""
import torch
from torch_geometric.transforms import BaseTransform

from pynuml.data import NuGraphData

class PositionFeatures(BaseTransform):
    """
    Append node position tensor to node feature tensor
    
    Args:
        planes: list of detector plane names
    """
    def __init__(self, planes: list[str]):
        super().__init__()
        self.planes = planes

    def __call__(self, data: NuGraphData) -> NuGraphData:
        """
        Apply transform to concatenate node position onto node feature tensor

        Args:
           data: NuGraph data object to transform
        """

        # in second-generation inputs, hit nodes live in a single node store
        if "hit" in data.node_types:
            node_types = ("hit",)

        # in first-generation inputs, hit nodes are separated out by plane
        else:
            node_types = self.planes

        for node_type in node_types:
            n = data[node_type]

            # --- NEW: if we're on the merged hit store and still only have
            #     the old 5 features, pad with 2 zeros to reach 7,
            #     so that after adding pos (3) we get 10 total.
            if node_type == "hit" and n.x.size(-1) == 5:
                extra = torch.zeros(
                    (n.x.size(0), 2),
                    dtype=n.x.dtype,
                    device=n.x.device,
                )
                n.x = torch.cat((n.x, extra), dim=-1)

            # concatenate position tensor onto node features
            n.x = torch.cat((n.pos, n.x), dim=-1)

        return data
