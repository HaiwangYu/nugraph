"""NuGraph3 data transform"""
import os
import torch
from torch_geometric.transforms import BaseTransform
from pynuml.data import NuGraphData


class Transform(BaseTransform):
    """
    NuGraph3 data transform

    Args:
        planes: Tuple of detector plane names
        in_features: total feature dimension expected by the model AFTER transform
                     (i.e. final hit.x dimension fed into Encoder)
    """
    def __init__(self, planes: tuple[str], in_features: int = 4):
        super().__init__()
        self.planes = planes
        self.in_features = int(in_features)

    def __call__(self, data: NuGraphData) -> NuGraphData:
        # --- (unchanged) old planar -> hierarchical conversion ---
        if "hit" not in data.node_types:
            edge_plane = []
            edge_nexus = []
            for i, p in enumerate(self.planes):
                offset = 0
                for j in range(i):
                    offset += data[self.planes[j]].num_nodes
                edge_plane.append(data[p, "plane", p].edge_index + offset)
                del data[p, "plane", p]
                edge_nexus.append(data[p, "nexus", "sp"].edge_index)
                edge_nexus[-1][0] += offset
                del data[p, "nexus", "sp"]
            data["hit", "delaunay-planar", "hit"].edge_index = torch.cat(edge_plane, dim=1)
            data["hit", "nexus", "sp"].edge_index = torch.cat(edge_nexus, dim=1)

            for i, p in enumerate(self.planes):
                data[p].plane = torch.empty_like(data[p].x[:, 0], dtype=int).fill_(i)
                data[p].x = torch.cat([data[p].x, data[p].plane.unsqueeze(1)], dim=1)

            for attr in data[self.planes[0]].node_attrs():
                data["hit"][attr] = torch.cat([data[p][attr] for p in self.planes], dim=0)
            for p in self.planes:
                del data[p]

            if hasattr(data["hit"], "y_instance"):
                data["hit"].pid = data["hit"].y_instance.clone()
                y = data["hit"].y_instance
                mask = y != -1
                y = y[mask]
                instances = y.unique()
                imax = instances.max() + 1 if instances.size(0) else 0
                if instances.size(0) != imax:
                    remap = torch.full((imax,), -1, dtype=torch.long)
                    remap[instances] = torch.arange(instances.size(0))
                    y = remap[y]
                data["particle-truth"].x = torch.empty(instances.size(0), 0)
                edges = torch.stack((mask.nonzero().squeeze(1), y), dim=0).long()
                data["hit", "cluster-truth", "particle-truth"].edge_index = edges

            data["evt"].x = torch.empty((1, 0))
            lo = torch.arange(data["hit"].num_nodes, dtype=torch.long)
            hi = torch.zeros(data["hit"].num_nodes, dtype=torch.long)
            data["hit", "in", "evt"].edge_index = torch.stack((lo, hi), dim=0)
            lo = torch.arange(data["sp"].num_nodes, dtype=torch.long)
            hi = torch.zeros(data["sp"].num_nodes, dtype=torch.long)
            data["sp", "in", "evt"].edge_index = torch.stack((lo, hi), dim=0)

        if "c" in data["hit"].keys():
            data["hit"].y_position = data["hit"].c
            del data["hit"].c

        evt = data["evt"]
        if not evt.y.ndim:
            evt.y = evt.y.reshape([1])

        # --- NEW: enforce that final hit.x has exactly self.in_features ---
        h = data["hit"]

        pos_dim = int(h.pos.size(-1))
        if pos_dim > self.in_features:
            raise RuntimeError(f"pos_dim={pos_dim} > in_features={self.in_features}")

        # how many non-pos features we want to keep
        feat_keep = self.in_features - pos_dim

        # If h.x doesn't exist, create it
        if "x" not in h.keys():
            h.x = torch.zeros((h.pos.size(0), 0), dtype=h.pos.dtype, device=h.pos.device)

        # Slice or pad h.x to exactly feat_keep
        if h.x.size(-1) > feat_keep:
            h.x = h.x[:, :feat_keep]
        elif h.x.size(-1) < feat_keep:
            need = feat_keep - h.x.size(-1)
            pad = torch.zeros((h.x.size(0), need), dtype=h.x.dtype, device=h.x.device)
            h.x = torch.cat((h.x, pad), dim=-1)

        # Finally prepend pos
        h.x = torch.cat((h.pos, h.x), dim=-1)

        # Optional debug
        if os.environ.get("NUGRAPH_DEBUG_FEATURES", "0") in ("1", "true", "yes"):
            print(f"[Transform] pos_dim={pos_dim}, feat_keep={feat_keep}, final hit.x={tuple(h.x.shape)}")

        return data
