"""NuGraph3 encoder module"""
import torch
from torch import nn
from torch_scatter import scatter_mean

from .types import Data
from ...util import InputNorm  # InputNorm is in nugraph.util.input_norm


class Encoder(torch.nn.Module):
    """
    NuGraph3 encoder
    
    Args:
        in_features: Number of input node features
        planar_features: Number of planar node features
        nexus_features: Number of nexus node features
        interaction_features: Number of interaction node features
        use_sp_features: Whether to incorporate sp.features into SP embeddings
        use_vtx_features: Whether to include vertex-related features (cols 2-5)
                          Only relevant if use_sp_features=True
        sp_feature_dim: Expected dimension of sp.features (default 6)
    """
    def __init__(self,
                 in_features: int,
                 planar_features: int,
                 nexus_features: int,
                 interaction_features: int,
                 use_sp_features: bool = True,
                 use_vtx_features: bool = False,
                 sp_feature_dim: int = 6):
        super().__init__()
        
        self.input_norm = InputNorm(in_features)
        self.planar_net = torch.nn.Linear(in_features, planar_features)
        self.nexus_features = nexus_features
        self.interaction_features = interaction_features
        
        # SP feature control
        self.use_sp_features = use_sp_features
        self.use_vtx_features = use_vtx_features
        self.sp_feature_dim = sp_feature_dim
        
        # Project aggregated hit embeddings to nexus dimension
        self.hit_to_nexus = torch.nn.Linear(planar_features, nexus_features)
        
        # SP feature processing (if enabled)
        if self.use_sp_features:
            # Determine how many SP features we'll actually use
            # cols 0-1: charge, hit_count (always fair)
            # cols 2-5: vtx_dist, vtx_dx, vtx_dy, vtx_dz (cheating if used)
            if self.use_vtx_features:
                self.sp_feat_cols = list(range(sp_feature_dim))  # all 6
            else:
                self.sp_feat_cols = [0, 1]  # only charge and hit_count
            
            n_sp_feats = len(self.sp_feat_cols)
            
            # Normalize and project SP features
            self.sp_feat_norm = nn.LayerNorm(n_sp_feats)
            
            # Combine aggregated hit embeddings + SP features -> nexus_features
            # Input: nexus_features (from hit agg) + n_sp_feats
            self.sp_combine = nn.Sequential(
                nn.Linear(nexus_features + n_sp_feats, nexus_features),
                nn.GELU(),
            )
            
            # One-time logging
            self._logged_sp_features = False
    
    def forward(self, data: Data) -> None:
        """
        NuGraph3 encoder forward pass
        
        Args:
            data: Graph data object
        """
        # Encode hit features
        data["hit"].x = self.input_norm(data["hit"].x)
        data["hit"].x = self.planar_net(data["hit"].x)
        
        # Initialize SP nodes with nexus aggregation
        if ("hit", "nexus", "sp") in data.edge_types:
            nexus_edge_index = data["hit", "nexus", "sp"].edge_index
            src_hits = nexus_edge_index[0]
            dst_sp = nexus_edge_index[1]
            
            # Aggregate hit embeddings to SP nodes
            sp_aggregated = scatter_mean(
                data["hit"].x[src_hits], 
                dst_sp, 
                dim=0, 
                dim_size=data["sp"].num_nodes
            )
            
            # Project to nexus dimension (planar_features → nexus_features)
            sp_from_hits = self.hit_to_nexus(sp_aggregated)
            
            # Optionally incorporate SP-level features
            if self.use_sp_features and hasattr(data["sp"], "features"):
                sp_feat = data["sp"].features
                
                # One-time logging
                if not self._logged_sp_features:
                    print(f"[Encoder] SP features enabled:")
                    print(f"  sp.features shape: {tuple(sp_feat.shape)}")
                    print(f"  use_vtx_features: {self.use_vtx_features}")
                    print(f"  using columns: {self.sp_feat_cols}")
                    self._logged_sp_features = True
                
                # Select appropriate columns
                if sp_feat.size(-1) >= max(self.sp_feat_cols) + 1:
                    sp_feat_selected = sp_feat[:, self.sp_feat_cols]
                    
                    # Normalize SP features
                    sp_feat_normed = self.sp_feat_norm(sp_feat_selected)
                    
                    # Combine with aggregated hit embeddings
                    combined = torch.cat([sp_from_hits, sp_feat_normed], dim=-1)
                    data["sp"].x = self.sp_combine(combined)
                else:
                    # Fallback if sp.features doesn't have expected columns
                    if not self._logged_sp_features:
                        print(f"[Encoder] Warning: sp.features has {sp_feat.size(-1)} cols, "
                              f"expected at least {max(self.sp_feat_cols) + 1}. Using hit-only.")
                    data["sp"].x = sp_from_hits
            else:
                # No SP features - just use aggregated hit embeddings
                data["sp"].x = sp_from_hits
        else:
            # Fallback to zeros if no nexus edges
            data["sp"].x = torch.zeros(
                data["sp"].num_nodes,
                self.nexus_features,
                device=data["hit"].x.device
            )
        
        # Event nodes initialized to zeros
        data["evt"].x = torch.zeros(
            data["evt"].num_nodes,
            self.interaction_features,
            device=data["hit"].x.device
        )