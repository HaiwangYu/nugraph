# filename: nugraph/models/nugraph4.py
import torch
import torch.nn.functional as F
from torch import nn

from ..nugraph3.nugraph3 import NuGraph3


class NuGraph4(NuGraph3):
    def __init__(
        self,
        *args,
        edge_hidden_dim: int = 32,
        embed_dim: int = 64,
        lambda_edge: float = 0.0,     # default off for v1
        edge_pos_weight: float = 0.3,
        lambda_embed: float = 0.2,    # NEW: weight for instance embedding loss
        lambda_coh: float = 0.0,      # NEW: cluster coherence loss weight
        coh_edge_thr: float = 0.7,    # threshold for p_same when forming clusters
        coh_min_cluster: int = 2,     # min cluster size for coherence loss
        use_sp_features: bool = True,       # NEW: passed to parent
        use_vtx_features: bool = False,     # NEW: passed to parent
        **kwargs,
    ):
        """
        NuGraph4 extends NuGraph3 by:
          - replacing the original semantic decoder with an embedding-based head
          - adding edge MLP + edge loss (optional)
          - adding instance embedding loss + coherence loss (optional)

        IMPORTANT for sidecar features:
        --------------------------------
        The new semantic sidecar features are appended to hit.x in
        NuGraphDataset, then PositionFeatures concatenates pos → final
        per-hit input dimension = `in_features` (CLI arg).

        NuGraph3's encoder uses `in_features` to map those raw channels
        to `hit_features` (the learned embedding dim).

        Here, `in_feat = self.hit_features` is that embedding dim, so
        NuGraph4 automatically "sees" the sidecar features through x.
        No extra plumbing is needed here.
        
        NEW: SP-level features (charge, hit count, optionally vertex distance)
        can be incorporated via use_sp_features and use_vtx_features flags.
        """
        # Pass SP feature flags to parent NuGraph3
        super().__init__(
            *args, 
            use_sp_features=use_sp_features,
            use_vtx_features=use_vtx_features,
            **kwargs
        )

        # Hit embedding dimension from the encoder/core (NOT raw input channels)
        in_feat = getattr(self, "hit_features", 256)

        # Optional one-time debug: confirm we're using the expected embedding dim
        if not hasattr(self, "_nug4_init_logged"):
            print(f"[NuGraph4] Initialized with hit_features (embedding dim) = {in_feat}")
            print(f"[NuGraph4] semantic_classes = {getattr(self, 'semantic_classes', None)}")
            print(f"[NuGraph4] use_sp_features = {use_sp_features}")
            print(f"[NuGraph4] use_vtx_features = {use_vtx_features}")
            if use_vtx_features:
                print(f"[NuGraph4] ⚠️  WARNING: Vertex features enabled - using MC truth info!")
            self._nug4_init_logged = True

        # Semantic classes (optionally read out_features from existing decoder)
        out_feat = len(getattr(self, "semantic_classes", []) or [0, 1])
        if hasattr(self, "semantic_decoder"):
            try:
                net = getattr(self.semantic_decoder, "net", None)
                if isinstance(net, nn.Sequential) and isinstance(net[-1], nn.Linear):
                    out_feat = net[-1].out_features
            except Exception:
                pass

        # Embedding-based semantic head (auxiliary)
        self.embedding_decoder = nn.Sequential(
            nn.Linear(in_feat, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.semantic_head = nn.Linear(embed_dim, out_feat)

        # Edge MLP uses learned embeddings + geometry.
        # in_edge_dim = 2 * in_feat (z_src, z_dst) + 7 geom features
        in_edge_dim = 2 * in_feat + 7
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, 1),
        )

        # SP-specific edge MLP for supervision edges (uses nexus_features not hit_features)
        sp_feat = getattr(self, "nexus_features", 64)
        in_edge_dim_sp = 2 * sp_feat + 4  # z_src, z_dst, dx, dy, dz, dr (no plane/charge/time for SP)
        self.edge_mlp_sp = nn.Sequential(
            nn.Linear(in_edge_dim_sp, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, 1),
        )

        # Loss weights
        self.lambda_edge = lambda_edge       # edge BCE (kept for later phases)
        self.edge_pos_weight = edge_pos_weight
        self.lambda_embed = lambda_embed     # instance embedding loss
        self.lambda_coh = lambda_coh         # semantic coherence loss
        self.coh_edge_thr = coh_edge_thr
        self.coh_min_cluster = coh_min_cluster

        # Internal flags for one-time logging (edge + embed)
        self._edge_stats_logged = False
        self._edge_balance_logged = False
        self._warmup_phase_logged_edge = False
        self._rampup_phase_logged_edge = False
        self._full_phase_logged_edge = False

        self._embed_stats_logged = False
        self._warmup_phase_logged_embed = False
        self._rampup_phase_logged_embed = False
        self._full_phase_logged_embed = False

        self._coh_phase_logged_warmup = False
        self._coh_phase_logged_rampup = False
        self._coh_phase_logged_full = False

    # ----------------------------------------------------------------------
    # Helper: rank-0 check to avoid DDP spam
    # ----------------------------------------------------------------------
    def _is_rank0(self) -> bool:
        """
        Returns True only on global rank 0 (if trainer is attached).
        Before trainer is set, we default to True.
        """
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            try:
                trainer = super().trainer  # may raise if not attached
            except Exception:
                trainer = None
        if trainer is None:
            return True
        return getattr(trainer, "global_rank", 0) == 0

    # ----------------------------------------------------------------------
    # Helper: build per-edge features from embeddings + geometry
    # ----------------------------------------------------------------------
    def _build_edge_attr(self, h, x, edge_index):
        """
        Build edge features per edge:

        concat( z_src, z_dst, dx, dy, dz, dr, plane_src, charge_diff, time_diff )

        where z_src/z_dst are the learned hit embeddings (x), and the rest
        are geometric / low-level quantities.

        NOTE: The new sidecar semantic features are already absorbed into
        the embeddings x via the NuGraph3 encoder, so they influence
        z_src/z_dst automatically.
        """
        pos = h.pos  # [N, D]
        plane = getattr(h, "plane", None)
        charge = getattr(h, "q", None)
        time = getattr(h, "t", None)

        if plane is None:
            plane = torch.zeros(pos.size(0), device=pos.device, dtype=torch.float)

        src, dst = edge_index[0], edge_index[1]

        # Geometric differences
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

        # Learned node embeddings (from core)
        z_src = x[src]  # [E, in_feat]
        z_dst = x[dst]  # [E, in_feat]

        edge_attr = torch.cat(
            [z_src, z_dst, dx, dy, dz, dr, p_src, qc, tc],
            dim=1,
        )
        return edge_attr  # [E, 2*in_feat + 7]


    # ----------------------------------------------------------------------
    # Helper: build per-edge features from SP embeddings + geometry
    # ----------------------------------------------------------------------
    def _build_edge_attr_sp(self, sp, x, edge_index):
        """
        Build edge features for SP-to-SP edges:
        concat( z_src, z_dst, dx, dy, dz, dr )
        
        SP nodes don't have plane/charge/time, so we only use position differences.
        """
        pos = sp.pos  # [N, 3] for SP nodes
        src, dst = edge_index[0], edge_index[1]

        # Geometric differences
        d = pos[dst] - pos[src]  # [E, 3]
        dx = d[:, 0:1]
        dy = d[:, 1:2]
        dz = d[:, 2:3]
        dr = torch.linalg.vector_norm(d, ord=2, dim=1, keepdim=True)

        # Learned SP embeddings
        z_src = x[src]  # [E, sp_feat=64]
        z_dst = x[dst]  # [E, sp_feat=64]

        edge_attr = torch.cat(
            [z_src, z_dst, dx, dy, dz, dr],
            dim=1,
        )
        return edge_attr  # [E, 2*64 + 4 = 132]
    # ----------------------------------------------------------------------
    # Helper: compute edge logits + labels from instance IDs
    # ----------------------------------------------------------------------
    def _edge_logits_and_labels(self, x, edge_index, batch):
        """
        Edge labels: positive if both endpoints belong to same particle instance.
        NOTE: This method is unused - kept for backward compatibility
        """
        h = batch["sp"]  # Use sp store (though this method isn't called)

        # Correct field: particle instance ID
        y_instance = getattr(h, "pid", None)
        if y_instance is None:
            y_instance = getattr(h, "y_instance", None)
        if y_instance is None:
            return None, None

        src, dst = edge_index
        y_src_inst = y_instance[src]
        y_dst_inst = y_instance[dst]

        valid = (y_src_inst >= 0) & (y_dst_inst >= 0)

        # ==================== DEBUG STATS (FIRST BATCH ONLY) ====================
        if self._is_rank0() and not self._edge_stats_logged:
            total_hits = len(y_instance)
            labeled_hits = (y_instance >= 0).sum().item()
            total_edges = edge_index.shape[1]
            valid_edges = valid.sum().item()

            print(f"\n{'=' * 70}")
            print(f"[EDGE LOSS DIAGNOSTICS - First Batch]")
            print(f"{'=' * 70}")
            print(f"  Total hits: {total_hits}")
            print(f"  Labeled hits: {labeled_hits} ({100*labeled_hits/total_hits:.1f}%)")
            print(f"  Total edges: {total_edges}")
            print(f"  Valid edges (both labeled): {valid_edges} ({100*valid_edges/total_edges:.1f}%)")
            self._edge_stats_logged = True

        if not valid.any():
            return None, None

        edge_index_valid = edge_index[:, valid]
        src, dst = edge_index_valid

        y_edge = (y_instance[src] == y_instance[dst]).float()

        # ==================== MORE DEBUG STATS (FIRST BATCH ONLY) ====================
        if self._is_rank0() and not self._edge_balance_logged:
            pos_edges = y_edge.sum().item()
            neg_edges = len(y_edge) - pos_edges
            pos_frac = (100 * pos_edges / len(y_edge)) if len(y_edge) > 0 else 0.0
            neg_frac = 100.0 - pos_frac

            print(f"  Positive edges (same instance): {pos_edges} ({pos_frac:.1f}%)")
            print(f"  Negative edges (diff instance): {neg_edges} ({neg_frac:.1f}%)")

            actual_imbalance = (neg_edges / pos_edges) if pos_edges > 0 else 0.0
            print(f"  Actual class imbalance (neg:pos): {actual_imbalance:.1f}:1")
            print(f"  Your pos_weight setting: {self.edge_pos_weight}")
            print(f"{'=' * 70}\n")
            self._edge_balance_logged = True
        # =======================================================================

        # Build edge attributes using embeddings + geometry
        edge_attr = self._build_edge_attr(h, x, edge_index_valid)
        edge_logit = self.edge_mlp(edge_attr).squeeze(-1)

        return edge_logit, y_edge, edge_index_valid

    # ----------------------------------------------------------------------
    # NEW: build clusters from high-confidence edge probabilities
    # ----------------------------------------------------------------------
    def _build_clusters_from_edges(
        self,
        edge_probs: torch.Tensor,
        edge_index_valid: torch.Tensor,
        batch_idx: torch.Tensor,
        pid: torch.Tensor,
        edge_thr: float,
        min_cluster_size: int,
    ):
        """
        Build per-graph clusters using edges with p_same >= edge_thr.
        Only labeled hits (pid >= 0) are considered.
        Returns a list of 1D LongTensors of global hit indices (one per cluster).
        """
        keep = edge_probs >= edge_thr
        if keep.sum() == 0:
            return []

        ei = edge_index_valid[:, keep]
        src, dst = ei

        # Only consider labeled hits
        labeled = pid >= 0

        clusters = []
        for g in batch_idx.unique():
            g = int(g.item())
            node_mask = (batch_idx == g) & labeled
            if node_mask.sum() < min_cluster_size:
                continue
            idx_g = node_mask.nonzero(as_tuple=False).view(-1)
            # Map global -> local
            global_to_local = {int(gi): li for li, gi in enumerate(idx_g.tolist())}

            # Edges inside this graph with labeled endpoints
            edge_mask_g = (
                (batch_idx[src] == g)
                & (batch_idx[dst] == g)
                & labeled[src]
                & labeled[dst]
            )
            if edge_mask_g.sum() == 0:
                continue

            src_g = src[edge_mask_g].tolist()
            dst_g = dst[edge_mask_g].tolist()

            # Union-find
            parent = list(range(idx_g.numel()))
            size = [1] * idx_g.numel()

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: int, b: int):
                ra, rb = find(a), find(b)
                if ra == rb:
                    return
                if size[ra] < size[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                size[ra] += size[rb]

            for s, d in zip(src_g, dst_g):
                if s in global_to_local and d in global_to_local:
                    union(global_to_local[s], global_to_local[d])

            # Collect clusters
            root_to_nodes = {}
            for gi in idx_g.tolist():
                li = global_to_local[gi]
                r = find(li)
                root_to_nodes.setdefault(r, []).append(gi)

            for nodes in root_to_nodes.values():
                if len(nodes) >= min_cluster_size:
                    clusters.append(torch.tensor(nodes, device=pid.device, dtype=torch.long))

        return clusters

    # ----------------------------------------------------------------------
    # NEW: instance embedding contrastive loss
    # ----------------------------------------------------------------------
    def _instance_embedding_loss(
        self,
        x,
        batch,
        max_nodes: int = 512,
        margin: float = 0.5,
    ):
        """
        Contrastive-style loss on SP embeddings x using true instance IDs.

        IMPORTANT: operates per-graph (per event) using h.batch, so we never
        mix nodes from different events that happen to share the same pid.
        """
        h = batch["sp"]

        # True instance ID per hit
        y_instance = getattr(h, "pid", None)
        if y_instance is None:
            y_instance = getattr(h, "y_instance", None)
        if y_instance is None:
            return None

        # Graph indices (one event per graph)
        batch_idx = getattr(h, "batch", None)
        if batch_idx is None:
            # Fallback: treat entire batch as single graph
            batch_idx = torch.zeros_like(y_instance, dtype=torch.long)

        device = x.device
        y_instance = y_instance.to(device)
        batch_idx = batch_idx.to(device)

        per_graph_losses = []
        first_debug_done = False

        # Loop over graphs in this mini-batch
        for g in batch_idx.unique():
            g_mask = (batch_idx == g) & (y_instance >= 0)
            if g_mask.sum() < 2:
                continue

            idx = g_mask.nonzero(as_tuple=False).view(-1)

            # Subsample to control O(N^2) cost
            if idx.numel() > max_nodes:
                perm = torch.randperm(idx.numel(), device=device)[:max_nodes]
                idx = idx[perm]

            z = x[idx]                      # [M, D] embeddings
            inst_ids = y_instance[idx]      # [M]
            M = z.size(0)
            if M < 2:
                continue

            # Optional: L2-normalize embeddings for more stable distances
            z = F.normalize(z, p=2, dim=1)

            # Pairwise distances
            dist = torch.cdist(z, z, p=2)   # [M, M]

            same = (inst_ids.unsqueeze(0) == inst_ids.unsqueeze(1))
            eye = torch.eye(M, dtype=torch.bool, device=device)
            same = same & (~eye)
            diff = (~same) & (~eye)

            if not same.any() or not diff.any():
                continue

            pos_d = dist[same]
            neg_d = dist[diff]

            # Contrastive loss with margin
            loss_pos = (pos_d ** 2).mean()
            loss_neg = F.relu(margin - neg_d).pow(2).mean()
            embed_loss_g = loss_pos + loss_neg
            per_graph_losses.append(embed_loss_g)

            # One-time debug print (first graph that contributes)
            if self._is_rank0() and not self._embed_stats_logged and not first_debug_done:
                print("\n[EMBED LOSS DIAGNOSTICS - First Contributing Graph]")
                print(f"  M (sampled hits) = {M}")
                print(f"  #pos pairs = {pos_d.numel()}, mean dist = {pos_d.mean().item():.3f}")
                print(f"  #neg pairs = {neg_d.numel()}, mean dist = {neg_d.mean().item():.3f}")
                print(f"  margin = {margin}")
                first_debug_done = True

        if not per_graph_losses:
            return None

        embed_loss = torch.stack(per_graph_losses).mean()
        if self._is_rank0() and not self._embed_stats_logged:
            print(f"[EMBED LOSS] Using {len(per_graph_losses)} graphs in this batch.")
            self._embed_stats_logged = True

        return embed_loss

    # ----------------------------------------------------------------------
    # Forward with embedding loss + optional edge loss, plus edge stashing
    # ----------------------------------------------------------------------
    def forward(self, batch, stage=None):
        """
        Forward pass with:

        1. Call NuGraph3.forward to compute the usual loss + metrics.
        2. For 'train' and 'val', optionally add:
           - instance embedding loss (weighted by lambda_embed, with warm-up)
           - edge loss (weighted by lambda_edge, with warm-up)
        3. For 'test' (and other stages), keep base loss/metrics but still
           compute edge logits and stash them on batch['hit'] for evaluation.

        NOTE: The new semantic sidecar features are already folded into the
        hit embeddings h.x by the time we get here. This method just works
        on those embeddings; no special handling of sidecar channels is needed.
        """
        base = super().forward(batch, stage)

        # If parent returns something unexpected, don't break.
        if not (isinstance(base, tuple) and len(base) == 2):
            return base

        loss, metrics = base

        # If no sp store or no x, we can't do embed/edge logic
        # CRITICAL: Use SP store because supervision edges connect SP nodes
        if "sp" not in batch.node_types:
            return loss, metrics
        h = batch["sp"]
        if not hasattr(h, "x"):
            return loss, metrics

        x = h.x  # learned SP embeddings (64-dim from nexus aggregation)
        total_loss = loss
        # -------------------------
        # Optional hygiene (in-memory only):
        # y_instance/pid must be -1 on ghosts so nothing downstream can treat them as valid.
        # -------------------------
        ys = getattr(h, "y_semantic", None)
        if ys is not None:
            ys = ys.long()
        
        pid = getattr(h, "pid", None)
        if pid is None:
            pid = getattr(h, "y_instance", None)
        
        if ys is not None and pid is not None:
            pid = pid.long()
            ghost = (ys == -1)
            if ghost.any():
                pid = pid.clone()
                pid[ghost] = -1
                # write back to the same attribute name used by the dataset
                if hasattr(h, "pid"):
                    h.pid = pid
                else:
                    h.y_instance = pid

        # ------------------------------------------------------------------
        # 1) Instance embedding loss (ONLY compute when enabled)
        # ------------------------------------------------------------------
        embed_loss = None
        embed_weight = 0.0
        
        if self.lambda_embed > 0.0:
            embed_loss = self._instance_embedding_loss(x, batch)
        
        if embed_loss is not None:

            if stage in ("train", "val"):
                try:
                    current_epoch = self.trainer.current_epoch
                except (AttributeError, RuntimeError):
                    current_epoch = 0

                WARMUP_EPOCHS = 10
                RAMPUP_EPOCHS = 10

                if current_epoch < WARMUP_EPOCHS:
                    embed_weight = 0.0
                    if self._is_rank0() and not self._warmup_phase_logged_embed:
                        print(
                            f"\n[EMBED WARM-UP] Epochs 0-{WARMUP_EPOCHS-1}: "
                            f"Embedding loss DISABLED (semantic-only training)"
                        )
                        self._warmup_phase_logged_embed = True
                elif current_epoch < WARMUP_EPOCHS + RAMPUP_EPOCHS:
                    ramp_progress = (current_epoch - WARMUP_EPOCHS) / RAMPUP_EPOCHS
                    embed_weight = self.lambda_embed * ramp_progress
                    if self._is_rank0() and not self._rampup_phase_logged_embed:
                        print(
                            f"\n[EMBED RAMP-UP] Epochs {WARMUP_EPOCHS}-"
                            f"{WARMUP_EPOCHS+RAMPUP_EPOCHS-1}: "
                            f"Embedding loss ramping from 0 to {self.lambda_embed}"
                        )
                        self._rampup_phase_logged_embed = True
                else:
                    embed_weight = self.lambda_embed
                    if self._is_rank0() and not self._full_phase_logged_embed:
                        print(
                            f"\n[EMBED FULL] Epoch {WARMUP_EPOCHS+RAMPUP_EPOCHS}+: "
                            f"Full embedding loss (λ_embed={self.lambda_embed})"
                        )
                        self._full_phase_logged_embed = True

                embed_weight = max(0.0, min(self.lambda_embed, float(embed_weight)))
                total_loss = total_loss + embed_weight * embed_loss
            else:
                # In test/inference: compute embed_loss for diagnostics if desired,
                # but do not change total_loss.
                embed_weight = 0.0

        # ------------------------------------------------------------------
        # 2) Edge logits + loss (USE supervision edges; only labelable==1)
        # ------------------------------------------------------------------
        edge_loss = None
        edge_weight = 0.0
        edge_logit = None
        edge_index_valid = None
        edge_probs = None

        need_edges = (self.lambda_edge > 0.0) or (self.lambda_coh > 0.0)

        if need_edges:
            sup_key = ("sp", "supervision", "sp")
            if sup_key in batch.edge_types:
                es = batch[sup_key]
                edge_index = es.edge_index.long()          # (2, E)
                edge_labelable = es.edge_labelable.long()  # (E,)
                y_edge = es.edge_y.float()                 # (E,)

                # Only labelable edges contribute to edge loss
                m = (edge_labelable == 1)
                if m.any():
                    edge_index_valid = edge_index[:, m]
                    y_edge_valid = y_edge[m]

                    # -----------------------
                    # Batch-time hard asserts
                    # NOTE: Use h (sp store) since supervision edges connect SP nodes
                    # -----------------------
                    ys = h.y_semantic.long()
                    a = edge_index_valid[0]
                    b = edge_index_valid[1]
                    assert (((ys[a] == -1) | (ys[b] == -1)).sum().item() == 0), "Labelable edges touch ghosts"

                    pid = getattr(h, "pid", None)
                    if pid is None:
                        pid = getattr(h, "y_instance", None)
                    if pid is not None:
                        pid = pid.long()
                        assert (((pid[a] < 0) | (pid[b] < 0)).sum().item() == 0), "Labelable edges have yi<0 endpoints"

                    # Build edge logits from learned embeddings + geometry
                    # Use SP-specific edge builder for supervision edges
                    edge_attr = self._build_edge_attr_sp(h, x, edge_index_valid)
                    edge_logit = self.edge_mlp_sp(edge_attr).squeeze(-1)
                    edge_probs = torch.sigmoid(edge_logit)

                    # Always stash for eval
                    h.edge_logits = edge_logit.detach()
                    h.edge_index = edge_index_valid.detach()

                    if self.lambda_edge > 0.0 and stage in ("train", "val"):
                        # No warmup complexity for smoke test; keep it simple
                        edge_weight = float(self.lambda_edge)

                        pos_w = torch.tensor(float(self.edge_pos_weight), device=edge_logit.device)
                        edge_loss = F.binary_cross_entropy_with_logits(
                            edge_logit,
                            y_edge_valid,
                            pos_weight=pos_w,
                        )
                        total_loss = total_loss + edge_weight * edge_loss
            # else: no supervision edges; edge loss remains None


        # ------------------------------------------------------------------
        # Extend metrics
        # ------------------------------------------------------------------
        if not isinstance(metrics, dict):
            metrics = {}
        else:
            metrics = dict(metrics)

        # Embedding loss metrics
        if embed_loss is not None and self.lambda_embed > 0.0:
            metrics["loss/embed"] = embed_loss.detach()
            metrics["loss/embed_weight"] = torch.tensor(
                embed_weight, device=x.device
            )

        # Edge loss metrics (only if active and we actually computed it)
        if edge_loss is not None and self.lambda_edge > 0.0:
            metrics["loss/edge"] = edge_loss.detach()
            metrics["loss/edge_weight"] = torch.tensor(
                edge_weight, device=x.device
            )

        # ------------------------------------------------------------------
        # 3) Coherence loss: make semantic logits consistent within clusters
        # ------------------------------------------------------------------
        coh_loss = None
        coh_weight = 0.0
        if (
            self.lambda_coh > 0.0
            and edge_probs is not None
            and edge_index_valid is not None
            and hasattr(h, "x_semantic")
        ):
            # Build clusters per graph from high-confidence edges
            batch_idx = getattr(h, "batch", None)
            if batch_idx is None:
                batch_idx = torch.zeros(h.x.size(0), dtype=torch.long, device=h.x.device)

            pid = getattr(h, "pid", None)
            if pid is None:
                pid = getattr(h, "y_instance", None)

            if pid is not None:
                clusters = self._build_clusters_from_edges(
                    edge_probs=edge_probs,
                    edge_index_valid=edge_index_valid,
                    batch_idx=batch_idx,
                    pid=pid,
                    edge_thr=self.coh_edge_thr,
                    min_cluster_size=self.coh_min_cluster,
                )
                if clusters:
                    # Use semantic logits/probabilities; coherence is variance within cluster
                    s = h.x_semantic  # [N, C] (softmaxed)
                    losses = []
                    for idx in clusters:
                        # Avoid CPU transfers; idx lives on device
                        s_c = s[idx]
                        if s_c.size(0) < 2:
                            continue
                        mean_c = s_c.mean(dim=0, keepdim=True)
                        losses.append(((s_c - mean_c) ** 2).mean())
                    if losses:
                        coh_loss = torch.stack(losses).mean()

            if stage in ("train", "val") and coh_loss is not None:
                try:
                    current_epoch = self.trainer.current_epoch
                except (AttributeError, RuntimeError):
                    current_epoch = 0

                WARMUP_EPOCHS = 10
                RAMPUP_EPOCHS = 10

                if current_epoch < WARMUP_EPOCHS:
                    coh_weight = 0.0
                    if self._is_rank0() and not self._coh_phase_logged_warmup:
                        print(
                            f"\n[COH WARM-UP] Epochs 0-{WARMUP_EPOCHS-1}: "
                            f"Coherence loss DISABLED"
                        )
                        self._coh_phase_logged_warmup = True
                elif current_epoch < WARMUP_EPOCHS + RAMPUP_EPOCHS:
                    ramp_progress = (current_epoch - WARMUP_EPOCHS) / RAMPUP_EPOCHS
                    coh_weight = self.lambda_coh * ramp_progress
                    if self._is_rank0() and not self._coh_phase_logged_rampup:
                        print(
                            f"\n[COH RAMP-UP] Epochs {WARMUP_EPOCHS}-"
                            f"{WARMUP_EPOCHS+RAMPUP_EPOCHS-1}: "
                            f"Coherence loss ramping from 0 to {self.lambda_coh}"
                        )
                        self._coh_phase_logged_rampup = True
                else:
                    coh_weight = self.lambda_coh
                    if self._is_rank0() and not self._coh_phase_logged_full:
                        print(
                            f"\n[COH FULL] Epoch {WARMUP_EPOCHS+RAMPUP_EPOCHS}+: "
                            f"Full coherence loss (λ_coh={self.lambda_coh})"
                        )
                        self._coh_phase_logged_full = True

                coh_weight = max(0.0, min(self.lambda_coh, float(coh_weight)))
                total_loss = total_loss + coh_weight * coh_loss

        if coh_loss is not None and self.lambda_coh > 0.0:
            metrics["loss/coh"] = coh_loss.detach()
            metrics["loss/coh_weight"] = torch.tensor(coh_weight, device=x.device)

        metrics["loss/total"] = total_loss.detach()

        # ------------------------------------------------------------------
        # 4) Edge metrics for validation/test (instance performance tracking)
        # ------------------------------------------------------------------
        if stage in ("val", "test") and edge_logit is not None and edge_index_valid is not None:
            # Only compute metrics if we have edge predictions
            if hasattr(h, 'edge_logits') and hasattr(h, 'edge_index'):
                sup_key = ("sp", "supervision", "sp")
                if sup_key in batch.edge_types:
                    es = batch[sup_key]
                    edge_labelable = es.edge_labelable.long()
                    y_edge_full = es.edge_y.float()
                    
                    labelable_mask = (edge_labelable == 1)
                    if labelable_mask.any():
                        y_edge_true = y_edge_full[labelable_mask]
                        edge_probs_val = torch.sigmoid(h.edge_logits)
                        edge_pred = (edge_probs_val > 0.5).float()
                        
                        with torch.no_grad():
                            # Accuracy
                            edge_acc = (edge_pred == y_edge_true).float().mean()
                            metrics["edge/accuracy"] = edge_acc
                            
                            # Precision, Recall, F1
                            tp = ((edge_pred == 1) & (y_edge_true == 1)).sum().float()
                            fp = ((edge_pred == 1) & (y_edge_true == 0)).sum().float()
                            fn = ((edge_pred == 0) & (y_edge_true == 1)).sum().float()
                            tn = ((edge_pred == 0) & (y_edge_true == 0)).sum().float()
                            
                            precision = tp / (tp + fp + 1e-8)
                            recall = tp / (tp + fn + 1e-8)
                            edge_f1 = 2 * precision * recall / (precision + recall + 1e-8)
                            
                            metrics["edge/f1"] = edge_f1
                            metrics["edge/precision"] = precision
                            metrics["edge/recall"] = recall

        return total_loss, metrics