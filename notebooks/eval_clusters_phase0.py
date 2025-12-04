#!/usr/bin/env python
"""
Phase 0: Evaluate NuGraph4 instance clustering quality from edge affinities.

- Loads a trained NuGraph{3,4} model + HDF5 via NuGraphDataModule.
- Runs a forward pass to get hit embeddings.
- Uses the hetero edge ("hit","delaunay-planar","hit") + pid/y_instance
  to build per-graph clusters from high-confidence edges (p_same >= thr).
- Computes per-graph Adjusted Rand Index (ARI), purity, and completeness
  vs. true pid (on labeled hits), and reports weighted averages.
"""

import argparse
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

torch.set_float32_matmul_precision("high")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Phase-0 clustering eval (ARI/purity) from NuGraph4 edge affinities"
    )
    p.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint (.ckpt)")
    p.add_argument("--data-path", required=True, help="HDF5 file used in training")
    p.add_argument(
        "--model",
        default="nugraph4",
        choices=["nugraph3", "nugraph4"],
        help="Which model class to load from nugraph.models",
    )
    p.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Which split to evaluate on",
    )
    p.add_argument(
        "--edge-thr",
        type=float,
        default=0.8,
        help="Threshold on p_same for building clusters (default: 0.8)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override of DataModule batch size",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional override of DataModule num_workers",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of batches to process (for quick tests)",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model + data",
    )
    p.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum cluster size to include in coherence stats (default: 2)",
    )
    return p.parse_args()


# ----------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def purity_completeness(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute purity and completeness for a single graph.

    - y_true: 1D array of true instance IDs (pid or y_instance), length N
    - y_pred: 1D array of predicted cluster IDs (arbitrary ints), length N
    """
    # Map to 0..T-1 and 0..K-1
    t_unique, t_inv = np.unique(y_true, return_inverse=True)
    p_unique, p_inv = np.unique(y_pred, return_inverse=True)

    T = t_unique.size
    K = p_unique.size
    contingency = np.zeros((T, K), dtype=np.int64)
    for ti, pi in zip(t_inv, p_inv):
        contingency[ti, pi] += 1

    total = contingency.sum()
    if total == 0:
        return np.nan, np.nan

    # Purity: sum over clusters of majority true class / N
    purity = contingency.max(axis=0).sum() / total

    # Completeness: sum over true labels of majority cluster / N
    completeness = contingency.max(axis=1).sum() / total

    return purity, completeness


# ----------------------------------------------------------------------
# DataModule helper (mirrors eval_semantic.py style)
# ----------------------------------------------------------------------
def make_datamodule(ng, data_path, model_cls, batch_size=None, num_workers=None):
    Data = ng.data.NuGraphDataModule
    dm = Data(
        model=model_cls,
        data_path=data_path,
        # If your NuGraphDataModule has min_nu_hits / nu_cut_plane, add args here
    )

    if batch_size is not None and hasattr(dm, "batch_size"):
        dm.batch_size = batch_size
    if num_workers is not None:
        for attr in ("num_workers", "num_workers_train", "num_workers_eval"):
            if hasattr(dm, attr):
                setattr(dm, attr, num_workers)

    dm.setup("test")

    try:
        n_val = len(dm.val_dataset)
        n_test = len(dm.test_dataset)
        print(f"[Info] Datasets: val={n_val}, test={n_test}")
    except Exception:
        pass

    return dm


# ----------------------------------------------------------------------
# Main clustering evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def eval_clusters(
    nugraph,
    loader,
    device,
    edge_thr: float,
    min_cluster_size: int = 2,
    limit: int | None = None,
):
    """
    For each batch:
      - run model forward to get hit embeddings,
      - build high-confidence edges (p_same >= edge_thr) on labeled hits,
      - find connected components per graph via union-find,
      - compute per-graph ARI, purity, completeness.

    Returns weighted averages over all graphs, weighted by # labeled hits.
    """
    from tqdm import tqdm

    nugraph.eval().to(device)

    total_labeled_hits = 0

    # Weighted sums over graphs
    ari_weighted = 0.0
    purity_weighted = 0.0
    completeness_weighted = 0.0

    n_graphs_used = 0

    edge_key = ("hit", "delaunay-planar", "hit")

    print(
        f"[Phase0] Evaluating clusters with edge_thr={edge_thr:.3f}, "
        f"min_cluster_size={min_cluster_size}"
    )

    for ib, batch in enumerate(tqdm(loader)):
        if limit is not None and ib >= limit:
            break

        batch = batch.to(device)

        if "hit" not in batch.node_types:
            continue

        h = batch["hit"]

        # Forward to populate embeddings & semantic stuff
        _loss, _metrics = nugraph(batch, stage="test")

        # Hit embeddings after core
        if not hasattr(h, "x"):
            raise RuntimeError("batch['hit'] is missing .x (hit embeddings).")

        x = h.x  # [N, D]
        N = x.size(0)

        # Instance IDs (truth)
        pid = getattr(h, "pid", None)
        if pid is None:
            pid = getattr(h, "y_instance", None)
        if pid is None:
            raise RuntimeError("Need either hit.pid or hit.y_instance for clustering truth.")

        pid = pid.to(device)

        # Graph indices (event index per hit)
        batch_idx = getattr(h, "batch", None)
        if batch_idx is None:
            batch_idx = torch.zeros(N, dtype=torch.long, device=device)
        else:
            batch_idx = batch_idx.to(device)

        # Hetero edge index: (hit, delaunay-planar, hit)
        if not hasattr(batch, "edge_index_dict") or edge_key not in batch.edge_index_dict:
            raise RuntimeError(
                f"Batch missing hetero edge key {edge_key}. "
                "Is this the 3D ppedges dataset?"
            )

        edge_index = batch[edge_key].edge_index  # [2, E]
        src, dst = edge_index

        # Valid edges: both endpoints labeled
        valid_nodes = (pid >= 0)
        valid_edges = valid_nodes[src] & valid_nodes[dst]
        if valid_edges.sum() == 0:
            continue

        edge_index_valid = edge_index[:, valid_edges]

        # Build edge features with the *current* embeddings
        # (mirror NuGraph4._build_edge_attr logic)
        edge_attr = nugraph._build_edge_attr(h, x, edge_index_valid)
        edge_logits = nugraph.edge_mlp(edge_attr).squeeze(-1)  # [E_valid]
        edge_probs = torch.sigmoid(edge_logits)

        # Threshold on p_same
        keep = edge_probs >= edge_thr
        if keep.sum() == 0:
            # No high-confidence edges in this batch; skip
            continue

        edge_index_thr = edge_index_valid[:, keep]
        src_thr, dst_thr = edge_index_thr

        # ------------------------------------------------------------------
        # Per-graph clustering and metrics
        # ------------------------------------------------------------------
        # Work graph by graph so ARI/purity aren't polluted by event boundaries
        graphs = batch_idx.unique()
        for g in graphs:
            g = int(g.item())
            mask_g = (batch_idx == g) & (pid >= 0)
            idx_g = mask_g.nonzero(as_tuple=False).view(-1)
            M = idx_g.numel()
            if M < 2:
                continue

            # Map global hit indices -> local 0..M-1
            global_to_local = {int(gi.item()): li for li, gi in enumerate(idx_g)}

            # Edges whose endpoints are both in this graph + labeled
            # (edge_index_thr is already labeled; just select graph)
            edge_mask_g = (
                (batch_idx[src_thr] == g)
                & (batch_idx[dst_thr] == g)
            )
            if edge_mask_g.sum() == 0:
                continue

            src_g = src_thr[edge_mask_g]
            dst_g = dst_thr[edge_mask_g]

            # Union-Find over local nodes
            uf = UnionFind(M)
            for s, d in zip(src_g.tolist(), dst_g.tolist()):
                if s in global_to_local and d in global_to_local:
                    uf.union(global_to_local[s], global_to_local[d])

            # Build per-node predicted cluster labels (local)
            roots = [uf.find(i) for i in range(M)]
            roots = np.array(roots, dtype=np.int64)

            # Optionally filter tiny clusters (set them to singleton IDs)
            if min_cluster_size > 1:
                # Count cluster sizes
                counts = Counter(roots.tolist())
                remap = {}
                next_id = 0
                for r in np.unique(roots):
                    if counts[int(r)] >= min_cluster_size:
                        remap[int(r)] = next_id
                        next_id += 1
                    else:
                        # treat as its own singleton
                        remap[int(r)] = next_id
                        next_id += 1
                pred_clusters = np.array([remap[int(r)] for r in roots], dtype=np.int64)
            else:
                # compress to 0..K-1
                _, pred_clusters = np.unique(roots, return_inverse=True)

            # Ground truth instance IDs for this graph
            y_true = pid[idx_g].cpu().numpy().astype(np.int64)

            # Sanity check: need at least 2 distinct true labels for ARI
            if np.unique(y_true).size < 2:
                continue

            # Compute per-graph ARI, purity, completeness
            ari_g = adjusted_rand_score(y_true, pred_clusters)
            purity_g, completeness_g = purity_completeness(y_true, pred_clusters)

            if np.isnan(purity_g) or np.isnan(completeness_g):
                continue

            n_graphs_used += 1
            total_labeled_hits += M
            ari_weighted += ari_g * M
            purity_weighted += purity_g * M
            completeness_weighted += completeness_g * M

    if total_labeled_hits == 0 or n_graphs_used == 0:
        raise RuntimeError("No graphs contributed to ARI/purity; check thresholds / labels.")

    ari_avg = ari_weighted / total_labeled_hits
    purity_avg = purity_weighted / total_labeled_hits
    completeness_avg = completeness_weighted / total_labeled_hits

    print("\n================ PHASE-0 CLUSTERING SUMMARY ================")
    print(f"Graphs used:           {n_graphs_used}")
    print(f"Total labeled hits:    {total_labeled_hits}")
    print(f"Edge threshold p_same: {edge_thr:.3f}")
    print(f"Weighted ARI:          {ari_avg:.4f}")
    print(f"Weighted purity:       {purity_avg:.4f}")
    print(f"Weighted completeness: {completeness_avg:.4f}")
    print("============================================================\n")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    # Import your project just like in eval_semantic.py
    import nugraph as ng

    if args.model == "nugraph4":
        Model = ng.models.NuGraph4
    else:
        Model = ng.models.NuGraph3

    dm = make_datamodule(
        ng,
        data_path=args.data_path,
        model_cls=Model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    nugraph = Model.load_from_checkpoint(args.ckpt, map_location="cpu")
    print("Loaded checkpoint:", args.ckpt)

    loader = dm.val_dataloader() if args.split == "val" else dm.test_dataloader()

    eval_clusters(
        nugraph=nugraph,
        loader=loader,
        device=args.device,
        edge_thr=args.edge_thr,
        min_cluster_size=args.min_cluster_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
