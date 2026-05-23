#!/usr/bin/env python
# notebooks/eval_semantic.py
"""
Evaluate NuGraph4 semantic and instance segmentation performance.

FIXED VERSION with corrections for:
- Variable naming bug in collect_edge_split (batch -> b)
- Edge prediction retrieval from correct store (SP node store, not edge store)
- Added debug output for first batch
- Length mismatch handling between ground truth and predictions
- Added --in-features argument to match training transform
"""
import os
import argparse
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import (
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision("high")


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate NuGraph semantic head (and optional edge head)"
    )
    p.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint (.ckpt)")
    p.add_argument(
        "--data-path",
        required=True,
        help="HDF5 used in training (same file as train.py)",
    )
    p.add_argument("--split", default="val", choices=["val", "validation", "test"])
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of batches to evaluate (for a quick pass)",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--nu-thr",
        type=float,
        default=None,
        help="Decision threshold for p(nu). If set, we use this value directly.",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=None,
        help=(
            "If provided, choose threshold that maximizes Fβ on the chosen split "
            "(β<1 favors precision; β>1 favors recall). If omitted, uses best-F1."
        ),
    )

    # NEW: neutrino-hit cut controls (must match your DataModule)
    p.add_argument(
        "--min-nu-hits",
        type=int,
        default=None,
        help=(
            "If set, only evaluate events with at least this many neutrino hits "
            "(as defined by nu_cut_plane/nu_class_index)."
        ),
    )
    p.add_argument(
        "--nu-cut-plane",
        type=str,
        default="any",
        choices=["any", "sum", "u", "v", "y"],
        help=(
            "How to count neutrino hits for the cut: "
            "'any' = ≥min on any plane; 'sum' = sum(U+V+Y) ≥min; "
            "or restrict to a specific plane."
        ),
    )
    p.add_argument(
        "--nu-class-index",
        type=int,
        default=0,
        help="Index of the 'nu' class inside semantic_classes (default 0).",
    )
    p.add_argument("--model", default="nugraph4", choices=["nugraph3", "nugraph4"])
    
    # NEW: in_features to match training transform
    p.add_argument(
        "--in-features",
        type=int,
        default=None,
        help=(
            "Number of input features for the transform. "
            "If not set, will try to read from checkpoint hparams. "
            "MUST match the value used during training (e.g., 4 or 18)."
        ),
    )

    # === EDGE EVAL OPTIONS ===================================================
    p.add_argument(
        "--eval-edges",
        action="store_true",
        help="If set (and model=nugraph4), also evaluate edge head using pid/y_instance.",
    )
    p.add_argument(
        "--eval-instances",
        action="store_true",
        help=(
            "If set (and model=nugraph4), cluster SP nodes from predicted same-instance "
            "edges and report ARI/purity/completeness against sp/y_instance."
        ),
    )
    p.add_argument(
        "--edge-thr",
        type=float,
        default=0.5,
        help="Threshold on edge prob for binary edge predictions / SP clustering.",
    )
    p.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum connected-component size kept as a cluster for instance metrics.",
    )
    p.add_argument(
        "--max-edges",
        type=int,
        default=None,
        help=(
            "Optional cap on total number of edges collected for metrics "
            "(subsampled if exceeded, just to keep memory in check)."
        ),
    )
    # ========================================================================
    return p.parse_args()


def normalize_split(split):
    return "val" if split == "validation" else split


def make_datamodule(
    ng,
    data_path,
    model_cls,
    batch_size=None,
    num_workers=None,
    min_nu_hits=None,
    nu_cut_plane="any",
    nu_class_index=0,
    in_features=None,
):
    Data = ng.data.NuGraphDataModule
    
    # Build kwargs, only include in_features if provided
    dm_kwargs = dict(
        model=model_cls,
        data_path=data_path,
        min_nu_hits=min_nu_hits,
    )
    
    # CRITICAL: Pass in_features to get the same transform as training
    if in_features is not None:
        dm_kwargs["in_features"] = in_features
        print(f"[Info] Using in_features={in_features} for transform (must match training)")
    
    dm = Data(**dm_kwargs)

    # Optional overrides
    if batch_size is not None and hasattr(dm, "batch_size"):
        dm.batch_size = batch_size
    if num_workers is not None:
        for attr in ("num_workers", "num_workers_train", "num_workers_eval"):
            if hasattr(dm, attr):
                setattr(dm, attr, num_workers)

    dm.setup("test")

    # Small heads-up on what we're actually evaluating on
    try:
        n_val = len(dm.val_dataset)
        n_test = len(dm.test_dataset)
        print(
            "[Info] Eval datasets after cut "
            f"(min_nu_hits={min_nu_hits}, nu_cut_plane='{nu_cut_plane}', "
            f"nu_class_index={nu_class_index}):"
        )
        print(f"       val:  {n_val} samples | test: {n_test} samples")
    except Exception:
        pass

    return dm


def require_hit_x_or_die(batch, expected_width=None):
    if "hit" not in batch.node_types:
        raise RuntimeError("Batch has no 'hit' node store.")
    keys = list(batch["hit"].keys())
    if not hasattr(batch["hit"], "x"):
        raise RuntimeError(
            "[error] batch['hit'] is missing .x (node features).\n"
            f"Available hit attributes: {keys}\n"
            "Recreate NuGraphDataModule exactly like training (model=Model, data_path=...)."
        )
    if expected_width is not None and batch["hit"].x.size(-1) != expected_width:
        raise RuntimeError(
            f"[error] hit.x has width {batch['hit'].x.size(-1)} but model expects "
            f"{expected_width}.\n"
            "You likely used a different transform; re-create the same DM as training.\n"
            "Try adding --in-features <N> to match your training config."
        )


# =============================================================================
# SEMANTIC EVAL
# =============================================================================
@torch.no_grad()
def collect_split(nugraph, loader, device, limit=None, expected_width=None):
    """
    Returns:
      y_true:    numpy int array with labels {0=nu, 1=cosmic}
      y_pred:    argmax predictions (for reference)
      y_score:   p(nu) probabilities
      evt_id:    global event id per hit (unique across split)
    """
    from tqdm import tqdm

    nugraph.eval().to(device)
    y_true, y_pred, y_score, evt_id = [], [], [], []
    evt_offset = 0

    print("Collecting predictions from the model (semantic)...")
    for i, b in enumerate(tqdm(loader)):
        if limit is not None and i >= limit:
            break
        b = b.to(device)

        require_hit_x_or_die(b, expected_width=expected_width)

        _loss, _ = nugraph(b, stage="test")

        # DEBUG: confirm where semantic predictions actually live
        if i == 0:
            for store in ["hit", "sp"]:
                if store in b.node_types:
                    has_pred = hasattr(b[store], "x_semantic")
                    has_y = hasattr(b[store], "y_semantic")
                    xshape = tuple(b[store].x_semantic.shape) if has_pred else None
                    yshape = tuple(b[store].y_semantic.shape) if has_y else None
                    print(
                        f"[DBG] {store}: has x_semantic={has_pred} shape={xshape} | "
                        f"has y_semantic={has_y} shape={yshape}"
                    )

        # Event ids (graph index) within this batch
        evt = b["hit"].batch.detach()  # [Nhit]

        # IMPORTANT: advance offset per batch, even if we skip due to masks
        batch_max_evt = int(evt.max().item()) if evt.numel() > 0 else -1
        next_offset = evt_offset + (batch_max_evt + 1)

        p = b["hit"].x_semantic.detach()  # [N, C], probs or logits
        p = p.float()

        # If these are logits, convert to probabilities
        row_sum = p.sum(dim=1)
        looks_like_probs = (
            (p.min() >= -1e-3)
            and (p.max() <= 1.0 + 1e-3)
            and torch.isfinite(row_sum).all()
            and (0.9 < row_sum.mean().item() < 1.1)
        )
        if not looks_like_probs:
            p = torch.softmax(p, dim=1)

        y = b["hit"].y_semantic.detach()

        mask = y >= 0
        if mask.sum() == 0:
            evt_offset = next_offset
            continue

        y_np = y[mask].long().cpu().numpy()
        p_np = p[mask].cpu().numpy()
        e_np = (evt[mask].long().cpu().numpy() + evt_offset)

        y_true.append(y_np)
        y_pred.append(p_np.argmax(1))
        y_score.append(p_np[:, 0])  # p(nu)
        evt_id.append(e_np)

        evt_offset = next_offset

    if not y_true:
        raise RuntimeError(
            "No labeled hits found in evaluated batches; check your split/limit."
        )

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_score = np.concatenate(y_score)
    evt_id = np.concatenate(evt_id)
    return y_true, y_pred, y_score, evt_id


# =============================================================================
# EDGE EVAL HELPERS
# =============================================================================
def _get_sp_supervision_edge_truth_and_scores(batch):
    """
    FIXED VERSION: Reads ground-truth supervision edges and model predictions.
    
    Ground truth lives on: batch[('sp','supervision','sp')]
      - edge_y: [E] ground truth labels (0/1 for different/same instance)
      - edge_labelable: [E] mask for which edges are labelable
    
    Model predictions live on: batch["sp"] (stored by NuGraph4.forward)
      - edge_logits: [E_valid] logits for labelable edges only
      - edge_index: [2, E_valid] edge indices for predictions
    
    Returns:
      y_true:  numpy array of ground truth labels for labelable edges
      y_score: numpy array of predicted probabilities for same-instance
    """
    E = ("sp", "supervision", "sp")
    
    # Check edge store exists
    if E not in batch.edge_types:
        return None, None

    e = batch[E]
    
    # Check ground truth exists
    if not hasattr(e, "edge_y") or not hasattr(e, "edge_labelable"):
        return None, None

    # Check predictions exist on SP node store (where NuGraph4 stores them)
    if "sp" not in batch.node_types:
        return None, None
    
    sp = batch["sp"]
    if not hasattr(sp, "edge_logits"):
        # Model didn't compute edge predictions (maybe semantic-only run?)
        return None, None

    # NuGraph4 stores predictions only for labelable edges
    pred = sp.edge_logits  # [E_labelable]
    
    # Filter ground truth to labelable edges
    labelable_mask = (e.edge_labelable > 0)
    if labelable_mask.sum() == 0:
        return None, None
    
    y_true = e.edge_y[labelable_mask].long()
    
    # Sanity check: lengths should match
    if y_true.shape[0] != pred.shape[0]:
        print(
            f"[WARNING] Edge prediction length mismatch: "
            f"y_true={y_true.shape[0]}, pred={pred.shape[0]}"
        )
        # Take minimum to avoid crash (should investigate if this happens)
        min_len = min(y_true.shape[0], pred.shape[0])
        y_true = y_true[:min_len]
        pred = pred[:min_len]
    
    # Ensure pred is 1D
    pred = pred.float()
    if pred.dim() > 1:
        pred = pred.squeeze(-1)

    # Convert logits to probabilities
    if pred.min().item() < -1e-3 or pred.max().item() > 1.0 + 1e-3:
        y_score = torch.sigmoid(pred)
    else:
        y_score = pred

    return y_true.cpu().numpy(), y_score.cpu().numpy()


@torch.no_grad()
def collect_edge_split(nugraph, loader, device, limit=None, max_edges=None):
    """
    FIXED VERSION: Collect edge predictions across the dataset.
    
    Returns:
      y_edge_true:  np array of {0,1} (0=different instance, 1=same instance)
      y_edge_score: np array of predicted prob(edge connects same instance)
    """
    from tqdm import tqdm

    nugraph.eval().to(device)
    y_true_all = []
    y_score_all = []
    total_edges = 0
    
    first_batch = True  # For debug output

    print("Collecting predictions from the model (edges)...")
    for i, b in enumerate(tqdm(loader)):
        if limit is not None and i >= limit:
            break

        b = b.to(device)
        _loss, _ = nugraph(b, stage="test")
        
        # DEBUG OUTPUT FOR FIRST BATCH
        if first_batch and "sp" in b.node_types:
            sp = b["sp"]
            print("\n" + "="*70)
            print("[DEBUG] First batch SP store attributes:")
            print(f"  has edge_logits: {hasattr(sp, 'edge_logits')}")
            print(f"  has edge_index: {hasattr(sp, 'edge_index')}")
            if hasattr(sp, 'edge_logits'):
                print(f"  edge_logits shape: {sp.edge_logits.shape}")
                print(f"  edge_logits range: [{sp.edge_logits.min():.3f}, {sp.edge_logits.max():.3f}]")
            if hasattr(sp, 'edge_index'):
                print(f"  edge_index shape: {sp.edge_index.shape}")
            
            E = ("sp", "supervision", "sp")
            if E in b.edge_types:
                e = b[E]
                print(f"\n[DEBUG] Supervision edge store:")
                if hasattr(e, 'edge_y'):
                    print(f"  edge_y shape: {e.edge_y.shape}")
                    print(f"  edge_y distribution: 0={int((e.edge_y==0).sum())}, 1={int((e.edge_y==1).sum())}")
                else:
                    print(f"  edge_y: N/A")
                    
                if hasattr(e, 'edge_labelable'):
                    print(f"  edge_labelable shape: {e.edge_labelable.shape}")
                    print(f"  labelable edges: {int((e.edge_labelable > 0).sum())}")
                else:
                    print(f"  edge_labelable: N/A")
            print("="*70 + "\n")
            first_batch = False

        # FIXED: Changed 'batch' to 'b' (was causing NameError)
        y_edge_true, y_edge_score = _get_sp_supervision_edge_truth_and_scores(b)
        
        if y_edge_true is None or y_edge_score is None:
            continue

        y_true_all.append(y_edge_true)
        y_score_all.append(y_edge_score)

        total_edges += int(y_edge_true.shape[0])
        if (max_edges is not None) and (total_edges >= max_edges):
            print(f"\n[Info] Reached max_edges={max_edges}, stopping collection.")
            break

    if not y_true_all:
        raise RuntimeError(
            "No valid edges collected for edge evaluation. "
            "Possible reasons:\n"
            "  1. Model checkpoint is from Phase 1 (lambda_edge=0, no edge training)\n"
            "  2. No supervision edges in the dataset\n"
            "  3. Model didn't store edge_logits (check forward() implementation)"
        )

    y_edge_true = np.concatenate(y_true_all)
    y_edge_score = np.concatenate(y_score_all)

    # Subsample if we collected more than max_edges
    if (max_edges is not None) and (y_edge_true.shape[0] > max_edges):
        print(f"[Info] Subsampling {y_edge_true.shape[0]} edges down to {max_edges}")
        idx = np.random.choice(y_edge_true.shape[0], size=max_edges, replace=False)
        y_edge_true = y_edge_true[idx]
        y_edge_score = y_edge_score[idx]

    return y_edge_true, y_edge_score


def report_edge_metrics(y_true, y_score, edge_thr, split_name):
    """Report comprehensive edge classification metrics."""
    print("\n" + "="*70)
    print(f"[{split_name}] EDGE METRICS (same-instance edge = positive)")
    print("="*70)
    
    pos_frac = float((y_true == 1).mean())
    neg_frac = 1.0 - pos_frac
    print(f"Total edges: {len(y_true)}")
    print(f"  Positive (same-instance): {int((y_true==1).sum())} ({pos_frac:.4f})")
    print(f"  Negative (diff-instance): {int((y_true==0).sum())} ({neg_frac:.4f})")
    print(f"  Class imbalance ratio (neg:pos): {neg_frac/pos_frac:.2f}:1")

    # ROC-AUC and PR-AUC
    try:
        auc = roc_auc_score(y_true, y_score)
        print(f"\nROC-AUC  (edge): {auc:.4f}")
    except ValueError as e:
        print(f"\nROC-AUC  (edge): N/A ({e})")
    
    try:
        ap = average_precision_score(y_true, y_score)
        print(f"PR-AUC   (edge): {ap:.4f}")
    except ValueError as e:
        print(f"PR-AUC   (edge): N/A ({e})")

    # Thresholded metrics
    y_pred = (y_score >= edge_thr).astype(int)
    print(f"\n[EDGE] Thresholded at p_same >= {edge_thr:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print("                 pred_diff  pred_same")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"  true_diff:  {cm[0,0]:8d}  {cm[0,1]:8d}")
    print(f"  true_same:  {cm[1,0]:8d}  {cm[1,1]:8d}")
    
    print("\nClassification report (edges):")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["diff", "same"],
            digits=4,
        )
    )
    print("="*70 + "\n")


# =============================================================================
# INSTANCE CLUSTERING EVAL HELPERS
# =============================================================================
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def purity_completeness(y_true, y_pred):
    t_unique, t_inv = np.unique(y_true, return_inverse=True)
    p_unique, p_inv = np.unique(y_pred, return_inverse=True)
    contingency = np.zeros((t_unique.size, p_unique.size), dtype=np.int64)
    for ti, pi in zip(t_inv, p_inv):
        contingency[ti, pi] += 1

    total = contingency.sum()
    if total == 0:
        return np.nan, np.nan

    purity = contingency.max(axis=0).sum() / total
    completeness = contingency.max(axis=1).sum() / total
    return purity, completeness


def clusters_from_edges(num_nodes, edge_index, edge_score, edge_thr, min_cluster_size):
    uf = UnionFind(num_nodes)
    keep = edge_score >= edge_thr
    if keep.any():
        for src, dst in edge_index[:, keep].T.tolist():
            uf.union(int(src), int(dst))

    roots = np.array([uf.find(i) for i in range(num_nodes)], dtype=np.int64)
    counts = Counter(roots.tolist())
    pred = np.empty_like(roots)
    remap = {}
    next_id = 0
    for i, root in enumerate(roots):
        root = int(root)
        if counts[root] < min_cluster_size:
            pred[i] = next_id
            next_id += 1
        else:
            if root not in remap:
                remap[root] = next_id
                next_id += 1
            pred[i] = remap[root]
    return pred


@torch.no_grad()
def collect_instance_cluster_metrics(
    nugraph,
    loader,
    device,
    edge_thr,
    min_cluster_size=2,
    limit=None,
):
    from tqdm import tqdm

    nugraph.eval().to(device)

    total_points = 0
    weighted_ari = 0.0
    weighted_purity = 0.0
    weighted_completeness = 0.0
    graphs_used = 0

    print("Collecting predictions from the model (SP instance clusters)...")
    for i, b in enumerate(tqdm(loader)):
        if limit is not None and i >= limit:
            break

        b = b.to(device)
        _loss, _ = nugraph(b, stage="test")

        if "sp" not in b.node_types:
            continue
        sp = b["sp"]
        if not hasattr(sp, "edge_logits") or not hasattr(sp, "edge_index"):
            continue
        if not hasattr(sp, "y_instance"):
            raise RuntimeError("SP store has no y_instance; cannot evaluate instance clusters.")

        edge_logits = sp.edge_logits.float()
        if edge_logits.dim() > 1:
            edge_logits = edge_logits.squeeze(-1)
        if edge_logits.min().item() < -1e-3 or edge_logits.max().item() > 1.0 + 1e-3:
            edge_score = torch.sigmoid(edge_logits).detach().cpu().numpy()
        else:
            edge_score = edge_logits.detach().cpu().numpy()
        edge_index = sp.edge_index.detach().cpu().numpy()
        y_instance = sp.y_instance.detach().cpu().numpy().astype(np.int64)

        if hasattr(sp, "batch"):
            batch_index = sp.batch.detach().cpu().numpy().astype(np.int64)
        else:
            num_sp_nodes = int(getattr(sp, "num_nodes", y_instance.shape[0]))
            batch_index = np.zeros(num_sp_nodes, dtype=np.int64)

        for graph_id in np.unique(batch_index):
            node_mask = batch_index == graph_id
            node_idx = np.flatnonzero(node_mask)
            if node_idx.size < 2:
                continue

            local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(node_idx)}
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            if not edge_mask.any():
                continue

            edge_index_g_global = edge_index[:, edge_mask]
            edge_score_g = edge_score[edge_mask]
            edge_index_g = np.array(
                [[local[int(src)], local[int(dst)]] for src, dst in edge_index_g_global.T],
                dtype=np.int64,
            ).T

            y_true = y_instance[node_idx]
            labeled = y_true >= 0
            if labeled.sum() < 2 or np.unique(y_true[labeled]).size < 2:
                continue

            y_pred_all = clusters_from_edges(
                num_nodes=node_idx.size,
                edge_index=edge_index_g,
                edge_score=edge_score_g,
                edge_thr=edge_thr,
                min_cluster_size=min_cluster_size,
            )

            y_true_labeled = y_true[labeled]
            y_pred_labeled = y_pred_all[labeled]
            weight = int(labeled.sum())

            ari = adjusted_rand_score(y_true_labeled, y_pred_labeled)
            purity, completeness = purity_completeness(y_true_labeled, y_pred_labeled)
            if np.isnan(purity) or np.isnan(completeness):
                continue

            graphs_used += 1
            total_points += weight
            weighted_ari += ari * weight
            weighted_purity += purity * weight
            weighted_completeness += completeness * weight

    if graphs_used == 0 or total_points == 0:
        raise RuntimeError(
            "No graphs contributed to instance clustering metrics. "
            "Check --edge-thr, --limit, and that the checkpoint has lambda_edge > 0."
        )

    return {
        "graphs_used": graphs_used,
        "total_labeled_sp": total_points,
        "ari": weighted_ari / total_points,
        "purity": weighted_purity / total_points,
        "completeness": weighted_completeness / total_points,
    }


def report_instance_cluster_metrics(metrics, edge_thr, min_cluster_size, split_name):
    print("\n" + "="*70)
    print(f"[{split_name}] SP INSTANCE CLUSTERING METRICS")
    print("="*70)
    print(f"Graphs used:            {metrics['graphs_used']}")
    print(f"Total labeled SP:       {metrics['total_labeled_sp']}")
    print(f"Edge threshold p_same:  {edge_thr:.3f}")
    print(f"Min cluster size:       {min_cluster_size}")
    print(f"Weighted ARI:           {metrics['ari']:.4f}")
    print(f"Weighted purity:        {metrics['purity']:.4f}")
    print(f"Weighted completeness:  {metrics['completeness']:.4f}")
    print("="*70 + "\n")


# =============================================================================
# SEMANTIC REPORT HELPERS
# =============================================================================
def report_argmax(y_true, y_pred, y_score, split_name):
    print(f"\n[{split_name}] ARGMAX baseline")
    print("labels:", Counter(y_true), "preds:", Counter(y_pred))
    print("\nConfusion matrix (rows=true [nu, cosmic], cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["nu", "cosmic"],
            digits=4,
        )
    )
    print(
        "\nnu-score stats (mean,std,min,max):",
        f"{float(y_score.mean()):.4f}",
        f"{float(y_score.std()):.4f}",
        f"{float(y_score.min()):.4f}",
        f"{float(y_score.max()):.4f}",
    )


def report_thresholded(y_true, y_score, nu_thr, header):
    """
    Apply threshold on p(nu). If p(nu) >= nu_thr => predict nu(0), else cosmic(1).
    """
    y_pred_thr_bin = (y_score >= nu_thr).astype(int)
    y_pred_thr = np.where(y_pred_thr_bin == 1, 0, 1)

    print(f"\n{header}")
    print(f"nu-threshold = {nu_thr:.3f}")
    print("preds:", Counter(y_pred_thr))
    print("\nConfusion matrix @thr (rows=true [nu, cosmic], cols=pred):")
    print(confusion_matrix(y_true, y_pred_thr, labels=[0, 1]))
    print("\nClassification report @thr:")
    print(
        classification_report(
            y_true,
            y_pred_thr,
            target_names=["nu", "cosmic"],
            digits=4,
        )
    )


def pick_best_f1_threshold(y_true, y_score):
    y_bin = (y_true == 0).astype(int)
    prec, rec, thr = precision_recall_curve(y_bin, y_score)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    idx = np.nanargmax(f1[:-1])
    best_thr = thr[idx] if idx < len(thr) else 0.5
    return best_thr, float(prec[idx]), float(rec[idx])


def pick_best_fbeta_threshold(y_true, y_score, beta=0.5):
    y_bin = (y_true == 0).astype(int)
    prec, rec, thr = precision_recall_curve(y_bin, y_score)
    beta2 = beta * beta
    fbeta = (1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-12)
    idx = np.nanargmax(fbeta[:-1])
    best_thr = thr[idx] if idx < len(thr) else 0.5
    return best_thr, float(prec[idx]), float(rec[idx])


def plot_pr_curve(y_true, y_score, filename, beta=None):
    print(f"\nGenerating Precision-Recall curve for '{filename}'...")
    y_bin = (y_true == 0).astype(int)
    precision, recall, _ = precision_recall_curve(y_bin, y_score)

    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(recall, precision, marker=".", markersize=3)
    plt.title("Precision-Recall Curve for Neutrino Class")
    plt.xlabel("Neutrino Recall (Efficiency)")
    plt.ylabel("Neutrino Precision (Purity)")
    plt.grid(True)
    plt.xlim([0, 1.02])
    plt.ylim([0, 1.02])

    f1_thr, p1, r1 = pick_best_f1_threshold(y_true, y_score)
    plt.plot(
        r1,
        p1,
        "ro",
        markersize=7,
        label=f"Best F1 (thr={f1_thr:.3f})\nP={p1:.2f}, R={r1:.2f}",
    )

    if beta is not None:
        fbeta_thr, pb, rb = pick_best_fbeta_threshold(y_true, y_score, beta=beta)
        plt.plot(
            rb,
            pb,
            "gs",
            markersize=7,
            label=f"Best F{beta:.2f} (thr={fbeta_thr:.3f})\nP={pb:.2f}, R={rb:.2f}",
        )

    plt.legend()
    plt.savefig(filename)
    print(f"--> Saved plot to {filename}")


def report_event_level(y_true, y_score, evt_id, nu_thr, split_name):
    """
    Event is positive if it contains ANY true nu hits (y_true==0).
    Event is predicted positive if it contains ANY predicted nu hits above threshold.
    """
    order = np.argsort(evt_id)
    evt_id_s = evt_id[order]
    y_s = y_true[order]
    s_s = y_score[order]

    uniq, start_idx = np.unique(evt_id_s, return_index=True)
    start_idx = np.append(start_idx, len(evt_id_s))

    y_evt_true = []
    y_evt_pred = []

    for i in range(len(uniq)):
        lo, hi = start_idx[i], start_idx[i + 1]
        y_e = y_s[lo:hi]
        s_e = s_s[lo:hi]

        true_event_nu = np.any(y_e == 0)
        pred_event_nu = np.any(s_e >= nu_thr)

        y_evt_true.append(1 if true_event_nu else 0)
        y_evt_pred.append(1 if pred_event_nu else 0)

    y_evt_true = np.array(y_evt_true, dtype=int)
    y_evt_pred = np.array(y_evt_pred, dtype=int)

    print(f"\n[{split_name}] EVENT-LEVEL @thr={nu_thr:.3f}")
    print(f"Events: {len(y_evt_true)} | true nu-events: {int(y_evt_true.sum())}")

    cm = confusion_matrix(y_evt_true, y_evt_pred, labels=[0, 1])
    print("Confusion matrix (rows=true [no-nu, nu], cols=pred):")
    print(cm)

    print("\nClassification report (event-level):")
    print(
        classification_report(
            y_evt_true,
            y_evt_pred,
            target_names=["no-nu", "nu"],
            digits=4,
        )
    )


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    split = normalize_split(args.split)
    split_name = split.upper()

    import nugraph as ng

    if args.model == "nugraph4":
        Model = ng.models.NuGraph4
    else:
        Model = ng.models.NuGraph3

    # Try to get in_features from checkpoint if not provided
    in_features = args.in_features
    if in_features is None:
        # Peek at checkpoint to get in_features
        try:
            ckpt = torch.load(args.ckpt, map_location="cpu")
            if "hyper_parameters" in ckpt:
                in_features = ckpt["hyper_parameters"].get("in_features", None)
                if in_features is not None:
                    print(f"[Info] Auto-detected in_features={in_features} from checkpoint")
        except Exception as e:
            print(f"[Warning] Could not read in_features from checkpoint: {e}")
    
    # If still None, warn and use a default
    if in_features is None:
        print("[Warning] in_features not specified and not found in checkpoint.")
        print("          Using default=4. If this fails, add --in-features <N>.")
        in_features = 4

    dm = make_datamodule(
        ng,
        data_path=args.data_path,
        model_cls=Model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_nu_hits=args.min_nu_hits,
        # nu_cut_plane=args.nu_cut_plane,
        nu_class_index=args.nu_class_index,
        in_features=in_features,
    )

    nugraph = Model.load_from_checkpoint(args.ckpt, map_location="cpu")
    print("Loaded checkpoint:", args.ckpt)
    
    # Print checkpoint hyperparameters for debugging
    if hasattr(nugraph, 'hparams'):
        hparams = nugraph.hparams
        print("\n[Info] Checkpoint hyperparameters:")
        for key in ['in_features', 'lambda_edge', 'lambda_embed', 'edge_pos_weight', 'num_iters', 
                    'use_sp_features', 'use_vtx_features']:
            if hasattr(hparams, key):
                print(f"  {key}: {getattr(hparams, key)}")

    expected_in_features = in_features

    loader = dm.val_dataloader() if split == "val" else dm.test_dataloader()

    # =========================================================================
    # SEMANTIC EVALUATION
    # =========================================================================
    y_true, y_pred, y_score, evt_id = collect_split(
        nugraph,
        loader,
        args.device,
        args.limit,
        expected_width=expected_in_features,
    )

    report_argmax(y_true, y_pred, y_score, split_name)

    # Choose threshold
    if args.nu_thr is not None:
        chosen_thr = float(args.nu_thr)
        header = f"[{split_name}] THRESHOLDED (user)"
    else:
        if args.beta is not None:
            chosen_thr, p, r = pick_best_fbeta_threshold(y_true, y_score, beta=args.beta)
            print(
                f"\nBest-F{args.beta:.2f} ν-threshold found: {chosen_thr:.3f} "
                f"(precision={p:.3f}, recall={r:.3f})"
            )
            header = f"[{split_name}] THRESHOLDED (best-F{args.beta:.2f})"
        else:
            chosen_thr, p, r = pick_best_f1_threshold(y_true, y_score)
            print(
                f"\nBest-F1 ν-threshold found: {chosen_thr:.3f} "
                f"(precision={p:.3f}, recall={r:.3f})"
            )
            header = f"[{split_name}] THRESHOLDED (best-F1)"

    # Report BOTH hit-level thresholded and event-level at the chosen threshold
    report_thresholded(y_true, y_score, chosen_thr, header=header)
    report_event_level(y_true, y_score, evt_id, chosen_thr, split_name)

    plot_filename = f"{split}_{os.path.basename(args.ckpt).replace('.ckpt', '_pr_curve.png')}"
    plot_pr_curve(y_true, y_score, plot_filename, beta=args.beta)

    # =========================================================================
    # EDGE EVALUATION (OPTIONAL)
    # =========================================================================
    if args.eval_edges and args.model == "nugraph4":
        print("\n" + "="*70)
        print("[EDGE] Starting edge evaluation using pid/y_instance labels...")
        print("="*70)
        
        # Check if checkpoint actually has edge training
        if hasattr(nugraph, 'hparams'):
            lambda_edge = getattr(nugraph.hparams, 'lambda_edge', 0.0)
            if lambda_edge == 0.0:
                print(
                    "\n[WARNING] Checkpoint has lambda_edge=0.0 (no edge training).\n"
                    "This is likely from Phase 1 (semantic-only). For edge evaluation,\n"
                    "use a Phase 2 checkpoint with lambda_edge > 0.\n"
                )
                print("Skipping edge evaluation.")
            else:
                try:
                    y_edge_true, y_edge_score = collect_edge_split(
                        nugraph,
                        loader,
                        args.device,
                        limit=args.limit,
                        max_edges=args.max_edges,
                    )
                    report_edge_metrics(
                        y_edge_true,
                        y_edge_score,
                        args.edge_thr,
                        split_name,
                    )
                except RuntimeError as e:
                    print(f"\n[ERROR] Edge evaluation failed: {e}")
        else:
            # Try anyway if hparams not available
            try:
                y_edge_true, y_edge_score = collect_edge_split(
                    nugraph,
                    loader,
                    args.device,
                    limit=args.limit,
                    max_edges=args.max_edges,
                )
                report_edge_metrics(
                    y_edge_true,
                    y_edge_score,
                    args.edge_thr,
                    split_name,
                )
            except RuntimeError as e:
                print(f"\n[ERROR] Edge evaluation failed: {e}")
                
    elif args.eval_edges:
        print(
            "\n[EDGE] --eval-edges was set but model!=nugraph4; skipping edge eval."
        )

    # =========================================================================
    # INSTANCE CLUSTERING EVALUATION (OPTIONAL)
    # =========================================================================
    if args.eval_instances and args.model == "nugraph4":
        print("\n" + "="*70)
        print("[INSTANCE] Starting SP instance clustering evaluation...")
        print("="*70)

        lambda_edge = 0.0
        lambda_coh = 0.0
        if hasattr(nugraph, "hparams"):
            lambda_edge = float(getattr(nugraph.hparams, "lambda_edge", 0.0))
            lambda_coh = float(getattr(nugraph.hparams, "lambda_coh", 0.0))

        if lambda_edge == 0.0 and lambda_coh == 0.0:
            print(
                "\n[WARNING] Checkpoint appears to have no edge/cohesion head enabled "
                "(lambda_edge=0 and lambda_coh=0). Skipping instance clustering."
            )
        else:
            try:
                instance_metrics = collect_instance_cluster_metrics(
                    nugraph,
                    loader,
                    args.device,
                    edge_thr=args.edge_thr,
                    min_cluster_size=args.min_cluster_size,
                    limit=args.limit,
                )
                report_instance_cluster_metrics(
                    instance_metrics,
                    args.edge_thr,
                    args.min_cluster_size,
                    split_name,
                )
            except RuntimeError as e:
                print(f"\n[ERROR] Instance clustering evaluation failed: {e}")
    elif args.eval_instances:
        print(
            "\n[INSTANCE] --eval-instances was set but model!=nugraph4; skipping instance eval."
        )


if __name__ == "__main__":
    main()
