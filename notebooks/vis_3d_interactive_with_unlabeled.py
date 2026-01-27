#!/usr/bin/env python
# filename: vis_3d_interactive_with_unlabeled.py
# UPDATED: Added --in-features argument to match training transform

import os
from pathlib import Path
import argparse
import traceback
import sys
from collections import defaultdict, Counter

import numpy as np
import torch
torch.set_float32_matmul_precision("high")


def log(msg: str):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive 3D visualization of NuGraph4 results (creates HTML files)"
    )
    p.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint (.ckpt)")
    p.add_argument("--data-path", required=True, help="HDF5 used in training")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--nu-thr", type=float, default=0.635)
    p.add_argument("--limit-events", type=int, default=5)
    p.add_argument("--skip-events", type=int, default=0)
    p.add_argument("--point-size", type=float, default=3.0)
    p.add_argument("--outfile-dir", type=str, default="event_viz_3d_interactive_val1")
    p.add_argument("--cluster-distance-thr", type=float, default=0.20)
    p.add_argument("--both-tpc-only", action="store_true",
                   help="Only visualize events that span both TPCs")
    p.add_argument("--hide-unlabeled", action="store_true",
                   help="Hide unlabeled points (original behavior)")
    p.add_argument("--no-shuffle", action="store_true",
                   help="Try to disable shuffle/samplers so batch_idx maps to dataset index")
    # NEW: in_features to match training
    p.add_argument("--in-features", type=int, default=None,
                   help="Number of input features (must match training). "
                        "If not set, tries to read from checkpoint.")
    return p.parse_args()


# =============================================================================
# Metadata helpers
# =============================================================================

def write_meta_sidecar(meta_path: str, meta_dict: dict):
    with open(meta_path, "w") as f:
        for k, v in meta_dict.items():
            f.write(f"{k}: {v}\n")


def sanitize_tag_for_filename(tag: str, maxlen: int = 140) -> str:
    safe = "".join([c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(tag)])
    return safe[:maxlen]


def try_get_dataset_event_key(ds, idx: int):
    """
    Best-effort: return the exact H5 event key string for dataset element `idx`.
    Different NuGraph forks store this list under different attribute names.
    """
    for attr in ["keys", "event_keys", "sample_keys", "samples", "ids", "_keys", "_samples"]:
        if hasattr(ds, attr):
            v = getattr(ds, attr)
            try:
                if isinstance(v, (list, tuple, np.ndarray)):
                    if idx < len(v):
                        return str(v[idx])
            except Exception:
                pass

    for attr in ["samples_by_split", "keys_by_split", "samples", "keys"]:
        if hasattr(ds, attr):
            v = getattr(ds, attr)
            if isinstance(v, dict):
                for split_name in ["train", "val", "test"]:
                    if split_name in v and idx < len(v[split_name]):
                        return str(v[split_name][idx])

    for meth in ["get_key", "key", "event_key", "sample_key"]:
        if hasattr(ds, meth) and callable(getattr(ds, meth)):
            try:
                return str(getattr(ds, meth)(idx))
            except Exception:
                pass

    return None


# =============================================================================
# Color Utilities
# =============================================================================

def get_tab20_colors(n):
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('tab20')
    colors = []
    for i in range(n):
        rgba = cmap(i % 20)
        colors.append(f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})')
    return colors


def remap_to_consecutive(labels):
    remapped = np.full_like(labels, -1)
    valid_mask = labels >= 0
    if not valid_mask.any():
        return remapped
    unique_labels = np.unique(labels[valid_mask])
    label_to_idx = {orig: idx for idx, orig in enumerate(unique_labels)}
    for i, label in enumerate(labels):
        if label >= 0:
            remapped[i] = label_to_idx[label]
    return remapped


# =============================================================================
# Metrics
# =============================================================================

def compute_ari_nmi(true_labels, pred_labels):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    labeled = (true_labels >= 0) & (pred_labels >= 0)
    if labeled.sum() < 2:
        return None, None
    t = true_labels[labeled]
    p = pred_labels[labeled]
    return adjusted_rand_score(t, p), normalized_mutual_info_score(t, p)


def compute_pq(true_labels, pred_labels, iou_threshold=0.5):
    from scipy.optimize import linear_sum_assignment
    labeled = (true_labels >= 0) & (pred_labels >= 0)
    if labeled.sum() < 2:
        return None
    true_lab = true_labels[labeled]
    pred_lab = pred_labels[labeled]
    true_ids = np.unique(true_lab)
    pred_ids = np.unique(pred_lab)
    if len(true_ids) == 0 or len(pred_ids) == 0:
        return {"PQ": 0}
    iou_matrix = np.zeros((len(true_ids), len(pred_ids)))
    for i, t in enumerate(true_ids):
        t_mask = (true_lab == t)
        for j, p in enumerate(pred_ids):
            p_mask = (pred_lab == p)
            inter = (t_mask & p_mask).sum()
            union = t_mask.sum() + p_mask.sum() - inter
            iou_matrix[i, j] = inter / union if union > 0 else 0
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_ious = []
    matched_true, matched_pred = set(), set()
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matched_ious.append(iou_matrix[r, c])
            matched_true.add(r)
            matched_pred.add(c)
    TP = len(matched_ious)
    FP = len(pred_ids) - len(matched_pred)
    FN = len(true_ids) - len(matched_true)
    SQ = np.mean(matched_ious) if matched_ious else 0
    RQ = TP / (TP + 0.5*FP + 0.5*FN) if (TP + FP + FN) > 0 else 0
    return {"PQ": SQ * RQ}


# =============================================================================
# Projection: Hit predictions -> SP level
# =============================================================================

def project_hit_semantic_to_sp(batch, nu_thr=0.635):
    hit = batch['hit']
    sp = batch['sp']

    hit_probs = hit.x_semantic.cpu().numpy()
    hit_pred = (hit_probs[:, 0] >= nu_thr).astype(int)
    hit_pred = 1 - hit_pred  # 0=nu, 1=cosmic

    hit_to_sp_key = ('hit', 'nexus', 'sp')
    if hit_to_sp_key not in batch.edge_types:
        return None

    nexus_edges = batch[hit_to_sp_key].edge_index.cpu().numpy()
    hit_indices = nexus_edges[0]
    sp_indices = nexus_edges[1]

    batch_hit = hit.batch.cpu().numpy() if hasattr(hit, 'batch') else np.zeros(hit.num_nodes, dtype=int)
    batch_sp = sp.batch.cpu().numpy() if hasattr(sp, 'batch') else np.zeros(sp.num_nodes, dtype=int)

    sp_to_hit_preds = defaultdict(list)
    for h_i, sp_i in zip(hit_indices, sp_indices):
        if batch_hit[h_i] == batch_sp[sp_i]:
            sp_to_hit_preds[sp_i].append(hit_pred[h_i])

    sp_semantic_pred = np.full(sp.num_nodes, -1, dtype=np.int64)
    for sp_i, preds in sp_to_hit_preds.items():
        if len(preds) > 0:
            sp_semantic_pred[sp_i] = Counter(preds).most_common(1)[0][0]

    return sp_semantic_pred


# =============================================================================
# Clustering
# =============================================================================

def cluster_sp_embeddings_labeled_only(embeddings, batch_sp, true_instance_sp, distance_threshold=0.20):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize

    num_sp = len(embeddings)
    embeddings_norm = normalize(embeddings, norm='l2')
    sp_clusters = np.full(num_sp, -1, dtype=np.int64)
    cluster_offset = 0

    for event_id in np.unique(batch_sp):
        event_mask = (batch_sp == event_id)
        labeled_mask = event_mask & (true_instance_sp >= 0)
        n_labeled = labeled_mask.sum()

        if n_labeled < 2:
            if n_labeled == 1:
                sp_clusters[labeled_mask] = cluster_offset
                cluster_offset += 1
            continue

        emb_labeled = embeddings_norm[labeled_mask]
        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='euclidean',
            linkage='average'
        )
        labels_labeled = agg.fit_predict(emb_labeled)
        sp_clusters[labeled_mask] = labels_labeled + cluster_offset
        cluster_offset = sp_clusters.max() + 1

    return sp_clusters


# =============================================================================
# Interactive 3D Visualization
# =============================================================================

def create_interactive_3d_html(
    x, y, z,
    sem_true, sem_pred,
    inst_true, inst_pred,
    event_id, metrics,
    output_path,
    point_size=3.0,
    show_unlabeled=True
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
               [{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Semantic Truth', 'Semantic Prediction', 'True Instances', 'Predicted Clusters'),
        horizontal_spacing=0.05,
        vertical_spacing=0.08
    )

    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=1.5, z=1.5)
    )

    scene_layout = dict(
        xaxis=dict(title='X (drift) [mm]', range=[-2500, 2500]),
        yaxis=dict(title='Y (vertical) [mm]', range=[-2500, 2500]),
        zaxis=dict(title='Z (beam) [mm]', range=[0, 5500]),
        camera=camera,
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=1.1)
    )

    # Semantic Truth
    if sem_true is not None:
        cosmic_mask = (sem_true == 1)
        if cosmic_mask.any():
            fig.add_trace(
                go.Scatter3d(
                    x=x[cosmic_mask], y=y[cosmic_mask], z=z[cosmic_mask],
                    mode='markers',
                    marker=dict(size=point_size, color='orange', opacity=0.7),
                    name='Cosmic (truth)',
                ),
                row=1, col=1
            )

        nu_mask = (sem_true == 0)
        if nu_mask.any():
            fig.add_trace(
                go.Scatter3d(
                    x=x[nu_mask], y=y[nu_mask], z=z[nu_mask],
                    mode='markers',
                    marker=dict(size=point_size, color='blue', opacity=0.9),
                    name='ν (truth)',
                ),
                row=1, col=1
            )

    # Semantic Prediction
    if sem_pred is not None:
        valid = (sem_pred >= 0)
        cosmic_mask = (sem_pred == 1) & valid
        nu_mask = (sem_pred == 0) & valid

        if cosmic_mask.any():
            fig.add_trace(
                go.Scatter3d(
                    x=x[cosmic_mask], y=y[cosmic_mask], z=z[cosmic_mask],
                    mode='markers',
                    marker=dict(size=point_size, color='orange', opacity=0.7),
                    name='Cosmic (pred)',
                    showlegend=False
                ),
                row=1, col=2
            )

        if nu_mask.any():
            fig.add_trace(
                go.Scatter3d(
                    x=x[nu_mask], y=y[nu_mask], z=z[nu_mask],
                    mode='markers',
                    marker=dict(size=point_size, color='blue', opacity=0.9),
                    name='ν (pred)',
                    showlegend=False
                ),
                row=1, col=2
            )

    # True Instances
    if show_unlabeled:
        unlabeled_mask = (inst_true < 0)
        if unlabeled_mask.any() and sem_true is not None:
            unlabeled_nu = unlabeled_mask & (sem_true == 0)
            if unlabeled_nu.any():
                fig.add_trace(
                    go.Scatter3d(
                        x=x[unlabeled_nu], y=y[unlabeled_nu], z=z[unlabeled_nu],
                        mode='markers',
                        marker=dict(size=point_size * 0.8, color='black', opacity=0.5),
                        name=f'ν unlabeled ({unlabeled_nu.sum()})',
                        showlegend=True
                    ),
                    row=2, col=1
                )

            unlabeled_cosmic = unlabeled_mask & (sem_true == 1)
            if unlabeled_cosmic.any():
                fig.add_trace(
                    go.Scatter3d(
                        x=x[unlabeled_cosmic], y=y[unlabeled_cosmic], z=z[unlabeled_cosmic],
                        mode='markers',
                        marker=dict(size=point_size * 0.6, color='moccasin', opacity=0.3),
                        name=f'cosmic unlabeled ({unlabeled_cosmic.sum()})',
                        showlegend=True
                    ),
                    row=2, col=1
                )

    inst_true_remapped = remap_to_consecutive(inst_true)
    labeled = (inst_true >= 0)
    if labeled.any():
        n_instances = int(inst_true_remapped[labeled].max()) + 1
        colors = get_tab20_colors(n_instances)
        for inst_id in range(n_instances):
            mask = (inst_true_remapped == inst_id)
            if mask.any():
                fig.add_trace(
                    go.Scatter3d(
                        x=x[mask], y=y[mask], z=z[mask],
                        mode='markers',
                        marker=dict(size=point_size, color=colors[inst_id], opacity=0.8),
                        name=f'Inst {inst_id}',
                        showlegend=False
                    ),
                    row=2, col=1
                )

    # Predicted Clusters
    if show_unlabeled:
        unlabeled_mask = (inst_true < 0)
        if unlabeled_mask.any() and sem_true is not None:
            unlabeled_nu = unlabeled_mask & (sem_true == 0)
            if unlabeled_nu.any():
                fig.add_trace(
                    go.Scatter3d(
                        x=x[unlabeled_nu], y=y[unlabeled_nu], z=z[unlabeled_nu],
                        mode='markers',
                        marker=dict(size=point_size * 0.8, color='black', opacity=0.5),
                        name='ν unlabeled',
                        showlegend=False
                    ),
                    row=2, col=2
                )

    inst_pred_remapped = remap_to_consecutive(inst_pred)
    labeled_pred = labeled & (inst_pred >= 0)
    if labeled_pred.any():
        n_clusters = int(inst_pred_remapped[labeled_pred].max()) + 1
        colors = get_tab20_colors(n_clusters)
        for clust_id in range(n_clusters):
            mask = (inst_pred_remapped == clust_id) & labeled_pred
            if mask.any():
                fig.add_trace(
                    go.Scatter3d(
                        x=x[mask], y=y[mask], z=z[mask],
                        mode='markers',
                        marker=dict(size=point_size, color=colors[clust_id], opacity=0.8),
                        name=f'Cluster {clust_id}',
                        showlegend=False
                    ),
                    row=2, col=2
                )

    title_parts = [f"Event {event_id}"]
    if metrics.get("h5_event_key"):
        title_parts.append(f"h5={metrics['h5_event_key']}")
    if metrics.get("dataset_index_in_split") is not None:
        title_parts.append(f"idx={metrics['dataset_index_in_split']}")
    if 'n_nu_true' in metrics:
        title_parts.append(f"truth: ν={metrics['n_nu_true']}, cosmic={metrics['n_cosmic_true']}")
    if 'n_nu_pred' in metrics:
        title_parts.append(f"pred: ν={metrics['n_nu_pred']}, cosmic={metrics['n_cosmic_pred']}")
    if 'ari' in metrics and metrics['ari'] is not None:
        title_parts.append(f"ARI={metrics['ari']:.2f}")
    if 'pq' in metrics and metrics['pq'] is not None:
        title_parts.append(f"PQ={metrics['pq']:.2f}")

    fig.update_layout(
        title=dict(text=" | ".join(title_parts), x=0.5, font=dict(size=14)),
        scene=scene_layout,
        scene2=scene_layout,
        scene3=scene_layout,
        scene4=scene_layout,
        height=900,
        width=1400,
        showlegend=True,
        margin=dict(l=0, r=150, t=50, b=0)
    )

    fig.write_html(output_path, include_plotlyjs='cdn')
    return output_path


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    outdir = Path(args.outfile_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"[info] Output directory: {outdir.resolve()}")

    log("[info] Loading model and data...")
    import nugraph as ng
    Model = ng.models.NuGraph4

    # Try to get in_features from checkpoint if not provided
    in_features = args.in_features
    if in_features is None:
        try:
            ckpt = torch.load(args.ckpt, map_location="cpu")
            if "hyper_parameters" in ckpt:
                in_features = ckpt["hyper_parameters"].get("in_features", None)
                if in_features is not None:
                    log(f"[info] Auto-detected in_features={in_features} from checkpoint")
        except Exception as e:
            log(f"[warn] Could not read in_features from checkpoint: {e}")
    
    if in_features is None:
        log("[warn] in_features not specified. Using default=4. Add --in-features <N> if this fails.")
        in_features = 4
    
    log(f"[info] Using in_features={in_features}")

    # Create DataModule with correct in_features
    dm = ng.data.NuGraphDataModule(
        model=Model, 
        data_path=args.data_path,
        in_features=in_features,  # CRITICAL: must match training
    )

    # knobs BEFORE setup
    if hasattr(dm, "batch_size"):
        dm.batch_size = args.batch_size
    if hasattr(dm, "num_workers"):
        dm.num_workers = args.num_workers

    # Try to disable shuffle/samplers so batch_idx is meaningful
    if args.no_shuffle:
        if hasattr(dm, "shuffle"):
            dm.shuffle = "none"
            log("[info] Set dm.shuffle='none'")
        else:
            log("[warn] dm has no attribute 'shuffle'; dataloader may still shuffle. "
                "If so, batch_idx will NOT map stably to H5 keys.")

    dm.setup("fit")
    dm.setup("test")

    def get_loader(split: str):
        if split == "train":
            return dm.train_dataloader()
        if split == "val":
            return dm.val_dataloader()
        if split == "test":
            return dm.test_dataloader()
        raise ValueError(split)

    model = Model.load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval().to(args.device)

    from tqdm import tqdm
    show_unlabeled = not args.hide_unlabeled

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        loader = get_loader(split)
        ds = getattr(loader, "dataset", None)

        split_outdir = outdir / split
        split_outdir.mkdir(parents=True, exist_ok=True)

        saved = 0
        seen = 0

        log(f"[info] ==== Split={split} -> {split_outdir} ====")
        if args.batch_size != 1:
            log("[warn] batch_size != 1. Mapping batch_idx -> single H5 event is ambiguous. "
                "Set --batch-size 1 if you want exact H5 mapping.")

        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Visualizing {split}")):
            batch = batch.to(args.device)

            with torch.inference_mode():
                model(batch, stage="test")

            sp = batch["sp"]
            pos_3d = sp.pos.cpu().numpy()
            embeddings = sp.x.cpu().numpy()
            batch_sp = sp.batch.cpu().numpy()

            sp_semantic_pred = project_hit_semantic_to_sp(batch, nu_thr=args.nu_thr)
            sp_semantic_true = sp.y_semantic.cpu().numpy() if hasattr(sp, "y_semantic") else None

            if hasattr(sp, "y_instance"):
                true_instance_sp = sp.y_instance.cpu().numpy()
            else:
                true_instance_sp = np.full(sp.num_nodes, -1, dtype=np.int64)

            pred_clusters = cluster_sp_embeddings_labeled_only(
                embeddings, batch_sp, true_instance_sp, args.cluster_distance_thr
            )

            sp_ptr = sp.ptr.cpu().numpy() if hasattr(sp, "ptr") else None
            if sp_ptr is None:
                continue

            num_graphs = len(sp_ptr) - 1

            h5_event_key = None
            if ds is not None and args.batch_size == 1:
                h5_event_key = try_get_dataset_event_key(ds, batch_idx)

            for g in range(num_graphs):
                if seen < args.skip_events:
                    seen += 1
                    continue
                if args.limit_events is not None and saved >= args.limit_events:
                    log(f"[done] Split={split}: Saved {saved} events to {split_outdir}/")
                    break

                a, b = int(sp_ptr[g]), int(sp_ptr[g + 1])

                x_ev = pos_3d[a:b, 0]
                y_ev = pos_3d[a:b, 1]
                z_ev = pos_3d[a:b, 2]

                if args.both_tpc_only:
                    in_left = (x_ev < 0).any()
                    in_right = (x_ev > 0).any()
                    if not (in_left and in_right):
                        seen += 1
                        continue

                true_inst_ev = true_instance_sp[a:b]
                pred_clust_ev = pred_clusters[a:b]
                sem_true_ev = sp_semantic_true[a:b] if sp_semantic_true is not None else None
                sem_pred_ev = sp_semantic_pred[a:b] if sp_semantic_pred is not None else None

                ari_ev, _ = compute_ari_nmi(true_inst_ev, pred_clust_ev)
                pq_result = compute_pq(true_inst_ev, pred_clust_ev)
                pq_ev = pq_result["PQ"] if pq_result else None

                n_nu_true = int((sem_true_ev == 0).sum()) if sem_true_ev is not None else 0
                n_cosmic_true = int((sem_true_ev == 1).sum()) if sem_true_ev is not None else 0
                n_nu_pred = int((sem_pred_ev == 0).sum()) if sem_pred_ev is not None else 0
                n_cosmic_pred = int((sem_pred_ev == 1).sum()) if sem_pred_ev is not None else 0

                n_nu_unlabeled = 0
                if sem_true_ev is not None:
                    n_nu_unlabeled = int(((sem_true_ev == 0) & (true_inst_ev < 0)).sum())

                metrics = {
                    "dataset_index_in_split": int(batch_idx) if args.batch_size == 1 else None,
                    "h5_event_key": h5_event_key,
                    "n_nu_true": n_nu_true,
                    "n_cosmic_true": n_cosmic_true,
                    "n_nu_pred": n_nu_pred,
                    "n_cosmic_pred": n_cosmic_pred,
                    "n_nu_unlabeled": n_nu_unlabeled,
                    "ari": ari_ev,
                    "pq": pq_ev,
                }

                tag = None
                if h5_event_key is not None:
                    tag = sanitize_tag_for_filename(h5_event_key)

                if tag:
                    outfile = split_outdir / f"{split}_event_3d_{seen:06d}_idx{batch_idx:06d}_{tag}.html"
                else:
                    outfile = split_outdir / f"{split}_event_3d_{seen:06d}_idx{batch_idx:06d}.html"

                meta_path = str(outfile).replace(".html", ".meta.txt")
                write_meta_sidecar(meta_path, {
                    "split": split,
                    "viz_index_seen": seen,
                    "dataset_index_in_split": metrics["dataset_index_in_split"],
                    "h5_event_key": metrics["h5_event_key"],
                    "n_sp_nodes": int(b - a),
                    "truth_counts": f"nu={n_nu_true} cosmic={n_cosmic_true}",
                    "pred_counts": f"nu={n_nu_pred} cosmic={n_cosmic_pred}",
                    "n_nu_unlabeled": n_nu_unlabeled,
                    "ari": ari_ev,
                    "pq": pq_ev,
                })

                if metrics["h5_event_key"] is None:
                    log(f"[warn] Could not recover H5 key for batch_idx={batch_idx}. "
                        f"Your dataset likely does not expose the event key list.")

                create_interactive_3d_html(
                    x=x_ev, y=y_ev, z=z_ev,
                    sem_true=sem_true_ev,
                    sem_pred=sem_pred_ev,
                    inst_true=true_inst_ev,
                    inst_pred=pred_clust_ev,
                    event_id=seen,
                    metrics=metrics,
                    output_path=str(outfile),
                    point_size=args.point_size,
                    show_unlabeled=show_unlabeled
                )

                saved += 1
                seen += 1

            if args.device == "cuda":
                torch.cuda.empty_cache()

        log(f"[done] Split={split}: Saved {saved} interactive HTML files to {split_outdir}/")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)