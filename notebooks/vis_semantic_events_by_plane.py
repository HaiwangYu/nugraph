#!/usr/bin/env python
# filename: vis_semantic_events_by_plane.py

import os
from pathlib import Path
import argparse
import traceback
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
torch.set_float32_matmul_precision("high")


def log(msg: str):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Event-by-event per-plane Truth vs Prediction + Instances + Clusters (NuGraph4)"
    )
    # Data/model
    p.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint (.ckpt)")
    p.add_argument("--data-path", required=True, help="HDF5 used in training")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader overrides for memory control
    p.add_argument("--batch-size", type=int, default=1,
                   help="Eval batch size (default 1 to avoid OOM).")
    p.add_argument("--num-workers", type=int, default=0,
                   help="Eval dataloader workers (default 0).")

    # Memory/precision knobs
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none",
                   help="Autocast precision on GPU. Use 'bf16' on A100/H100 or 'fp16' if needed.")
    p.add_argument("--disable-checkpointing", action="store_true",
                   help="Bypass model core_net.checkpoint wrapper during inference.")

    # Threshold selection for semantics
    p.add_argument("--nu-thr", type=float, default=None,
                   help="Decision threshold for p(nu). If unset and --beta is given, "
                        "pick best-Fbeta on the chosen split; else 0.5.")
    p.add_argument("--beta", type=float, default=None,
                   help="Find threshold maximizing F_beta on the chosen split.")

    # Event selection
    p.add_argument("--limit-events", type=int, default=20)
    p.add_argument("--skip-events", type=int, default=0)

    # Plane detection / override
    p.add_argument("--plane-field", type=str, default=None,
                   help="Name of tensor on hit store that encodes plane (e.g. plane/view/pid).")
    p.add_argument("--plane-from-x-col", type=int, default=None,
                   help="If no plane field, take plane id from this x feature column.")
    p.add_argument("--plane-values", type=int, nargs=3, default=[0, 1, 2],
                   help="Integer values that correspond to planes [U V Y] (default 0 1 2).")
    p.add_argument("--plane-names", type=str, nargs=3, default=["u", "v", "y"],
                   help="Names of planes for labels (default: u v y).")

    # Coordinates
    p.add_argument("--x-col", type=int, default=None,
                   help="Feature index to use for X if no pos/xy present.")
    p.add_argument("--y-col", type=int, default=None,
                   help="Feature index to use for Y if no pos/xy present.")

    # Plotting
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--outfile-dir", type=str, default="event_viz_by_plane")

    # Performance / clarity
    p.add_argument("--max-points", type=int, default=None,
                   help="Randomly subsample at most this many hits per event for speed/clarity.")
    p.add_argument("--debug", action="store_true")

    # Physics selection
    p.add_argument("--min-nu-hits", type=int, default=0,
                   help="Keep only events with ≥ this many ν hits (sum over U+V+Y) in the chosen split.")

    # Cluster-building
    p.add_argument("--edge-thr", type=float, default=0.8,
                   help="Threshold on p_same for building predicted clusters.")

    return p.parse_args()


def maybe_subsample(idx_array, max_points=None, rng=None):
    if max_points is None or len(idx_array) <= max_points:
        return idx_array
    if rng is None:
        rng = np.random.default_rng(123)
    pick = rng.choice(len(idx_array), size=max_points, replace=False)
    return idx_array[pick]


def get_hit_xy(hit_store, x_col=None, y_col=None):
    if hasattr(hit_store, "pos") and hit_store.pos is not None:
        pos = hit_store.pos
        if pos.dim() == 2 and pos.size(-1) >= 2:
            pos_np = pos[:, :2].detach().cpu().numpy()
            return pos_np[:, 0], pos_np[:, 1]
    if hasattr(hit_store, "xy") and hit_store.xy is not None:
        xy = hit_store.xy
        if xy.dim() == 2 and xy.size(-1) >= 2:
            xy_np = xy.detach().cpu().numpy()
            return xy_np[:, 0], xy_np[:, 1]
    if not hasattr(hit_store, "x") or hit_store.x is None:
        raise RuntimeError("No hit.pos/xy and hit.x missing—need --x-col/--y-col.")
    if x_col is None or y_col is None:
        raise RuntimeError("No hit.pos/xy—please pass --x-col and --y-col.")
    x = hit_store.x[:, x_col].detach().cpu().numpy()
    y = hit_store.x[:, y_col].detach().cpu().numpy()
    return x, y


def detect_plane_tensor(hit_store, prefer=None, debug=False):
    candidates = []
    if prefer:
        candidates.append(prefer)
    candidates += ["plane", "view", "pid", "plane_id", "wire_plane", "p"]
    for name in candidates:
        if hasattr(hit_store, name):
            t = getattr(hit_store, name)
            try:
                arr = t.detach().cpu().numpy()
            except Exception:
                continue
            if arr.ndim == 1 and arr.shape[0] == hit_store.num_nodes:
                if debug:
                    uniq = np.unique(arr)
                    log(f"[debug] found plane-like '{name}', unique={uniq[:10]}")
                return arr.astype(int), name
    return None, None


def pick_best_threshold(y_true, y_score, beta=1.0):
    from sklearn.metrics import precision_recall_curve
    y_pos = (y_true == 0).astype(int)  # 'nu' is positive
    p, r, thr = precision_recall_curve(y_pos, y_score)
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r + 1e-12)
    idx = np.nanargmax(f[:-1])
    return float(thr[idx] if idx < len(thr) else 0.5)


@torch.no_grad()
def collect_split_scores(model, loader, device, expected_in_features=None, debug=False, amp="none"):
    y_true, y_score = [], []
    from tqdm import tqdm
    autocast = torch.autocast if (device == "cuda" and amp in {"bf16", "fp16"}) else None
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    for b in tqdm(loader, desc="Collecting scores (for threshold)"):
        b = b.to(device)
        if expected_in_features is not None:
            if not hasattr(b["hit"], "x") or b["hit"].x.size(-1) != expected_in_features:
                raise RuntimeError(
                    f"hit.x width {b['hit'].x.size(-1) if hasattr(b,'hit') else 'N/A'} "
                    f"!= expected {expected_in_features}"
                )
        if autocast:
            with autocast(device_type="cuda", dtype=amp_dtype):
                _loss, _metrics = model(b, stage="test")
        else:
            _loss, _metrics = model(b, stage="test")
        probs = b["hit"].x_semantic.detach()
        y = b["hit"].y_semantic.detach()
        m = y >= 0
        if m.sum() == 0:
            continue
        y_true.append(y[m].long().cpu().numpy())
        y_score.append(probs[m][:, 0].cpu().numpy())
        if device == "cuda":
            torch.cuda.empty_cache()
    if not y_true:
        raise RuntimeError("No labeled hits found while collecting scores.")
    return np.concatenate(y_true), np.concatenate(y_score)


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


def build_clusters_for_batch(model, batch, edge_thr: float):
    """
    Build per-hit cluster IDs using NuGraph4 edge head, similar to eval_clusters_phase0.py.
    Returns:
        cluster_all: np.ndarray of shape [N_hits], with integer cluster ids per hit.
    """
    device = next(model.parameters()).device
    edge_key = ("hit", "delaunay-planar", "hit")

    if "hit" not in batch.node_types:
        return None

    h = batch["hit"]
    if not hasattr(h, "x"):
        raise RuntimeError("batch['hit'] is missing .x (hit embeddings).")

    x = h.x  # [N, D]
    N = x.size(0)

    batch_idx = getattr(h, "batch", None)
    if batch_idx is None:
        batch_idx = torch.zeros(N, dtype=torch.long, device=device)
    else:
        batch_idx = batch_idx.to(device)

    if not hasattr(batch, "edge_index_dict") or edge_key not in batch.edge_index_dict:
        raise RuntimeError(
            f"Batch missing hetero edge key {edge_key}. "
            "Is this the 3D ppedges dataset?"
        )

    edge_index = batch[edge_key].edge_index  # [2, E]
    src, dst = edge_index

    edge_attr = model._build_edge_attr(h, x, edge_index)
    edge_logits = model.edge_mlp(edge_attr).squeeze(-1)  # [E]
    edge_probs = torch.sigmoid(edge_logits)

    keep = edge_probs >= edge_thr
    if keep.sum() == 0:
        return np.arange(N, dtype=int)

    edge_index_thr = edge_index[:, keep]
    src_thr, dst_thr = edge_index_thr

    cluster_all = np.full(N, -1, dtype=int)

    graphs = batch_idx.unique()
    for g in graphs:
        g = int(g.item())
        mask_g = (batch_idx == g)
        idx_g = mask_g.nonzero(as_tuple=False).view(-1)
        M = idx_g.numel()
        if M < 1:
            continue

        global_to_local = {int(gi.item()): li for li, gi in enumerate(idx_g)}

        edge_mask_g = (
            (batch_idx[src_thr] == g)
            & (batch_idx[dst_thr] == g)
        )
        if edge_mask_g.sum() == 0:
            for li, gi in enumerate(idx_g.tolist()):
                cluster_all[gi] = gi
            continue

        src_g = src_thr[edge_mask_g]
        dst_g = dst_thr[edge_mask_g]

        uf = UnionFind(M)
        for s, d in zip(src_g.tolist(), dst_g.tolist()):
            if s in global_to_local and d in global_to_local:
                uf.union(global_to_local[s], global_to_local[d])

        roots = [uf.find(i) for i in range(M)]
        roots = np.array(roots, dtype=np.int64)
        _, pred_clusters = np.unique(roots, return_inverse=True)

        for li, gi in enumerate(idx_g.tolist()):
            cluster_all[gi] = int(pred_clusters[li])

    mask_unassigned = (cluster_all < 0)
    if mask_unassigned.any():
        next_id = cluster_all.max() + 1 if cluster_all.max() >= 0 else 0
        for i in np.where(mask_unassigned)[0]:
            cluster_all[i] = next_id
            next_id += 1

    return cluster_all


def main():
    args = parse_args()

    outdir = Path(args.outfile_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "_STARTED.txt").write_text("script started\n")
    log(f"[info] Outdir: {outdir.resolve()}")

    log("[info] Importing nugraph and loading model/datamodule...")
    import nugraph as ng
    Model = ng.models.NuGraph4
    Data = ng.data.NuGraphDataModule

    # Build DataModule and override eval batch size & workers
    dm = Data(model=Model, data_path=args.data_path, min_nu_hits=args.min_nu_hits)
    dm.setup("test")
    for attr in ("batch_size", "batch_size_eval"):
        if hasattr(dm, attr):
            setattr(dm, attr, args.batch_size)
    for attr in ("num_workers", "num_workers_eval"):
        if hasattr(dm, attr):
            setattr(dm, attr, args.num_workers)

    loader = dm.val_dataloader() if args.split == "val" else dm.test_dataloader()

    model = Model.load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval().to(args.device)
    expected_in_features = getattr(getattr(model, "hparams", None), "in_features", None)

    if args.disable_checkpointing or args.disable_checkpointing:
        try:
            if hasattr(model, "core_net") and hasattr(model.core_net, "checkpoint"):
                model.core_net.checkpoint = (lambda f, *a, **k: f(*a, **k))
                log("[info] Disabled core_net.checkpoint wrapper for inference.")
        except Exception:
            pass

    if args.nu_thr is not None:
        nu_thr = float(args.nu_thr)
        log(f"[info] Using user ν-threshold: {nu_thr:.3f}")
    elif args.beta is not None:
        log(f"[info] Computing best-F{args.beta:.2f} threshold on {args.split}...")
        y_true_all, y_score_all = collect_split_scores(
            model, loader, args.device,
            expected_in_features=expected_in_features,
            debug=args.debug, amp=args.amp
        )
        nu_thr = pick_best_threshold(y_true_all, y_score_all, beta=args.beta)
        log(f"[info] Best-F{args.beta:.2f} ν-threshold: {nu_thr:.3f}")
    else:
        nu_thr = 0.5
        log("[warn] No --nu-thr/--beta; defaulting to 0.5")

    from tqdm import tqdm
    rng = np.random.default_rng(123)
    saved = 0
    seen = 0

    use_autocast = (args.device == "cuda" and args.amp in {"bf16", "fp16"})
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_autocast else torch.cuda.amp.autocast(enabled=False)
    )

    log(f"[info] Visualizing split={args.split}  limit={args.limit_events}  "
        f"skip={args.skip_events}  batch_size={args.batch_size}  "
        f"workers={args.num_workers}  amp={args.amp}  edge_thr={args.edge_thr}")

    for batch in tqdm(loader, desc=f"Visualizing {args.split} events"):
        batch = batch.to(args.device)

        # Check input feature width BEFORE forward (h.x will be encoder output after forward)
        if expected_in_features is not None:
            if not hasattr(batch["hit"], "x") or batch["hit"].x.size(-1) != expected_in_features:
                raise RuntimeError(
                    f"hit.x width {batch['hit'].x.size(-1) if hasattr(batch['hit'],'x') else 'N/A'} "
                    f"!= expected {expected_in_features}"
                )

        with torch.inference_mode(), autocast_ctx:
            _loss, _metrics = model(batch, stage="test")

        hit = batch["hit"]

        probs = hit.x_semantic.detach().cpu().numpy()         # [N,2]
        labels = hit.y_semantic.detach().cpu().numpy()        # [N]
        p_nu = probs[:, 0]

        X, Y = get_hit_xy(hit, x_col=args.x_col, y_col=args.y_col)

        plane_np, used_name = detect_plane_tensor(hit, prefer=args.plane_field, debug=args.debug)
        if plane_np is None:
            if args.plane_from_x_col is None:
                raise RuntimeError(
                    "Could not find a plane tensor on hit store. "
                    "Pass --plane-field (e.g. plane/view/pid) or --plane-from-x-col."
                )
            plane_np = hit.x[:, args.plane_from_x_col].detach().cpu().numpy().astype(int)
            used_name = f"x[:,{args.plane_from_x_col}]"
        if args.debug and saved == 0:
            log(f"[debug] using plane field: {used_name}")

        if hasattr(hit, "y_instance") and hit.y_instance is not None:
            y_inst = hit.y_instance.detach().cpu().numpy()
        elif hasattr(hit, "pid") and hit.pid is not None:
            y_inst = hit.pid.detach().cpu().numpy()
        else:
            y_inst = None

        try:
            cluster_all = build_clusters_for_batch(model, batch, edge_thr=args.edge_thr)
        except Exception as e:
            if args.debug:
                log(f"[debug] Could not build clusters for this batch: {e}")
            cluster_all = None

        if not hasattr(hit, "ptr") or hit.ptr is None:
            raise RuntimeError("batch['hit'].ptr missing, cannot slice per event.")
        ptr = hit.ptr.cpu().numpy()
        num_graphs = len(ptr) - 1

        for g in range(num_graphs):
            if seen < args.skip_events:
                seen += 1
                continue
            if args.limit_events is not None and saved >= args.limit_events:
                log(f"[done] Saved {saved} events to {outdir}/")
                return

            a, b = int(ptr[g]), int(ptr[g + 1])
            loc = slice(a, b)

            x_ev, y_ev = X[loc], Y[loc]
            y_true_ev = labels[loc]
            p_nu_ev = p_nu[loc]
            y_pred_ev = np.where(p_nu_ev >= nu_thr, 0, 1)  # 0=nu, 1=cosmic
            plane_ev = plane_np[loc]

            if y_inst is not None:
                y_inst_ev = y_inst[loc]
            else:
                y_inst_ev = None

            if cluster_all is not None:
                cluster_ev = cluster_all[loc]
            else:
                cluster_ev = None

            if args.max_points is not None and len(x_ev) > args.max_points:
                pick = maybe_subsample(np.arange(len(x_ev)), args.max_points, rng)
                x_ev = x_ev[pick]
                y_ev = y_ev[pick]
                y_true_ev = y_true_ev[pick]
                y_pred_ev = y_pred_ev[pick]
                plane_ev = plane_ev[pick]
                if y_inst_ev is not None:
                    y_inst_ev = y_inst_ev[pick]
                if cluster_ev is not None:
                    cluster_ev = cluster_ev[pick]

            fig, axes = plt.subplots(4, 3, figsize=(12, 10), dpi=args.dpi, constrained_layout=True)
            row_titles = [
                "Semantic Truth (ν vs cosmic)",
                f"Semantic Prediction (thr={nu_thr:.3f})",
                "True Instances",
                "Predicted Clusters"
            ]
            col_titles = args.plane_names
            pv = args.plane_values

            plane_masks = [
                (plane_ev == pv[0]),
                (plane_ev == pv[1]),
                (plane_ev == pv[2]),
            ]

            for c in range(3):
                mask_p = plane_masks[c]

                ax = axes[0, c]
                m_lab = (y_true_ev >= 0) & mask_p
                if m_lab.any():
                    ax.scatter(x_ev[m_lab & (y_true_ev == 0)], y_ev[m_lab & (y_true_ev == 0)],
                               s=args.point_size, alpha=0.85, label="ν (truth)")
                    ax.scatter(x_ev[m_lab & (y_true_ev == 1)], y_ev[m_lab & (y_true_ev == 1)],
                               s=args.point_size, alpha=0.85, label="cosmic (truth)")
                ax.set_title(f"{row_titles[0]} — {col_titles[c]}")
                ax.set_xlabel("X"); ax.set_ylabel("Y")
                ax.legend(markerscale=3, loc="best")

                ax = axes[1, c]
                mp = mask_p
                if mp.any():
                    ax.scatter(x_ev[mp & (y_pred_ev == 0)], y_ev[mp & (y_pred_ev == 0)],
                               s=args.point_size, alpha=0.85, label="ν (pred)")
                    ax.scatter(x_ev[mp & (y_pred_ev == 1)], y_ev[mp & (y_pred_ev == 1)],
                               s=args.point_size, alpha=0.85, label="cosmic (pred)")
                ax.set_title(f"{row_titles[1]} — {col_titles[c]}")
                ax.set_xlabel("X"); ax.set_ylabel("Y")

                ax = axes[2, c]
                if y_inst_ev is not None:
                    m_inst = (y_inst_ev >= 0) & mask_p
                    if m_inst.any():
                        ax.scatter(
                            x_ev[m_inst], y_ev[m_inst],
                            c=y_inst_ev[m_inst],
                            s=args.point_size, alpha=0.85, cmap="tab20"
                        )
                    ax.set_title(f"{row_titles[2]} — {col_titles[c]}")
                    ax.set_xlabel("X"); ax.set_ylabel("Y")
                else:
                    ax.set_title(f"{row_titles[2]} — {col_titles[c]} (no y_instance)")
                    ax.set_xlabel("X"); ax.set_ylabel("Y")

                ax = axes[3, c]
                if cluster_ev is not None:
                    m_cl = mask_p
                    if m_cl.any():
                        ax.scatter(
                            x_ev[m_cl], y_ev[m_cl],
                            c=cluster_ev[m_cl],
                            s=args.point_size, alpha=0.85, cmap="tab20"
                        )
                    ax.set_title(f"{row_titles[3]} — {col_titles[c]}")
                    ax.set_xlabel("X"); ax.set_ylabel("Y")
                else:
                    ax.set_title(f"{row_titles[3]} — {col_titles[c]} (no clusters)")
                    ax.set_xlabel("X"); ax.set_ylabel("Y")

            n_true_nu = int((y_true_ev == 0).sum())
            n_true_cos = int((y_true_ev == 1).sum())
            n_pred_nu = int((y_pred_ev == 0).sum())
            n_pred_cos = int((y_pred_ev == 1).sum())
            fig.suptitle(
                f"Event {seen} | truth ν={n_true_nu}, cosmic={n_true_cos} | "
                f"pred ν={n_pred_nu}, cosmic={n_pred_cos}",
                fontsize=11
            )

            outfile = outdir / f"{args.split}_event_byplane_{seen:06d}.png"

            fig.savefig(outfile)
            plt.close(fig)

            if saved == 0:
                log(f"[info] Saved first image to: {outfile}")
            saved += 1
            seen += 1

        if args.device == "cuda":
            torch.cuda.empty_cache()

    log(f"[done] Saved {saved} events to {outdir}/")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        try:
            Path("event_viz_by_plane").mkdir(parents=True, exist_ok=True)
            Path("event_viz_by_plane/_FATAL.txt").write_text("".join(traceback.format_exc()))
        except Exception:
            pass
        sys.exit(1)
