#!/usr/bin/env python
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
        description="Event-by-event per-plane Truth vs Prediction (NuGraph3)"
    )
    # Data/model
    p.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint (.ckpt)")
    p.add_argument("--data-path", required=True, help="HDF5 used in training")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # NEW: dataloader overrides for memory control
    p.add_argument("--batch-size", type=int, default=1,
                   help="Eval batch size (default 1 to avoid OOM).")
    p.add_argument("--num-workers", type=int, default=0,
                   help="Eval dataloader workers (default 0).")

    # NEW: memory/precision knobs
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none",
                   help="Autocast precision on GPU. Use 'bf16' on A100/H100 or 'fp16' if needed.")
    p.add_argument("--disable-checkpointing", action="store_true",
                   help="Bypass model core_net.checkpoint wrapper during inference.")

    # Threshold selection
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
                    f"hit.x width {b['hit'].x.size(-1) if hasattr(b['hit'],'x') else 'N/A'} "
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


def main():
    args = parse_args()

    outdir = Path(args.outfile_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "_STARTED.txt").write_text("script started\n")
    log(f"[info] Outdir: {outdir.resolve()}")

    log("[info] Importing nugraph and loading model/datamodule...")
    import nugraph as ng
    Model = ng.models.NuGraph3

    # Build DataModule and OVERRIDE eval batch size & workers
    Data = ng.data.NuGraphDataModule
    dm = Data(model=Model, data_path=args.data_path)
    dm.setup("test")
    # Override batch size/num_workers consistently
    for attr in ("batch_size", "batch_size_eval"):
        if hasattr(dm, attr):
            setattr(dm, attr, args.batch_size)
    for attr in ("num_workers", "num_workers_eval"):
        if hasattr(dm, attr):
            setattr(dm, attr, args.num_workers)

    loader = dm.val_dataloader() if args.split == "val" else dm.test_dataloader()

    model = Model.load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval().to(args.device)
    expected_in_features = getattr(getattr(model, "hparams", None), "in_features", 8)

    # Optionally bypass checkpointing wrapper during inference
    if args.disable_checkpointing or args.disable_checkpointing:  # tolerate typo
        try:
            if hasattr(model, "core_net") and hasattr(model.core_net, "checkpoint"):
                model.core_net.checkpoint = (lambda f, *a, **k: f(*a, **k))
                log("[info] Disabled core_net.checkpoint wrapper for inference.")
        except Exception:
            pass

    # Decide threshold
    if args.nu_thr is not None:
        nu_thr = float(args.nu_thr)
        log(f"[info] Using user ν-threshold: {nu_thr:.3f}")
    elif args.beta is not None:
        log(f"[info] Computing best-F{args.beta:.2f} threshold on {args.split}...")
        y_true_all, y_score_all = collect_split_scores(
            model, loader, args.device, expected_in_features=expected_in_features, debug=args.debug, amp=args.amp
        )
        nu_thr = pick_best_threshold(y_true_all, y_score_all, beta=args.beta)
        log(f"[info] Best-F{args.beta:.2f} ν-threshold: {nu_thr:.3f}")
    else:
        nu_thr = 0.5
        log("[warn] No --nu-thr/--beta; defaulting to 0.5")

    # Prepare helpers
    from tqdm import tqdm
    rng = np.random.default_rng(123)
    saved = 0
    seen = 0

    # Autocast context for low-memory GPU
    use_autocast = (args.device == "cuda" and args.amp in {"bf16", "fp16"})
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if use_autocast else torch.cuda.amp.autocast(enabled=False)
    )

    log(f"[info] Visualizing split={args.split}  limit={args.limit_events}  skip={args.skip_events}  "
        f"batch_size={args.batch_size}  workers={args.num_workers}  amp={args.amp}")

    for batch in tqdm(loader, desc=f"Visualizing {args.split} events"):
        batch = batch.to(args.device)

        with torch.inference_mode(), autocast_ctx:
            _loss, _metrics = model(batch, stage="test")

        hit = batch["hit"]
        probs = hit.x_semantic.detach().cpu().numpy()         # [N,2]
        labels = hit.y_semantic.detach().cpu().numpy()        # [N]
        p_nu = probs[:, 0]

        # Coordinates
        X, Y = get_hit_xy(hit, x_col=args.x_col, y_col=args.y_col)

        # Plane detection
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

        # Event slicing
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

            # Optional subsample
            if args.max_points is not None and len(x_ev) > args.max_points:
                pick = maybe_subsample(np.arange(len(x_ev)), args.max_points, rng)
                x_ev, y_ev = x_ev[pick], y_ev[pick]
                y_true_ev, y_pred_ev, plane_ev = y_true_ev[pick], y_pred_ev[pick], plane_ev[pick]

            # Build figure: 2 rows (Truth, Pred), 3 cols (U,V,Y)
            fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=args.dpi, constrained_layout=True)
            row_titles = ["Truth", f"Prediction (thr={nu_thr:.3f})"]
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
                # ax.legend(markerscale=3, loc="best")

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

        # free per-batch GPU cache
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
