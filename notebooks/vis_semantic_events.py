#!/usr/bin/env python
import os
import sys
import argparse
from pathlib import Path
import traceback

# Always use a non-GUI backend (important on login/compute nodes)
import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision("high")


def log(msg: str):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Event-by-event truth vs predicted semantic visualization (NuGraph3)"
    )
    p.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint (.ckpt)")
    p.add_argument("--data-path", required=True, help="HDF5 used in training")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--limit-events", type=int, default=20)
    p.add_argument("--skip-events", type=int, default=0)

    # Threshold selection
    p.add_argument("--nu-thr", type=float, default=None,
                   help="Decision threshold for p(nu). If unset, and --beta is given, "
                        "the script will find best-Fβ on the chosen split; else fallback 0.5.")
    p.add_argument("--beta", type=float, default=None,
                   help="If set (and --nu-thr not set), choose threshold that maximizes Fβ.")

    # Coordinates
    p.add_argument("--x-col", type=int, default=None,
                   help="Column index in hit.x to use for X if no pos/xy is available")
    p.add_argument("--y-col", type=int, default=None,
                   help="Column index in hit.x to use for Y if no pos/xy is available")

    # Plotting
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--outfile-dir", type=str, default="event_viz")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--debug", action="store_true", help="Print per-batch hit attributes")

    return p.parse_args()


def pick_best_fbeta_threshold(y_true, y_score, beta=0.5):
    from sklearn.metrics import precision_recall_curve
    y_bin = (y_true == 0).astype(int)
    prec, rec, thr = precision_recall_curve(y_bin, y_score)
    beta2 = beta * beta
    fbeta = (1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-12)
    idx = np.nanargmax(fbeta[:-1])  # exclude sentinel
    best_thr = thr[idx] if idx < len(thr) else 0.5
    return float(best_thr)


def make_datamodule(ng, data_path, model_cls):
    Data = ng.data.NuGraphDataModule
    dm = Data(model=model_cls, data_path=data_path)
    dm.setup("test")  # this sets up splits; we’ll pick later
    return dm


def get_hit_xy(hit_store, x_col=None, y_col=None):
    # Try pos
    if hasattr(hit_store, "pos") and hit_store.pos is not None:
        pos = hit_store.pos
        if pos.dim() == 2 and pos.size(-1) >= 2:
            xy = pos[:, :2]
            return xy[:, 0].detach().cpu().numpy(), xy[:, 1].detach().cpu().numpy()
    # Try xy
    if hasattr(hit_store, "xy") and hit_store.xy is not None:
        xy = hit_store.xy
        if xy.dim() == 2 and xy.size(-1) >= 2:
            return xy[:, 0].detach().cpu().numpy(), xy[:, 1].detach().cpu().numpy()
    # Fallback to features
    if not hasattr(hit_store, "x") or hit_store.x is None:
        raise RuntimeError("No hit.pos/xy and hit.x missing—need --x-col/--y-col.")
    if x_col is None or y_col is None:
        raise RuntimeError("No hit.pos/xy—please pass --x-col and --y-col.")
    x = hit_store.x[:, x_col]
    y = hit_store.x[:, y_col]
    return x.detach().cpu().numpy(), y.detach().cpu().numpy()


@torch.no_grad()
def collect_split_scores(nugraph, loader, device, expected_in_features=None, debug=False):
    y_true, y_score = [], []
    from tqdm import tqdm
    for b in tqdm(loader, desc="Collecting scores (for threshold)"):
        b = b.to(device)
        if expected_in_features is not None:
            if not hasattr(b["hit"], "x") or b["hit"].x.size(-1) != expected_in_features:
                raise RuntimeError(
                    f"hit.x width {b['hit'].x.size(-1) if hasattr(b['hit'],'x') else 'N/A'} "
                    f"!= expected {expected_in_features}"
                )
        _loss, _metrics = nugraph(b, stage="test")
        p = b["hit"].x_semantic.detach()      # [N, 2] probs
        y = b["hit"].y_semantic.detach()
        m = y >= 0
        if m.sum() == 0:
            continue
        y_true.append(y[m].long().cpu().numpy())
        y_score.append(p[m][:, 0].cpu().numpy())
        if debug:
            log(f"[debug] batch hits={b['hit'].num_nodes}, labeled={int(m.sum())}")
    if not y_true:
        raise RuntimeError("No labeled hits found while collecting scores.")
    return np.concatenate(y_true), np.concatenate(y_score)


@torch.no_grad()
def main():
    args = parse_args()

    # Create outdir immediately + sentinel to prove we started
    outdir = Path(args.outfile_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "_STARTED.txt").write_text("script started\n")
    log(f"[info] Outdir: {outdir.resolve()}")

    log("[info] Importing nugraph and loading model/datamodule...")
    import nugraph as ng
    Model = ng.models.NuGraph3

    dm = make_datamodule(ng, args.data_path, Model)
    loader = dm.val_dataloader() if args.split == "val" else dm.test_dataloader()

    nugraph = Model.load_from_checkpoint(args.ckpt, map_location="cpu")
    nugraph.eval().to(args.device)
    log(f"[info] Loaded ckpt: {args.ckpt}")
    log(f"[info] Using device: {args.device}")

    expected_in_features = getattr(getattr(nugraph, "hparams", None), "in_features", 8)

    # Threshold
    if args.nu_thr is not None:
        nu_thr = float(args.nu_thr)
        log(f"[info] Using user ν-threshold: {nu_thr:.3f}")
    else:
        if args.beta is not None:
            log(f"[info] Computing best-F{args.beta:.2f} threshold on {args.split}...")
            y_true_all, y_score_all = collect_split_scores(
                nugraph, loader, args.device, expected_in_features=expected_in_features, debug=args.debug
            )
            nu_thr = pick_best_fbeta_threshold(y_true_all, y_score_all, beta=args.beta)
            log(f"[info] Best-F{args.beta:.2f} ν-threshold: {nu_thr:.3f}")
        else:
            nu_thr = 0.5
            log("[warn] No --nu-thr/--beta; defaulting to 0.5")

    # Visualize
    from tqdm import tqdm
    saved = 0
    seen = 0
    log(f"[info] Beginning visualization on split={args.split} (limit={args.limit_events}, skip={args.skip_events})")

    for batch in tqdm(loader, desc=f"Visualizing {args.split} events"):
        batch = batch.to(args.device)

        # Quick debug on first batch
        if args.debug and saved == 0:
            keys = list(batch["hit"].keys())
            log(f"[debug] hit keys: {keys}")
            log(f"[debug] has pos? {hasattr(batch['hit'],'pos')}  has xy? {hasattr(batch['hit'],'xy')}")
            if hasattr(batch['hit'], 'x'):
                log(f"[debug] hit.x shape: {tuple(batch['hit'].x.shape)}")

        _loss, _metrics = nugraph(batch, stage="test")
        probs = batch["hit"].x_semantic.detach()  # [N, 2]
        labels = batch["hit"].y_semantic.detach()

        # Coordinates for the whole batch
        try:
            Xall, Yall = get_hit_xy(batch["hit"], x_col=args.x_col, y_col=args.y_col)
        except Exception as e:
            (outdir / "_ERROR.txt").write_text(f"Coordinate extraction failed:\n{e}\n")
            raise

        if not hasattr(batch["hit"], "ptr") or batch["hit"].ptr is None:
            raise RuntimeError("batch['hit'].ptr missing, cannot slice per event.")
        ptr = batch["hit"].ptr
        num_graphs = ptr.numel() - 1

        for g in range(num_graphs):
            if seen < args.skip_events:
                seen += 1
                continue
            if args.limit_events is not None and saved >= args.limit_events:
                log(f"[done] Saved {saved} events to {outdir}/")
                return

            start, end = ptr[g].item(), ptr[g + 1].item()
            idx = slice(start, end)

            x = Xall[idx]
            y = Yall[idx]
            y_true = labels[idx].detach().long().cpu().numpy()
            p_nu = probs[idx][:, 0].detach().cpu().numpy()
            y_pred = np.where(p_nu >= nu_thr, 0, 1)  # 0=nu, 1=cosmic

            # Only labeled for truth panel
            m = y_true >= 0
            if m.sum() == 0:
                seen += 1
                continue

            fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=args.dpi, constrained_layout=True)
            axL, axR = axes

            # Truth
            axL.scatter(x[m][y_true[m] == 0], y[m][y_true[m] == 0], s=args.point_size, alpha=0.85, label="ν (truth)")
            axL.scatter(x[m][y_true[m] == 1], y[m][y_true[m] == 1], s=args.point_size, alpha=0.85, label="cosmic (truth)")
            axL.set_title("Truth")
            axL.set_xlabel("X"); axL.set_ylabel("Y")
            axL.legend(markerscale=3, loc="best")

            # Prediction
            axR.scatter(x[y_pred == 0], y[y_pred == 0], s=args.point_size, alpha=0.85, label=f"ν (pred, thr={nu_thr:.3f})")
            axR.scatter(x[y_pred == 1], y[y_pred == 1], s=args.point_size, alpha=0.85, label="cosmic (pred)")
            axR.set_title("Prediction")
            axR.set_xlabel("X"); axR.set_ylabel("Y")
            axR.legend(markerscale=3, loc="best")

            n_nu_true = int((y_true == 0).sum())
            n_cos_true = int((y_true == 1).sum())
            n_nu_pred = int((y_pred == 0).sum())
            n_cos_pred = int((y_pred == 1).sum())
            fig.suptitle(
                f"Event {seen} | truth ν={n_nu_true}, cosmic={n_cos_true} | pred ν={n_nu_pred}, cosmic={n_cos_pred}",
                fontsize=10
            )

            outfile = outdir / f"{args.split}_event_{seen:06d}.png"
            fig.savefig(outfile)
            plt.close(fig)
            if saved == 0:
                log(f"[info] Saved first image to: {outfile}")
            saved += 1
            seen += 1

    log(f"[done] Saved {saved} events to {outdir}/")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Always show any error and write it to outdir if possible
        traceback.print_exc()
        try:
            # best effort: write a copy to the most likely outdir name
            Path("event_viz").mkdir(parents=True, exist_ok=True)
            Path("event_viz/_FATAL.txt").write_text("".join(traceback.format_exc()))
        except Exception:
            pass
        sys.exit(1)
