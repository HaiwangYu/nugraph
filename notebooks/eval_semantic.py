#!/usr/bin/env python
# notebooks/eval_semantic.py
import os
import argparse
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import (
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
    p.add_argument("--split", default="val", choices=["val", "test"])
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
    p.add_argument("--model", default="nugraph3", choices=["nugraph3", "nugraph4"])

    # === EDGE EVAL OPTIONS ===================================================
    p.add_argument(
        "--eval-edges",
        action="store_true",
        help="If set (and model=nugraph4), also evaluate edge head using pid/y_instance.",
    )
    p.add_argument(
        "--edge-thr",
        type=float,
        default=0.5,
        help="Threshold on edge prob for binary edge predictions (same-instance=1).",
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


def make_datamodule(
    ng,
    data_path,
    model_cls,
    batch_size=None,
    num_workers=None,
    min_nu_hits=None,
    nu_cut_plane="any",
    nu_class_index=0,
):
    Data = ng.data.NuGraphDataModule
    dm = Data(
        model=model_cls,
        data_path=data_path,
        # neutrino-hit cut args that your DataModule understands
        min_nu_hits=min_nu_hits,
        # nu_cut_plane=nu_cut_plane,
        # nu_hit_class_index=nu_class_index,
    )

    # Optional overrides
    if batch_size is not None and hasattr(dm, "batch_size"):
        dm.batch_size = batch_size
    if num_workers is not None:
        for attr in ("num_workers", "num_workers_train", "num_workers_eval"):
            if hasattr(dm, attr):
                setattr(dm, attr, num_workers)

    dm.setup("test")

    # Small heads-up on what we’re actually evaluating on
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
            "You likely used a different transform; re-create the same DM as training."
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
    """
    from tqdm import tqdm

    nugraph.eval().to(device)
    y_true, y_pred, y_score = [], [], []

    print("Collecting predictions from the model (semantic)...")
    for i, b in enumerate(tqdm(loader)):
        if limit is not None and i >= limit:
            break
        b = b.to(device)

        require_hit_x_or_die(b, expected_width=expected_width)

        _loss, _ = nugraph(b, stage="test")
        p = b["hit"].x_semantic.detach()  # [N, C], probs
        y = b["hit"].y_semantic.detach()

        mask = y >= 0
        if mask.sum() == 0:
            continue

        y = y[mask].long().cpu().numpy()
        p = p[mask].cpu().numpy()

        y_true.append(y)
        y_pred.append(p.argmax(1))
        y_score.append(p[:, 0])  # score for class index 0 ('nu')

    if not y_true:
        raise RuntimeError(
            "No labeled hits found in evaluated batches; check your split/limit."
        )

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_score = np.concatenate(y_score)
    return y_true, y_pred, y_score


# =============================================================================
# EDGE EVAL HELPERS
# =============================================================================
def _get_edge_truth_and_scores_from_batch(batch):
    """
    After NuGraph4.forward(stage='test'):

      - batch['hit'].edge_index   : [2, E]
      - batch['hit'].edge_logits  : [E] or [E,1]
      - batch['hit'].pid or y_instance : [N] (instance IDs, -1 = ignore)

    Build:
      y_edge_true  : np array of {0,1} (same-instance edge = 1)
      y_edge_score : np array of predicted prob(edge is same-instance)
    """
    if "hit" not in batch.node_types:
        return None, None

    h = batch["hit"]

    if not hasattr(h, "edge_index") or not hasattr(h, "edge_logits"):
        # Edge head might be disabled or lambda_edge=0.0
        return None, None

    edge_index = h.edge_index
    edge_logits = h.edge_logits

    # Handle [E,1] vs [E]
    if edge_logits.dim() > 1:
        edge_logits = edge_logits.squeeze(-1)

    # Instance IDs: pid preferred, y_instance as fallback
    y_inst = getattr(h, "pid", None)
    if y_inst is None:
        y_inst = getattr(h, "y_instance", None)
    if y_inst is None:
        raise RuntimeError(
            "[edge-eval] Neither 'pid' nor 'y_instance' found on batch['hit']; "
            "cannot construct edge ground truth."
        )

    src, dst = edge_index
    y_src = y_inst[src]
    y_dst = y_inst[dst]

    valid = (y_src >= 0) & (y_dst >= 0)
    if valid.sum() == 0:
        return None, None

    y_src = y_src[valid]
    y_dst = y_dst[valid]
    logits = edge_logits[valid]

    y_edge_true = (y_src == y_dst).long()  # 1 if same instance, else 0
    y_edge_score = torch.sigmoid(logits)

    return y_edge_true.cpu().numpy(), y_edge_score.cpu().numpy()


@torch.no_grad()
def collect_edge_split(nugraph, loader, device, limit=None, max_edges=None):
    """
    Aggregate edge labels + scores over the split.

    Returns:
      y_edge_true:  np array of {0,1} (0 = different instance, 1 = same instance)
      y_edge_score: np array of predicted prob(edge is same-instance)
    """
    from tqdm import tqdm

    nugraph.eval().to(device)
    y_true_all = []
    y_score_all = []

    total_edges = 0

    print("Collecting predictions from the model (edges)...")
    for i, b in enumerate(tqdm(loader)):
        if limit is not None and i >= limit:
            break

        b = b.to(device)
        # This call populates batch['hit'].edge_index / edge_logits via NuGraph4.forward
        _loss, _ = nugraph(b, stage="test")

        y_edge_true, y_edge_score = _get_edge_truth_and_scores_from_batch(b)
        if y_edge_true is None or y_edge_score is None:
            continue

        y_true_all.append(y_edge_true)
        y_score_all.append(y_edge_score)

        total_edges += int(y_edge_true.shape[0])
        if (max_edges is not None) and (total_edges >= max_edges):
            # We still finish this batch, but will downsample globally later
            break

    if not y_true_all:
        raise RuntimeError("No valid edges collected for edge evaluation.")

    y_edge_true = np.concatenate(y_true_all)
    y_edge_score = np.concatenate(y_score_all)

    # If we hit max_edges, randomly downsample so metrics are unbiased-ish
    if (max_edges is not None) and (y_edge_true.shape[0] > max_edges):
        idx = np.random.choice(y_edge_true.shape[0], size=max_edges, replace=False)
        y_edge_true = y_edge_true[idx]
        y_edge_score = y_edge_score[idx]

    return y_edge_true, y_edge_score


def report_edge_metrics(y_true, y_score, edge_thr, split_name):
    """
    Print ROC-AUC, PR-AUC, and thresholded confusion for the edge head.
    """
    print(f"\n[{split_name}] EDGE METRICS (same-instance edge = positive)")
    pos_frac = float((y_true == 1).mean())
    print(f"Total edges: {len(y_true)}  |  positive fraction: {pos_frac:.4f}")

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    try:
        ap = average_precision_score(y_true, y_score)
    except ValueError:
        ap = float("nan")

    print(f"ROC-AUC  (edge): {auc:.4f}")
    print(f"PR-AUC   (edge): {ap:.4f}")

    # Thresholded summary
    y_pred = (y_score >= edge_thr).astype(int)
    print(f"\n[EDGE] Thresholded at p_same >= {edge_thr:.3f}")
    print("Confusion matrix (rows=true [0: diff, 1: same], cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print("\nClassification report (edges):")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["diff", "same"],
            digits=4,
        )
    )


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
        float(y_score.mean()),
        float(y_score.std()),
        float(y_score.min()),
        float(y_score.max()),
    )


def report_thresholded(y_true, y_score, nu_thr, header):
    """
    Apply threshold on p(nu). If p(nu) >= nu_thr => predict nu(0), else cosmic(1).
    """
    y_bin = (y_true == 0).astype(int)  # 1 means "nu-positive"
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
    """
    Compute PR curve treating 'nu' as positive, return threshold with best F1.
    """
    y_bin = (y_true == 0).astype(int)
    prec, rec, thr = precision_recall_curve(y_bin, y_score)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    idx = np.nanargmax(f1[:-1])  # exclude last sentinel
    best_thr = thr[idx] if idx < len(thr) else 0.5
    return best_thr, float(prec[idx]), float(rec[idx])


def pick_best_fbeta_threshold(y_true, y_score, beta=0.5):
    """
    Compute PR curve and return threshold that maximizes Fβ.
    β < 1 favors precision; β > 1 favors recall.
    """
    y_bin = (y_true == 0).astype(int)
    prec, rec, thr = precision_recall_curve(y_bin, y_score)
    beta2 = beta * beta
    fbeta = (1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-12)
    idx = np.nanargmax(fbeta[:-1])  # exclude last sentinel
    best_thr = thr[idx] if idx < len(thr) else 0.5
    return best_thr, float(prec[idx]), float(rec[idx])


def plot_pr_curve(y_true, y_score, filename, beta=None):
    """
    Plot the precision-recall curve for the 'nu' class.
    Marks best-F1, and best-Fβ if beta is provided.
    """
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

    # Best F1
    f1_thr, p1, r1 = pick_best_f1_threshold(y_true, y_score)
    plt.plot(
        r1,
        p1,
        "ro",
        markersize=7,
        label=f"Best F1 (thr={f1_thr:.3f})\nP={p1:.2f}, R={r1:.2f}",
    )

    # Best Fβ (optional)
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


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()

    # --- import your project exactly as in training ---
    import nugraph as ng

    if args.model == "nugraph4":
        Model = ng.models.NuGraph4
    else:
        Model = ng.models.NuGraph3

    # DataModule exactly as training, now with neutrino-hit cuts forwarded
    dm = make_datamodule(
        ng,
        data_path=args.data_path,
        model_cls=Model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_nu_hits=args.min_nu_hits,
        # nu_cut_plane=args.nu_cut_plane,
        nu_class_index=args.nu_class_index,
    )

    # Load model
    nugraph = Model.load_from_checkpoint(args.ckpt, map_location="cpu")
    print("Loaded checkpoint:", args.ckpt)

    # Expected input feature width
    expected_in_features = getattr(
        getattr(nugraph, "hparams", None),
        "in_features",
        None,
    )
    if expected_in_features is None:
        expected_in_features = 8  # fallback used in training

    loader = dm.val_dataloader() if args.split == "val" else dm.test_dataloader()

    # -------------------------------------------------------------------------
    # 1) SEMANTIC EVAL
    # -------------------------------------------------------------------------
    y_true, y_pred, y_score = collect_split(
        nugraph,
        loader,
        args.device,
        args.limit,
        expected_width=expected_in_features,
    )

    # Argmax baseline
    report_argmax(y_true, y_pred, y_score, args.split.upper())

    # Thresholded evaluation
    chosen_thr = None
    if args.nu_thr is not None:
        chosen_thr = args.nu_thr
        header = f"[{args.split.upper()}] THRESHOLDED (user)"
        report_thresholded(y_true, y_score, chosen_thr, header=header)
    else:
        if args.beta is not None:
            chosen_thr, p, r = pick_best_fbeta_threshold(
                y_true,
                y_score,
                beta=args.beta,
            )
            print(
                f"\nBest-F{args.beta:.2f} ν-threshold found: {chosen_thr:.3f} "
                f"(precision={p:.3f}, recall={r:.3f})"
            )
            header = f"[{args.split.upper()}] THRESHOLDED (best-F{args.beta:.2f})"
        else:
            chosen_thr, p, r = pick_best_f1_threshold(y_true, y_score)
            print(
                f"\nBest-F1 ν-threshold found: {chosen_thr:.3f} "
                f"(precision={p:.3f}, recall={r:.3f})"
            )
            header = f"[{args.split.upper()}] THRESHOLDED (best-F1)"

        report_thresholded(y_true, y_score, chosen_thr, header=header)

    # Plot PR curve
    plot_filename = os.path.basename(args.ckpt).replace(".ckpt", "_pr_curve.png")
    plot_pr_curve(y_true, y_score, plot_filename, beta=args.beta)

    # -------------------------------------------------------------------------
    # 2) EDGE EVAL (NuGraph4, optional)
    # -------------------------------------------------------------------------
    if args.eval_edges and args.model == "nugraph4":
        print("\n[EDGE] Starting edge evaluation using pid/y_instance labels...")
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
            args.split.upper(),
        )
    elif args.eval_edges:
        print(
            "\n[EDGE] --eval-edges was set but model!=nugraph4; "
            "skipping edge eval."
        )


if __name__ == "__main__":
    main()
