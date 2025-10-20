# notebooks/eval_semantic.py
import os
import argparse
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision("high")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate NuGraph semantic head")
    p.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint (.ckpt)")
    p.add_argument("--data-path", required=True,
                   help="HDF5 used in training (same file as train.ipynb)")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of batches to evaluate (for a quick pass)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--nu-thr", type=float, default=None,
                   help="Decision threshold for p(nu). If unset, pick best-F1 on validation data.")
    return p.parse_args()


def make_datamodule(ng, data_path, model_cls, batch_size=None, num_workers=None):
    """
    Recreate exactly what you did in train.ipynb:
        Data = ng.data.NuGraphDataModule
        Model = ng.models.NuGraph3
        nudata = Data(model=Model, data_path=...)
    """
    Data = ng.data.NuGraphDataModule
    dm = Data(model=model_cls, data_path=data_path)

    # Optional overrides
    if batch_size is not None and hasattr(dm, "batch_size"):
        dm.batch_size = batch_size
    if num_workers is not None:
        for attr in ("num_workers", "num_workers_train", "num_workers_eval"):
            if hasattr(dm, attr):
                setattr(dm, attr, num_workers)

    dm.setup("test")
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
            f"[error] hit.x has width {batch['hit'].x.size(-1)} but model expects {expected_width}.\n"
            "You likely used a different transform; re-create the same DM as training."
        )


@torch.no_grad()
def collect_split(nugraph, loader, device, limit=None, expected_width=None):
    """
    Returns:
      y_true:    numpy int array with labels {0=nu, 1=cosmic}
      y_pred:    argmax predictions (for reference)
      y_score:   p(nu) probabilities
    """
    nugraph.eval().to(device)
    y_true, y_pred, y_score = [], [], []

    # Use a progress bar for long evaluations
    from tqdm import tqdm
    print("Collecting predictions from the model...")
    for i, b in enumerate(tqdm(loader)):
        if limit is not None and i >= limit:
            break
        b = b.to(device)

        # Make sure features exist before forward
        require_hit_x_or_die(b, expected_width=expected_width)

        # Forward (decoder writes softmax probs to b["hit"].x_semantic)
        _loss, _ = nugraph(b, stage="test")
        p = b["hit"].x_semantic.detach()      # [N, C], probs
        y = b["hit"].y_semantic.detach()

        # Keep only labeled
        mask = y >= 0
        if mask.sum() == 0:
            continue

        y = y[mask].long().cpu().numpy()
        p = p[mask].cpu().numpy()

        y_true.append(y)
        y_pred.append(p.argmax(1))
        y_score.append(p[:, 0])  # score for class index 0 ('nu')

    if not y_true:
        raise RuntimeError("No labeled hits found in evaluated batches; check your split/limit.")

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_score = np.concatenate(y_score)
    return y_true, y_pred, y_score


def report_argmax(y_true, y_pred, y_score, split_name):
    print(f"\n[{split_name}] ARGMAX baseline")
    print("labels:", Counter(y_true), "preds:", Counter(y_pred))
    print("\nConfusion matrix (rows=true [nu, cosmic], cols=pred):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["nu", "cosmic"], digits=4))
    print("\nnu-score stats (mean,std,min,max):",
          float(y_score.mean()), float(y_score.std()),
          float(y_score.min()), float(y_score.max()))


def report_thresholded(y_true, y_score, nu_thr, header):
    """
    Apply threshold on p(nu). If p(nu) >= nu_thr => predict nu(0), else cosmic(1).
    """
    y_bin = (y_true == 0).astype(int)  # 1 means "nu-positive" for PR-style thinking
    y_pred_thr_bin = (y_score >= nu_thr).astype(int)
    # Map back to original label ids: 1->nu(0), 0->cosmic(1)
    y_pred_thr = np.where(y_pred_thr_bin == 1, 0, 1)

    print(f"\n{header}")
    print(f"nu-threshold = {nu_thr:.3f}")
    print("preds:", Counter(y_pred_thr))
    print("\nConfusion matrix @thr (rows=true [nu, cosmic], cols=pred):")
    print(confusion_matrix(y_true, y_pred_thr, labels=[0, 1]))
    print("\nClassification report @thr:")
    print(classification_report(y_true, y_pred_thr, target_names=["nu", "cosmic"], digits=4))


def pick_best_f1_threshold(y_true, y_score):
    """
    Compute PR curve treating 'nu' as positive, return threshold with best F1.
    """
    y_bin = (y_true == 0).astype(int)
    prec, rec, thr = precision_recall_curve(y_bin, y_score)
    # Add a small epsilon to avoid division by zero
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    # Find the index of the maximum F1 score
    idx = np.nanargmax(f1[:-1]) # Exclude the last value which can be problematic
    best_thr = thr[idx] if idx < len(thr) else 0.5
    return best_thr, prec[idx], rec[idx]

def plot_pr_curve(y_true, y_score, filename):
    """
    Calculate and plot the precision-recall curve for the 'nu' class.
    """
    print(f"\nGenerating Precision-Recall curve for '{filename}'...")
    y_bin = (y_true == 0).astype(int)  # Treat 'nu' (class 0) as the positive class
    precision, recall, _ = precision_recall_curve(y_bin, y_score)

    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(recall, precision, marker='.', markersize=3)
    plt.title("Precision-Recall Curve for Neutrino Class")
    plt.xlabel("Neutrino Recall (Efficiency)")
    plt.ylabel("Neutrino Precision (Purity)")
    plt.grid(True)
    plt.xlim([0, 1.02])
    plt.ylim([0, 1.02])
    
    # Find and plot the best F1-score point
    best_thr, p, r = pick_best_f1_threshold(y_true, y_score)
    plt.plot(r, p, 'ro', markersize=8, label=f'Best F1-Score (thr={best_thr:.3f})\nPrecision={p:.2f}, Recall={r:.2f}')
    plt.legend()
    
    plt.savefig(filename)
    print(f"--> Saved plot to {filename}")


def main():
    args = parse_args()

    # --- import your project exactly as in training ---
    import nugraph as ng
    Model = ng.models.NuGraph3

    # DataModule exactly as training
    dm = make_datamodule(
        ng,
        data_path=args.data_path,
        model_cls=Model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Load model
    nugraph = Model.load_from_checkpoint(args.ckpt, map_location="cpu")
    print("Loaded checkpoint:", args.ckpt)

    # Expected input feature width
    expected_in_features = getattr(getattr(nugraph, "hparams", None), "in_features", None)
    if expected_in_features is None:
        expected_in_features = 8  # fallback used in training

    loader = dm.val_dataloader() if args.split == "val" else dm.test_dataloader()

    # Collect predictions/scores
    y_true, y_pred, y_score = collect_split(
        nugraph, loader, args.device, args.limit, expected_width=expected_in_features
    )

    # 1) Argmax baseline
    report_argmax(y_true, y_pred, y_score, args.split.upper())

    # 2) Thresholded evaluation
    if args.nu_thr is not None:
        # User-specified threshold
        report_thresholded(y_true, y_score, args.nu_thr,
                           header=f"[{args.split.upper()}] THRESHOLDED (user)")
    else:
        # Auto-pick best-F1 threshold
        best_thr, p, r = pick_best_f1_threshold(y_true, y_score)
        print(f"\nBest-F1 ν-threshold found: {best_thr:.3f} "
              f"(precision={p:.3f}, recall={r:.3f})")
        report_thresholded(y_true, y_score, best_thr,
                           header=f"[{args.split.upper()}] THRESHOLDED (best-F1)")

    # 3) Plot and save the PR curve
    plot_filename = os.path.basename(args.ckpt).replace(".ckpt", "_pr_curve.png")
    plot_pr_curve(y_true, y_score, plot_filename)


if __name__ == "__main__":
    main()