#!/usr/bin/env python
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
                   help="Decision threshold for p(nu). If set, we use this value directly.")
    p.add_argument("--beta", type=float, default=None,
                   help="If provided, choose threshold that maximizes Fβ on the chosen split "
                        "(β<1 favors precision; β>1 favors recall). If omitted, uses best-F1.")

    # NEW: neutrino-hit cut controls (must match your DataModule)
    p.add_argument("--min-nu-hits", type=int, default=None,
                   help="If set, only evaluate events with at least this many neutrino hits "
                        "(as defined by nu_cut_plane/nu_class_index).")
    p.add_argument("--nu-cut-plane", type=str, default="any",
                   choices=["any", "sum", "u", "v", "y"],
                   help="How to count neutrino hits for the cut: "
                        "'any' = ≥min on any plane; 'sum' = sum(U+V+Y) ≥min; "
                        "or restrict to a specific plane.")
    p.add_argument("--nu-class-index", type=int, default=0,
                   help="Index of the 'nu' class inside semantic_classes (default 0).")
    return p.parse_args()


def make_datamodule(ng, data_path, model_cls,
                    batch_size=None, num_workers=None,
                    min_nu_hits=None, nu_cut_plane="any", nu_class_index=0):
    """
    Recreate exactly what you did in training:
        Data = ng.data.NuGraphDataModule
        Model = ng.models.NuGraph3
        nudata = Data(model=Model, data_path=..., min_nu_hits=..., nu_cut_plane=..., nu_hit_class_index=...)
    """
    Data = ng.data.NuGraphDataModule
    dm = Data(
        model=model_cls,
        data_path=data_path,
        # forward neutrino-hit cut args if present in your DataModule
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
        print(f"[Info] Eval datasets after cut (min_nu_hits={min_nu_hits}, nu_cut_plane='{nu_cut_plane}', nu_class_index={nu_class_index}):")
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

    from tqdm import tqdm
    print("Collecting predictions from the model...")
    for i, b in enumerate(tqdm(loader)):
        if limit is not None and i >= limit:
            break
        b = b.to(device)

        require_hit_x_or_die(b, expected_width=expected_width)

        _loss, _ = nugraph(b, stage="test")
        p = b["hit"].x_semantic.detach()      # [N, C], probs
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
    y_bin = (y_true == 0).astype(int)  # 1 means "nu-positive"
    y_pred_thr_bin = (y_score >= nu_thr).astype(int)
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
    plt.plot(recall, precision, marker='.', markersize=3)
    plt.title("Precision-Recall Curve for Neutrino Class")
    plt.xlabel("Neutrino Recall (Efficiency)")
    plt.ylabel("Neutrino Precision (Purity)")
    plt.grid(True)
    plt.xlim([0, 1.02])
    plt.ylim([0, 1.02])

    # Best F1
    f1_thr, p1, r1 = pick_best_f1_threshold(y_true, y_score)
    plt.plot(r1, p1, 'ro', markersize=7, label=f'Best F1 (thr={f1_thr:.3f})\nP={p1:.2f}, R={r1:.2f}')

    # Best Fβ (optional)
    if beta is not None:
        fbeta_thr, pb, rb = pick_best_fbeta_threshold(y_true, y_score, beta=beta)
        plt.plot(rb, pb, 'gs', markersize=7, label=f'Best F{beta:.2f} (thr={fbeta_thr:.3f})\nP={pb:.2f}, R={rb:.2f}')

    plt.legend()
    plt.savefig(filename)
    print(f"--> Saved plot to {filename}")


def main():
    args = parse_args()

    # --- import your project exactly as in training ---
    import nugraph as ng
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
    chosen_thr = None
    if args.nu_thr is not None:
        chosen_thr = args.nu_thr
        header = f"[{args.split.upper()}] THRESHOLDED (user)"
        report_thresholded(y_true, y_score, chosen_thr, header=header)
    else:
        if args.beta is not None:
            chosen_thr, p, r = pick_best_fbeta_threshold(y_true, y_score, beta=args.beta)
            print(f"\nBest-F{args.beta:.2f} ν-threshold found: {chosen_thr:.3f} "
                  f"(precision={p:.3f}, recall={r:.3f})")
            header = f"[{args.split.upper()}] THRESHOLDED (best-F{args.beta:.2f})"
        else:
            chosen_thr, p, r = pick_best_f1_threshold(y_true, y_score)
            print(f"\nBest-F1 ν-threshold found: {chosen_thr:.3f} "
                  f"(precision={p:.3f}, recall={r:.3f})")
            header = f"[{args.split.upper()}] THRESHOLDED (best-F1)"

        report_thresholded(y_true, y_score, chosen_thr, header=header)

    # 3) Plot and save the PR curve (marks best-F1; also best-Fβ if provided)
    plot_filename = os.path.basename(args.ckpt).replace(".ckpt", "_pr_curve.png")
    plot_pr_curve(y_true, y_score, plot_filename, beta=args.beta)


if __name__ == "__main__":
    main()
