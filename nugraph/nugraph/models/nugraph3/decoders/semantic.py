# models/nugraph3/decoders/semantic.py
"""NuGraph3 semantic decoder"""

from typing import Any, List
import torch
from torch import nn
import torchmetrics as tm
from torch_geometric.data import Batch
from pytorch_lightning.loggers import Logger, WandbLogger

import matplotlib.pyplot as plt

from ....util import ConfusionMatrixLogger, RecallLoss
from ..types import Data


class SemanticDecoder(nn.Module):
    """
    Convolve planar hit embeddings down to class logits for each semantic class.
    """

    def __init__(self,
                 hit_features: int,
                 semantic_classes: List[str],
                 class_weight: list[float] | None = None):
        super().__init__()

        # ---- Loss (handles imbalance; keeps dynamic recall & focal modulation via RecallLoss) ----
        # For classes ['nu','cosmic'] use [w_nu, w_cosmic]
        self.loss = RecallLoss(class_weight=class_weight, label_smoothing=0.05)

        # Temperature parameter (as in the original)
        self.temp = nn.Parameter(torch.tensor(0.0))

        # ---- Metrics ----
        metric_args = {
            "task": "multiclass",
            "num_classes": len(semantic_classes),
            "ignore_index": -1,
        }
        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)

        # Helpful aggregate metrics to spot class-imbalance progress
        self.recall_macro = tm.Recall(average="macro", **metric_args)
        self.recall_weighted = tm.Recall(average="weighted", **metric_args)
        self.precision_macro = tm.Precision(average="macro", **metric_args)
        self.precision_weighted = tm.Precision(average="weighted", **metric_args)
        self.f1_macro = tm.F1Score(average="macro", **metric_args)

        # Confusion-matrix accumulators (normalized variants for readable heatmaps)
        self.cm_logger = ConfusionMatrixLogger(semantic_classes)  # kept for compatibility
        self.cm_recall = tm.ConfusionMatrix(normalize="true", **metric_args)
        self.cm_precision = tm.ConfusionMatrix(normalize="pred", **metric_args)

        # ---- Network ----
        # A tiny MLP head (LayerNorm before ReLU is slightly stabler; either order is fine)
        # self.net = nn.Sequential(
        #     nn.Linear(hit_features, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, len(semantic_classes)),
        # )
        self.net = nn.Sequential(
            nn.Linear(hit_features, 256),  # Wider first layer
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(p=0.2),             # Slightly more dropout
            nn.Linear(256, 128),           # Add another layer
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(p=0.2),
            nn.Linear(128, len(semantic_classes)),
        )

        self.classes = semantic_classes

    def forward(self, data: Data, stage: str | None = None) -> dict[str, Any]:
        """
        Args:
            data: graph batch
            stage: "train"|"val"|"test"|None (controls logging)
        """
        # ---- Logits ----
        logits = self.net(data["hit"].x)          # [N, C] logits
        data["hit"].x_semantic_logits = logits    # keep logits for anyone who needs them
        data["hit"].x_semantic = logits           # keep backward-compat for now

        # bookkeeping for batched heterogeneous graphs
        if isinstance(data, Batch):
            # pylint: disable=protected-access
            data._slice_dict["hit"]["x_semantic"] = data["hit"].ptr
            inc = torch.zeros(data.num_graphs, device=data["hit"].x.device)
            data._inc_dict["hit"]["x_semantic"] = inc

        # ---- Loss ----
        x = data["hit"].x_semantic  # logits [N, C]
        y = data["hit"].y_semantic.long()  # labels [N], with ghosts = -1

        # IMPORTANT: ensure ghosts NEVER contribute to the loss, even if RecallLoss
        # does not implement ignore_index internally.
        valid = (y != -1)
        if valid.any():
            x_valid = x[valid]
            y_valid = y[valid]
            w = 2 * (-1.0 * self.temp).exp()
            loss = w * self.loss(x_valid, y_valid) + self.temp
        else:
            # all-ghost batch; return zero loss (keeps graph valid)
            loss = x.sum() * 0.0


        # ---- Metrics ----
        metrics: dict[str, Any] = {}
        if stage:
            metrics[f"semantic/loss-{stage}"] = loss
            metrics[f"semantic/recall-{stage}"] = self.recall(x, y)
            metrics[f"semantic/precision-{stage}"] = self.precision(x, y)
            metrics[f"semantic/recall-macro-{stage}"] = self.recall_macro(x, y)
            metrics[f"semantic/recall-weighted-{stage}"] = self.recall_weighted(x, y)
            metrics[f"semantic/precision-macro-{stage}"] = self.precision_macro(x, y)
            metrics[f"semantic/precision-weighted-{stage}"] = self.precision_weighted(x, y)
            metrics[f"semantic/f1-macro-{stage}"] = self.f1_macro(x, y)

        if stage == "train":
            metrics["temperature/semantic"] = self.temp
        if stage in ("val", "test"):
            # accumulate confusion matrices for end-of-epoch logging
            self.cm_recall.update(x, y)
            self.cm_precision.update(x, y)

        # ---- Convert logits to probabilities for downstream use ----
        data["hit"].x_semantic = logits.softmax(dim=1)  # [N, C] probabilities

        return loss, metrics

    # ---------- Confusion matrix image logging to W&B ----------
    @staticmethod
    def _plot_cm(cm: torch.Tensor, title: str, class_names: List[str]):
        cm_np = cm.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        im = ax.imshow(cm_np, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        # annotate cells
        for i in range(cm_np.shape[0]):
            for j in range(cm_np.shape[1]):
                ax.text(j, i, f"{cm_np[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    def _log_confusions(self, logger: Logger | list[Logger], stage: str, epoch: int):
        # compute frozen tensors
        cm_rec = self.cm_recall.compute()
        cm_pre = self.cm_precision.compute()

        fig1 = self._plot_cm(cm_rec, f"Recall-normalized ({stage})", self.classes)
        fig2 = self._plot_cm(cm_pre, f"Precision-normalized ({stage})", self.classes)

        # support single or multiple loggers
        loggers = logger if isinstance(logger, list) else [logger]
        for lg in loggers:
            if isinstance(lg, WandbLogger):
                import wandb
                lg.experiment.log({
                    f"semantic/confusion_recall_{stage}": wandb.Image(fig1),
                    f"semantic/confusion_precision_{stage}": wandb.Image(fig2),
                    "epoch": epoch,
                })

        plt.close(fig1); plt.close(fig2)

        # IMPORTANT: reset accumulators for next epoch
        self.cm_recall.reset()
        self.cm_precision.reset()

    def on_epoch_end(self, logger: Logger | list[Logger], stage: str, epoch: int) -> None:
        """
        Called by the Lightning module after each stage's epoch.
        """
        # keep your existing PNG logging (if ConfusionMatrixLogger writes artifacts),
        # and ALSO push nice images directly to W&B:
        self._log_confusions(logger, stage, epoch)