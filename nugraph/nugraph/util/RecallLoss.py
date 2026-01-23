# # as described in https://arxiv.org/abs/2106.14917

# import torch
# from torch import Tensor
# import torch.nn.functional as F
# from torchmetrics.functional import recall

# class RecallLoss(torch.nn.Module):
#     def __init__(self, ignore_index: int = -1):
#         super().__init__()
#         self.ignore_index = ignore_index

#     def forward(self, input: Tensor, target: Tensor) -> Tensor:
#         weight = 1 - recall(input, target, 'multiclass',
#                             num_classes=input.size(1),
#                             average='none',
#                             ignore_index=self.ignore_index)
#         ce = F.cross_entropy(input, target, reduction='none',
#                              ignore_index=self.ignore_index)
#         loss = weight[target] * ce
#         return loss.mean()
# util/RecallLoss.py
# util/RecallLoss.py
# Recall-weighted cross-entropy with optional class weights.
# Keeps the original "dynamic" weighting (1 - per-class recall),
# and relies on PyTorch CE's built-in class weighting for imbalance.

import torch
from torch import Tensor
import torch.nn.functional as F
from torchmetrics.functional import recall


class RecallLoss(torch.nn.Module):
    def __init__(self, ignore_index: int = -1, class_weight: list[float] | None = None, label_smoothing: float = 0.05):
        """
        Args:
            ignore_index: label index to ignore (e.g., -1 for unlabeled)
            class_weight: list of per-class weights in label-index order
                          (e.g., for ['nu','cosmic'] use [w_nu, w_cosmic])
        """
        super().__init__()
        self.ignore_index = ignore_index
        if class_weight is not None:
            # Stored as a buffer so it moves with .to(device) and is saved in state_dict
            self.register_buffer("cw", torch.tensor(class_weight, dtype=torch.float))
        else:
            self.cw = None  # type: ignore

        self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        input: logits [N, C]
        target: labels [N], with ignore_index (e.g. -1) meaning "unlabeled/ghost"
        """
        target = target.long()
        C = int(input.size(1))

        # If everything is ignored, return a clean 0 loss (keeps graph/batch valid).
        valid = (target != self.ignore_index)
        if not valid.any():
            return input.sum() * 0.0

        # Per-sample CE (ignored entries get 0 by PyTorch)
        ce = F.cross_entropy(
            input,
            target,
            reduction="none",
            ignore_index=self.ignore_index,
            weight=self.cw,
            label_smoothing=self.label_smoothing,
        )

        # ---- Dynamic per-class weighting: (1 - recall_k) ----
        # Compute recall only on valid labels to avoid empty/ignore edge cases.
        # Also nan_to_num to handle "class absent in batch" situations.
        dyn_w = 1.0 - recall(
            input[valid],
            target[valid],
            "multiclass",
            num_classes=C,
            average="none",
            ignore_index=self.ignore_index,
        )
        dyn_w = torch.nan_to_num(dyn_w, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0, 1.0)

        # Map per-class dyn_w -> per-sample dyn (only for valid samples)
        dyn = torch.ones_like(target, dtype=input.dtype, device=input.device)
        dyn[valid] = dyn_w[target[valid]]

        # ---- Focal modulation (only valid labels matter) ----
        with torch.no_grad():
            pt = torch.ones_like(target, dtype=input.dtype, device=input.device)
            probs = F.softmax(input[valid], dim=1)
            pt[valid] = probs.gather(1, target[valid].unsqueeze(1)).squeeze(1)

        gamma = 2.0
        focal = (1.0 - pt).pow(gamma)

        loss = ce * dyn * focal

        # Mean only over valid labels (ignored labels are forced to 0 anyway,
        # but this keeps scaling consistent if the ignored fraction varies).
        return loss[valid].mean()
    
