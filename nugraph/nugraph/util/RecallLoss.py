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
        # Always register the buffer so `self.cw` exists in all instances / state_dicts.
        # register_buffer accepts None, which keeps the attribute but no tensor is stored.
        cw_tensor = torch.tensor(class_weight, dtype=torch.float) if class_weight is not None else None
        self.register_buffer("cw", cw_tensor)
 
         self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # logits -> per-sample CE
        ce = F.cross_entropy(input, target, reduction='none',
                             ignore_index=self.ignore_index, weight=self.cw, label_smoothing=self.label_smoothing)
    
        # dynamic recall weights (unchanged)
        dyn_w = 1 - recall(input, target, 'multiclass',
                           num_classes=input.size(1), average='none',
                           ignore_index=self.ignore_index)
        dyn = dyn_w[target.clamp_min(0)]
    
        # focal modulation
        with torch.no_grad():
            pt = F.softmax(input, dim=1).gather(1, target.clamp_min(0).unsqueeze(1)).squeeze(1)
            pt[target < 0] = 1.0
        gamma = 2.0 #was 2.0
        focal = (1 - pt).pow(gamma)
    
        loss = ce * dyn * focal
        return loss.mean()