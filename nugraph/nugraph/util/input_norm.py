"""Input feature normalization module"""
import torch

P = torch.nn.Parameter

class InputNorm(torch.nn.Module):
    """
    PyTorch module to normalize input features

    Args:
        num_feats: Number of tensor features
    """
    def __init__(self, num_features: int):
        super().__init__()

        # hold onto the running averages for the mean and variance
        self.norm = torch.nn.ParameterDict({
            "mean": P(torch.zeros(num_features), requires_grad=False),
            "var": P(torch.zeros(num_features), requires_grad=False),
            "count": P(torch.zeros(1, dtype=torch.long), requires_grad=False),
        })

        # whether to continue updating running averages
        self.update = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        """
        Forward pass for InputNorm module

        Args:
            x: Tensor to normalize
        """

        # update running average
        if self.update and self.training:
            n1 = self.norm["count"]
            m1 = self.norm["mean"]
            v1 = self.norm["var"]

            n2 = x.shape[0]
            m2 = x.mean(dim=0)
            v2 = x.var(dim=0)

            # HOTFIX: feature dimension changed (e.g. from 8 -> 10); reset running stats
            if m1.shape != m2.shape:
                print(
                    f"[InputNorm] WARNING: resetting running stats due to feature dim change: "
                    f"running={tuple(m1.shape)}, batch={tuple(m2.shape)}",
                    flush=True,
                )

                # Reset running statistics to current batch
                # Make count a tensor matching the existing shape/dtype/device
                n_reset = torch.full_like(n1, fill_value=n2)
                self.norm["count"] = P(n_reset, requires_grad=False)
                self.norm["mean"] = P(m2.detach(), requires_grad=False)
                self.norm["var"] = P(v2.detach(), requires_grad=False)

            else:
                # Normal Welford-style update when feature dims match
                n = n1 + n2
                mean = ((n1 * m1) + (n2 * m2)) / n
                d1 = v1 + (mean - m1).square()
                d2 = v2 + (mean - m2).square()
                var = ((n1 * d1) + (n2 * d2)) / n

                self.norm["count"] = P(n, requires_grad=False)
                self.norm["mean"] = P(mean, requires_grad=False)
                self.norm["var"] = P(var, requires_grad=False)

        # return normalized tensor
        mean = self.norm["mean"][None, :]
        var = self.norm["var"][None, :] + 1e-5
        return (x - mean) / var.sqrt()
