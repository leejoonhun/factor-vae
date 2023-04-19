import torch
from torch import nn


class KLDivLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        prior_mean: torch.Tensor,
        prior_std: torch.Tensor,
        post_mean: torch.Tensor,
        post_std: torch.Tensor,
    ):
        return (
            (prior_std / post_std).log()
            + (post_std**2 + (post_mean - prior_mean) ** 2) / (2 * prior_std**2)
            - 0.5
        ).sum()


class NLLLoss(nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__()
