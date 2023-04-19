from typing import Tuple

import torch
from torch import nn


class FactorEncoder(nn.Module):
    """Factor encoder for FactorVAE

    Args:
        num_pfs (int): number of portfolios $M$
        num_facts (int): number of facts $K$
        num_feats (int): number of features $H$
    """

    def __init__(self, num_facts: int, num_feats: int, num_pfs: int):
        super().__init__()
        self.pf_layer = nn.Sequential(nn.Linear(num_feats, num_pfs), nn.Softmax(dim=0))
        self.mapping_layer = MappingLayer(num_facts, num_pfs)

    def forward(
        self, rets: torch.Tensor, feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mapping_layer(
            torch.einsum("bs,bsm->bm", rets, self.pf_layer(feats))
        )


class MappingLayer(nn.Module):
    def __init__(self, num_facts: int, num_pfs: int):
        super().__init__()
        self.mean_layer = nn.Linear(num_pfs, num_facts)
        self.std_layer = nn.Sequential(nn.Linear(num_pfs, num_facts), nn.Softplus())

    def forward(self, pf_rets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.mean_layer(pf_rets), self.std_layer(pf_rets)
        return mean, std.clip(min=0)
