from typing import Tuple

import torch
from torch import nn

from .factor_decoder import AlphaLayer


class FactorPredictor(nn.Module):
    """Factor predictor for FactorVAE

    Args:
        num_facts (int): number of facts $K$
        num_feats (int): number of features $H$
    """

    def __init__(self, num_facts: int, num_feats: int):
        super().__init__()
        self.attn_layer = MHA3d(num_facts, num_feats)
        self.alpha_layer = AlphaLayer(num_feats)

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.alpha_layer(self.attn_layer(feats))


class MHA3d(nn.Module):
    def __init__(self, num_facts: int, num_feats: int):
        super().__init__()
        self.query = nn.Parameter(torch.rand(num_facts, num_feats))
        self.key_layer = nn.Linear(num_feats, num_feats, bias=False)
        self.val_layer = nn.Linear(num_feats, num_feats, bias=False)
        self.attn_layer = nn.MultiheadAttention(num_feats, num_facts, batch_first=True)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        attn = torch.stack(
            [
                self.attn_layer(
                    self.query,
                    self.key_layer(feat),
                    self.val_layer(feat),
                )[0]
                for feat in feats
            ]
        )
        return attn
