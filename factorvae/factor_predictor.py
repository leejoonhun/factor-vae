import torch
from torch import nn

from .factor_decoder import AlphaLayer


class FactorPredictor(nn.Module):
    """Factor predictor for FactorVAE

    Args:
        num_factors (int): number of factors $K$
        num_feats (int): number of features $H$
    """

    def __init__(self, num_factors: int, num_feats: int):
        super().__init__()
        self.attn_layer = AttentionLayer(num_factors, num_feats)
        self.alpha_layer = AlphaLayer(num_feats)

    def forward(
        self,
        feats: torch.Tensor,  # (num_stocks, num_feats)
    ) -> torch.Tensor:
        return self.alpha_layer(self.attn_layer(feats))


class AttentionLayer(nn.Module):
    def __init__(self, num_factors: int, num_feats: int):
        super().__init__()
        self.query = nn.Parameter(torch.rand(num_factors, num_feats))
        self.key_layer = nn.Linear(num_feats, num_feats, bias=False)
        self.val_layer = nn.Linear(num_feats, num_feats, bias=False)
        self.attn_layer = nn.MultiheadAttention(
            num_feats, num_factors, batch_first=True
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        attn, _ = self.attn_layer(
            self.query, self.key_layer(feats), self.val_layer(feats)
        )
        return attn


if __name__ == "__main__":
    num_stocks, len_hist, num_chars = 5, 10, 16
    num_factors, num_feats, num_pfs = 32, 128, 64
    factors, feats = torch.rand(num_factors), torch.rand(num_stocks, num_feats)
    model = FactorPredictor(num_factors, num_feats)
    factors = model(feats)
    assert factors.shape == (num_factors,)
    print("passed test")
