import torch
from torch import nn


class FactorDecoder(nn.Module):
    """Factor decoder for FactorVAE

    Args:
        num_factors (int): number of factors $K$
        num_feats (int): number of features $H$
    """

    def __init__(self, num_factors: int, num_feats: int):
        super().__init__()
        self.alpha_layer = AlphaLayer(num_feats)
        self.beta_layer = nn.Linear(num_feats, num_factors)

    def forward(self, factors: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            "bsk,bk->bs", self.beta_layer(feats), factors
        ) + self.alpha_layer(feats)


class AlphaLayer(nn.Module):
    def __init__(self, num_feats: int):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(num_feats, num_feats), nn.LeakyReLU()
        )
        self.mean_layer = nn.Linear(num_feats, 1)
        self.std_layer = nn.Sequential(nn.Linear(num_feats, 1), nn.Softplus())

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        hidden = self.linear_layer(feats)
        mean, std = self.mean_layer(hidden), self.std_layer(hidden)
        return torch.normal(mean, std).flatten(-2)


if __name__ == "__main__":
    batch_size, num_stocks, len_hist, num_chars = 1, 5, 10, 16
    num_factors, num_feats, num_pfs = 32, 128, 64
    factors, feats = (
        torch.rand(batch_size, num_factors),
        torch.rand(batch_size, num_stocks, num_feats),
    )
    model = FactorDecoder(num_factors, num_feats)
    returns = model(factors, feats)
    assert returns.shape == (batch_size, num_stocks)
    print("passed test")
