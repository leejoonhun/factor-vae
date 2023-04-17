import torch
from torch import nn


class FactorEncoder(nn.Module):
    """Factor encoder for FactorVAE

    Args:
        num_pfs (int): number of portfolios $M$
        num_factors (int): number of factors $K$
        num_feats (int): number of features $H$
    """

    def __init__(self, num_pfs: int, num_factors: int, num_feats: int):
        super().__init__()
        self.pf_layer = nn.Sequential(nn.Linear(num_feats, num_pfs), nn.Softmax(dim=0))
        self.mapping_layer = MappingLayer(num_pfs, num_factors)

    def forward(self, returns: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        pf_returns = returns @ self.pf_layer(feats)
        return self.mapping_layer(pf_returns)


class MappingLayer(nn.Module):
    def __init__(self, num_pfs, num_factors):
        super().__init__()
        self.mean_layer = nn.Linear(num_pfs, num_factors)
        self.std_layer = nn.Sequential(nn.Linear(num_pfs, num_factors), nn.Softplus())

    def forward(self, pf_returns: torch.Tensor) -> torch.Tensor:
        mean, std = self.mean_layer(pf_returns), self.std_layer(pf_returns)
        return torch.normal(mean, std)


if __name__ == "__main__":
    num_stocks, len_hist, num_chars = 5, 10, 16
    num_factors, num_feats, num_pfs = 32, 128, 64
    feats, returns = torch.rand(num_stocks, num_feats), torch.rand(num_stocks)
    model = FactorEncoder(num_pfs, num_factors, num_feats)
    factors = model(returns, feats)
    assert factors.shape == (num_factors,)
    print("passed test")
