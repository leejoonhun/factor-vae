import torch
from torch import nn


class FactorEncoder(nn.Module):
    """Factor encoder for FactorVAE

    Args:
        num_pfs (int): number of portfolios $M$
        num_factors (int): number of factors $K$
        dim_hidden (int): dimension of features $H$
    """

    def __init__(self, num_pfs: int, num_factors: int, dim_hidden: int):
        super().__init__()
        self.pf_layer = nn.Sequential(nn.Linear(dim_hidden, num_pfs), nn.Softmax(dim=0))
        self.mean_layer = nn.Linear(num_pfs, num_factors)
        self.std_layer = nn.Sequential(nn.Linear(num_pfs, num_factors), nn.Softplus())

    def forward(
        self,
        returns: torch.Tensor,  # (num_stocks,)
        feats: torch.Tensor,  # (num_stocks, dim_hidden)
    ) -> torch.Tensor:
        pf_returns = self.pf_layer(feats).T @ returns
        mean, std = self.mean_layer(pf_returns), self.std_layer(pf_returns)
        return torch.normal(mean, std)  # (num_factors,)


if __name__ == "__main__":
    num_stocks, num_factors, num_pfs, dim_hidden = 5, 30, 64, 128
    feats, returns = torch.rand(num_stocks, dim_hidden), torch.rand(num_stocks)
    model = FactorEncoder(num_pfs, num_factors, dim_hidden)
    factors = model(returns, feats)
    assert factors.shape == (num_factors,)
    print("passed test")
