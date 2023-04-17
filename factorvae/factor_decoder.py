import torch
from torch import nn


class FactorDecoder(nn.Module):
    def __init__(self, num_stocks: int, num_factors: int, dim_hidden: int):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden), nn.LeakyReLU()
        )
        self.alpha_mean_layer = nn.Linear(dim_hidden, 1)
        self.alpha_std_layer = nn.Sequential(nn.Linear(dim_hidden, 1), nn.Softplus())
        self.beta_layer = nn.Linear(dim_hidden, num_factors)

    def forward(
        self,
        factors: torch.Tensor,  # (num_factors,)
        feats: torch.Tensor,  # (num_stocks, dim_hidden)
    ) -> torch.Tensor:
        hidden = self.linear_layer(feats)
        alpha = torch.normal(
            self.alpha_mean_layer(hidden), self.alpha_std_layer(hidden)
        ).flatten()
        return self.beta_layer(feats) @ factors + alpha


if __name__ == "__main__":
    num_stocks, num_factors, dim_hidden = 5, 30, 128
    factors, feats = torch.rand(num_factors), torch.rand(num_stocks, dim_hidden)
    model = FactorDecoder(num_stocks, num_factors, dim_hidden)
    returns = model(factors, feats)
    assert returns.shape == (num_stocks,)
    print("passed test")
