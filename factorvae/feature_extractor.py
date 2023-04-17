import torch
from torch import nn


class FeatureExtractor(nn.Module):
    """Feature extractor for FactorVAE

    Args:
        num_char (int): number of characteristics $C$
        dim_hidden (int): dimension of features $H$
    """

    def __init__(self, num_char: int, dim_hidden: int):
        super().__init__()
        self.linear_layer = nn.Sequential(nn.Linear(num_char, num_char), nn.LeakyReLU())
        self.gru = nn.GRUCell(num_char, dim_hidden)

    def forward(
        self, chars: torch.Tensor  # (len_hist, num_stocks, num_chars)
    ) -> torch.Tensor:
        feats = None
        for char in chars:
            feats = self.gru(self.linear_layer(char), feats)
        return feats  # (num_stocks, dim_hidden)


if __name__ == "__main__":
    num_stocks, len_hist, num_chars, dim_hidden = 5, 10, 16, 128
    chars = torch.rand(len_hist, num_stocks, num_chars)
    model = FeatureExtractor(num_chars, dim_hidden)
    feature = model(chars)
    assert feature.shape == (num_stocks, dim_hidden)
    print("passed test")
