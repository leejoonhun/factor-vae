import torch
from torch import nn


class FeatureExtractor(nn.Module):
    """Feature extractor for FactorVAE

    Args:
        num_char (int): number of characteristics $C$
        num_feats (int): dimension of features $H$
    """

    def __init__(self, num_chars: int, num_feats: int):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(num_chars, num_chars), nn.LeakyReLU()
        )
        self.gru_cell = nn.GRUCell(num_chars, num_feats)

    def forward(
        self, chars: torch.Tensor  # (len_hist, num_stocks, num_chars)
    ) -> torch.Tensor:
        feats = None
        for char in chars:
            feats = self.gru_cell(self.linear_layer(char), feats)
        return feats


if __name__ == "__main__":
    num_stocks, len_hist, num_chars = 5, 10, 16
    num_factors, num_feats, num_pfs = 32, 128, 64
    chars = torch.rand(len_hist, num_stocks, num_chars)
    model = FeatureExtractor(num_chars, num_feats)
    feature = model(chars)
    assert feature.shape == (num_stocks, num_feats)
    print("passed test")
