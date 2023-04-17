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
        self.num_feats = num_feats
        self.linear_layer = nn.Sequential(
            nn.Linear(num_chars, num_chars), nn.LeakyReLU()
        )
        self.gru_cell = GRUCell3d(num_chars, num_feats)

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        chars = chars.permute(1, 0, 2, 3)
        feats = torch.zeros(*chars.shape[1:-1], self.num_feats)
        for char in chars:
            feats = self.gru_cell(char, feats)
        return feats


class GRUCell3d(nn.Module):
    def __init__(self, num_chars: int, num_feats: int):
        super().__init__()
        self.gru_cell = nn.GRUCell(num_chars, num_feats)

    def forward(self, data: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.gru_cell(d, h) for d, h in zip(data, hidden)])


if __name__ == "__main__":
    batch_size, num_stocks, len_hist, num_chars = 1, 5, 10, 16
    num_factors, num_feats, num_pfs = 32, 128, 64
    chars = torch.rand(batch_size, len_hist, num_stocks, num_chars)
    model = FeatureExtractor(num_chars, num_feats)
    feature = model(chars)
    assert feature.shape == (batch_size, num_stocks, num_feats)
    print("passed test")
