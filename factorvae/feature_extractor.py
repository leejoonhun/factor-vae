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
        feats = torch.zeros(*chars.shape[1:-1], self.num_feats, device=chars.device)
        for char in chars:
            feats = self.gru_cell(char, feats)
        return feats


class GRUCell3d(nn.Module):
    def __init__(self, num_chars: int, num_feats: int):
        super().__init__()
        self.gru_cell = nn.GRUCell(num_chars, num_feats)

    def forward(self, data: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.gru_cell(d, h) for d, h in zip(data, hidden)])
