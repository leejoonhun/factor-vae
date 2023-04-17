import torch
from torch import nn

from .factor_decoder import FactorDecoder
from .factor_encoder import FactorEncoder
from .factor_predictor import FactorPredictor
from .feature_extractor import FeatureExtractor


class FactorVAE(nn.Module):
    def __init__(self, num_chars: int, num_factors: int, num_feats: int, num_pfs: int):
        super().__init__()
        self.feature_extractor = FeatureExtractor(num_chars, num_feats)
        self.factor_encoder = FactorEncoder(num_pfs, num_factors, num_feats)
        self.factor_decoder = FactorDecoder(num_factors, num_feats)
        self.factor_predictor = FactorPredictor(num_factors, num_feats)

    def forward(self, chars: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(chars)
        factors = self.factor_encoder(returns, feats)
        return self.factor_decoder(factors, feats)

    def predict(self, chars: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(chars)
        factors = self.factor_predictor(feats)
        return self.factor_decoder(factors, feats)
