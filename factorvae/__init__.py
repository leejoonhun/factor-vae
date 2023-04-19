from argparse import Namespace
from typing import Tuple

import torch
from torch import nn

from .factor_decoder import FactorDecoder
from .factor_encoder import FactorEncoder
from .factor_predictor import FactorPredictor
from .feature_extractor import FeatureExtractor


class FactorVAE(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.feature_extractor = FeatureExtractor(args.num_chars, args.num_feats)
        self.factor_encoder = FactorEncoder(
            args.num_facts, args.num_feats, args.num_pfs
        )
        self.factor_predictor = FactorPredictor(args.num_facts, args.num_feats)
        self.factor_decoder = FactorDecoder(args.num_facts, args.num_feats)

    def forward(
        self, chars: torch.Tensor, rets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.feature_extractor(chars)
        post_mean, post_std = self.factor_encoder(rets, feats)
        prior_mean, prior_std = self.factor_predictor(feats)
        return (
            self.factor_decoder(torch.normal(post_mean, post_std), feats),
            post_mean,
            post_std,
            prior_mean,
            prior_std,
        )

    def predict(self, chars: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(chars)
        prior_mean, prior_std = self.factor_predictor(feats)
        return self.factor_decoder(torch.normal(prior_mean, prior_std), feats)
