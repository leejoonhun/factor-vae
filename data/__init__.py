from argparse import Namespace
from typing import Tuple

from torch import cuda
from torch.utils import data as dt

from .dataset import StockReturnDataset


def get_dataloaders(
    args: Namespace,
) -> Tuple[dt.DataLoader, dt.DataLoader, dt.DataLoader]:
    trainset, validset, testset = dt.random_split(
        StockReturnDataset(args.locale, args.len_hist, args.num_stocks), [0.8, 0.1, 0.1]
    )
    args.num_chars = trainset[0][0].shape[-1]
    return (
        dt.DataLoader(
            trainset,
            args.batch_size,
            shuffle=True,
            num_workers=cuda.device_count() * 4,
        ),
        dt.DataLoader(
            validset,
            args.batch_size,
            shuffle=False,
            num_workers=cuda.device_count() * 4,
        ),
        dt.DataLoader(
            testset,
            args.batch_size,
            shuffle=False,
            num_workers=cuda.device_count() * 4,
        ),
    )
