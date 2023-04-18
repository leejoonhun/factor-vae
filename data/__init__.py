from typing import Tuple

from torch import cuda
from torch.utils import data as dt

from .dataset import StockReturnDataset


def get_dataloaders(
    locale: str, len_hist: int, batch_size: int
) -> Tuple[dt.DataLoader, dt.DataLoader]:
    trainset, validset = dt.random_split(
        StockReturnDataset(locale, len_hist), [0.9, 0.1]
    )
    return (
        dt.DataLoader(
            trainset, batch_size, shuffle=True, num_workers=cuda.device_count() * 4
        ),
        dt.DataLoader(
            validset, batch_size, shuffle=False, num_workers=cuda.device_count() * 4
        ),
    )
