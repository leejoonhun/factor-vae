from typing import Tuple

import numpy as np
import polars as pl
import torch
from torch.utils import data as dt

from .path import DATA_DIR


class StockReturnDataset(dt.Dataset):
    def __init__(self, locale: str, len_hist: int):
        super().__init__()
        self.len_hist = len_hist

        df = (
            pl.read_csv(DATA_DIR / f"{locale}.csv")
            .select(
                ["<DATE>", "<TICKER>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"]
            )
            .with_columns(pl.col("<CLOSE>").pct_change().alias("<RETURN>"))
            .drop_nulls()
            .groupby(["<TICKER>"])
            .apply(lambda df: df.head(-len_hist + 1))
            .sort(["<DATE>", "<TICKER>"])
        )
        self.data = np.stack(
            [
                df.pivot(
                    val_col, "<DATE>", "<TICKER>", aggregate_function="first"
                ).to_numpy()
                for val_col in [
                    "<OPEN>",
                    "<HIGH>",
                    "<LOW>",
                    "<CLOSE>",
                    "<VOL>",
                    "<RETURN>",
                ]
            ],
            axis=2,
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.data[idx : idx + self.len_hist]),
            torch.tensor(self.data[idx + self.len_hist, :, -1]),
        )

    def __len__(self) -> int:
        return self.data.shape[0]
