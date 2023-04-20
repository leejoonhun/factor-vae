from typing import List

import numpy as np


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def calc_rankic(preds: List[np.ndarray], futrs: List[np.ndarray]) -> np.ndarray:
    """Calculates rank information coefficient"""
    rankic = _calc_rankic(preds, futrs)
    return rankic.mean()


def calc_rankicir(preds: List[np.ndarray], futrs: List[np.ndarray]) -> np.ndarray:
    """Calculates information ratio of Rank IC"""
    rankic = _calc_rankic(preds, futrs)
    return safe_div(rankic.mean(), rankic.std())


def _calc_rankic(preds: List[np.ndarray], futrs: List[np.ndarray]) -> np.ndarray:
    """Calculates rank information coefficient by element"""
    preds, futrs = (
        np.argsort(np.concatenate(preds), axis=1),
        np.argsort(np.concatenate(futrs), axis=1),
    )
    num_stocks = preds.shape[-1]
    return 1 - safe_div(
        6 * ((preds - futrs) ** 2).sum(axis=1),
        num_stocks * (num_stocks**2 - 1),
    )
