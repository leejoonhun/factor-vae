from typing import List

import numpy as np


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def calc_rankic(preds: List[np.ndarray], futrs: List[np.ndarray]) -> np.ndarray:
    """Calculates rank information coefficient"""
    rankic = _calc_rankic(preds, futrs)
    return safe_div(rankic.mean(), rankic)


def calc_rankicir(preds: List[np.ndarray], futrs: List[np.ndarray]) -> np.ndarray:
    """Calculates information ratio of Rank IC"""
    ric = _calc_rankic(preds, futrs)
    return safe_div(ric.mean(), ric.std())


def _calc_rankic(preds: List[np.ndarray], futrs: List[np.ndarray]) -> np.ndarray:
    """Calculates rank information coefficient by element"""
    preds, futrs = np.stack(preds), np.stack(futrs)
    ...
