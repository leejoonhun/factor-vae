import os
import random

import numpy as np
import torch
from torch import cuda
from torch.backends import cudnn


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
