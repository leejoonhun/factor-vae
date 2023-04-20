import os
import random
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from torch import cuda
from torch.backends import cudnn


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--locale", type=str, default="us", choices=["hk", "hu", "jp", "pl", "uk", "us"]
    )
    parser.add_argument("--len_hist", type=int, default=20)
    parser.add_argument("--num_stocks", type=int, default=1_000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_facts", type=int, default=32)
    parser.add_argument("--num_feats", type=int, default=64)
    parser.add_argument("--num_pfs", type=int, default=100)
    parser.add_argument("--metric", type=str, default="rankic")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=718)
    args = parser.parse_args()
    return args


def seed_everything(args: Namespace):
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
