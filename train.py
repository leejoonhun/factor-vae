import logging

import torch
import wandb
from torch import cuda, nn, optim
from tqdm import trange

import metrics
from data import get_dataloaders
from factorvae import FactorVAE
from losses import KLDivLoss, NLLLoss
from utils import parse_args, seed_everything


def train(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Start training with args: {vars(args)}")
    seed_everything(args)
    torch.set_default_dtype(torch.float64)
    device = "cuda" if cuda.is_available() else "cpu"

    logger.info("Loading data")
    trainloader, validloader, testloader = get_dataloaders(args)
    wandb.init(config=args)

    logger.info("Building model and configuring optimizer")
    model = FactorVAE(args).to(device)

    nll_loss = NLLLoss()
    kld_loss = KLDivLoss()
    enc_optim = optim.Adam(
        list(model.feature_extractor.parameters())
        + list(model.factor_encoder.parameters())
        + list(model.factor_predictor.parameters()),
        lr=args.lr,
    )
    dec_optim = optim.Adam(model.factor_decoder.parameters(), lr=args.lr)

    if cuda.device_count() > 1:
        model = nn.DataParallel(model)
        enc_optim = optim.Adam(
            list(model.module.feature_extractor.parameters())
            + list(model.module.factor_encoder.parameters())
            + list(model.module.factor_predictor.parameters()),
            lr=args.lr,
        )
        dec_optim = optim.Adam(model.module.factor_decoder.parameters(), lr=args.lr)

    for epoch in trange(args.epochs):
        model.train()
        for hist, futr in trainloader:
            hist, futr = hist.to(device), futr.to(device)
            pred, post_mean, post_std, prior_mean, prior_std = model(hist, futr)

            loss = nll_loss(pred, futr) + args.gamma * kld_loss(
                post_mean, post_std, prior_mean, prior_std
            )
            enc_optim.zero_grad()
            dec_optim.zero_grad()
            loss.backward()
            enc_optim.step()
            dec_optim.step()
            wandb.log({"epoch": epoch, "train_loss": loss.item()})

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                for hist, futr in validloader:
                    hist, futr = hist.to(device), futr.to(device)
                    pred, post_mean, post_std, prior_mean, prior_std = model(hist, futr)

                    loss = nll_loss(pred, futr) + args.gamma * kld_loss(
                        post_mean, post_std, prior_mean, prior_std
                    )
                    wandb.log({"epoch": epoch, "valid_loss": loss.item()})
    logger.info("Training finished")
    wandb.finish()

    logger.info("Start testing")
    metric_fn = getattr(metrics, args.metric)()
    model.eval()
    preds, futrs = [], []
    with torch.no_grad():
        for hist, futr in testloader:
            hist, futr = hist.to(device), futr.to(device)
            pred, *_ = model(hist, futr)
            preds.append(pred.detach().cpu().numpy())
            futrs.append(futr.detach().cpu().numpy())
    logger.info(f"{args.metric}: {metric_fn(preds, futrs)}")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
