import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from disco_split.models.bart import BartDataModule, BartFinetuner

if __name__ == '__main__':
    """
    Train a BART generative model.
    """

    # prepare argument parser
    parser = argparse.ArgumentParser()

    parser = BartFinetuner.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # prepare data module and trainer class
    if args.checkpoint is None:
        model = BartFinetuner(hparams=args)
    else:
        model = BartFinetuner.load_from_checkpoint(args.checkpoint, hparams=args)
    model.update_mt_tokens()
    dm = BartDataModule(model.tokenizer, hparams=args)

    if args.name is None:
        args.name = f"{args.max_samples}_{args.batch_size}_{args.learning_rate}"

    wandb_logger = WandbLogger(
        name=args.name, project=args.project, save_dir=args.save_dir)

    trainer = pl.Trainer.from_argparse_args(
        args,
        val_check_interval=args.val_check_interval,
        logger=wandb_logger,
        accelerator="ddp")

    trainer.fit(model, dm)
