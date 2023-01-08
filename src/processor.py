from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import KFold
from torch.utils.data import random_split
import wandb

from src.data_module import ClimateNetDataModule
from src.models import get_model


class Processor:
    @staticmethod
    def fit(files, config, args, max_epochs=100, precision=16):
        wandb.init(project="ClimateNet")

        train_files, val_files = random_split(files, [0.8, 0.2])

        datamodule = ClimateNetDataModule(
            train_files,
            val_files,
            config.features,
            args.batch_size,
            num_workers=args.num_workers,
            shuffle=args.shuffle,
        )

        model = get_model(args.model)

        logger = WandbLogger(project="ClimateNet", log_model="all")

        callbacks = [
            ModelCheckpoint(monitor="val/mean_iou", mode="max"),
        ]

        trainer = Trainer(
            accelerator="gpu",
            callbacks=callbacks,
            devices=-1,
            enable_progress_bar=False,
            logger=logger,
            log_every_n_steps=1,
            max_epochs=max_epochs,
            precision=precision,
        )

        trainer.fit(model, datamodule)

    @staticmethod
    def cv(files, config, args, max_epochs=100, precision=16):
        kf = KFold(n_splits=5, shuffle=True)
        splits = list(kf.split(files))

        for f in range(5):
            wandb.init(project="ClimateNet")
            train_files = [files[i] for i in splits[f][0]]
            val_files = [files[i] for i in splits[f][1]]
            model = get_model(args.model)
            datamodule = ClimateNetDataModule(
                train_files,
                val_files,
                config.features,
                args.batch_size,
                num_workers=args.num_workers,
                shuffle=args.shuffle,
            )
            logger = WandbLogger(project="ClimateNet", log_model="all")
            callbacks = [
                ModelCheckpoint(monitor="val/mean_iou", mode="max"),
            ]
            trainer = Trainer(
                accelerator="gpu",
                callbacks=callbacks,
                devices=-1,
                enable_progress_bar=False,
                logger=logger,
                log_every_n_steps=1,
                max_epochs=max_epochs,
                precision=precision,
            )
            trainer.fit(model, datamodule)
            wandb.finish()
