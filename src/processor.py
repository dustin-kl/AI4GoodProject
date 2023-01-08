from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger


class Processor:
    @staticmethod
    def fit(model, datamodule, max_epochs=100, precision=16):
        logger = WandbLogger(project="ClimateNet", log_model="all")

        callbacks = [
            ModelCheckpoint(monitor="val/mean_iou", mode="max"),
            #EarlyStopping(monitor="val/mean_iou", mode="max", patience=10),
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
