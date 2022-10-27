import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary

from src.models.CNN import CNN
from src.models.Model import Model
from src.utils import Logger


def get_model(model, parameters):
    if model == "CNN":
        return CNN(parameters)
    if model == "model":
        return Model(parameters)
    else:
        Logger.log_error(f"Model {model} not found.")
        raise NotImplementedError


def run_model(model, data_module, logger, log_dir):
    logger.log_hyperparams(model.params)

    trainer = pl.Trainer(
        # accelerator="gpu",  # cpu or gpu
        # devices=-1,  # -1: use all available gpus, for cpu e.g. 4
        enable_progress_bar=False,
        logger=[logger],
        max_epochs=model.params["epochs"],  # max number of epochs
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10),  # early stopping
            ModelSummary(max_depth=1),  # model summary
            ModelCheckpoint(
                log_dir + "checkpoint/", monitor="val_loss", save_top_k=1
            ),  # save best model
        ],
        auto_lr_find=True,  # automatically find learning rate
        log_every_n_steps=1
    )

    Logger.log_info("Training model...")
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())
    Logger.log_info("Finished training.")

    Logger.log_info("Testing model...")
    trainer.test(model, data_module.test_dataloader())
    Logger.log_info("Finished testing.")
