from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from src.models.CNN import CNN
from src.models.Model import Model
from src.models.baseline import DeepLabv3_plus


def get_model(model, n_channels, params):
    if model == "cnn":
        return CNN(params)
    if model == "model":
        return Model(params)
    if model == "baseline":
        return DeepLabv3_plus(
            params, nInputChannels=n_channels, n_classes=3, _print=False
        )
    else:
        raise NotImplementedError


def train_model(model, model_name, data_module, trainer=None):
    log_dir = "./runs/"
    log_name = f"{model_name} - lr={model.lr}"
    logger = TensorBoardLogger(log_dir, name=log_name)

    logger.log_hyperparams(model.params)
    logger.log_graph(model)

    if trainer is None:
        trainer = Trainer(
            accelerator="cpu",
            enable_progress_bar=False,
            max_epochs=1,
            logger=[logger],
            log_every_n_steps=1,
        )
    else:
        trainer.logger = [logger]

    trainer.fit(model, datamodule=data_module)
