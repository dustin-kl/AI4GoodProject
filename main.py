import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from config import config
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.hyperparameters import params
from datamodule_factory import get_datamodule
import time
from src.utils import Logger, log_params
import os
import logging
from datamodule_factory import custom_Datamodule
import argparse

# change config['model'] to args['model'] to select which model to run

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('model', type=str, help='select which model to run')
args = parser.parse_args()


def main():
    
    # Create model directory and Logger
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/{run_id}_{config['model']}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # sys.stdout = Logger(print_fp=os.path.join(log_dir, 'out.txt'))
    # Create logging file
    logging.basicConfig(filename=f"{log_dir}/info.log", encoding='utf-8', level=logging.INFO)
    logging.info("Started logging.")

    # Obtain datamodule based on config settings for dataset
    data_module = custom_Datamodule(batch_size=params[config['model']]['batch_size'])
    logging.info("Created data module.")

    # Create model based on config.py and hyperparameters.py settings
    model = get_model(params['CNN'])       # use args['model'] to select which model to run
    logging.info("Created model.")
    # print model summary
    # summary(model, (params['input_height'], params['input_width']))

    # Log hyperparameters and config file
    log_params(log_dir)

    # Run the model
    tb_logger = TensorBoardLogger("./runs/", name=f"{run_id}_{config['model']}")
    tb_logger.log_hyperparams(params[config['model']])  # log hyperparameters

    trainer = pl.Trainer(accelerator="gpu",  # cpu or gpu
                         devices=-1,  # -1: use all available gpus, for cpu e.g. 4
                         enable_progress_bar=False,  # disable progress bar
                         # progress_bar_refresh_rate=500, # show progress bar every 500 iterations
                         # precision=16, # 16 bit float precision for training
                         #logger=[tb_logger, wandb_logger],  # log to tensorboard and wandb
                         logger = [tb_logger],

                         max_epochs=params[config['model']]['epochs'],  # max number of epochs
                         callbacks=[EarlyStopping(monitor="Validation Loss", patience=10),  # early stopping
                                    ModelSummary(max_depth=1),  # model summary
                                    ModelCheckpoint(log_dir, monitor='Validation Loss', save_top_k=1)  # save best model
                                    ],
                         auto_lr_find=True  # automatically find learning rate
                         )
    
    # Train the model
    logging.info("Start training.")
    trainer.fit(model, data_module)  # train the model
    logging.info("Finished training.")

    # Test the model
    logging.infor("Start testing.")
    trainer.test(model, data_module)  # test the model
    logging.info("Finished testing.")

def get_model(model_params):
    if model_params['name'] == 'CNN':
        from src.models.CNN import CNN
        model = CNN(model_params)
    return model


if __name__ == '__main__':
    main()


    ''''
    Give list to dataloader of which features to use
    Metric: IoU, Accuracy, F1 score
    '''