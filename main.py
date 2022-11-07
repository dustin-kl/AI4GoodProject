from parser import args
from src.models.hyperparameters import params
from src.data_module import DataModule
from src.utils import Logger
from src.models.manager import get_model, run_model
from src.utils import Logger


def main():
    model = args["model"]

    tb_logger, log_dir = Logger.setup_logger(model)
    Logger.log_info(params[model])

    # Obtain datamodule based on config settings for dataset
    data_module = DataModule(batch_size=params[model]["batch_size"])

    try:
        model = get_model(model, params[model])
    except NotImplementedError:
        exit()

    run_model(model, data_module, tb_logger, log_dir)


if __name__ == "__main__":
    main()

    """'
    Give list to dataloader of which features to use
    Metric: IoU, Accuracy, F1 score
    """
