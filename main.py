import os

from config import config
from parser import args
from src.models.hyperparameters import params
from src.data_module import ClimateNetDataModule
from src.models.manager import get_model, train_model
from src.utils import Logger, Generic


def main():
    files = Generic.list_files("./dataset/")

    if len(files) < 10:
        raise Exception(
            "The amount of data is not enough, please download the data before running again."
        )

    model_name = args.model
    model_params = params[model_name]

    feature_list = config["feature_list"]
    batch_size = model_params["batch_size"]

    if args.test:
        files = files[:10]
    print('create Datamodule')
    data_module = ClimateNetDataModule(files, feature_list, batch_size)

    print(f'Create Model {model_name}')
    model = get_model(model_name, len(feature_list), params[model_name])

    print('Start Training')
    if args.test:
        model_name = "test"
    train_model(model, model_name, data_module)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
