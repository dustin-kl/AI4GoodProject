import logging
import os
import sys
import time

from netCDF4 import Dataset
from pytorch_lightning.loggers import TensorBoardLogger


class Generic:
    @staticmethod
    def get_directory(file):
        return os.path.dirname(os.path.realpath(file))

    @staticmethod
    def remove_file(file):
        return os.remove(file)


class NetCDF:
    @staticmethod
    def load_data(path, features):
        dataset = Dataset(path)
        data = []
        for feature in features:
            data.append(dataset["data"][feature][:])
        return data

    @staticmethod
    def load_labels(path):
        dataset = Dataset(path)
        labels = []
        labels.append(dataset["labels"]["label_0_ar"][:])
        #labels.append(dataset["labels"]["label_0_tc"][:])
        return labels


class Logger:
    @staticmethod
    def setup_logger(model):
        log_id = time.strftime("%Y%m%d-%H%M%S")
        log_dir = f"{Generic.get_directory(__file__)}/../logs/{model}/{log_id}/"

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            os.makedirs(log_dir + "tensor_board/")
            os.makedirs(log_dir + "checkpoint/")

        logging.basicConfig(filename=f"{log_dir}/info.log", level=logging.INFO)

        Logger.log_info(f"Started logging.")

        return TensorBoardLogger(log_dir, "tensor_board"), log_dir

    @staticmethod
    def log_info(message):
        message = Logger.format_message(message)
        logging.info(message)
        print(message)

    @staticmethod
    def log_error(message):
        message = Logger.format_message(message)
        logging.error(message)
        print(message)

    @staticmethod
    def format_message(message):
        now = time.strftime("[%Y-%m-%d %H:%M:%S]")
        return f"{now} {message}"
