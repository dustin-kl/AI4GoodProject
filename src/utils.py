import argparse
import json
import logging
import os
from os.path import join
import time

from lightning_lite.utilities.seed import seed_everything
from netCDF4 import Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np


class Generic:
    @staticmethod
    def get_directory(file):
        return os.path.dirname(os.path.realpath(file))

    @staticmethod
    def remove_file(file):
        return os.remove(file)

    @staticmethod
    def list_files(directory):
        file_list = os.listdir(directory)
        for idx, file in enumerate(file_list):
            file_list[idx] = join(directory, file)
        return file_list

    @staticmethod
    def to_json(dictionary, path):
        with open(path, "w") as f:
            json.dump(dictionary, f)

    @staticmethod
    def split_data(data_list):
        data_list = shuffle(data_list)
        data_train, data_test = train_test_split(data_list, test_size=0.2)
        return data_train, data_test


class NetCDF:
    @staticmethod
    def load_data(path, features):
        dataset = Dataset(path)
        data = []
        for feature in features:
            data.append(dataset[feature][:][0])
        dataset.close()
        data = np.array(data)
        return data

    @staticmethod
    def load_labels(path):
        dataset = Dataset(path)
        labels = dataset["LABELS"][:]
        labels = np.array(labels)
        dataset.close()
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


class Config:
    @staticmethod
    def init_config(seed=6):
        seed_everything(seed)

        config = Config()
        config.data_dir = "./dataset/"
        config.features = ["TMQ", "U850", "V850", "PRECT"]
        config.height = 768
        config.width = 1152

        return config

class Parser:
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()

        # Data module
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--small_dataset", action="store_true")
        parser.add_argument("--no_small_dataset", action="store_false", dest="small_dataset")
        parser.add_argument("--shuffle", action="store_true")
        parser.add_argument("--no_shuffle", action="store_false", dest="shuffle")
        parser.set_defaults(small_dataset=False, shuffle=True)
        # Model
        parser.add_argument("--model", type=str, default="unet")

        args = parser.parse_args()

        return args
