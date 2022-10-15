"""
Create dataloaders depending on settings in config.py
"""

from typing import Optional
from config import config

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.utils import Generic, NetCDF, Logger


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.data_dir = Generic.get_directory(__file__) + "/../dataset/"
        self.batch_size = batch_size
        self.num_workers = 8
        self.setup()
        Logger.log_info("Data module set up.")

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.train_set = [self.load_data("train"), self.load_labels("train")]
        self.val_set = [self.load_data("val"), self.load_labels("val")]
        self.test_set = [self.load_data("test"), self.load_labels("test")]

    def load_data(self, mode):
        data = []
        for dataset in config[f"{mode}_dataset"]:
            data.append(NetCDF.load_data(self.data_dir + dataset, config["features"]))
        data = torch.tensor(np.array(data))
        return data

    def load_labels(self, mode):
        labels = []
        for dataset in config[f"{mode}_dataset"]:
            labels.append(NetCDF.load_labels(self.data_dir + dataset))
        labels = torch.tensor(np.array(labels)) 
        return labels

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)
