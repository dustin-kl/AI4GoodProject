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
        self.train_files, self.val_files = Generic.split_data(config["train_dataset"])
        self.test_files = config["test_dataset"]
        self.dataset = {
            "train": self.train_files,
            "val": self.val_files,
            "test": self.test_files,
        }
        self.setup()
        Logger.log_info("Data module set up.")

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.train_set = self.load_data("train")
        self.val_set = self.load_data("val")
        self.test_set = self.load_data("test")

    def load_data(self, mode):
        data = []
        for dataset in self.dataset[mode]:
            sample = (
                torch.tensor(
                    NetCDF.load_data(self.data_dir + dataset, config["features"]),
                    dtype=torch.float32,
                ),
                torch.tensor(
                    NetCDF.load_labels(self.data_dir + dataset), dtype=torch.long
                ),
            )
            data.append(sample)
        return data

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=self.num_workers
        )
