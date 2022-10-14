"""
Create dataloaders depending on settings in config.py
"""

from config import config
from src.models.hyperparameters import params
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import netCDF4 as nc
from src.utils import get_directory



class custom_Datamodule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.data_dir = get_directory(__file__) + "/dataset/"
        self.batch_size = batch_size
        
    def prepare_data(self):
        # download data if necessary
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        self.train_set = self.load_data('train')
        self.val_set = self.load_data('val')
        self.test_set = self.load_data('test')

    def load_data(self, mode):
        data = []
        for dataset in config[f'{mode}_dataset']:
            ds = nc.Dataset(self.data_dir + dataset)
            x_image = []
            for feature in config['feature_list']:
                x_image.append(ds[feature])
            data.append(torch.tensor(x_image))
        data = torch.tensor(data)
        return data


    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)
