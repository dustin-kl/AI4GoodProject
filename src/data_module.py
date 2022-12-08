from netCDF4 import Dataset
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split


class ClimateNetDataModule(pl.LightningDataModule):
    def __init__(self, files, feature_list, batch_size, num_workers=4, shuffle=True):
        super().__init__()
        self.files = files
        self.batch_size = batch_size
        self.feature_list = feature_list
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.train_files = self.val_files = None
        self.train_ds = self.val_ds = None

    def prepare_data(self):
        self.train_files, self.val_files = random_split(self.files, [0.8, 0.2])

    def setup(self, stage: str = None):
        if stage == "fit":
            self.train_ds = []
            self.val_ds = []
            for file in self.train_files:
                features, labels = self.load_data(file, self.feature_list)
                self.train_ds.append(self.transform(features, labels))
            for file in self.val_files:
                features, labels = self.load_data(file, self.feature_list)
                self.val_ds.append(self.transform(features, labels))

    @staticmethod
    def load_data(file, feature_list):
        ncdf = Dataset(file)
        features = []
        for feature in feature_list:
            features.append(ncdf[feature][0])
        labels = ncdf["LABELS"]
        features = np.array(features)
        labels = np.array(labels)
        ncdf.close()
        return features, labels

    @staticmethod
    def transform(features, labels):
        features = torch.tensor(features) #.to(torch.float16)
        labels = torch.tensor(labels)
        labels = F.one_hot(labels, num_classes=3)
        labels = labels.permute(2, 0, 1) # batch, classes, (x,y)
        labels = labels.to(torch.float16)
        return features, labels

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )
