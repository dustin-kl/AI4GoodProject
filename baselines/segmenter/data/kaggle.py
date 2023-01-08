import mmcv
import torch
import torchvision
import torchvision.io as io
import os
import os.path as osp
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2 as cv
from data.base import BaseMMSeg
from config import dataset_dir

from mmseg.datasets import DATASETS
from mmseg.datasets.custom import CustomDataset


from pathlib import Path
from data import utils

KAGGLE_CONFIG_PATH = Path(__file__).parent / "config" / "kaggle.py"
KAGGLE_CATS_PATH = Path(__file__).parent / "config" / "kaggle.yml"


class KaggleDataset(BaseMMSeg):
    # def __init__(self, prefix, with_gt=True, use_canny=False):
    def __init__(self, image_size, crop_size, split, **kwargs):
        super().__init__(
            image_size,
            crop_size,
            split,
            KAGGLE_CONFIG_PATH,
            **kwargs,
        )
        self.names, self.colors = utils.dataset_cat_description(KAGGLE_CATS_PATH)
        self.n_cls = 2
        self.ignore_label = 0
        self.reduce_zero_label = True

    def update_default_config(self, config):
        root_dir = dataset_dir()
        path = Path(root_dir) / "kaggle"
        config.data_root = path
        if self.split == "train":
            config.data.train.data_root = path
        elif self.split == "trainval":
            config.data.trainval.data_root = path
        elif self.split == "val":
            config.data.val.data_root = path
        elif self.split == "test":
            config.data.test.data_root = path
        config = super().update_default_config(config)
        return config


@DATASETS.register_module('kaggle')
class KaggleDataset2(CustomDataset):
    """Chase_db1 dataset.

    In segmentation map annotation for Chase_db1, 0 stands for background,
    which is included in 2 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_1stHO.png'.
    """

    def __init__(self, **kwargs):
        super(KaggleDataset2, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            classes=('road', 'bg'),
            palette=[[255,], [0,]],
            **kwargs)
        assert osp.exists(self.img_dir)
