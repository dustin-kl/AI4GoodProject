from argparse import ArgumentParser
from pytorch_lightning.core.mixins import hparams_mixin
from pytorch_lightning.callbacks import ModelCheckpoint
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch
from torch import nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms

from kaggle_dataset import KaggleDataset
from massachusetts_roads_dataset import MassachusettsRoadsDataset
from deepglobe_dataset import DeepglobeDataset
from google_dataset import GoogleDataset

from config import get_hparams_defaults

from utils.metrics import accuracy_fn, patch_accuracy_fn, patch_f1_score
import os
import cv2
import pandas as pd
import numpy as np
from utils.kaggle_data_utils import get_training_augmentation as get_training_augmentation0
from utils.kaggle_data_utils import get_validation_augmentation as get_validation_augmentation0 
from utils.kaggle_data_utils import get_preprocessing as get_preprocessing0
from utils.kaggle_data_utils import load_all_from_path, reverse_one_hot

from utils.massachusetts_roads_data_utils import get_training_augmentation as get_training_augmentation1
from utils.massachusetts_roads_data_utils import get_validation_augmentation as get_validation_augmentation1 
from utils.massachusetts_roads_data_utils import get_preprocessing as get_preprocessing1

from utils.deepglobe_data_utils import get_training_augmentation as get_training_augmentation2
from utils.deepglobe_data_utils import get_validation_augmentation as get_validation_augmentation2 
from utils.deepglobe_data_utils import get_preprocessing as get_preprocessing2

from losses import DiceLoss, DiceBCELoss, FocalLoss

from utils.submission_utils import create_submission

from glob import glob

import segmentation_models_pytorch as smp

import warnings
warnings.filterwarnings("ignore")

        
class DeepLabV3Plus(pl.LightningModule):
    def __init__(self, lr, encoder, encoder_weights, num_classes, activation):
        super().__init__()
        self.lr = lr

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder, 
            encoder_weights=encoder_weights, 
            classes=num_classes, 
            activation=activation,
        )

        self.metric_fns = {
          'patch_f1_score': patch_f1_score,
          'patch_acc': patch_accuracy_fn
        }

    def forward(self, x):

        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss_fn1 = nn.BCELoss()
        loss_fn2 = DiceLoss()
        loss_fn3 = FocalLoss()
        w1 = 0.3
        w2 = 1.0
        w3 = 0.7

        loss = w1*loss_fn1(y_hat, y) + w2*loss_fn2(y_hat, y) + w3*loss_fn3(y_hat, y)

        # log partial metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss_fn1 = nn.BCELoss()
        loss_fn2 = DiceLoss()
        loss_fn3 = FocalLoss()

        w1 = 0.3
        w2 = 1.0
        w3 = 0.7

        loss = w1*loss_fn1(y_hat, y) + w2*loss_fn2(y_hat, y) + w3*loss_fn3(y_hat, y)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for k, fn in self.metric_fns.items():
            self.log('val_'+k, fn(y_hat, y, 0.25), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        SAVE_IMAGES = True
        if SAVE_IMAGES and batch_idx == 0:
            save_dir = os.path.join('lightning_logs', 'example_outputs_val')
            os.makedirs(save_dir, exist_ok=True)
            out_filename = os.path.join(save_dir, f'result_{batch_idx:05d}.obj')

            images_in = x[0].clone().cpu().numpy().transpose(1, 2, 0) * 255
            images_in = np.clip(images_in, 0, 255).astype(np.uint8)

            cv2.imwrite(
                os.path.join(save_dir, f'result_in_{batch_idx:05d}.jpg'),
                cv2.cvtColor(images_in, cv2.COLOR_BGR2RGB)
            )

            images_gt = y[0].clone().cpu().numpy().transpose(1, 2, 0) * 255
            images_gt = np.clip(images_gt, 0, 255).astype(np.uint8)

            cv2.imwrite(
                os.path.join(save_dir, f'result_gt_{batch_idx:05d}.jpg'),
                cv2.cvtColor(images_gt, cv2.COLOR_BGR2RGB)
            )

            images_y_hat = y_hat[0].clone().cpu().numpy().transpose(1, 2, 0) * 255
            images_y_hat = np.clip(images_y_hat, 0, 255).astype(np.uint8)

            cv2.imwrite(
                os.path.join(save_dir, f'result_pred_{batch_idx:05d}.jpg'),
                cv2.cvtColor(images_y_hat, cv2.COLOR_BGR2RGB)
            )


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def cli_main():
    
    hparams = get_hparams_defaults()

    pl.seed_everything(hparams.SEED_VALUE)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ENCODER = 'mobilenet_v2' # can also use 'tu-xception65' or 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    if hparams.DATASET.CHOICE == 0:
        train_dataset = KaggleDataset(
            'data/kaggle_dataset/training', 
            use_patches=False,
            augmentation=get_training_augmentation0(),
            preprocessing=get_preprocessing0(preprocessing_fn)
        )
        val_dataset = KaggleDataset(
            'data/kaggle_dataset/validation',
            use_patches=False,
            augmentation=None,
            preprocessing=get_preprocessing0(preprocessing_fn)
        )

        train_loader = DataLoader(train_dataset, batch_size=hparams.DATASET.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hparams.DATASET.BATCH_SIZE, shuffle=True)
    
    elif hparams.DATASET.CHOICE == 1:
        DATA_DIR = 'data/massachusetts_roads_dataset/tiff/'

        x_train_dir = os.path.join(DATA_DIR, 'train')
        y_train_dir = os.path.join(DATA_DIR, 'train_labels')

        x_valid_dir = os.path.join(DATA_DIR, 'val')
        y_valid_dir = os.path.join(DATA_DIR, 'val_labels')

        x_test_dir = os.path.join(DATA_DIR, 'test')
        y_test_dir = os.path.join(DATA_DIR, 'test_labels')



        class_dict = pd.read_csv("data/massachusetts_roads_dataset/label_class_dict.csv")
        # Get class names
        class_names = class_dict['name'].tolist()
        # Get class RGB values
        class_rgb_values = class_dict[['r','g','b']].values.tolist()

        print('All dataset classes and their corresponding RGB values in labels:')
        print('Class Names: ', class_names)
        print('Class RGB values: ', class_rgb_values)

        # Useful to shortlist specific classes in datasets with large number of classes
        select_classes = ['background', 'road']

        # Get RGB values of required classes
        select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
        select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

        print('Selected classes and their corresponding RGB values in labels:')
        print('Class Names: ', class_names)
        print('Class RGB values: ', class_rgb_values)

        train_dataset = MassachusettsRoadsDataset(
            x_train_dir, y_train_dir,
            augmentation=get_training_augmentation1(),
            preprocessing=get_preprocessing1(preprocessing_fn),
            class_rgb_values=select_class_rgb_values,
        )

        val_dataset = MassachusettsRoadsDataset(
            x_valid_dir, y_valid_dir,
            augmentation=get_validation_augmentation1(), 
            preprocessing=get_preprocessing1(preprocessing_fn),
            class_rgb_values=select_class_rgb_values,
        )

        # Get train and val data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)

    elif hparams.DATASET.CHOICE == 2:
        DATA_DIR = 'data/deepglobe_dataset/'

        metadata_df = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))
        metadata_df = metadata_df[metadata_df['split']=='train']
        metadata_df = metadata_df[['image_id', 'sat_image_path', 'mask_path']]
        metadata_df['sat_image_path'] = metadata_df['sat_image_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
        metadata_df['mask_path'] = metadata_df['mask_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
        # Shuffle DataFrame
        metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)

        # Perform 90/10 split for train / val
        valid_df = metadata_df.sample(frac=0.1, random_state=hparams.SEED_VALUE)
        train_df = metadata_df.drop(valid_df.index)

        class_dict = pd.read_csv(os.path.join(DATA_DIR, 'class_dict.csv'))
        # Get class names
        class_names = class_dict['name'].tolist()
        # Get class RGB values
        class_rgb_values = class_dict[['r','g','b']].values.tolist()

        print('All dataset classes and their corresponding RGB values in labels:')
        print('Class Names: ', class_names)
        print('Class RGB values: ', class_rgb_values)

        # Useful to shortlist specific classes in datasets with large number of classes
        select_classes = ['background', 'road']

        # Get RGB values of required classes
        select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
        select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

        print('Selected classes and their corresponding RGB values in labels:')
        print('Class Names: ', class_names)
        print('Class RGB values: ', class_rgb_values)

        train_dataset = DeepglobeDataset(
            train_df,
            augmentation=get_training_augmentation1(),
            preprocessing=get_preprocessing2(preprocessing_fn),
            class_rgb_values=select_class_rgb_values,
        )

        val_dataset = DeepglobeDataset(
            valid_df,
            augmentation=get_validation_augmentation1(), 
            preprocessing=get_preprocessing2(preprocessing_fn),
            class_rgb_values=select_class_rgb_values,
        )

        # Get train and val data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)

    elif hparams.DATASET.CHOICE == 3:
        train_dataset = GoogleDataset(
            '../../road-segmentation-eth-cil-2020/cil-road-segmentation/google-maps-data', 
            use_patches=False,
            augmentation=get_training_augmentation0(),
            preprocessing=get_preprocessing0(preprocessing_fn)
        )
        val_dataset = GoogleDataset(
            'data/kaggle_dataset/training',
            use_patches=False,
            augmentation=None,
            preprocessing=get_preprocessing0(preprocessing_fn)
        )

        train_loader = DataLoader(train_dataset, batch_size=hparams.DATASET.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hparams.DATASET.BATCH_SIZE, shuffle=True)
    # ------------
    # model
    # ------------

    model = DeepLabV3Plus(
      lr=hparams.OPTIMIZER.LR,
      encoder=ENCODER,
      encoder_weights=ENCODER_WEIGHTS,
      num_classes=1,
      activation='sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
    )

    # TRAINING.PRETRAINED_LIT points to the checkpoint files trained using this repo
    # This has a separate cfg value since in some cases we use checkpoint files from different repos
    if hparams.TRAINING.PRETRAINED_LIT is not None:
        print(f'Loading pretrained model from {hparams.TRAINING.PRETRAINED_LIT}')
        ckpt = torch.load(hparams.TRAINING.PRETRAINED_LIT)['state_dict']

        model.load_state_dict(ckpt, strict=False)
        print(f'Loading done.')

    # When training gets interrupted you can resume it from a checkpoint by setting
    # the hparams.TRAINING.RESUME param
    if hparams.TRAINING.RESUME is not None:
        print('Loading resuming info')
        resume_ckpt = torch.load(hparams.TRAINING.RESUME)
        print('Done loading resuming info')

    # ------------
    # training
    # ------------
    # trainer = pl.Trainer.from_argparse_args(args)
    # this callback saves best 2 checkpoint based on the validation loss
    ckpt_callback = ModelCheckpoint(
        monitor=hparams.TRAINING.METRIC_TO_MONITOR,
        verbose=True,
        save_top_k=2, # reduce this if you don't have enough storage
        mode='max',
    )
    
    if hparams.RUN_TRAINING:
        print('Starting training')
        trainer = pl.trainer = pl.Trainer(
            gpus=1 if torch.cuda.is_available() else 0,
            max_epochs=hparams.TRAINING.MAX_EPOCHS, # total number of epochs
            callbacks=[ckpt_callback],
            resume_from_checkpoint=hparams.TRAINING.RESUME
        )
        trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    if hparams.RUN_TEST:
        print('Running test')
        test_path = 'data/kaggle_dataset/test'
        
        PATCH_SIZE = 16
        CUTOFF = 0.25
        model.eval()

        if hparams.TEST_AUGM:
            i_path = './lightning_logs/results_aug'
            o_path = './lightning_logs/results'

            os.system(f'rm -r {o_path}')
            os.makedirs(o_path, exist_ok=True)
            
            os.system(f'rm -r {i_path}')
            os.makedirs(i_path, exist_ok=True)
            test_dataset = GoogleDataset(
                test_path, 
                use_patches=False,
                augmentation=None,
                preprocessing=get_preprocessing0(preprocessing_fn),
                is_test=True,
                test_augm=True
            )

            test_filenames = sorted(glob(os.path.join(test_path, 'images_aug') + '/*.png'))

            for idx in range(len(test_dataset)):
                print(f'Predicting {idx}/{len(test_dataset)}')
                t = test_dataset[idx]
                t = torch.unsqueeze(t, dim=0)
                pred = model(t).detach().cpu().numpy()

                pred = np.clip(pred[0][0] * 255.0, 0, 255).astype(np.uint8)

                test_filename = os.path.basename(test_filenames[idx])

                cv2.imwrite(os.path.join(i_path, test_filename), pred)

        else:
            test_dataset = KaggleDataset(
                test_path, 
                use_patches=False,
                augmentation=None,
                preprocessing=get_preprocessing0(preprocessing_fn),
                is_test=True
            )
            test_filenames = sorted(glob(os.path.join(test_path, 'images') + '/*.png'))
            test_images = load_all_from_path(os.path.join(test_path, 'images')) 
            batch_size = test_images.shape[0]
            size = test_images.shape[1:3]

            test_pred = []
            for idx in range(len(test_dataset)):
                print(f'Predicting {idx}/{len(test_dataset)}')
                t = test_dataset[idx]
                t = torch.unsqueeze(t, dim=0)
                pred = model(t).detach().cpu().numpy()
                test_pred.append(pred)

            test_pred = np.concatenate(test_pred, 0)
            test_pred= np.moveaxis(test_pred, 1, -1)  # CHW to HWC

            test_pred = test_pred.reshape((-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE))
            test_pred = np.moveaxis(test_pred, 2, 3)
            test_pred = np.round(np.mean(test_pred, (-1, -2)) > CUTOFF)
            create_submission(test_pred, test_filenames, submission_filename='deeplabv3p_submission.csv')
            print('Submission file created')


if __name__ == '__main__':
    cli_main()