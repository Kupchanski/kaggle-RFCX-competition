import os
import cv2
import glob
import argparse
import numpy as np
import pandas as pd
import albumentations as A
import albumentations.pytorch as AT
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

import matplotlib
matplotlib.use('Agg')
import seaborn as sn
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from dataset import MelTrainDataset, MelInferenceDataset
from loss import lsep_loss_stable, focal_loss
from metrics import lwlrap, lrap
from batch_mixer import BatchMixer


class MixedAdaptivePool2d(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.p1 = torch.nn.AdaptiveAvgPool2d(1)
        self.p2 = torch.nn.AdaptiveMaxPool2d(1)
        self.register_parameter("weight", torch.nn.Parameter(torch.tensor(1.)))

    def forward(self, x):
        return self.weight * self.p1(x) + self.p2(x)


class Model(pl.LightningModule):

    def __init__(self, arch, n_classes=24, learning_rate=3e-4, mixup_p=0., y_coord=False, num_epochs=30, saving_args=None):

        super().__init__()
        self.save_hyperparameters(saving_args)

        self.mixup_p = mixup_p
        self.arch = arch
        self.n_classes = n_classes
        self.y_coord = y_coord
        self.num_epochs = num_epochs
        
        in_channels = 1 if not y_coord else 2
        weights = "imagenet"
        if "efficientnet" in arch:
            weights = "noisy-student"
        self.model = smp.encoders.get_encoder(name=arch, in_channels=in_channels, weights="imagenet")
        self.head = smp.base.heads.ClassificationHead(
            in_channels=self.model.out_channels[-1],
            classes=n_classes,
            pooling="avg",
            dropout=0.2,
        )

        self.head[0] = MixedAdaptivePool2d()

    def _forward(self, x):
        
        if self.y_coord:
        # add y-position coordinate gradient 
            b, c, h, w = x.shape
            grad = torch.linspace(0, 0.1, h, dtype=x.dtype).expand(b, 1, w, h).transpose(2, 3).to(x.device)
            x = torch.cat([x, grad], axis=1)

        x = self.model(x)[-1]
        x = self.head(x)
        return x

    def forward(self, x):
        return self._forward(x)

    def training_step(self, batch, batch_idx):
        images = batch["image"].float()
        y_true = batch["label"]
        images, y_true = BatchMixer(p=self.mixup_p)(images=images, labels=y_true)
        y_pred = self._forward(images)
        loss = lsep_loss_stable(y_pred, y_true)
        # loss = focal_loss(y_pred, y_true)
        # loss = 100 * focal_loss(y_pred, y_true) + lsep_loss_stable(y_pred, y_true)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images = batch["image"].float()
        y_true = batch["label"]
        y_pred = self._forward(images)
        recording_id = batch["recording_id"]
        return dict(
            y_pred=y_pred.sigmoid().cpu().numpy().round(4),
            y_true=y_true.cpu().numpy(),
            recording_id=recording_id,
        )

    def test_step(self, batch, batch_idx):
        recording_id = batch["recording_id"]
        images = batch["image"].float()
        y_pred = self._forward(images)
        return dict(
            recording_id=recording_id,
            y_pred=y_pred.sigmoid().cpu().numpy().round(4).astype("float16"),
        )
    
    def validation_epoch_end(self, validation_step_outputs):
        
        all_data = []
        for data in validation_step_outputs:
            n_samples = data["y_pred"].shape[0]
            for i in range(n_samples):
                all_data.append({
                    "recording_id": data["recording_id"][i],
                    "y_true": data["y_true"][i],
                    "y_pred": data["y_pred"][i],
                })
        
        df = pd.DataFrame(all_data)
        y_true = df.groupby("recording_id")["y_true"].apply(lambda x: np.stack(x, axis=1).max(axis=1))
        y_pred = df.groupby("recording_id")["y_pred"].apply(lambda x: np.stack(x, axis=1).max(axis=1))

        y_true = np.stack(y_true.values)
        y_pred = np.stack(y_pred.values)

        # compute main metric
        lwlrap_metric = lwlrap(y_true, y_pred)
        self.log(name="lwlrap", value=lwlrap_metric, prog_bar=True)

        # compute accuracy
        gt = np.argmax(y_true, axis=1)
        pr = np.argmax(y_pred, axis=1)
        acc = (gt == pr).sum() * 1.0 / y_true.shape[0]
        self.log(name="acc", value=acc, prog_bar=True)

        # compute f1 score per class
        for cls_id in range(self.n_classes):
            cls_gt = gt == cls_id
            cls_pr = pr == cls_id
            cls_f1 = ((2.0 * cls_pr * cls_gt).sum()) / (cls_gt.sum() + cls_pr.sum())
            self.log(name="f1_{}".format(str(cls_id).zfill(2)), value=cls_f1, prog_bar=True)

        # compute confusion matrix
        y_pred_round = np.zeros([pr.size, self.n_classes])
        y_pred_round[np.arange(pr.size), pr] = 1.
        conf_matrix = y_true.T @ y_pred_round
        fig = plt.figure(figsize=(12, 10))
        ax = sn.heatmap(conf_matrix, cmap='Oranges', annot=True, linewidths=1, linecolor="white", square=True)
        fig.savefig(os.path.join(self.logger.log_dir, "cm_{}.png".format(str(self.current_epoch).zfill(2))))
        self.logger.experiment.add_figure("conf_matrix", fig, global_step=0)
    
    def test_epoch_end(self, test_step_outputs):

        all_data = []
        for data in test_step_outputs:
            n_samples = data["y_pred"].shape[0]
            for i in range(n_samples):
                all_data.append({
                    "recording_id": data["recording_id"][i],
                    "y_pred": data["y_pred"][i],
                })
        
        df = pd.DataFrame(all_data)
        df = pd.DataFrame(df.groupby("recording_id")["y_pred"].apply(lambda x: np.stack(x, axis=1).max(axis=1)))
        df = pd.DataFrame(df.y_pred.to_list(), columns=[f"s{i}" for i in range(self.n_classes)], index=df.index).reset_index()
        df.to_csv(os.path.join(self.logger.log_dir, "sub.csv"), index=False)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # opt = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9, weight_decay=1e-5)
        # sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=self.num_epochs, T_mult=1)
        sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=8, T_mult=1)
        return [opt], [sch]


def main(args):

    exps_name = "debug" if args.debug or args.epochs == 1 else "season-02"
    exps_name = "debug"
    logger = pl.loggers.TensorBoardLogger(args.log_dir, name=exps_name, log_graph=False)
    checkpoints_dir = f"{logger.log_dir}"
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=checkpoints_dir, verbose=True, mode="max", monitor="lwlrap")
    callbacks = [
        pl.callbacks.LearningRateMonitor()
    ]

    trainer = pl.Trainer(
        gpus=args.gpus, 
        checkpoint_callback=checkpoint_callback,
        weights_summary=None,
        benchmark=True,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=callbacks,
        precision=32,
        fast_dev_run=args.debug,
    )

    model = Model(
        arch=args.arch,
        learning_rate=args.lr,
        mixup_p=args.mixup_p,
        y_coord=args.y_coord,
        num_epochs=args.epochs,
        saving_args=args,
    )

    # define dataloaders
    h, w = args.sample_size
    train_transform = A.Compose(
        [
            A.OneOf([
                A.CropNonEmptyMaskIfExists(h, w, p=1.),
                A.RandomCrop(h, w, p=0.5),
            ], p=1.0),

            A.Normalize(mean=[0], std=[-80.]),

            # A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=-80, p=1),
            # A.Downscale(scale_max=0.95, scale_min=0.75, interpolation=cv2.INTER_LINEAR, p=0.5),
            
            # A.GaussNoise(var_limit=(0.01, 0.05), p=0.25),
            # A.IAAAdditiveGaussianNoise(loc=0, scale=(0.01, 0.05), p=0.25),

            AT.ToTensorV2(),
        ]
    )
    valid_transform=A.Compose([
        A.Normalize(mean=[0], std=[-80.]),
        AT.ToTensorV2(),
    ])

    df = pd.read_csv(args.csv_path)
    
    train_df = df[(df.fold != 5) & (df.fold != args.fold)]
    valid_df = df[(df.fold != 5) & (df.fold == args.fold)]
    
    assert set(valid_df.recording_id.unique()).intersection(set(train_df.recording_id.unique())) == set()

    valid_paths = [os.path.join(args.images_dir, recording_id + ".tif") for recording_id in valid_df.recording_id.unique()]
    valid_labels = valid_df.groupby("recording_id")["species_id"].apply(list).to_dict()
    
    train_dataset = MelTrainDataset(train_df, args.images_dir, args.masks_dir, num_classes=args.num_classes, transform=train_transform)
    valid_dataset = MelInferenceDataset(
        paths=valid_paths,
        image_width=5626,
        window_width=args.sample_size[1],
        step=args.sample_size[1] // 2,
        transform=valid_transform,
        labels=valid_labels,
    )
    test_dataset = MelInferenceDataset(
        paths=glob.glob(os.path.join(args.test_dir, "*.tif")),
        image_width=5626,
        window_width=args.sample_size[1],
        step=args.sample_size[1] // 2,
        transform=valid_transform,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.bs * 4, shuffle=False, num_workers=12, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs * 8, shuffle=False, num_workers=4, pin_memory=True)

    # training
    trainer.fit(model=model, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader)
    trainer.test(model=model, test_dataloaders=test_dataloader)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default="../data/mel_320_p2/train/")
    parser.add_argument("--masks_dir", type=str, default="../data/mel_320_p2/train_tp_masks/")
    parser.add_argument("--test_dir", type=str, default="../data/mel_320_p2/test/")
    parser.add_argument("--csv_path", type=str, default="../data/train_tp_folds_v3.csv")
    parser.add_argument("--log_dir", type=str, default="../logs/")
    parser.add_argument("--num_classes", type=int, default=24)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--sample_size", nargs=2, type=int, default=(320, 512))
    parser.add_argument("--arch", type=str, default="resnet34")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--gpus", type=str, required=True)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--mixup_p", type=float, default=0.)
    parser.add_argument("--y_coord", type=int, default=0)
    args = parser.parse_args()

    main(args)
