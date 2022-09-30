import os

import torch
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from resnet18 import Resnet18
from vit import ViT
from pytorch_lightning.loggers import WandbLogger

seed_everything(1234)

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./data")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)

train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

cifar10_dm = CIFAR10DataModule(
    data_dir=PATH_DATASETS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

cifar10_dm.train_transforms = train_transforms
cifar10_dm.test_transforms = test_transforms
cifar10_dm.val_transforms  = test_transforms

# model
model = ViT(dim=192, patch_size=4, seq_len=64, depth=12, heads=3, dropout=0.1)
#model = Resnet18()

wandb_logger = WandbLogger(project='2022320002_bokyeong_pl_cifar10', name='ViT-Tiny/4')

trainer = Trainer(
    max_epochs=200,
    accelerator="auto",
    logger=wandb_logger,
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
)

trainer.fit(model, cifar10_dm)
trainer.test(model, datamodule=cifar10_dm)
