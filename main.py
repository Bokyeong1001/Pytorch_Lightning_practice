
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from vit import ViT
from resnet50 import Resnet18

# data
transform_train = transforms.Compose([
        transforms.RandomResizedCrop((128, 128), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

trainset = CIFAR10(root="./data", train=True, download=True, transform=transform_train)
testset = CIFAR10(root="./data", train=False, download=True, transform=transform_test)

train_loader = DataLoader(trainset, batch_size=512)
val_loader = DataLoader(testset, batch_size=512)

# model
#model = ViT(dim=384, patch_size=16, seq_len=64, depth=6, heads=6, dropout=0.1)
model = Resnet18()

# training
wandb_logger = WandbLogger()
trainer = pl.Trainer(gpus=1,logger=wandb_logger)
trainer.fit(model, train_loader, val_loader)
    
#trainer.validate(dataloaders=val_loader)