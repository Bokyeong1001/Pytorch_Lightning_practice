import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchmetrics.functional import accuracy

class Resnet18(pl.LightningModule):
    def __init__(self):
         super().__init__()
         self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)
         self.output_layer = nn.Linear(1000,10)

    def forward(self, x):
        output = self.resnet18(x)
        out = self.output_layer(output)
        return F.log_softmax(out, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")
    
