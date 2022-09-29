import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights

class Resnet18(pl.LightningModule):
    def __init__(self):
         super().__init__()
         self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)
         self.output_layer = nn.Linear(1000,10)

    def forward(self, x):
        output = self.resnet18(x)
        out = self.output_layer(output)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.forward(x)    
        out = out.view(out.size(0), -1)
        loss = F.cross_entropy(out, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self.forward(x)
        out = out.view(out.size(0), -1)
        loss = F.cross_entropy(out, y)
        _, predicted = out.max(1)
        correct = predicted.eq(y).sum().item()
        self.log('val_loss', loss)
        return correct, out.size(0)
    
    def validation_epoch_end(self, validation_step_outputs):
        corrects = (list(zip(*validation_step_outputs))[0])
        total = (list(zip(*validation_step_outputs))[1])
        self.log('val_acc', sum(corrects)/sum(total)*100)
