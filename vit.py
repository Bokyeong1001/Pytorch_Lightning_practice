import torch
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from einops import rearrange

class PreNorm(pl.LightningModule):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fn = fn
    def forward(self,x):
        return self.fn(self.norm(x))

class FeedForward(pl.LightningModule):
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
      nn.Linear(dim, mlp_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(mlp_dim, dim),
      nn.Dropout(dropout))
    def forward(self,x):
        return self.ffn(x)

class Attention(pl.LightningModule):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim*3)
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax()
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self,x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(pl.LightningModule):
    def __init__(self,dim,depth,heads,dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = int(dim/heads), dropout = dropout)),
                PreNorm(dim, FeedForward(dim, int(dim*4), dropout = dropout))
            ]))
    def forward(self,x):
        for attn,ff in self.layers:
            x = attn(x)+x
            x = ff(x)+x
        return x

class ViT(pl.LightningModule):
    def __init__(self, dim, patch_size, seq_len, depth, heads, dropout):
         super().__init__()
         self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=patch_size, stride=patch_size)
         self.position_embedding = nn.Parameter(torch.randn(1, seq_len, dim))
         self.transformer=Transformer(dim, depth, heads, dropout)
         self.out_layer = nn.Linear(dim, 10)

    def forward(self, x):
        embedding = self.patch_embedding(x) #B,D,H,W
        embedding = embedding.flatten(2) #B,D,H*W
        embedding = embedding.transpose(-1, -2) #B,H*W,D
        input = embedding + self.position_embedding
        out = self.transformer(input)
        output = self.out_layer(out)
        return output

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
