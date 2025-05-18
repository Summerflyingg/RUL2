import torch
import pytorch_lightning as pl
from torch import nn
from models.rul_net import RULNet
from losses.ua_infonce import ua_infonce
from losses.mono_reg import mono_loss

class SSLLit(pl.LightningModule):
    def __init__(self, adj):
        super().__init__()
        self.model = RULNet(adj=adj)
        self.lr = 1e-4
        self.save_hyperparameters()

    def forward(self, x_time, x_spec):
        _, h_t, h_f = self.model(x_time, x_spec)
        return h_t, h_f

    def training_step(self, batch, batch_idx):
        x_time, x_spec = batch
        h_t, h_f = self(x_time, x_spec)
        loss_c = ua_infonce(h_t, h_f)
        seq_mean = x_time.mean(1)  # (B,L)
        loss_mono = mono_loss(seq_mean, warmup_steps=50, global_step=self.global_step)
        loss = loss_c + 0.05*loss_mono
        self.log_dict({'ssl_loss':loss, 'contrast':loss_c, 'mono':loss_mono})
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
