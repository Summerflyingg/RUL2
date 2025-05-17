# lit_modules/finetune_lit.py
import torch
import pytorch_lightning as pl

from models.rul_net import RULNet
from utils.metrics import nasa_score

class FineTuneLit(pl.LightningModule):
    def __init__(self, adj, encoder_ckpt: str, lr: float = 2e-3):
        super().__init__()
        self.model = RULNet(adj=adj)
        ckpt = torch.load(encoder_ckpt, map_location='cpu')
        self.model.load_state_dict(ckpt['state_dict'], strict=False)

        # 冻结编码器，只让 MLP 头训练
        for m in [self.model.gat,
                  self.model.time_branch,
                  self.model.freq_branch]:
            for p in m.parameters():
                p.requires_grad = False

        self.lr = lr
        self.save_hyperparameters(ignore=['adj'])  # Lightning log

        # 累计 NASA score 用
        self.val_preds, self.val_trues = [], []

    # ---------- forward ----------
    def forward(self, x_time, x_spec):
        y, _, _ = self.model(x_time, x_spec)
        return y

    # ---------- training ----------
    def training_step(self, batch, batch_idx):
        x_time, x_spec, rul = batch
        pred = self(x_time, x_spec).squeeze()
        rmse = torch.sqrt(((pred - rul) ** 2).mean())
        self.log("train/RMSE", rmse, prog_bar=True)
        return rmse   # 直接用 RMSE 当 loss

    # ---------- validation ----------
    def validation_step(self, batch, batch_idx):
        x_time, x_spec, rul = batch
        pred = self(x_time, x_spec).squeeze()
        self.val_preds.append(pred.detach())
        self.val_trues.append(rul.detach())

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds)
        trues = torch.cat(self.val_trues)
        rmse = torch.sqrt(((preds - trues) ** 2).mean())
        score = nasa_score(preds, trues)

        self.log_dict({
            "val/RMSE": rmse,
            "val/NASA": score
        }, prog_bar=True)

        # 清空缓存
        self.val_preds.clear()
        self.val_trues.clear()

    # ---------- optim ----------
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
