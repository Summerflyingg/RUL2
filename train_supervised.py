import pytorch_lightning as pl
from datasets.cmapss_dm import CMAPSSDataModule
from lit_modules.finetune_lit import FineTuneLit
import torch
from utils.adjacency import build_pearson_adj

if __name__ == '__main__':
    ckpt = 'ssl_ckpt.ckpt'  # replace with real path
    dm = CMAPSSDataModule(data_root='data', batch_size=32, stage='supervised')
    adj = build_pearson_adj('data', subset='FD001')
    model = FineTuneLit(adj=adj, encoder_ckpt=ckpt)
    trainer = pl.Trainer(max_epochs=5, accelerator='cpu', limit_train_batches=5)
    trainer.fit(model, dm)
