import pytorch_lightning as pl
from lit_modules.ssl_lit import SSLLit
from datasets.cmapss_dm import CMAPSSDataModule
import torch
from utils.adjacency import build_pearson_adj

if __name__ == '__main__':
    dm = CMAPSSDataModule(data_root='data', batch_size=32, stage='ssl')
    adj = build_pearson_adj('data', subset='FD001')
    model = SSLLit(adj=adj)
    trainer = pl.Trainer(max_epochs=1, accelerator='cpu', limit_train_batches=5)
    trainer.fit(model, dm)
