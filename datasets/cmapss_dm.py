import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.cmapss import CMAPSSUnlabeledDataset, CMAPSSLabeledDataset

class CMAPSSDataModule(pl.LightningDataModule):
    def __init__(self, data_root:str, batch_size:int=32, num_workers:int=4, window_len:int=30, subset:str='FD001', stage='ssl'):
        super().__init__()
        self.data_root = data_root
        self.bs = batch_size
        self.nw = num_workers
        self.window_len = window_len
        self.subset = subset
        self.stage = stage
    def setup(self, stage=None):
        if self.stage=='ssl':
            self.train_set = CMAPSSUnlabeledDataset(self.data_root, subset=self.subset, window_len=self.window_len)
        elif self.stage=='supervised':
            self.train_set = CMAPSSLabeledDataset(self.data_root, subset=self.subset, window_len=self.window_len)
        else:
            raise ValueError('unknown stage')
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bs, shuffle=True, num_workers=self.nw, pin_memory=True)
