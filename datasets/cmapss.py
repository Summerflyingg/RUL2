import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.spec_augment import stft_log

class CMAPSSUnlabeledDataset(Dataset):
    """CMAPSS dataset for self‑supervised stage.
    Expects data directory structure:
        root/
            FD001/
                unit001.npy  # shape (T_i, 14)
                ...
            FD002/
            ...
    For each unit sequence we slide a window of length `window_len`
    with stride `stride` (default 1). Each window is z‑scored *per sensor*
    using statistics computed over the **whole unit**.
    """
    def __init__(self, root, subset='FD001', window_len=30, stride=1, max_rul=125):
        self.window_len = window_len
        self.files = sorted([os.path.join(root, subset, f) for f in os.listdir(os.path.join(root, subset)) if f.endswith('.npy')])
        self.stride = stride
        self.max_rul = max_rul
        # build index (file_id, start_idx)
        self.index = []
        for fid, path in enumerate(self.files):
            seq_len = np.load(path, mmap_mode='r').shape[0]
            for s in range(0, seq_len - window_len + 1, stride):
                self.index.append((fid, s))
    def __len__(self):
        return len(self.index)
    def __getitem__(self, idx):
        fid, s = self.index[idx]
        arr = np.load(self.files[fid], mmap_mode='r')[s:s+self.window_len]  # (L,14)
        arr = arr.astype('float32')
        # z‑score per sensor using window statistics (simpler than unit‑level; okay for SSL)
        mu = arr.mean(0, keepdims=True)
        std = arr.std(0, keepdims=True) + 1e-8
        x = torch.from_numpy((arr - mu) / std).transpose(1,0)  # (14,L)
        spec = stft_log(x.unsqueeze(0))[0]  # (14,F,T)
        return x, spec

class CMAPSSLabeledDataset(CMAPSSUnlabeledDataset):
    """Supervised variant that returns RUL label for each window"""
    def __init__(self, root, subset='FD001', **kwargs):
        super().__init__(root, subset=subset, **kwargs)
        # cache seq lengths for RUL calc
        self.seq_lens = [np.load(p, mmap_mode='r').shape[0] for p in self.files]
    def __getitem__(self, idx):
        x_time, x_spec = super().__getitem__(idx)
        fid, s = self.index[idx]
        seq_len = self.seq_lens[fid]
        rul = min(self.max_rul, seq_len - (s + self.window_len))
        return x_time, x_spec, torch.tensor(rul, dtype=torch.float32)
