import os
import numpy as np
import torch


def build_pearson_adj(root: str, subset: str = "FD001", thresh: float = 0.2) -> torch.Tensor:
    """Compute sensor adjacency via Pearson correlation."""
    subset_dir = os.path.join(root, subset)
    files = [os.path.join(subset_dir, f) for f in os.listdir(subset_dir) if f.endswith('.npy')]
    all_arr = [np.load(f, mmap_mode='r') for f in files]
    arr = np.concatenate(all_arr, axis=0)
    corr = np.corrcoef(arr, rowvar=False)
    adj = (np.abs(corr) > thresh).astype('float32')
    np.fill_diagonal(adj, 0)
    return torch.tensor(adj)
