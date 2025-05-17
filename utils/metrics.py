# utils/metrics.py
import torch

def nasa_score(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """
    pred, true: (N,) 1-D tensor, already moved to same device.
    Return: scalar tensor
    """
    d = pred - true
    pos = torch.exp(-d.clamp(min=0) / 13) - 1          # over-estimate part
    neg = torch.exp( d.clamp(max=0).abs() / 10) - 1    # under-estimate part
    return (pos + neg).sum()
