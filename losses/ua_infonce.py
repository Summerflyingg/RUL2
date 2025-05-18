import torch
import torch.nn.functional as F

def ua_infonce(z1, z2, dropout_p=0.2, repeats=5, temperature=0.1):
    """Uncertainty‑aware InfoNCE.

    Args:
        z1, z2: (B,D) positive pairs
    Returns:
        scalar loss
    """
    B, _ = z1.shape
    # variance-based weight
    z1_drop = torch.stack([F.dropout(z1, dropout_p, training=True) for _ in range(repeats)], 0)
    var = z1_drop.var(dim=0).mean(-1)
    weight = torch.exp(-var.detach())  # (B,)

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = torch.mm(z1, z2.t()) / temperature  # (B,B)
    labels = torch.arange(B, device=z1.device)
    ce = F.cross_entropy(logits, labels, reduction='none')
    return (weight * ce).mean()
