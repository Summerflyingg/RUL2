import torch
import torch.nn.functional as F

def ua_infonce(z1, z2, dropout_p=0.2, repeats=5, temperature=0.1):
    """Uncertainty‑aware InfoNCE.

    Args:
        z1, z2: (B,D) positive pairs
    Returns:
        scalar loss
    """
    B, D = z1.shape
    zs = []
    for _ in range(repeats):
        zs.append(F.dropout(z1, dropout_p, training=True))
    z1s = torch.stack(zs)  # K,B,D
    sigma2 = torch.var(z1s, dim=0).mean(-1, keepdim=True)  # B,1
    w = torch.exp(-sigma2.detach())

    z = torch.cat([z1, z2], 0)
    sim = torch.mm(z, z.t()) / temperature
    labels = torch.arange(B, device=z1.device)
    logits = sim[:B, B:]  # positive
    logit_all = torch.cat([sim[:B, :B], sim[:B, B:]], 1)
    ce = F.cross_entropy(logit_all, labels.repeat(1), reduction='none')
    return (w.squeeze()*ce).mean()
