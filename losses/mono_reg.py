import torch
def mono_loss(seq_mean, warmup_steps, global_step):
    """Penalise decreasing trend: ReLU(mean[t-1]-mean[t])."""
    if global_step < warmup_steps:
        return torch.tensor(0., device=seq_mean.device)
    diff = seq_mean[:,:-1] - seq_mean[:,1:]
    return torch.relu(diff).mean()
