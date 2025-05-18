import torch


def mono_loss(seq, warmup_steps: int, global_step: int) -> torch.Tensor:
    """Penalise decreasing trend in a 1-D sequence.

    Args:
        seq: tensor of shape (B, L) representing mean sensor value per time step.
    """

    if global_step < warmup_steps:
        return torch.tensor(0.0, device=seq.device)

    diff = seq[:, :-1] - seq[:, 1:]
    return torch.relu(diff).mean()
