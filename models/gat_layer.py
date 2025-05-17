import torch
from torch import nn
try:
    from torch_geometric.nn import GATConv
except ImportError:
    GATConv = None

class SensorGAT(nn.Module):
    """Graph‑attention layer for 14 correlated sensors.

    Args:
        in_dim: input feature dim (window length)
        out_dim: hidden dim
        adj (torch.Tensor): 14×14 binary adjacency (no self loops)
    """
    def __init__(self, in_dim: int, out_dim: int, adj: torch.Tensor):
        super().__init__()
        if GATConv is None:
            raise ImportError('torch_geometric is required for SensorGAT')
        # register fixed edge_index from adjacency
        src, dst = torch.nonzero(adj, as_tuple=True)
        self.register_buffer('edge_index', torch.stack([src, dst], dim=0))
        self.gat = GATConv(in_dim, out_dim, add_self_loops=False, heads=1, bias=False)

    def forward(self, x):
        # x: (B, 14, L) -> (B*L, 14, 1)
        B, N, L = x.shape
        x = x.permute(0, 2, 1).reshape(B*L, N, 1)  # feat=1
        out = self.gat(x.squeeze(-1), self.edge_index)  # (B*L, out_dim)
        out = out.view(B, L, -1).permute(0, 2, 1)  # (B, out_dim, L)
        return out
