import torch
from torch import nn
from .gat_layer import SensorGAT
from .patchtst import PatchTSTBranch, PatchEmbedding1D, PatchEmbedding3D

class RULNet(nn.Module):
    def __init__(self, window_len=30, d_model=128, adj=None):
        super().__init__()
        # Graph layer
        self.gat = SensorGAT(in_dim=1, out_dim=1, adj=adj)  # feature dim=1 sensor reading
        # Branches
        self.time_branch = PatchTSTBranch(
            PatchEmbedding1D(patch_len=4, in_ch=14, d_model=d_model), d_model=d_model)
        self.freq_branch = PatchTSTBranch(
            PatchEmbedding3D(patch_size=(4,4), in_ch=1, d_model=d_model), d_model=d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model*2),
            nn.Linear(d_model*2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x_time, x_spec):
        # x_time: (B,14,L) already z-score
        x_gat = self.gat(x_time)  # (B,14,L)
        h_t = self.time_branch(x_gat)
        h_f = self.freq_branch(x_spec)
        h = torch.cat([h_t,h_f], dim=-1)
        return self.head(h).squeeze(-1), h_t, h_f
