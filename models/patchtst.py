import math
import torch
from torch import nn
from torch.nn import functional as F

class PatchEmbedding1D(nn.Module):
    def __init__(self, patch_len: int, in_ch: int, d_model: int):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, d_model, kernel_size=patch_len, stride=patch_len)
    def forward(self, x):
        # x: (B,C,L)
        return self.proj(x).permute(0,2,1)  # B, Patches, D

class PatchEmbedding3D(nn.Module):
    """For STFT freq×time patches."""
    def __init__(self, patch_size=(4,4), in_ch=1, d_model=128):
        super().__init__()
        self.proj = nn.Conv3d(in_ch, d_model, kernel_size=patch_size+(patch_size[1],),
                              stride=patch_size+(patch_size[1],))
    def forward(self, x):
        # x: (B,C,F,T)
        out = self.proj(x.unsqueeze(1))  # B,d_model,f',t'
        B,D,Fp,Tp = out.shape
        out = out.reshape(B, D, Fp*Tp).permute(0,2,1)  # B,patches,D
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=128, depth=2, nhead=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, batch_first=True)
        self.tr = nn.TransformerEncoder(encoder_layer, depth)
        self.cls_token = nn.Parameter(torch.randn(1,1,d_model))
        self.pos = nn.Parameter(torch.randn(1,512,d_model))
    def forward(self, x):
        B = x.size(0)
        cls = self.cls_token.expand(B,-1,-1)
        x = torch.cat([cls,x],1)
        x = x + self.pos[:,:x.size(1)]
        x = self.tr(x)
        return x[:,0]  # CLS

class PatchTSTBranch(nn.Module):
    def __init__(self, patch_func, d_model=128):
        super().__init__()
        self.embed = patch_func
        self.enc = TransformerEncoder(d_model=d_model)
    def forward(self, x):
        tokens = self.embed(x)
        h = self.enc(tokens)
        return h
