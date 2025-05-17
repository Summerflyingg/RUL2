import torch
import torch.nn.functional as F
def stft_log(x, n_fft=32, hop_length=2):
    # x: (B,14,L)
    B,C,L = x.shape
    xs = x.reshape(B*C, L)
    spec = torch.stft(xs, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft, device=x.device), return_complex=True)
    mag = torch.log1p(spec.abs())
    spec = mag.reshape(B, C, mag.shape[-2], mag.shape[-1])
    return spec
