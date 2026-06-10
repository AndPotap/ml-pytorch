import torch
from torch.nn import Module


class RoPE(Module):
    def __init__(self, embd: int, max_seq: int):
        super().__init__()
        assert embd % 2 == 0, f"{embd=} is not divisible by 2"
        embd_half = embd // 2
        theta = 10**-(4.0 * torch.arange(embd_half) / embd_half)
        theta = torch.arange(max_seq)[:, None] * theta[None, :]
        self.register_buffer("cos", torch.cos(theta)[None], persistent=False)
        self.register_buffer("sin", torch.sin(theta)[None], persistent=False)

    def forward(self, x):
        # x: [...,S,2D]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        z1 = self.cos * x_even - self.sin * x_odd
        z2 = self.sin * x_even + self.cos * x_odd
        z = torch.stack([z1, z2], dim=-1)  # [B,S,D,2]
        return z.flatten(-2)  # [B,S,2D]
