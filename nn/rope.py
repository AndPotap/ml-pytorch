import torch
from torch.nn import Module


class RoPE(Module):
    def __init__(self, embd: int, max_seq: int):
        super().__init__()
        theta = 10 ** -(4 * torch.arange(embd) / embd)
        theta = torch.arange(max_seq)[:, None] * theta[None, :]

    def forward(x):
        return x
