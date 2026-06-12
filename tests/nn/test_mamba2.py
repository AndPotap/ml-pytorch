import torch
import torch.nn.functional as F
import numpy as np
from nn.mamba2 import ssd_minimal_discrete, mamba2_recursive_np


def test_mamba2_recursive_np():
    T, N = 3, 2
    dtype = np.float32
    rng = np.random.default_rng(seed=21)

    x = rng.normal(size=(T, )).astype(dtype)
    gamma = rng.uniform(size=(T, )).astype(dtype)
    A = np.abs(rng.normal(size=(T, )).astype(dtype))
    alpha = np.exp(-gamma * A)
    B = rng.normal(size=(T, N)).astype(dtype)
    C = rng.normal(size=(T, N)).astype(dtype)

    y = mamba2_recursive_np(x, alpha, gamma, B, C)
    assert y.shape == (T, )


def test_mamba2():
    torch.manual_seed(42)

    # Dimensions
    # Denoted (B, T, Q, D, P) in the paper
    batch, seqlen, chunk_size, dim, headdim = 1, 2048, 64, 2048, 64
    nheads = dim // headdim  # (H) in the paper
    ngroups = 1  # (G) in the paper
    dstate = 64  # (N) in the paper
    dtype = torch.float32
    # device = "cuda"
    device = "cpu"

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
    dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=dtype, device=device) - 4)
    A = -torch.exp(torch.rand(nheads, dtype=dtype, device=device))
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    # D = torch.randn(nheads, dtype=dtype, device=device)

    y_min, _ = ssd_minimal_discrete(X=x * dt.unsqueeze(-1), A=A * dt, B=B, C=C, block_len=chunk_size)
    assert y_min.shape == (batch, seqlen, nheads, headdim)
