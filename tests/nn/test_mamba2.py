import torch
import torch.nn.functional as F
import numpy as np
from nn.mamba2 import ssd_minimal_discrete, mamba2_recursive_np, create_L_np, mamba2_ssd_np, mamba2_chunked_np


def test_mamba2_recursive_np():
    T, N = 6, 4
    # T, N = 2048, 64
    dtype = np.float64
    rng = np.random.default_rng(seed=21)

    x = rng.normal(size=(T, )).astype(dtype)
    gamma = np.log(1.0 + np.exp(rng.normal(size=(T, )).astype(dtype) - 1.0))
    A = np.abs(rng.normal(size=(T, )).astype(dtype))
    alpha = np.exp(-gamma * A)
    B = rng.normal(size=(T, N)).astype(dtype)
    C = rng.normal(size=(T, N)).astype(dtype)

    y = mamba2_recursive_np(x, alpha, gamma, B, C)
    assert y.shape == (T, )

    ans = [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [alpha[1], 1.0, 0.0, 0.0, 0.0],
        [alpha[2] * alpha[1], alpha[2], 1.0, 0.0, 0.0],
        [alpha[3] * alpha[2] * alpha[1], alpha[3] * alpha[2], alpha[3], 1.0, 0.0],
        [alpha[4] * alpha[3] * alpha[2] * alpha[1], alpha[4] * alpha[3] * alpha[2], alpha[4] * alpha[3], alpha[4], 1.0],
    ]
    ans = np.array(ans, dtype=alpha.dtype)
    L = create_L_np(np.log(alpha[:ans.shape[0]]))
    abs_diff = np.linalg.norm(ans - L)
    print(f"{abs_diff=:.3e}")
    assert abs_diff < 1e-12

    approx = mamba2_ssd_np(x, alpha, gamma, B, C)
    abs_diff = np.linalg.norm(y - approx)
    print(f"{abs_diff=:.3e}")
    assert abs_diff < 1e-12

    for bl in [2, T]:
        y, _ = mamba2_chunked_np(X=gamma * x, A=np.log(alpha), B=B, C=C, block_len=bl)
        abs_diff = np.linalg.norm(y - approx)
        print(f"{abs_diff=:.3e}")
        assert abs_diff < 1e-12


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
