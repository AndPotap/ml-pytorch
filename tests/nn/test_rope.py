import torch
import numpy as np

from nn.rope import RoPE


def test_rope():
    # B, S, D = 1, 4, 6
    # B, S, D = 7, 32, 64
    B, S, D = 7, 512, 128
    rope = RoPE(embd=D, max_seq=S)
    assert rope.cos.shape == (1, S, D // 2)
    assert rope.sin.shape == (1, S, D // 2)

    x = torch.randn(B, S, D)
    z = rope(x)
    assert z.shape == (B, S, D)

    dtype = torch.float64
    x = x.to(dtype)
    norm_x = torch.linalg.norm(x)
    rope.to(dtype)
    norm_z = torch.linalg.norm(rope(x))
    abs_diff = torch.abs(norm_x - norm_z)
    print(f"{abs_diff=:1.3e}")
    assert abs_diff < 1e-6

    x_np = x.numpy()
    approx = rope_np_fast(x_np)
    ans = rope_np(x_np)
    abs_diff = np.max(np.abs(approx - ans))
    print(f"{abs_diff=:1.3e}")
    assert abs_diff < 1e-12

    ans = rope_np(x_np)
    approx = rope(x).numpy()
    abs_diff = np.max(np.abs(approx - ans))
    print(f"{abs_diff=:1.3e}")
    assert abs_diff < 1e-4


def rope_np(x):
    dtype = np.float128
    B, S, D = x.shape
    D2 = D // 2
    x = x.astype(dtype)

    phi = np.zeros(shape=(S, D2), dtype=dtype)
    R = np.zeros(shape=(S, D, D), dtype=dtype)
    z = np.empty(shape=(B, S, D), dtype=dtype)

    for s in range(S):
        for d in range(D2):
            phi[s, d] = s * 10**(-4. * (2. * (d / D)))

    for s in range(S):
        for idx in range(0, D, 2):
            d = idx // 2
            R[s, idx, idx] = np.cos(phi[s, d])
            R[s, idx, idx + 1] = -np.sin(phi[s, d])
            R[s, idx + 1, idx] = np.sin(phi[s, d])
            R[s, idx + 1, idx + 1] = np.cos(phi[s, d])

    for b in range(B):
        for s in range(S):
            z[b, s] = R[s] @ x[b, s]

    return z.astype(np.float64)


def rope_np_fast(x):
    dtype = np.float128
    B, S, D = x.shape
    D2 = D // 2
    x = x.astype(dtype)

    theta = 10**-(4 * np.arange(D2, dtype=dtype) / D2)
    phi = np.arange(S, dtype=dtype)[:, None] * theta[None, :]
    cos = np.cos(phi)
    sin = np.sin(phi)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    z1 = cos * x_even - sin * x_odd
    z2 = sin * x_even + cos * x_odd
    z = np.stack([z1, z2], axis=-1)  # [B,S,D,2]
    z = z.reshape(B, S, D)

    return z.astype(np.float64)
