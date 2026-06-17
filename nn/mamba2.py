import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np


def mamba2_np(X, A, B, C, block_len):
    # X: [T,]
    # A: [T,]
    # B: [T,N]
    # C: [T,N]
    assert X.shape[0] % block_len == 0
    X, A, B, C = [rearrange(v, "(c l) ... -> c l ...", l=block_len) for v in (X, A, B, C)]

    L = create_L_np(A)
    Y_diag = np.einsum("cln,csn,cls,cs->cl", C, B, L, X)

    A_cumsum = np.cumsum(A, axis=-1)
    decay_states = np.exp((A_cumsum[..., [-1]] - A_cumsum))
    states = np.einsum("cln,cl,cl->cn", B, decay_states, X)

    initial_states = np.zeros((1, states.shape[1]), dtype=states.dtype)
    states = np.concatenate([initial_states, states], axis=0)
    decay_chunk = create_L_np(np.pad(A_cumsum[..., -1], pad_width=(1, 0)))
    new_states = np.einsum("zc,cn->zn", decay_chunk, states)
    states = new_states[:-1]

    state_decay_out = np.exp(A_cumsum)
    Y_off = np.einsum("cln,cn,cl->cl", C, states, state_decay_out)

    Y = rearrange(Y_diag + Y_off, "c l -> (c l)")
    return Y


def mamba2_chunked_np(X, A, B, C, block_len, initial_states=None):
    # X: [T,]
    # A: [T,]
    # B: [T,N]
    # C: [T,N]
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[0] % block_len == 0

    X, A, B, C = [rearrange(x, "(c l) ... -> c l ...", l=block_len) for x in (X, A, B, C)]

    L = create_L_np(A)
    Y_diag = np.einsum("cln,csn,cls,cs->cl", C, B, L, X)

    A_cumsum = np.cumsum(A, axis=-1)
    decay_states = np.exp((A_cumsum[..., [-1]] - A_cumsum))
    states = np.einsum("cln,cl,cl->cn", B, decay_states, X)

    if initial_states is None:
        initial_states = np.zeros((1, states.shape[1]), dtype=states.dtype)
    states = np.concatenate([initial_states, states], axis=0)
    decay_chunk = create_L_np(np.pad(A_cumsum[..., -1], pad_width=(1, 0)))
    new_states = np.einsum("zc,cn->zn", decay_chunk, states)
    states, final_state = new_states[:-1], new_states[-1]

    state_decay_out = np.exp(A_cumsum)
    Y_off = np.einsum("cln,cn,cl->cl", C, states, state_decay_out)

    Y = rearrange(Y_diag + Y_off, "c l -> (c l)")
    return Y, final_state


def mamba2_ssd_np(x, alpha, gamma, B, C):
    # x: [T,]
    # alpha: [T,]
    # gamma: [T,]
    # B: [T,N]
    # C: [T,N]

    L = create_L_np(np.log(alpha))
    y = (L * (C @ B.T)) @ (gamma * x)
    return y


def create_L_np(log_alpha):
    # log_alpha: [T,]

    T = log_alpha.shape[-1]
    log_L = repeat(log_alpha, "... d -> ... d e", e=T)
    mask = np.tril(np.ones(shape=(T, T), dtype=log_alpha.dtype), k=-1)
    log_L = np.cumsum(log_L * mask, axis=-2)
    # mask = np.tril(np.ones(shape=(T, T), dtype=np.bool), k=0)
    # mask = np.tril(np.ones(shape=(T, T), dtype=log_alpha.dtype), k=0)
    mask = np.triu(np.full((T, T), -np.inf, dtype=log_alpha.dtype), k=1)
    log_L = log_L + mask
    # log_L[~mask] = -np.inf
    # log_L = log_L.masked_fill(~mask, -np.inf)
    return np.exp(log_L)


def mamba2_recursive_np(x, alpha, gamma, B, C):
    # x: [T,]
    # alpha: [T,]
    # gamma: [T,]
    # B: [T,N]
    # C: [T,N]

    dtype = x.dtype
    T, N = B.shape
    h = np.empty(shape=(T + 1, N), dtype=dtype)
    h[0] = 0.0  # initial condition
    y = np.empty_like(x)

    for t in range(T):
        h[t + 1] = alpha[t] * h[t] + gamma[t] * B[t] * x[t]
        y[t] = np.sum(C[t] * h[t + 1])
    return y


def segsum(x):
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)  # Broadcast matches dims from right to left
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state
