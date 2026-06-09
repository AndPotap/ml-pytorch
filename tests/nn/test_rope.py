from nn.rope import RoPE


def test_rope():
    S, D = 4, 3
    model = RoPE(embd=D, max_seq=S)
    assert model.cos.shape == (S, D)
