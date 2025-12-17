import torch
from nano_llm.mini_transformer import MiniTransformer

def test_mini_transformer_output_shape():
    B, T = 2, 4
    vocab_size = 100
    d_model = 32

    x = torch.randint(0, vocab_size, (B, T))
    model = MiniTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=4,
        ff_hidden=64,
        num_layers=2,
        max_len=16,
    )

    out = model(x)
    assert out.shape == (B, T, vocab_size)

def test_mini_transformer_causal_consistency():
    torch.manual_seed(0)

    vocab_size = 50
    x1 = torch.tensor([[1, 2, 3]])
    x2 = torch.tensor([[1, 2, 9]])

    model = MiniTransformer(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=4,
        ff_hidden=64,
        num_layers=1,
        max_len=8,
    )
    model.eval()

    out1 = model(x1)
    out2 = model(x2)

    assert torch.allclose(out1[:,:2], out2[:, :2], atol=1e-6)

def test_mini_transformer_deterministic_eval():
    torch.manual_seed(0)

    x = torch.randint(0, 20, (1, 5))
    model = MiniTransformer(
        vocab_size=20,
        d_model=16,
        num_heads=4,
        ff_hidden=32,
        num_layers=2,
        max_len=10,
        dropout=0.1,
    )
    model.eval()

    out1 = model(x)
    out2 = model(x)

    assert torch.allclose(out1, out2)
