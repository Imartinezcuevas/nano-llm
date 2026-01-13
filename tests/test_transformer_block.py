import torch
from nano_llm.transformer_block import TransformerBlock

def test_transformer_block_output_shape():
    B, T, D = 2, 3, 32
    x = torch.randn(B, T, D)
    block = TransformerBlock(d_model=D, num_heads=4, ff_hidden=64)
    out, _ = block(x)
    assert out.shape == (B, T, D)

def test_transformer_block_residual():
    x = torch.randn(2, 3, 32)
    block = TransformerBlock(d_model=32, num_heads=4, ff_hidden=64)
    out, _ = block(x)
    assert not torch.allclose(out, x)

def test_transformer_block_padding_mask():
    torch.manual_seed(0)

    B, T, D = 2, 4, 16
    x = torch.randn(B, T, D)

    padding_mask = torch.tensor([
        [1, 1, 1, 0],
        [1, 1, 0, 0],
    ], dtype=torch.bool)

    block = TransformerBlock(d_model=D, num_heads=4, ff_hidden=32)
    block.eval()

    out, _ = block(x, padding_mask=padding_mask)

    assert out.shape == (B, T, D)

def test_transformer_block_deterministic_eval():
    torch.manual_seed(0)

    x = torch.randn(1, 3, 16)
    block = TransformerBlock(d_model=16, num_heads=4, ff_hidden=32, dropout=0.1)
    block.eval()

    out1, _ = block(x)
    out2, _ = block(x)

    assert torch.allclose(out1, out2)

def test_transformer_block_backward():
    x = torch.randn(2, 3, 16, requires_grad=True)
    block = TransformerBlock(d_model=16, num_heads=4, ff_hidden=32)

    out, _ = block(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
