import pytest
import torch
from nano_llm.attention import MultiHeadAttention, scaled_dot_product_attention

def test_qkv_shapes():
    x = torch.randn(2, 5, 32)
    attn = MultiHeadAttention(d_model=32, num_heads=4)

    q, k, v = attn.project_qkv(x)

    assert q.shape == (2, 4, 5, 8)
    assert k.shape == (2, 4, 5, 8)
    assert v.shape == (2, 4, 5, 8)

def test_attention_output_shape():
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 4, 5, 8)
    v = torch.randn(2, 4, 5, 8)

    out = scaled_dot_product_attention(q, k, v)

    assert out.shape == (2, 4, 5, 8)

def test_invalid_num_heads():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=30, num_heads=8)
