import pytest
import torch
import math
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

def test_scaled_attention_shape_and_mask():
    B, H, T, D = 2, 2, 3, 4
    torch.manual_seed(0)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    out = scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, H, T, D)

    mask = torch.tril(torch.ones(T, T))
    out_masked = scaled_dot_product_attention(q, k, v, mask)
    assert out_masked.shape == (B, H, T, D)

    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    masked_scores = scores.masked_fill(mask == 0, float('-inf'))
    for i in range(T):
        for j in range(i + 1, T):
            assert masked_scores[0, 0, i, j] == float('-inf')

def test_scaled_attention_output_known_values():
    q = k = v = torch.tensor([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]])
    B, H, T, D = q.shape
    mask = torch.tril(torch.ones(T,T)).unsqueeze(0).unsqueeze(0).expand(B,H,T,T)

    out = scaled_dot_product_attention(q, k, v, mask)
    assert out.shape == (B, H, T, D)
