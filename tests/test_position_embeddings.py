import torch
import pytest
from nano_llm.position_embeddings import PositionEmbedding

def test_embedding_shape():
    max_len, d_model = 20, 16
    emb = PositionEmbedding(max_len, d_model)
    x = torch.zeros(10)
    out = emb(x)
    assert out.shape == (10, d_model)

def test_embedding_grad():
    max_len, d_model = 20, 16
    emb = PositionEmbedding(max_len, d_model)
    x = torch.zeros(5)

    out = emb(x).sum()
    out.backward()

    assert emb.weight.grad is not None
    assert emb.weight.grad.shape == emb.weight.shape

def test_embedding_out_of_range():
    max_len, d_model = 10, 16
    emb = PositionEmbedding(max_len, d_model)
    x = torch.zeros(12)

    with pytest.raises(IndexError):
        emb(x)
