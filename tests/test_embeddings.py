import pytest
import torch
from nano_llm.embeddings import TokenEmbedding

def test_embedding_shape():
    vocab, d_model = 10, 16
    emb = TokenEmbedding(vocab, d_model)
    x = torch.tensor([1, 3, 5])
    out = emb(x)
    assert out.shape == (3, d_model)

def test_embedding_consistency():
    vocab, d_model = 10, 16
    emb = TokenEmbedding(vocab, d_model)
    x = torch.tensor([2, 2, 2])
    out = emb(x)
    assert torch.allclose(out[0], out[1])
    assert torch.allclose(out[1], out[2])

def test_embedding_grad():
    vocab, d_model = 10, 16
    emb = TokenEmbedding(vocab, d_model)
    x = torch.tensor([1, 2, 3])

    out = emb(x).sum()
    out.backward()

    assert emb.weight.grad is not None
    assert emb.weight.grad.shape == emb.weight.shape

def test_embedding_out_of_range():
    vocab, d_model = 10, 16
    emb = TokenEmbedding(vocab, d_model)
    x = torch.tensor([0, 9, 10])

    with pytest.raises(IndexError):
        emb(x)

def test_matches_nn_embedding():
    vocab, d_model = 10, 16

    my_emb = TokenEmbedding(vocab, d_model)
    torch_emb = torch.nn.Embedding(vocab, d_model)

    torch_emb.weight.data = my_emb.weight.data.clone()

    x = torch.tensor([1, 3, 5])
    out1 = my_emb(x)
    out2 = torch_emb(x)
    assert torch.allclose(out1, out2)
