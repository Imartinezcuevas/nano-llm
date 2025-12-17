from nano_llm.generation import generate
from nano_llm.mini_transformer import MiniTransformer
import torch

def test_generate_shape():
    vocab_size = 20
    model = MiniTransformer(
        vocab_size=vocab_size,
        d_model=16,
        num_heads=4,
        ff_hidden=32,
        num_layers=1,
        max_len=10
    )
    model.eval()

    x = torch.randint(0, vocab_size, (1, 3))
    out = generate(model, x, max_new_tokens=5)

    assert out.shape == (1, 8)

def test_generate_deterministic_greedy():
    torch.manual_seed(0)

    vocab_size = 10
    model = MiniTransformer(
        vocab_size=vocab_size,
        d_model=16,
        num_heads=4,
        ff_hidden=32,
        num_layers=1,
        max_len=10,
    )
    model.eval()

    x = torch.randint(0, vocab_size, (1, 3))
    out1 = generate(model, x, max_new_tokens=3, temperature=0.0)
    out2 = generate(model, x, max_new_tokens=3, temperature=0.0)

    assert torch.equal(out1, out2)

def test_generate_prefix_preserved():
    vocab_size = 15
    model = MiniTransformer(
        vocab_size=vocab_size,
        d_model=16,
        num_heads=4,
        ff_hidden=32,
        num_layers=1,
        max_len=10,
    )
    model.eval()

    x = torch.tensor([[1, 2, 3]])
    out = generate(model, x, max_new_tokens=4)

    assert torch.equal(out[:, :3], x)
