import torch
import pytest
from nano_llm.mini_transformer import MiniTransformer
from nano_llm.generation import generate

@pytest.fixture
def model():
    torch.manual_seed(42)
    return MiniTransformer(
        vocab_size=50,
        d_model=32,
        num_heads=4,
        ff_hidden=64,
        num_layers=2,
        max_len=20
    )

def test_generate_deterministic_greedy(model):
    """Prueba que temperature=0 siempre produce la misma secuencia."""
    input_ids = torch.tensor([[1]]) # Start token

    # Generar dos veces
    out1 = generate(model, input_ids, max_new_tokens=5, temperature=0.0)
    out2 = generate(model, input_ids, max_new_tokens=5, temperature=0.0)

    assert torch.equal(out1, out2), "Greedy generation must be deterministic"

def test_generate_stochastic_randomness(model):
    """Prueba que temperature > 0 produce variaciones (estadísticamente)."""
    torch.manual_seed(42)
    input_ids = torch.tensor([[1]])

    # Con temperature alta, debería haber variedad
    out1 = generate(model, input_ids, max_new_tokens=10, temperature=2.0)

    torch.manual_seed(43) # Semilla distinta
    out2 = generate(model, input_ids, max_new_tokens=10, temperature=2.0)

    # Es muy improbable que sean idénticos con vocab=50 y len=10
    assert not torch.equal(out1, out2), "Stochastic generation should vary with seeds"

def test_generate_output_structure(model):
    """Valida que la salida tenga la forma correcta (Input + New Tokens)."""
    B, T = 2, 3
    input_ids = torch.randint(0, 50, (B, T))
    max_new = 4

    out = generate(model, input_ids, max_new_tokens=max_new, temperature=1.0)

    assert out.shape == (B, T + max_new)
    # Verificar que el prefijo no cambió
    assert torch.equal(out[:, :T], input_ids)

def test_kv_cache_equivalence_fixed(model):
    """
    Tu test original, que ahora debería pasar.
    Valida que pasar todo de golpe vs token-a-token da los mismos logits.
    """
    model.eval()
    x = torch.randint(0, 50, (1, 10))

    # 1. Full pass
    with torch.no_grad():
        logits_full, _ = model(x)

    # 2. Cached pass step-by-step
    past_kvs = None
    logits_steps = []

    with torch.no_grad():
        # Prefill (primer token)
        logits, past_kvs = model(x[:, 0:1], past_kvs=None)
        logits_steps.append(logits[:, -1])

        # Decode resto
        for i in range(1, x.size(1)):
            logits, past_kvs = model(x[:, i:i+1], past_kvs=past_kvs)
            logits_steps.append(logits[:, -1])

    logits_cached = torch.stack(logits_steps, dim=1)

    # Usamos una tolerancia ligeramente mayor por errores de punto flotante acumulados
    assert torch.allclose(logits_full, logits_cached, atol=1e-5)
