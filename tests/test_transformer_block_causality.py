import torch
from nano_llm.transformer_block import TransformerBlock

def test_causality_robustness():
    """
    Verifica la causalidad mediante perturbación de inputs (Black-box testing).
    """
    torch.manual_seed(42)
    B, T, D = 1, 5, 16

    block = TransformerBlock(D, 4, 32, is_causal=True)
    block.eval()

    # Input base vs Input modificado en el futuro (último token)
    x = torch.randn(B, T, D)
    x_mutated = x.clone()
    x_mutated[:, -1, :] += 100.0 # Perturbación masiva

    with torch.no_grad():
        out_1, _ = block(x)
        out_2, _ = block(x_mutated)

    # En un modelo causal, el token 0 NO debe enterarse del cambio en el token -1
    diff = (out_1[:, 0, :] - out_2[:, 0, :]).abs().max().item()
    assert diff == 0.0, f"Error Causal: El pasado cambió al tocar el futuro.Diff:{diff}"

    block.mha.is_causal = False

    with torch.no_grad():
        out_1, _ = block(x)
        out_2, _ = block(x_mutated)

    # En un modelo bidireccional, el token 0 DEBE cambiar
    diff = (out_1[:, 0, :] - out_2[:, 0, :]).abs().max().item()
    assert diff > 0.0, "Error Bidireccional: El pasado ignoró el futuro."
