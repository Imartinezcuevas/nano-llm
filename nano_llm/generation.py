from nano_llm.mini_transformer import MiniTransformer
import torch
from typing import Optional

def generate(
        model: MiniTransformer,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
):
    """
    Generate tokens autoregressively using the MiniTransformer with kv-cache.

    Args:
        model: MiniTransformer model
        input_ids: (B, T) starting tokens
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature (0 = greedy)
        top_k: optionally apply top-k filtering
        top_p: optionally apply top-p (nucleus) filtering

    Returns:
        generated: (B, T + max_new_tokens) token indices
    """
    model.eval()
    generated = input_ids.clone()
    past_kvs = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Model forward with kv cache
            logits, past_kvs = model(generated[:, -1:], past_kvs=past_kvs)
            # Take logits of the last token
            next_token_logits = logits[:, -1, :]

            # Apply temperature
            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = next_token_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)

                # Apply top-k
                if top_k is not None:
                    topk_vals, topk_ids = torch.topk(probs, top_k, dim=-1)
                    probs = torch.zeros_like(probs).scatter_(-1, topk_ids, topk_vals)
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                if top_p is not None:
                    sorted_probs, sorted_idx = torch.sort(
                        probs,
                        descending=True,
                        dim=-1
                    )
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cum_probs > top_p
                    mask[..., 1:] = mask[..., : -1].clone()
                    mask[..., 0] = False
                    sorted_probs[mask] = 0.0
                    probs = torch.zeros_like(probs).scatter_(
                        -1,
                        sorted_idx,
                        sorted_probs
                    )
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

    return generated
