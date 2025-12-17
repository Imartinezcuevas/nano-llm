from nano_llm.mini_transformer import MiniTransformer
import torch

def generate(
        model: MiniTransformer,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
):
    model.eval()
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_token_logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                scaled_logits = next_token_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

    return generated
