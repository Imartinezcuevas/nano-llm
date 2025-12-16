import torch
import torch.nn as nn
from nano_llm.embeddings import TokenEmbedding
from nano_llm.position_embeddings import PositionEmbedding
from nano_llm.transformer_block import TransformerBlock

class MiniTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        ff_hidden: int,
        num_layers: int,
        max_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionEmbedding(max_len, d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_hidden=ff_hidden,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor):
        B, T = x.shape
        device = x.device

        tok_emb = self.token_emb(x)
        pos_ids = torch.arange(T, device=device)
        pos_emb = self.pos_emb(pos_ids)

        h = tok_emb + pos_emb

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        logits = self.head(h)

        return logits
