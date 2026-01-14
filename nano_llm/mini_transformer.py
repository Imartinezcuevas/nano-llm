import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from nano_llm.embeddings import TokenEmbedding
from nano_llm.position_embeddings import PositionEmbedding
from nano_llm.transformer_block import TransformerBlock

class MiniTransformer(nn.Module):
    """
    Decoder-only transformer model.

    This module implements a minimal Transformer model composed of token and
    positional embeddings, a stack of Transformer blocks with causal
    self-attention, and a final linear projection to vocab logits.

    The model operates on token indices and returns unnormalized
    logits for each position in the input sequence.
    """
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
                    is_causal=True
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self,
                x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ):
        """
        Forward pass of the Transformer.

        Args:
            x (LongTensor): Input token indices of shape (B, T).
            padding_mask (BoolTensor, optional): Mask for padded tokens, shape (B, T)
            past_kvs (List, optional): list of (k, v) tuples from previous steps,
                one per block

        Returns:
            Tensor: Logits of shape (B, T, vocab_size)
            new_past_kvs: updated past key/values for each block
        """
        B, T = x.shape
        device = x.device

        tok_emb = self.token_emb(x)
        past_len = 0
        if past_kvs is not None:
            past_len = past_kvs[0][0].size(2)
        pos_ids = torch.arange(past_len, past_len + T, device=device)
        pos_emb = self.pos_emb(pos_ids)

        h = tok_emb + pos_emb

        new_past_kvs = []
        for i, block in enumerate(self.blocks):
            block_past_kv = past_kvs[i] if past_kvs is not None else None
            h, present_kv = block(h, padding_mask=padding_mask, past_kv=block_past_kv)
            new_past_kvs.append(present_kv)

        h = self.ln_f(h)
        logits = self.head(h)

        return logits, new_past_kvs
