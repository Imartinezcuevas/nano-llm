"""
Single transformer block with self-attention and feedforward network.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from nano_llm.attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self,
                 d_model:int,
                 num_heads:int,
                 ff_hidden:int,
                 dropout:float=0.0,
                 is_causal:bool=True):
        """
        Features:
            - Multi-head self-attention with causal masking.
            - Residual connection with LayerNorm.
            - Feedforward network with 2 linear layers and ReLU.
            - Optional dropout on attention and feedforward.

        Args:
            d_model (int): Input/output dimension.
            num_heads (int): Number of attention heads.
            ff_hidden (int): Hidden dimension of feedforward network.
            dropout (float): Dropout probability for attention and feedforward.
        """
        super().__init__()

        self.mha = MultiHeadAttention(d_model=d_model,
                                      num_heads=num_heads,
                                      dropout=dropout,
                                      is_causal=is_causal)
        self.ln1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]]=None):
        """
        Forward pass for a single Transformer block.

        Args:
            x (Tensor): Input embeddings of shape (B, T, d_model)
            padding_mask (BoolTensor, optional): Mask for padded tokens, shape (B, T)

        Returns:
            Tensor: Output embeddings, shape (B, T, d_model)
        """
        attn_out, present_kv = self.mha(x, padding_mask=padding_mask, past_kv=past_kv)
        h = self.ln1(x + self.dropout(attn_out))

        ff_out = self.ff(h)
        x = self.ln2(h + self.dropout(ff_out))

        return x, present_kv
