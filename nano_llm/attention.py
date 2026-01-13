'''
Multi-Head self-attention layer.

This module implements the core self-attention mechanism used in Transformers.
Each input token is projected into queries, keys, and values, split across multiple
heads, and then combined via scaled dot-product attention.
'''

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 is_causal: bool = True):
        """
        Features:
            - Supports causal (autoregressive) attention with a lower-triangular mask.
            - Optional dropout on attention weights for regularization.
            - Optional padding mask to ignore padded tokens in sequences.

        Args:
            d_model (int): Dimensionality of input and output embeddings.
            num_heads (int): Number of attention heads.
            dropout (float, optional): Dropout probability applied to attention
                weights. Default: 0.0
            is_causal (bool): Whether to use causal masking. Default: True

        Shape:
            - Input: Tensor of shape (B, T, D) where
                B = batch size
                T = sequence length
                D = d_model
            - Padding mask (optional): BoolTensor of shape (B, T), True=keep, False=pad
            - Output: Tensor of shape (B, T, D), same as input, after self-attention.
        """
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.is_causal = is_causal

        self.out_proj = nn.Linear(d_model, d_model)

    def project_qkv(self, x: torch.Tensor):
        """
        Project input tensor into queries, keys and values, then split into heads.

        Args:
            x (Tensor): Input of shape (B, T, D)

        Returns:
            tuple of three tensors (q, k, v) each of shape (B, num_heads, T, d_head)
        """
        B, T, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        return q, k, v

    def forward(self,
                x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Compute multi-head self-attention.

        Args:
            x (Tensor): Input embedding of shape (B, T, D)
            padding_mask (BoolTensor, optional): Mask for padded tokens, shape (B, T)
            past_kv (tuple, optional): tuple of (k_prev, v_prev) each
                (B, num_heads, T_prev, d_head)

        Returns:
            Tensor: Output embeddings after self-attention, shape (B, T, D)
            present_kv: (k, v) including previous
        """
        B, T, D = x.shape
        q, k, v = self.project_qkv(x)

        if past_kv is not None:
            k_prev, v_prev = past_kv
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
        present_kv = (k, v)

        # Build attention mask
        mask = None
        total_len = k.size(2)

        # Create causal mask if needed
        if self.is_causal:
            mask = torch.tril(
                torch.ones(T, total_len, device=x.device, dtype=torch.bool)
            ).unsqueeze(0).unsqueeze(0)

        # Handle padding mask
        if padding_mask is not None:
            if past_kv is not None:
                pad = torch.ones(B,
                                 past_kv[0].size(2),
                                 device=x.device,
                                 dtype=torch.bool)
                padding_mask = torch.cat([pad, padding_mask], dim=1)

            # Expand padding mask to (B, 1, 1, total_len) for broadcasting
            # This masks out KEY positions (columns in attention matrix)
            padding_mask = padding_mask[:, None, None, :]

            if mask is None:
                # No causal mask, only padding - use padding mask directly
                mask = padding_mask
            else:
                # Combine causal and padding masks
                mask = mask & padding_mask

        # Compute attention
        attn_out = scaled_dot_product_attention(q, k, v, mask, self.attn_dropout)

        # Concatenate heads: (B, num_heads, T, d_head) -> (B, T, d_model)
        B, num_heads, T, d_head = attn_out.shape
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)

        # Output projection
        out = self.out_proj(attn_out)

        return out, present_kv


def scaled_dot_product_attention(q: torch.Tensor,
                                 k: torch.Tensor,
                                 v: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None,
                                 dropout: Optional[nn.Dropout] = None):
    """
    Compute scaled dot-product attention.

    Args:
        q, k, v (Tensor): Queries, Keys, Values of shape (B, num_heads, T, d_head)
        mask (Tensor, optional): Mask tensor. Can be either:
            - BoolTensor where True=attend, False=mask out
            - FloatTensor where 1.0=attend, 0.0=mask out
        dropout (Dropout, optional): Dropout layer to apply to attention weights

    Returns:
        Tensor: Attention output, same shape as q (B, num_heads, T, d_head)
    """
    d_k = q.size(-1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        # Handle both boolean and float masks
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float('-inf'))
        else:
            # Float mask: 1.0 = attend, 0.0 = mask out
            scores = scores.masked_fill(mask == 0, float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    out = torch.matmul(attn, v)

    return out
