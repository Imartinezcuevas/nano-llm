'''
Multi-Head self-attention layer.

This module implements de core self-attention mechanism used in Transformers.
Each input token is projected into queries, keys and values, split across multiple
heads, and then combined via scaled dot-product attention.
'''

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, dropout:float=0.0):
        """
        Features:
            - Supports causal (autoregressive) attention with a lower-triangular mask.
            - Optional dropout on attention weights for regularization.
            - Optional padding mask to ignore padded tokens in sequences.

        Args:
            d_model (int): Dimensionality of input and output embeddings.
            num_heads (int): Number of attention heads.
            dropout (float, optional): Dropout probability applied to attention weights. Default: 0.0

        Shape:
            - Input: Tensor of shape (B, T, D) where
                B = batch size
                T = sequence length
                D = d_model
            - Padding mask (optional): BoolTensor of shape (B, T), True = keep, False = pad
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
    
    def forward(self, x: torch.Tensor, padding_mask:torch.Tensor=None):
        """
        Compute multi-head self-attention.

        Args:
            x (Tensor): Input embedding of shape (B, T, D)
            padding_mask (BoolTensor, optional): Mask for padded tokens, shape (B, T)

        Returns:
            Tensor: Output embeddings after self-attention, shape (B, T, D)
        """
        B, T, D = x.shape

        q, k, v = self.project_qkv(x)

        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        if padding_mask is not None:
            padding_mask = padding_mask[:, None, None, :]
            mask = mask.bool() & padding_mask
        else:
            mask = mask.bool()

        attn_out = self.scaled_dot_product_attention(q, k, v, mask)

        #concatenate heads: (B, num_heads, T, d_head) -> (B, T, d_model)
        B, num_heads, T, d_head = attn_out.shape
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)

        # output projection
        out = self.out_proj(attn_out)

        return out

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        """
        Compute scaled dot-product attention via optimal masking.

        Args:
            q, k, v (Tensor): Queries, Keys, Values of shape (B, num_heads, T, d_head)
            mask (BoolTensor, optional): Mask tensor of shape (B, num_heads, T, T)

        Returns:
            Tensor: Attention output, same shape as q (B, num_heads, T, d_head)
        """
        d_k = q.size(-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)

        return out
