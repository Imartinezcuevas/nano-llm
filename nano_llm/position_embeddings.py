"""
Positional embeddings provide the model with information about the order of
the tokens. Unlike token embeddings, which map discrete token IDs to
learnable vectors, positional embeddings assing each position in the
sequence a learnable representation.

This module implements a simple learned position embedding matrix of
shape [], where each row corresponds to the embedding of a
position index from 0 to max_len - 1.

The forward pass simply indexes into this matrix using the positions
provided. Forward does not perform any computation besides lookup.
"""

import torch
import torch.nn as nn

class PositionEmbedding(nn.Module):
    """
    Learnable positional embedding layer.

    Args:
        max_len (int): Maximum sequence length supported.
        d_model (int): Dimensionality of each positional vector.

    Shape:
        - Input: LongTensor of shape (...), containing position indices
            in the range [0, max_len - 1]
        - Output: Tensor of shape (..., d_model), where each position is
            mapped to its corresponding embedding vector.
    """
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(max_len, d_model) * (d_model ** -0.5))

    def forward(self, x):
        """
        Retrieve positional embeddings for the given indices.

        Args:
            x (Tensor): A tensor of position indices.

        Returns:
            Tensor: Positional embeddings correspondign to each index.
        """
        seq_len = x.size(0)
        if seq_len > self.weight.size(0):
            raise IndexError(f"""Sequence length {seq_len}
                             exceeds maximum {self.weight.size(0)}""")
        return self.weight[x]
