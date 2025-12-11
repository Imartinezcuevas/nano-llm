import torch
import torch.nn as nn

class PositionEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(max_len, d_model) * (d_model ** -0.5))

    def forward(self, x):
        seq_len = x.size(0)
        if seq_len > self.weight.size(0):
            raise IndexError(f"""Sequence length {seq_len}
                             exceeds maximum {self.weight.size(0)}""")
        return self.weight[:seq_len, :]
