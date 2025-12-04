"""
An embedding is a learneable lookup tabla that maps token IDs
to dense vectors. It's just a trainable matriz of shape 
[vocab_size, d_model] where each row is the representation of one token. 

The forward pass does nothing but index into that matrix. 
The updates happen during backprop, not in the forward.
"""

import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(vocab_size, d_model) * (d_model ** -0.5)
        )

    def forward(self, x):
        return self.weight[x]