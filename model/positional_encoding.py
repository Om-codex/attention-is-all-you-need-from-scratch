import math

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000, dropout= 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # position values
        position = torch.arange(max_len).unsqueeze(1) # this creates positions like 0,1,2,...., max_len -1
        # shape [max_len, 1]

        # frequency scaling
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model)
        )

        # sin and cos encoding

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])

        # add batch dimension
        pe = pe.unsqueeze(0) # shape [1, max_len, d_model]

        # this matches batched input shape [batch_size, seq_len, d_model]

        # Register buffer
        self.register_buffer("pe", pe)
        # This means pe is saved with the model and moved to GPU with the model, but it is not trainable.

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# Notes : 
# Because embeddings are vectors.
# We need positional information also as vectors of size d_model.