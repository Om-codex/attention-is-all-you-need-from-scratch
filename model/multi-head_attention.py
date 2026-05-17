import math
import torch
import torch.nn as nn

class MuliiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0 # checks if d_model is equally divided

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads # each head gets 64 dimensions

        # transform embeddings into Q,K,V vectors
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        # final output projection
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x, mask = None):
        
        batch_size, seq_len, _ = x.shape

        # creating Q, K, V
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # splitting the dim for the no of head_dim
        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1,2) # Changes shape: (32, 20, 8, 64) -> (32, 8, 20, 64)

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        # attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))

        scores = scores / math.sqrt(self.head_dim)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # attention weights
        attention_weights = torch.softmax(scores, dim = -1)

        # weighted sum
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads
        attention_output = attention_output.transpose(1,2)

        attention_output = attention_output.contiguous().view(
            batch_size,
            seq_len,
            self.d_model
        )

        output = self.Wo(attention_output)

        return output