import math
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask = None):

        # QKᵀ
        scores = torch.matmul( Q, K.transpose( -2, -1))

        # scaling
        scores = scores / math.sqrt(Q.size(-1))

        # applying padding mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0 , -1e9)

        # softmax
        attention_weights = torch.softmax(scores, dim = -1)

        # weighted sum
        output = torch.matmul(attention_weights, V)

        return output, attention_weights
