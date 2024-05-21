import torch
from torch import nn
import torch.nn.functional as K
import numpy as np

class MultiHeadAttention(nn.Module):
    """
    * An MHA takes an input x [batch_size, sequence_length, input_dim]
    * Raises x to three learnable parameters: q, k, v using mlp
        * This version uses one mlp to do it, then split the output to q, k, v
    * Calculates the weights using: attn = softmax(qk/sqrt(d_k)+M)
        * Mask is optional, often for decoders
    * Calculates the weighted sum of attn and v
    
    The "multi-head" comonent operates on the feature dimension. 

    If we treat MHA as a filter, it only alters the feature dimension of x.
    """
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model)
        self.linear = nn.Linear(d_model, d_model)

    @staticmethod
    def cal_attn(q, k, v, mask=None):
        scaled = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(k.size(-1))
        if mask is not None:
            scaled += mask
        attn = K.softmax(scaled, dim=-1)
        out = torch.matmul(attn, v)
        return out
        
    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x) # torch.Size([1, 4, 1536])
        batch_size, sequence_length, _ = x.size()
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3*self.head_dim) # torch.Size([1, 4, 8, 192])
        qkv = qkv.permute(0,2,1,3) # bs, n_heads, seq_len, 3*head_dim
        q, k, v = qkv.chunk(3, dim=-1) # 3 of torch.Size([1, 8, 4, 64]) 
        values = self.cal_attn(q,k,v, mask)
        return values.reshape((batch_size, sequence_length, self.d_model))
    
if __name__ == "__main__":
    input_dim = 1024
    d_model = 512
    num_heads = 8
    # d_model should be dividable by num_heads

    batch_size = 30
    sequence_length = 5
    x = torch.randn((batch_size, sequence_length, input_dim))
    mha = MultiHeadAttention(input_dim, d_model, num_heads)
    out = mha(x)
    out = mha.linear(out)
    print(out.shape) # Out[145]: torch.Size([30, 5, 512])

    