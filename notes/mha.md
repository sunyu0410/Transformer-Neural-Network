```python
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sequence_length = 4
batch_size = 1
input_dim = 512 # vocab dim
d_model = 512  # embedding dim
x = torch.randn((batch_size, sequence_length, input_dim)) # torch.Size([1, 4, 512])


qkv_layer = nn.Linear(input_dim, 3 * d_model)
qkv = qkv_layer(x) # torch.Size([1, 4, 1536])

num_heads = 8
head_dim = d_model // num_heads
qkv = qkv.reshape(batch_size, sequence_length, num_heads, 3*head_dim) # torch.Size([1, 4, 8, 192])

qkv = qkv.permute(0,2,1,3) # bs, n_heads, seq_len, 3*head_dim

q, k, v = qkv.chunk(3, dim=-1) # 3 of torch.Size([1, 8, 4, 64])

# Self-attention
# attn = softmax((QK)/sqrt(d_k) + Mask)
d_k = q.size(-1)
scaled = torch.matmul(
    q, k.transpose(-2, -1)
) / np.sqrt(d_k) # 1, 8, 4, 4

mask = torch.full(scaled.size(), float('-inf')) # # 1, 8, 4, 4
# float('-inf') defines -inf
mask = torch.triu(mask, diagonal=1)
# diagonal controls the offset of the digonal line

(scaled+mask)[0,0]
# In [96]: (scaled+mask)[0,0]
# Out[96]: 
# tensor([[ 0.4496,    -inf,    -inf,    -inf],
#         [-0.1858,  0.0240,    -inf,    -inf],
#         [-0.3487, -0.3538,  0.1245,    -inf],
#         [ 0.0288, -0.2639,  0.3080,  0.6101]], grad_fn=<SelectBackward0>)

scaled += mask

attention = F.softmax(scaled+mask, dim=-1)

values = torch.matmul(attention, v) # torch.Size([1, 8, 4, 64])
values = values.reshape((batch_size, sequence_length, num_heads*head_dim)) # [1, 4, 512]



```