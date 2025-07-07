import torch.nn as nn
import torch.nn.functional as F
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = F.norm(self.embed_dim)

    def forward(self, x):
        b = x.size(0) # b, num_seq, embed_dim
        q = self.q(x.transpose(1, 2))
        k = self.k(x.transpose(1, 2))
        v = self.v(x.transpose(1, 2))

        attention_score = F.softmax(torch.matual(q, k.T)/torch.sqrt(self.embed_dim))
        out = torch.matual(attention_score, v)

        return out

x = torch.randn(1, 20, 128)
attn = MultiHeadAttention()
x = attn(x)