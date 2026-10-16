from torch import nn as nn
from weird_ai.attention import CausalAttention, SelfAttention
from weird_ai.feed_forward import FeedForward
from weird_ai.layer_norm import LayerNorm

class TransformerBlock(nn.Module):
    # TODO 
    # Create a TransformerBlock class, inheriting from nn.Module
    def __init__(self, emb_dim, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        self.att = CausalAttention(emb_dim, emb_dim, context_length, qkv_bias)
        self.ff = FeedForward(emb_dim)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.drop_shortcut = nn.Dropout(dropout)

    
    # using 
    #  - LayerNorm
    #  - SelfAttention from previous assignment
    #  - FeedForward
    #  - Residual connections
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
