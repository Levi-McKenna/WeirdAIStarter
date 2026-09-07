import torch
import torch.nn as nn


class SimpleSelfAttention(nn.Module):
    """
    A simple self-attention module without trainable query, key, and value projections.

    This version is primarily for learning.
    """

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (num_tokens, embedding_dim)
            attention_weights: Tensor of shape (num_tokens, num_tokens)
        """

        # TODO:
        # 1. Compute attention scores using matrix multiplication.
        # 2. Normalize scores with softmax.
        # 3. Compute context vectors as weighted sums of input vectors.
        attention_score = x @ x.T
        attention_weights = torch.softmax(attention_score, dim=-1)
        context_vecs = attention_weights @ x
        return context_vecs, attention_weights

class SelfAttention(nn.Module):
    """
    Trainable self-attention using query, key, and value projections.
    """

    def __init__(self, embedding_dim, output_dim, qkv_bias=False):
        super().__init__()

        self.query = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.key = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.value = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (num_tokens, output_dim)
            attention_weights: Tensor of shape (num_tokens, num_tokens)
        """

        # TODO:
        # 1. Compute queries, keys, and values.
        # 2. Compute scaled attention scores.
        # 3. Apply softmax.
        # 4. Compute context vectors.
        query_v = self.query(x)
        key_v = self.key(x)
        values = self.value(x)

        attention_scores = query_v @ key_v.T
        embed_dim = key_v.shape[-1]
        attention_weights = torch.softmax(attention_scores / embed_dim**0.5,
                                          dim=-1)
        context_vecs = attention_weights @ values
        return context_vecs, attention_weights

class CausalAttention(nn.Module):
    """
    Self-attention with a causal mask so tokens cannot attend to future tokens.
    """

    def __init__(self, embedding_dim, output_dim, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        self.query = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.key = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.value = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (batch_size, num_tokens, output_dim)
        """

        # TODO:
        # 1. Compute keys, queries, and values.
        # 2. Compute scaled attention scores.
        # 3. Mask future tokens.
        # 4. Apply softmax.
        # 5. Apply dropout.
        # 6. Compute context vectors.
        batch, token_count, embedding_dim = x.shape
        query_v = self.query(x)
        key_v = self.key(x)
        values = self.value(x)

        attention_scores = query_v @ key_v.transpose(1, 2)
        attention_scores.masked_fill_(
            self.mask.bool()[:token_count, :token_count], -torch.inf
        )
        attention_weights = torch.softmax(
            attention_scores / embedding_dim**0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        context_vecs = attention_weights @ values
        return context_vecs
