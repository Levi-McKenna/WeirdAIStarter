import torch
import torch.nn as nn

from weird_ai.layer_norm import LayerNorm
from weird_ai.transformer import TransformerBlock


class WeirdAIModel(nn.Module):
    def __init__(self, vocab_size, context_length, emb_dim, layers=4, drop_rate=0.1):
        super().__init__()

        # TODO:
        # Create embeddings
        # Create attention layers
        # Create output head
        self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_emb = nn.Embedding(context_length, emb_dim)
        self.drop_emb = nn.Dropout(drop_rate)

        self.transformer = nn.Sequential(
            *[TransformerBlock(emb_dim, context_length, drop_rate) for _ in range(layers)]
        )
        self.norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_tokens, embedding_dim)
        Returns:
            out_head: Tensor of shape (batch_size, num_tokens, vocab_size)
        """
        batch_size, seq_len = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(
            # also this is .ARange not .arrange don't get confused idiot
            torch.arange(seq_len, device=x.device)
        )

        # these two tensors are broadcastable which means despite differing
        # dimensions they can still be added
        x = tok_emb + pos_emb
        x = self.drop_emb(x)
        x = self.transformer(x)
        x = self.norm(x)
        logits = self.out_head(x)
        # the final vector determines the next token in the sequence by
        # selecting the highest normalized probablity (the highest individual
        # element of the final vector)
        return logits
