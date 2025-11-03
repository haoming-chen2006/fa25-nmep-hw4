import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention, FeedForwardNN
from .encoder import PositionalEncoding
from seq2seq.data.fr_en import tokenizer


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        dropout: float = 0.1,
    ):
        """
        Each decoder layer will take in two embeddings of
        shape (B, T, C):

        1. The `target` embedding, which comes from the decoder
        2. The `source` embedding, which comes from the encoder

        and will output a representation
        of the same shape.

        The decoder layer will have three main components:
            1. A Masked Multi-Head Attention layer (you'll need to
               modify the MultiHeadAttention layer to handle this!)
            2. A Multi-Head Attention layer for cross-attention
               between the target and source embeddings.
            3. A Feed-Forward Neural Network layer.

        Remember that for each Multi-Head Attention layer, we
        need create Q, K, and V matrices from the input embedding(s)!
        """
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qk_length = qk_length
        self.value_length = value_length

        # Define layers: masked self-attention, cross-attention, feed-forward, norms
        self.self_attn = MultiHeadAttention(self.num_heads, self.embedding_dim, self.qk_length, self.value_length)
        self.cross_attn = MultiHeadAttention(self.num_heads, self.embedding_dim, self.qk_length, self.value_length)
        self.ffn = FeedForwardNN(self.embedding_dim, self.ffn_hidden_dim)

        self.norm1 = nn.LayerNorm(self.embedding_dim)
        self.norm2 = nn.LayerNorm(self.embedding_dim)
        self.norm3 = nn.LayerNorm(self.embedding_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(
        self,
        x: torch.Tensor,
        enc_x: torch.Tensor | None,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        The forward pass of the DecoderLayer.
        """
        # Masked self-attention (tgt_mask should include causal + pad masks)
        # x: (B, T, C)
        residual = x
        x = x + self.dropout(self.self_attn(x, x, x, mask=tgt_mask))
        x = self.norm1(x)

        # Cross-attention (if encoder output provided)
        if enc_x is not None:
            residual = x
            x = x + self.dropout(self.cross_attn(x, enc_x, enc_x, mask=src_mask))
            x = self.norm2(x)

        # Feed-forward
        x = x + self.dropout(self.ffn(x))
        x = self.norm3(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        max_length: int,
        dropout: float = 0.1,
    ):
        """
        Remember that the decoder will take in a sequence
        of tokens AND a source embedding
        and will output an encoded representation
        of shape (B, T, C).

        First, we need to create an embedding from the sequence
        of tokens. For this, we need the vocab size.

        Next, we want to create a series of Decoder layers.
        For this, we need to specify the number of layers
        and the number of heads.

        Additionally, for every Multi-Head Attention layer, we
        need to know how long each query/key is, and how long
        each value is.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        self.qk_length = qk_length
        self.value_length = value_length

        # embeddings and positional encoding
        self.token_embed = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.pos_embed = PositionalEncoding(self.embedding_dim, dropout=dropout, max_len=max_length)
        self.dropout = nn.Dropout(dropout)

        # decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                self.num_heads,
                self.embedding_dim,
                self.ffn_hidden_dim,
                self.qk_length,
                self.value_length,
                dropout,
            )
            for _ in range(self.num_layers)
        ])

        # final projection to vocab
        self.output_proj = nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        enc_x: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        The forward pass of the Decoder.
        """
        # x: token indices (B, T)
        x = self.token_embed(x)
        x = self.pos_embed(x)
        x = self.dropout(x)

        # pass through decoder layers
        for layer in self.layers:
            x = layer(x, enc_x, tgt_mask, src_mask)

        # project to vocab logits
        logits = self.output_proj(x)
        return logits

