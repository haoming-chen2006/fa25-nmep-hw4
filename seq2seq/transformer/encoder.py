from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention, FeedForwardNN
from seq2seq.data.fr_en import tokenizer


class PositionalEncoding(nn.Module):
    """
    The PositionalEncoding layer will take in an input tensor
    of shape (B, T, C) and will output a tensor of the same
    shape, but with positional encodings added to the input.

    We provide you with the full implementation for this
    homework.

    Based on:
        https://web.archive.org/web/20230315052215/https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10_000):
        """
        Initialize the PositionalEncoding layer.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape (B, T, C)
        """
        x = x.transpose(0, 1)
        x = x + self.pe[: x.size(0)]
        x = self.dropout(x)
        return x.transpose(0, 1)

class MLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = embedding_dim
        self.c_fc    = nn.Linear(self.embedding, hidden_dim)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(hidden_dim,self.embedding)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        dropout: float,
    ):
        """
        Each encoder layer will take in an embedding of
        shape (B, T, C) and will output an encoded representation
        of the same shape.

        The encoder layer will have a Multi-Head Attention layer
        and a Feed-Forward Neural Network layer.

        Remember that for each Multi-Head Attention layer, we
        need create Q, K, and V matrices from the input embedding!
        """
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qk_length = qk_length
        self.value_length = value_length

        self.q_proj = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.k_proj = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.output_proj = nn.Linear(self.embedding_dim,self.embedding_dim)
        self.dropout = nn.Dropout(0.1)
    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        The forward pass of the EncoderLayer.

        """
        B,T,C = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        score = (q@k.transpose(-2,-1))/(k.size(-1))**0.5
        score = F.softmax(score,dim = -1)
        y = score@v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.dropout(y)
        y = self.output_proj(y)
        return y


class EncoderLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        ffn_hidden_dim: int,
        qk_length: int,
        value_length: int,
        dropout: float,
    ):
        """
        Each encoder layer will take in an embedding of
        shape (B, T, C) and will output an encoded representation
        of the same shape.

        The encoder layer will have a Multi-Head Attention layer
        and a Feed-Forward Neural Network layer.

        Remember that for each Multi-Head Attention layer, we
        need create Q, K, and V matrices from the input embedding!
        """
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qk_length = qk_length
        self.value_length = value_length
        self.mlp = MLP(self.embedding_dim, self.ffn_hidden_dim)
        self.attention = CausalSelfAttention(        
        self.num_heads,
        self.embedding_dim,
        self.ffn_hidden_dim,
        self.qk_length,
        self.value_length,
        dropout
        )



    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        The forward pass of the EncoderLayer.

        """
        # Apply attention with residual connection
        x = x + self.attention(x, mask)
        
        # Apply MLP with residual connection
        x = x + self.mlp(x)
        
        return x


class Encoder(nn.Module):
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
        dropout: float,
    ):
        """
        Remember that the encoder will take in a sequence
        of tokens and will output an encoded representation
        of shape (B, T, C).

        First, we need to create an embedding from the sequence
        of tokens. For this, we need the vocab size.

        Next, we want to create a series of Encoder layers,
        each of which will have a Multi-Head Attention layer
        and a Feed-Forward Neural Network layer. For this, we
        need to specify the number of layers and the number of
        heads.

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

        # Create multiple encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                self.num_heads,
                self.embedding_dim,
                self.ffn_hidden_dim,
                self.qk_length,
                self.value_length,
                dropout
            )
            for _ in range(self.num_layers)
        ])

        self.pos_embed = PositionalEncoding(self.embedding_dim, dropout=dropout, max_len=max_length)
        self.token_embed = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(dropout) 



    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        The forward pass of the Encoder.
        """
        # x is token indices of shape (B, T)
        # Create token embeddings
        x = self.token_embed(x)
        
        # Add positional encodings
        x = self.pos_embed(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through all encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x


