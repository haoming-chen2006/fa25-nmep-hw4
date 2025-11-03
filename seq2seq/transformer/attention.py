from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self, num_heads: int, embedding_dim: int, qk_length: int, value_length: int
    ):
        """
        The Multi-Head Attention layer will take in Q, K, and V
        matrices and will output an attention matrix of shape <TODO>.

        First, Q, K, and V should be projected to have
        a shape of (B, T, C) where C = num_heads * qk_length
        (OR value_length). You are then expected to split
        the C dimension into num_heads different heads, each
        with shape (B, T, vec_length).

        Next, you will compute the scaled dot-product attention
        between Q, K, and V.

        Finally, you will concatenate the heads and project the
        output to have a shape of (B, T, C).

        Check out the `masked_fill` method in PyTorch to help
        you implement the masking step!
        """
        super().__init__()
        self.num_heads = num_heads
        self.qk_length = qk_length
        self.value_length = value_length

        # We'll use linear projections for Q, K, V and an output projection
        self.q_proj = nn.Linear(embedding_dim, num_heads * qk_length)
        self.k_proj = nn.Linear(embedding_dim, num_heads * qk_length)
        self.v_proj = nn.Linear(embedding_dim, num_heads * value_length)
        self.out_proj = nn.Linear(num_heads * value_length, embedding_dim)
        self.dropout = nn.Dropout(0.0)

    def split_heads(self, x: torch.Tensor, vec_length: int) -> torch.Tensor:
        """
        Split the C dimension of the input tensor into num_heads
        different heads, each with shape (B, T, vec_length).
        Hint: check out the `view` and 'permute` methods in PyTorch to help
        you reshape the tensor.

        Args:
            x: torch.Tensor of shape (B, T, C), where C = num_heads * vec_length
            vec_length: int, the length of the query/key/value vectors

        Returns:
            torch.Tensor of shape (B, num_heads, T, vec_length)
        """
        B, T, C = x.size()

        assert C // self.num_heads == vec_length, (
            "Input tensor does not have the correct shape for splitting."
        )
        # reshape to (B, T, num_heads, vec_length) then permute to (B, num_heads, T, vec_length)
        x = x.view(B, T, self.num_heads, vec_length).permute(0, 2, 1, 3)
        return x

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine the num_heads different heads into a single tensor.
        Hint: check out the `contiguous` method in PyTorch to help
        you reshape the tensor.

        Args:
            x: torch.Tensor of shape (B, num_heads, T, vec_length)

        Returns:
            torch.Tensor of shape (B, T, num_heads * vec_length)
        """
        B, num_heads, T, vec_length = x.size()
        # permute back to (B, T, num_heads, vec_length) then reshape to (B, T, num_heads * vec_length)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, num_heads * vec_length)
        return x

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the scaled dot-product attention given Q, K, and V.
        This is where the pad_mask and causal_mask are applied.

        Args:
            Q: torch.Tensor of shape (B, num_heads, T, qk_length)
            K: torch.Tensor of shape (B, num_heads, T, qk_length)
            V: torch.Tensor of shape (B, num_heads, T, value_length)
            mask: Optional boolean torch.Tensor, broadcastable to (B, num_heads, T, T).
        """
        # Q, K: (B, num_heads, T, qk_length)
        # V: (B, num_heads, T, value_length)
        # compute scores
        dk = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk ** 0.5)

        if mask is not None:
            # mask expected to be broadcastable to (B, num_heads, T, T)
            # mask positions with True should be set to -inf
            scores = scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return out

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        The forward pass of the Multi-Head Attention layer.

        Args:
            Q: torch.Tensor of shape (B, T, C)
            K: torch.Tensor of shape (B, T, C)
            V: torch.Tensor of shape (B, T, C)
            mask: Optional torch.Tensor of shape (B, T, T) or None

        Returns:
            torch.Tensor of shape (B, T, C)
        """
        # Project inputs
        B, T, C = Q.size()

        q = self.q_proj(Q)  # (B, T, num_heads * qk_length)
        k = self.k_proj(K)
        v = self.v_proj(V)

        # split heads
        q = self.split_heads(q, self.qk_length)
        k = self.split_heads(k, self.qk_length)
        v = self.split_heads(v, self.value_length)

        # apply scaled dot-product attention
        if mask is not None:
            # mask may be (B, T, T) or (B, 1, 1, T) or (B, num_heads, T, T)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (B, 1, T, T)
            # broadcast to (B, num_heads, T, T)
            mask = mask.expand(-1, self.num_heads, -1, -1)

        attn_out = self.scaled_dot_product_attention(q, k, v, mask=mask)

        # attn_out: (B, num_heads, T, value_length). Combine heads
        combined = self.combine_heads(attn_out)

        # final linear projection
        out = self.out_proj(combined)
        return out



class FeedForwardNN(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        """
        The Feed-Forward Neural Network layer will take in
        an input tensor of shape (B, T, C) and will output
        a tensor of the same shape.

        The FFNN will have two linear layers, with a ReLU
        activation function in between.

        Args:
            hidden_dim: int, the size of the hidden layer
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.fc1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the FeedForwardNN.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
