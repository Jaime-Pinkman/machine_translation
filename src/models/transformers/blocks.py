import torch
from torch import nn

from .multihead_attention import SelfAttention


class TransformerBlock(nn.Module):  # type: ignore
    def __init__(
        self, embed_size: int, heads: int, forward_expansion: int, dropout: int
    ):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention = self.attention(values, keys, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.ff(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class DecoderBlock(nn.Module):  # type: ignore
    def __init__(
        self, embed_size: int, heads: int, forward_expansion: int, dropout: int
    ):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, forward_expansion, dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        value: torch.Tensor,
        key: torch.Tensor,
        src_mask: torch.Tensor,
        trg_mask: torch.Tensor,
    ) -> torch.Tensor:
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out
