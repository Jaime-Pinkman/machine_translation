import torch
from torch import nn
import torch.nn.functional as F

from .multihead_attention import SelfAttention


class TransformerBlock(nn.Module):
  def __init__(self, embed_size, heads, forward_expansion, dropout):
    super(TransformerBlock, self).__init__()
    self.attention = SelfAttention(embed_size, heads)
    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)

    self.ff = nn.Sequential(
      nn.Linear(embed_size, forward_expansion*embed_size),
      nn.ReLU(),
      nn.Linear(forward_expansion*embed_size, embed_size)
    )
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, values, keys, query, mask=None):
    attention = self.attention(values, keys, query, mask)
    x = self.dropout(self.norm1(attention + query))
    forward = self.ff(x)
    out = self.dropout(self.norm2(forward + x))
    return out


class DecoderBlock(nn.Module):
  def __init__(self, embed_size, heads, forward_expansion, dropout):
    super(DecoderBlock, self).__init__()
    self.attention = SelfAttention(embed_size, heads)
    self.norm = nn.LayerNorm(embed_size)
    self.transformer_block = TransformerBlock(embed_size, heads, forward_expansion, dropout)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x, value, key, src_mask, trg_mask):
    attention = self.attention(x, x, x, trg_mask)
    query = self.dropout(self.norm(attention + x))
    out = self.transformer_block(value, key, query, src_mask)
    return out