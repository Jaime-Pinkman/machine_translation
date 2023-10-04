import torch
from torch import nn

from .blocks import DecoderBlock


class Decoder(nn.Module):  # type: ignore
    def __init__(
        self,
        trg_vocab_size: int,
        embed_size: int,
        num_layers: int,
        heads: int,
        forward_expansion: int,
        dropout: int,
        device: str,
        max_length: int,
    ):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size).to(device)
        self.position_embedding = nn.Embedding(max_length, embed_size).to(device)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout).to(device)
                for _ in range(self.num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        src_mask: torch.Tensor,
        trg_mask: torch.Tensor,
    ) -> torch.Tensor:
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out
