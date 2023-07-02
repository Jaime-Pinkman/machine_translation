import torch
from torch import nn


from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):  # type: ignore
    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        src_pad_idx: str,
        embed_size: int = 256,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        heads: int = 8,
        forward_expansion: int = 4,
        dropout: int = 0,
        max_length: int = 100,
        device: str = "cuda",
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_encoder_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_decoder_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = 1
        self.device = device

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N,
            1,
            trg_len,
            trg_len,
        )
        return trg_mask.to(self.device)

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
