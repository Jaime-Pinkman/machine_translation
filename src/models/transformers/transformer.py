import torch
from torch import nn
import torch.nn.functional as F


from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(
        self, 
        src_vocab_size, 
        trg_vocab_size, 
        src_pad_idx,
        embed_size=256, 
        num_encoder_layers=6,
        num_decoder_layers=6, 
        heads=8, 
        forward_expansion=4, 
        dropout=0, 
        max_length=100,
        device="cuda", 
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
            max_length
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_decoder_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = 1
        self.device = device
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len,
        )
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
