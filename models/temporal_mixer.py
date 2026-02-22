import torch
from torch import nn


class TemporalTokenMixer(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # tokens: (B, N, D) where N = T * M (token-preserving)
        return self.encoder(tokens, mask=attn_mask)
