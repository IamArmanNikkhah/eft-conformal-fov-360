# src/model.py

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        device = x.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.pe(positions)
        return x + pos_emb


class PooledFoVTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        d_model: int = 256,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = LearnedPositionalEncoding(d_model, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.prefetch_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )

        self.deadline_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )

    def forward(self, x: torch.Tensor):
        # x: (batch, seq len, input dim)
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.encoder(h)
        last_step = h[:, -1, :]
        y_prefetch = self.prefetch_head(last_step)
        y_deadline = self.deadline_head(last_step)
        return y_prefetch, y_deadline


if __name__ == "__main__":
    batch_size = 8
    seq_len = 15
    input_dim = 2 # (Yaw, Pitch)

    model = PooledFoVTransformer(input_dim=input_dim)
    dummy_x = torch.zeros(batch_size, seq_len, input_dim)
    y_prefetch, y_deadline = model(dummy_x)

    print("y_prefetch shape:", y_prefetch.shape)
    print("y_deadline shape:", y_deadline.shape)