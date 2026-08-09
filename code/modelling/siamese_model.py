"""Siamese DNN architecture shared by model_training.ipynb and model_validation.ipynb.

Encoder + 4-way interaction + head. Linear output (Fisher-z space).
Keep this file in sync across both notebooks — model_validation must mirror
model_training bit-for-bit to load the trained checkpoint correctly.
"""

import torch.nn as nn
import torch


class SiameseEncoder(nn.Module):
    """Shared MLP applied independently to both item embeddings."""

    def __init__(self, emb_dim, encoder_dims, dropout):
        super().__init__()
        layers, prev = [], emb_dim
        for h in encoder_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        return self.net(x)


class SiameseDNN(nn.Module):
    """Encoder + 4-way interaction + head. Linear output (Fisher-z space)."""

    def __init__(self, emb_dim, aux_dim, encoder_dims, head_dims, dropout, use_skip):
        super().__init__()
        self.encoder = SiameseEncoder(emb_dim, encoder_dims, dropout)
        e = self.encoder.out_dim

        # Interaction: concat[h1, h2, h1*h2, |h1-h2|]. h1*h2 and |h1-h2| are
        # symmetric in pair ordering, which matches the symmetry of Pearson r.
        head_in = 3 * e + aux_dim

        layers, prev = [], head_in
        for h in head_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.head = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)

        self.use_skip = use_skip
        if use_skip:
            # Linear shortcut from raw aux straight to the prediction. Lets the
            # model anchor on global_sim / cross-encoder signal even before the
            # head learns to mix everything.
            self.aux_skip = nn.Linear(aux_dim, 1)

    def forward(self, e1, e2, aux):
        h1 = self.encoder(e1)
        h2 = self.encoder(e2)
        inter = torch.cat([h1 + h2, h1 * h2, (h1 - h2).abs()], dim=-1)
        x = torch.cat([inter, aux], dim=-1)
        z = self.out(self.head(x)).squeeze(-1)
        if self.use_skip:
            z = z + self.aux_skip(aux).squeeze(-1)
        return z  # Fisher-z space
