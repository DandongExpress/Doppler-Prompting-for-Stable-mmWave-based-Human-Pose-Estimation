"""Spatial and Doppler tokenization (paper Sec. 3.3).

Spatial tokens: non-overlapping ``P_r x P_a`` patches of magnitude map S_t
via a convolutional patch encoder f_s.

Doppler tokens: each range-angle cell's Doppler spectrum is embedded
independently by an MLP f_v; a gate head f_g predicts motion relevance.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatialTokenizer(nn.Module):
    """Partition S into non-overlapping patches and project to d-dim tokens.

    Input:  S  [B, 1, R, A]
    Output: T_s [B, N_s, d]  with N_s = (R / P_r) * (A / P_a)
    """

    def __init__(
        self,
        embed_dim: int = 32,
        patch_size: int = 4,
        work_range: int = 64,
        work_angle: int = 64,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.grid_r = work_range // patch_size
        self.grid_a = work_angle // patch_size
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # [B, 1, R, A] -> [B, d, grid_r, grid_a] -> [B, N_s, d]
        t = self.proj(s)
        t = t.flatten(2).transpose(1, 2)
        return self.norm(t)


class DopplerTokenizer(nn.Module):
    """Embed per-cell Doppler spectra and predict confidence gates.

    Input:  V  [B, R, A, D]
    Output: tokens [B, R, A, d], gate [B, R, A, 1]
            where g_{t,j} = sigmoid(f_g(t^v_{t,j}))  (Eq. 8)
    """

    def __init__(self, doppler_bins: int = 16, embed_dim: int = 32) -> None:
        super().__init__()
        self.f_v = nn.Sequential(
            nn.Linear(doppler_bins, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.f_g = nn.Linear(embed_dim, 1)

    def forward(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.norm(self.f_v(v))
        gate = torch.sigmoid(self.f_g(tokens))
        return tokens, gate
