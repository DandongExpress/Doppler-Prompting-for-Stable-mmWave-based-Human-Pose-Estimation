"""Example: replace mmDiff front-end fusion with PULSE prompting.

mmDiff consumes multi-frame radar features. Here we keep a stub backbone and
show how to feed PULSE-prompted spatial tokens into it.

This file is illustrative — wire ``spatial_tokens`` into your real mmDiff
encoder instead of ``StubBackbone``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pulse import PULSEPrompting


class StubBackbone(nn.Module):
    """Stand-in for mmDiff's pose backbone (not the official model)."""

    def __init__(self, embed_dim: int = 32, num_joints: int = 17) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_joints * 3),
        )
        self.num_joints = num_joints

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        # spatial_tokens: [B, N_s, d] -> mean pool -> pose
        pooled = spatial_tokens.mean(dim=1)
        return self.head(pooled).view(-1, self.num_joints, 3)


class mmDiffWithPULSE(nn.Module):
    def __init__(self, k_frames: int = 9) -> None:
        super().__init__()
        self.pulse = PULSEPrompting(
            range_bins=64,
            angle_bins=64,
            doppler_bins=16,
            patch_size=(4, 4),
            embed_dim=32,
            neighborhood=3,
            beta=1.0,
        )
        self.backbone = StubBackbone(embed_dim=32)
        self.k_frames = k_frames

    def forward(self, h_window: torch.Tensor) -> torch.Tensor:
        # h_window: [B, K, R, A, D]
        tokens = self.pulse(h_window)
        return self.backbone(tokens)


if __name__ == "__main__":
    model = mmDiffWithPULSE(k_frames=9)
    x = torch.randn(2, 9, 64, 64, 16)
    y = model(x)
    print("plugin_mmdiff:", tuple(x.shape), "->", tuple(y.shape))
