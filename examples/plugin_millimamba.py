"""Example: replace milliMamba front-end fusion with PULSE prompting.

Keep the temporal / Mamba backbone unchanged; only the magnitude-Doppler
fusion stage is swapped for confidence-gated local Doppler prompting.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pulse import PULSEPrompting


class StubMilliMambaBackbone(nn.Module):
    """Stand-in for milliMamba's STCA / Mamba stack (not the official model)."""

    def __init__(self, embed_dim: int = 32, num_joints: int = 17) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.head = nn.Linear(embed_dim, num_joints * 3)
        self.num_joints = num_joints

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        h = self.proj(spatial_tokens).mean(dim=1)
        return self.head(h).view(-1, self.num_joints, 3)


class MilliMambaWithPULSE(nn.Module):
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
        self.backbone = StubMilliMambaBackbone(embed_dim=32)
        self.k_frames = k_frames

    def forward(self, h_window: torch.Tensor) -> torch.Tensor:
        tokens = self.pulse(h_window)  # multi-frame Doppler aggregation inside
        return self.backbone(tokens)


if __name__ == "__main__":
    model = MilliMambaWithPULSE(k_frames=9)
    x = torch.randn(2, 9, 64, 64, 16)
    y = model(x)
    print("plugin_millimamba:", tuple(x.shape), "->", tuple(y.shape))
