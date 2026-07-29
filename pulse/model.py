"""Full PULSE model: RAD tensor -> 3D joint coordinates (paper Sec. 3).

Pipeline
--------
1. Dual-domain features  S_t (Eq. 2), V_t (Eq. 3)
2. Tokenization          spatial patches + per-cell Doppler tokens (Sec. 3.3)
3. Controlled prompting  confidence-gated local cross-attention (Sec. 3.4)
4. Pose regression       Transformer backbone + MLP head (Sec. 3.6)

Multi-frame mode (Sec. 3.5) aggregates Doppler prompts over a K-frame window
while keeping the spatial branch on the current frame only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiframe import MultiFrameDopplerAggregator
from .prompting import ControlledPrompting
from .tokenizer import DopplerTokenizer, SpatialTokenizer


@dataclass
class PULSEConfig:
    """Hyperparameters aligned with paper Table 6 / README."""

    # Native input size (resampled to work_* before tokenization)
    in_range: int = 256
    in_angle: int = 128
    doppler_bins: int = 16

    work_range: int = 64
    work_angle: int = 64

    patch_size: int = 4
    embed_dim: int = 32
    num_heads: int = 4
    neighbor_patches: int = 3  # 3x3 patch neighbourhood N(i)
    beta: float = 1.0

    depth: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    num_joints: int = 17
    reg_hidden: int = 512

    # K=1 -> single-frame (1F); K>1 -> multi-frame Doppler aggregation (KF)
    k_frames: int = 1
    eps: float = 1e-6

    def __post_init__(self) -> None:
        assert self.work_range % self.patch_size == 0
        assert self.work_angle % self.patch_size == 0
        assert self.embed_dim % self.num_heads == 0
        assert self.neighbor_patches % 2 == 1
        assert self.k_frames >= 1


class PULSE(nn.Module):
    """Prompting Using Local Spectral Estimates.

    Input:
        single-frame  H_t      [B, R, A, D]
        multi-frame   H_window [B, K, R, A, D]  (K == cfg.k_frames)
    Output:
        pose P_t [B, J, 3]
    """

    def __init__(self, cfg: PULSEConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PULSEConfig()
        cfg = self.cfg

        self.spatial_tokenizer = SpatialTokenizer(
            embed_dim=cfg.embed_dim,
            patch_size=cfg.patch_size,
            work_range=cfg.work_range,
            work_angle=cfg.work_angle,
        )
        self.doppler_tokenizer = DopplerTokenizer(
            doppler_bins=cfg.doppler_bins, embed_dim=cfg.embed_dim
        )
        self.prompting = ControlledPrompting(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            patch_size=cfg.patch_size,
            work_range=cfg.work_range,
            work_angle=cfg.work_angle,
            neighborhood=cfg.neighbor_patches,
            beta=cfg.beta,
        )
        self.aggregator = MultiFrameDopplerAggregator(eps=cfg.eps)

        n_s = (cfg.work_range // cfg.patch_size) * (
            cfg.work_angle // cfg.patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, n_s, cfg.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=int(cfg.embed_dim * cfg.mlp_ratio),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.depth)
        self.norm = nn.LayerNorm(cfg.embed_dim)

        # Lightweight pooling + MLP regression head f_reg (Eq. 14)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        nn.init.trunc_normal_(self.pool_query, std=0.02)
        self.pool_attn = nn.MultiheadAttention(
            cfg.embed_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=True
        )
        self.reg_head = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.reg_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.reg_hidden, cfg.reg_hidden),
            nn.GELU(),
            nn.Linear(cfg.reg_hidden, cfg.num_joints * 3),
        )

    # ------------------------------------------------------------------ dual domain
    def _dual_domain(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """S_t = mean_d |H|; V_t = |H|; bilinear resample to work resolution."""
        cfg = self.cfg
        mag = h.abs()
        s = mag.mean(dim=-1).unsqueeze(1)                 # [B, 1, R, A]
        v = mag.permute(0, 3, 1, 2).contiguous()          # [B, D, R, A]
        size = (cfg.work_range, cfg.work_angle)
        if (h.shape[1], h.shape[2]) != size:
            s = F.interpolate(s, size=size, mode="bilinear", align_corners=False)
            v = F.interpolate(v, size=size, mode="bilinear", align_corners=False)
        v = v.permute(0, 2, 3, 1).contiguous()            # [B, R', A', D]
        return s, v

    # ------------------------------------------------------------------ prompting
    def _prompted_tokens(self, h: torch.Tensor) -> torch.Tensor:
        """Return motion-conditioned spatial tokens [B, N_s, d]."""
        if h.dim() == 5:
            # Multi-frame: spatial from last frame; Doppler confidence-weighted
            b, k, r, a, d = h.shape
            s, _ = self._dual_domain(h[:, -1])
            spatial = self.spatial_tokenizer(s) + self.pos_embed

            flat = h.reshape(b * k, r, a, d)
            _, v = self._dual_domain(flat)
            tokens, gate = self.doppler_tokenizer(v)
            wr, wa = tokens.shape[1], tokens.shape[2]
            tokens = tokens.view(b, k, wr, wa, -1)
            gate = gate.view(b, k, wr, wa, 1)
            agg_tokens, agg_gate = self.aggregator(tokens, gate)
            return self.prompting(spatial, agg_tokens, agg_gate)

        if h.dim() != 4:
            raise ValueError(f"Expected [B,R,A,D] or [B,K,R,A,D], got {tuple(h.shape)}")

        s, v = self._dual_domain(h)
        spatial = self.spatial_tokenizer(s) + self.pos_embed
        doppler, gate = self.doppler_tokenizer(v)
        return self.prompting(spatial, doppler, gate)

    # ------------------------------------------------------------------ forward
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        tokens = self._prompted_tokens(h)
        z = self.norm(self.transformer(tokens))

        query = self.pool_query.expand(z.shape[0], -1, -1)
        pooled, _ = self.pool_attn(query, z, z)
        out = self.reg_head(pooled.squeeze(1))
        return out.view(-1, self.cfg.num_joints, 3)

    @staticmethod
    def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-frame L_pos (Eq. 17): mean L2 over joints. No temporal loss."""
        return torch.norm(pred - target, dim=-1).mean()


if __name__ == "__main__":
    cfg = PULSEConfig()
    model = PULSE(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PULSE params: {n_params / 1e6:.2f} M")

    x = torch.randn(2, 64, 64, 16)
    y = model(x)
    print("1F in/out:", tuple(x.shape), "->", tuple(y.shape))

    cfg_k = PULSEConfig(k_frames=9)
    model_k = PULSE(cfg_k)
    xk = torch.randn(2, 9, 64, 64, 16)
    yk = model_k(xk)
    print("KF in/out:", tuple(xk.shape), "->", tuple(yk.shape))

    loss = PULSE.compute_loss(y, torch.randn_like(y))
    loss.backward()
    print("loss ok:", float(loss))
