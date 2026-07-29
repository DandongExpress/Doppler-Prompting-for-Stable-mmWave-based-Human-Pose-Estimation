"""Confidence-gated local cross-attention prompting (paper Sec. 3.4).

Doppler tokens act as screened motion prompts: each spatial token attends
only within a local neighbourhood N(i) on the shared R-A lattice, with
attention logits biased by the confidence gate g_{t,j}.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlledPrompting(nn.Module):
    """Locality-restricted, confidence-gated cross-attention.

    Attention (Eq. 9)::

        alpha_ij = softmax_{j in N(i)}(
            (t^s_i W_Q)(t^v_j W_K)^T / sqrt(d_k) + beta * g_j
        )

    Residual update (Eq. 12)::

        t~^s_i = t^s_i + lambda_i * c_i,  lambda_i = sigmoid(f_lambda(t^s_i))
    """

    def __init__(
        self,
        embed_dim: int = 32,
        num_heads: int = 4,
        patch_size: int = 4,
        work_range: int = 64,
        work_angle: int = 64,
        neighborhood: int = 3,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        assert neighborhood % 2 == 1, "neighborhood must be odd (e.g. 3)"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.beta = beta
        self.patch_size = patch_size
        self.work_range = work_range
        self.work_angle = work_angle
        self.grid_r = work_range // patch_size
        self.grid_a = work_angle // patch_size
        self.win = neighborhood * patch_size
        self.pad = (neighborhood // 2) * patch_size

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.f_lambda = nn.Linear(embed_dim, 1)

    def _gather_neighborhoods(self, grid: torch.Tensor) -> torch.Tensor:
        """Gather local cell windows for each spatial patch.

        Args:
            grid: [B, C, R, A]
        Returns:
            [B, N_s, L, C] where L = win * win
        """
        b, c, _, _ = grid.shape
        cols = F.unfold(
            grid,
            kernel_size=self.win,
            stride=self.patch_size,
            padding=self.pad,
        )
        n_s = cols.shape[-1]
        length = self.win * self.win
        return cols.view(b, c, length, n_s).permute(0, 3, 2, 1).contiguous()

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        doppler_tokens: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            spatial_tokens: [B, N_s, d]
            doppler_tokens: [B, R, A, d]
            gate:           [B, R, A, 1]
        Returns:
            Updated spatial tokens [B, N_s, d]
        """
        b, n_s, _ = spatial_tokens.shape
        h, hd = self.num_heads, self.head_dim

        dv = doppler_tokens.permute(0, 3, 1, 2).contiguous()
        gg = gate.permute(0, 3, 1, 2).contiguous()
        ones = torch.ones(
            b, 1, self.work_range, self.work_angle,
            device=dv.device, dtype=dv.dtype,
        )

        nbr_tok = self._gather_neighborhoods(dv)
        nbr_gate = self._gather_neighborhoods(gg)
        nbr_mask = self._gather_neighborhoods(ones)
        length = nbr_tok.shape[2]

        q = self.w_q(spatial_tokens).view(b, n_s, h, hd)
        k = self.w_k(nbr_tok).view(b, n_s, length, h, hd)
        v = self.w_v(nbr_tok).view(b, n_s, length, h, hd)

        scores = torch.einsum("bnhd,bnlhd->bnhl", q, k) * self.scale
        scores = scores + self.beta * nbr_gate.squeeze(-1).unsqueeze(2)

        mask = nbr_mask.squeeze(-1).unsqueeze(2)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)

        context = torch.einsum("bnhl,bnlhd->bnhd", attn, v).reshape(b, n_s, h * hd)
        context = self.proj(context)

        lam = torch.sigmoid(self.f_lambda(spatial_tokens))
        return spatial_tokens + lam * context


class PULSEPrompting(nn.Module):
    """Plug-in front-end: RAD tensor -> motion-conditioned spatial tokens.

    Drop-in replacement for naive magnitude/Doppler fusion in external
    backbones (mmDiff, milliMamba, ...). See ``examples/``.
    """

    def __init__(
        self,
        range_bins: int = 64,
        angle_bins: int = 64,
        doppler_bins: int = 16,
        patch_size: tuple[int, int] | int = (4, 4),
        embed_dim: int = 32,
        num_heads: int = 4,
        neighborhood: int = 3,
        beta: float = 1.0,
        multi_frame: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        from .tokenizer import DopplerTokenizer, SpatialTokenizer
        from .multiframe import MultiFrameDopplerAggregator

        if isinstance(patch_size, int):
            pr = pa = patch_size
        else:
            pr, pa = patch_size
        assert pr == pa, "Rectangular patches are not supported; use square P x P."

        self.range_bins = range_bins
        self.angle_bins = angle_bins
        self.doppler_bins = doppler_bins
        self.patch_size = pr
        self.embed_dim = embed_dim
        self.multi_frame = multi_frame
        self.eps = eps

        self.spatial_tokenizer = SpatialTokenizer(
            embed_dim=embed_dim,
            patch_size=pr,
            work_range=range_bins,
            work_angle=angle_bins,
        )
        self.doppler_tokenizer = DopplerTokenizer(
            doppler_bins=doppler_bins, embed_dim=embed_dim
        )
        self.prompting = ControlledPrompting(
            embed_dim=embed_dim,
            num_heads=num_heads,
            patch_size=pr,
            work_range=range_bins,
            work_angle=angle_bins,
            neighborhood=neighborhood,
            beta=beta,
        )
        n_s = (range_bins // pr) * (angle_bins // pr)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_s, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.aggregator = MultiFrameDopplerAggregator(eps=eps)

    def _dual_domain(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build S_t (Eq. 2) and V_t (Eq. 3); resample to work resolution."""
        mag = h.abs()
        s = mag.mean(dim=-1, keepdim=False).unsqueeze(1)
        v = mag.permute(0, 3, 1, 2).contiguous()
        size = (self.range_bins, self.angle_bins)
        if (h.shape[1], h.shape[2]) != size:
            s = F.interpolate(s, size=size, mode="bilinear", align_corners=False)
            v = F.interpolate(v, size=size, mode="bilinear", align_corners=False)
        v = v.permute(0, 2, 3, 1).contiguous()
        return s, v

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: single-frame [B, R, A, D] or multi-frame [B, K, R, A, D]
        Returns:
            Prompted spatial tokens [B, N_s, d]
        """
        if h.dim() == 5:
            return self._forward_multiframe(h)
        if h.dim() != 4:
            raise ValueError(f"Expected [B,R,A,D] or [B,K,R,A,D], got {tuple(h.shape)}")
        return self._forward_single(h)

    def _forward_single(self, h: torch.Tensor) -> torch.Tensor:
        s, v = self._dual_domain(h)
        spatial = self.spatial_tokenizer(s) + self.pos_embed
        doppler, gate = self.doppler_tokenizer(v)
        return self.prompting(spatial, doppler, gate)

    def _forward_multiframe(self, h_window: torch.Tensor) -> torch.Tensor:
        # Spatial branch uses the last frame only; Doppler is aggregated.
        b, k, r, a, d = h_window.shape
        h_last = h_window[:, -1]
        s, _ = self._dual_domain(h_last)
        spatial = self.spatial_tokenizer(s) + self.pos_embed

        flat = h_window.reshape(b * k, r, a, d)
        _, v = self._dual_domain(flat)
        tokens, gate = self.doppler_tokenizer(v)
        # [B*K, R', A', d] -> [B, K, R', A', d]
        wr, wa = tokens.shape[1], tokens.shape[2]
        tokens = tokens.view(b, k, wr, wa, -1)
        gate = gate.view(b, k, wr, wa, 1)
        agg_tokens, agg_gate = self.aggregator(tokens, gate)
        return self.prompting(spatial, agg_tokens, agg_gate)
