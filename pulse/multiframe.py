"""Multi-frame Doppler prompt aggregation (paper Sec. 3.5).

When a short window of K frames is available, Doppler tokens and gates are
computed per frame with the same f_v / f_g, then confidence-weighted
aggregated on the shared R-A lattice (Eq. 11). The spatial backbone is
unchanged and still consumes the current-frame magnitude map.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiFrameDopplerAggregator(nn.Module):
    """Confidence-weighted aggregation of Doppler tokens over K frames.

    bar{t}^v_{t,j} = sum_tau g_j^(tau) t^{v,(tau)}_j
                     / (sum_tau g_j^(tau) + eps)
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        tokens: torch.Tensor,
        gates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: [B, K, R, A, d]
            gates:  [B, K, R, A, 1]
        Returns:
            aggregated tokens [B, R, A, d],
            aggregated gate   [B, R, A, 1]  (mean gate over the window)
        """
        if tokens.dim() != 5:
            raise ValueError(
                f"Expected tokens [B, K, R, A, d], got {tuple(tokens.shape)}"
            )
        weight = gates  # [B, K, R, A, 1]
        numer = (weight * tokens).sum(dim=1)
        denom = weight.sum(dim=1) + self.eps
        agg_tokens = numer / denom
        agg_gate = gates.mean(dim=1)
        return agg_tokens, agg_gate
