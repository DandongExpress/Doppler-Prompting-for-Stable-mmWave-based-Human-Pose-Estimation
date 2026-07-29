"""PULSE: Prompting Using Local Spectral Estimates.

Doppler Prompting for Stable mmWave-based Human Pose Estimation (ICML 2026).

Public API
----------
- ``PULSE`` / ``PULSEConfig``: full pose estimator
- ``PULSEPrompting``: plug-in front-end (confidence-gated local cross-attention)
"""

from .model import PULSE, PULSEConfig
from .prompting import PULSEPrompting

__all__ = ["PULSE", "PULSEConfig", "PULSEPrompting"]
__version__ = "1.0.0"
