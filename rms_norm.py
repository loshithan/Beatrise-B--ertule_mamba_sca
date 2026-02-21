"""
RMSNorm — as specified in the paper (Section 4.3.2, Figure 4.4).
The paper explicitly states RMS Normalization is used inside each
MambaEncoderBlock, NOT LayerNorm.

Reference: "Deep Learning-based Side-Channel Attack: Mamba Approach"
           Beatrise Bertule, Radboud University, January 2026.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Used inside each MambaEncoderBlock as shown in Figure 4.4 of the paper.
    Applied BEFORE the Mamba block (pre-norm pattern).

    Formula:
        RMSNorm(x) = x / RMS(x) * weight
        where RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        d_model (int): Feature dimension to normalize over.
        eps (float):   Numerical stability constant. Default: 1e-5.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # learnable scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., d_model]
        Returns:
            [..., d_model]  normalized
        """
        # Compute RMS over last dimension
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight
