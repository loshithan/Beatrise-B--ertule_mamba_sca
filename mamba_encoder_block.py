"""
MambaEncoderBlock — One encoder block as shown in Figure 4.4.

Structure (directly from paper Figure 4.4 and Section 4.3.2):

    residual = input
    x = RMSNorm(input)          ← pre-norm (RMS Normalization label in Figure 4.4)
    x = CoreMambaBlock(x)       ← inner S6 block (no residual inside)
    output = x + residual       ← residual connection (labeled "Residual" in Figure 4.4)

The paper uses N_blocks = 2 encoder blocks per direction (forward/backward).
Total = 4 Mamba blocks (2 fwd + 2 bwd), each with INDEPENDENT weights.

Paper: "Deep Learning-based Side-Channel Attack: Mamba Approach"
       Beatrise Bertule, Radboud University, January 2026.
       Section 4.3.2, Figure 4.4, Algorithm 5.
"""

import torch
import torch.nn as nn

from rms_norm        import RMSNorm
from core_mamba_block import CoreMambaBlock


class MambaEncoderBlock(nn.Module):
    """
    Single encoder block: RMSNorm → CoreMambaBlock + residual.

    Args:
        d_model (int): Feature dimension. Paper specifies 64.
        d_state (int): SSM state size. Standard Mamba default: 16.
        d_conv  (int): Causal conv kernel. Standard Mamba default: 4.
        expand  (int): Inner expansion factor. Standard Mamba default: 2.
    """

    def __init__(
        self,
        d_model : int = 64,
        d_state : int = 16,
        d_conv  : int = 4,
        expand  : int = 2,
    ):
        super().__init__()

        # Pre-norm: RMSNorm — explicitly labeled in Figure 4.4
        self.norm  = RMSNorm(d_model)

        # Inner S6 block — no residual inside (residual is here)
        self.mamba = CoreMambaBlock(
            d_model = d_model,
            d_state = d_state,
            d_conv  = d_conv,
            expand  = expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]
        Returns:
            [B, L, d_model]
        """
        residual = x
        x = self.norm(x)           # RMSNorm (pre-norm)
        x = self.mamba(x)          # Core S6 (no internal residual)
        return x + residual        # residual connection
