"""
Paper-Faithful Mamba Model for Side-Channel Attack.

EXACT reconstruction of the architecture described in:
    "Deep Learning-based Side-Channel Attack: Mamba Approach"
    Beatrise Bertule, Radboud University Nijmegen, January 2026.

Every design decision, parameter value, and structural choice below
is directly sourced from the paper. No additions, no modifications.

─────────────────────────────────────────────────────────────────────
ARCHITECTURE OVERVIEW (Section 4.3.2, Figure 4.4, Algorithm 5)
─────────────────────────────────────────────────────────────────────

Input: power trace x ∈ R^(B × T)
    CW-Target : T = feature-selected window (OPOI, ~700 region, 12k raw)
    ASCADv1   : T = 700 (pre-selected)

Module 1 — Convolution Module (Section 4.3.2, Algorithm 5 lines 1-3):
    Reshape  : (B, T)       → (B, 1, T)
    Conv1d   : kernel=3, stride=3, out_channels=64, no padding
    Reshape  : (B, 64, L)   → (B, L, 64)
    L = floor((T - 3) / 3 + 1)   e.g. T=700 → L=233

Module 2 — Bidirectional Mamba (Section 4.3.2, Algorithm 5 lines 4-8):
    z_fwd = z                         (original order)
    z_bwd = Flip(z, dim=1)            (reversed order)

    For i in range(N_blocks=2):       (2 forward + 2 backward = 4 total)
        z_fwd = MambaEncoderBlock_fwd[i](z_fwd)
        z_bwd = MambaEncoderBlock_bwd[i](z_bwd)

    Each MambaEncoderBlock:
        residual = input
        x = RMSNorm(input)
        x = CoreMambaBlock(x)
        output = x + residual

    h_fwd = z_fwd[:, -1, :]          last token (B, 64)
    h_bwd = z_bwd[:, -1, :]          last token (B, 64)
    h = Concat([h_fwd, h_bwd], -1)   (B, 128)

Module 3 — Classification (Section 4.3.2):
    Linear(128, 256)                  logits over 256 S-box values

─────────────────────────────────────────────────────────────────────
HYPERPARAMETERS (Table 4.2)
─────────────────────────────────────────────────────────────────────
    d_model      = 64
    N_blocks     = 2   (per direction)
    conv kernel  = 3
    conv stride  = 3
    num_classes  = 256
    ~148,000 learnable parameters total

─────────────────────────────────────────────────────────────────────
TRAINING CONFIG (Tables 4.3, 4.4 and Section 4.4.2)
─────────────────────────────────────────────────────────────────────
    batch_size   = 768
    lr           = 5e-3
    optimizer    = Adam with weight decay
    max_epochs   = 100
    loss         = CrossEntropyLoss
    labels       = Identity (ID) leakage model → 256 classes
    val_metric   = mean Guessing Entropy (NOT loss, NOT accuracy)
    train traces = 40,000
    val traces   = 10,000
    test traces  = 10,000

─────────────────────────────────────────────────────────────────────
WHAT IS DELIBERATELY ABSENT (tested and excluded by paper, Sec 4.3.3)
─────────────────────────────────────────────────────────────────────
    - No positional encoding
    - No input LayerNorm / input scaling
    - No dropout anywhere
    - No multi-layer conv or BatchNorm after conv
    - No attention mechanism
    - No pooling of any kind (only last token)
    - No hidden layer in classifier
    - No GNN / GAT components
    - kernel=3 stride=1 was tested → rejected
    - kernel=5 stride=3 was tested → rejected
    - kernel=5 stride=5 was tested → rejected
    - d_model=32 was tested        → rejected
    - N_blocks=4 per direction     → rejected
"""

import torch
import torch.nn as nn

from mamba_encoder_block import MambaEncoderBlock


class MambaSCAModel(nn.Module):
    """
    Exact paper architecture for Mamba-based profiled side-channel attack.

    Args:
        trace_length (int): Number of input time samples T.
                            700 for ASCADv1 (pre-selected).
                            Feature-selected length for CW-Target.
        d_model      (int): Embedding dimension. Paper: 64. (Table 4.2)
        n_blocks     (int): Mamba encoder blocks per direction. Paper: 2. (Table 4.2)
        num_classes  (int): Output classes = S-box values. Paper: 256.
        d_state      (int): SSM hidden state size. Standard Mamba default: 16.
        d_conv       (int): Causal conv kernel in SSM. Standard Mamba default: 4.
        expand       (int): Inner expansion in SSM. Standard Mamba default: 2.
    """

    def __init__(
        self,
        trace_length : int = 700,
        d_model      : int = 64,     # Table 4.2 — exact value
        n_blocks     : int = 2,      # Table 4.2 — exact value
        num_classes  : int = 256,    # 256 S-box output values, ID model
        d_state      : int = 8,      # d_state=8 → ~151,808 params, closest to paper's ~148,000 (Table 4.2)
        d_conv       : int = 4,      # standard Mamba default
        expand       : int = 2,      # standard Mamba default
    ):
        super().__init__()

        self.d_model  = d_model
        self.n_blocks = n_blocks

        # ─────────────────────────────────────────────────────────────────
        # MODULE 1: Convolution Module
        # Paper Section 4.3.2, Algorithm 5 lines 1-3:
        #   "single one-dimensional convolution layer with kernel size and
        #    stride of 3" — Table 4.2, Section 4.3.3
        #
        # in_channels  = 1     (raw trace reshaped to single channel)
        # out_channels = 64    (= d_model, Table 4.2)
        # kernel_size  = 3     (Table 4.2 / Section 4.3.3)
        # stride       = 3     (Table 4.2 / Section 4.3.3)
        # padding      = 0     (not mentioned → default)
        # bias         = True  (Conv1d default, not changed by paper)
        #
        # Output length: L = floor((T - 3) / 3 + 1)
        #   T=700  → L = 233
        # ─────────────────────────────────────────────────────────────────
        self.conv = nn.Conv1d(
            in_channels  = 1,
            out_channels = d_model,   # 64
            kernel_size  = 3,
            stride       = 3,
            padding      = 0,
            bias         = True,
        )

        # ─────────────────────────────────────────────────────────────────
        # MODULE 2: Bidirectional Mamba Module
        # Paper Section 4.3.2, Figure 4.4, Algorithm 5 lines 4-8:
        #   "two parallel encoders — a forward Mamba encoder and a backward
        #    Mamba encoder, each composed of a stack of Mamba blocks followed
        #    by residual connections and layer normalization"
        #
        # - fwd_blocks and bwd_blocks have INDEPENDENT weights (not shared)
        # - n_blocks = 2 per direction → 4 total MambaEncoderBlocks
        # ─────────────────────────────────────────────────────────────────
        self.fwd_blocks = nn.ModuleList([
            MambaEncoderBlock(
                d_model = d_model,
                d_state = d_state,
                d_conv  = d_conv,
                expand  = expand,
            )
            for _ in range(n_blocks)    # 2 forward blocks
        ])

        self.bwd_blocks = nn.ModuleList([
            MambaEncoderBlock(
                d_model = d_model,
                d_state = d_state,
                d_conv  = d_conv,
                expand  = expand,
            )
            for _ in range(n_blocks)    # 2 backward blocks, independent weights
        ])

        # ─────────────────────────────────────────────────────────────────
        # MODULE 3: Classification Module
        # Paper Section 4.3.2:
        #   "single linear transformation layer that maps the high-dimensional
        #    representations to the target class space"
        #
        # Input:  128 = 2 * d_model  (fwd last token + bwd last token concatenated)
        # Output: 256 = num_classes  (all possible AES S-box output values)
        # No activation, no hidden layer, no dropout, no BatchNorm
        # ─────────────────────────────────────────────────────────────────
        self.classifier = nn.Linear(d_model * 2, num_classes)  # Linear(128, 256)

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Implements Algorithm 5 exactly.

        Args:
            x: Raw power trace, shape [B, T]
               B = batch size (paper uses 768)
               T = trace length (700 for ASCADv1)

        Returns:
            logits: [B, 256]  — unnormalized, passed to CrossEntropyLoss
        """
        # ── Algorithm 5, lines 1-3: Convolution Module ───────────────────
        # Line 1: Reshape (B, T) → (B, 1, T)
        x = x.unsqueeze(1)                          # (B, 1, T)

        # Line 2: Conv1d → (B, d_model, L)
        z = self.conv(x)                            # (B, 64, L)

        # Line 3: Reshape → (B, L, d_model)
        z = z.transpose(1, 2)                       # (B, L, 64)

        # ── Algorithm 5, lines 4-5: Set up fwd/bwd inputs ────────────────
        # Line 4: forward input = original
        z_fwd = z                                   # (B, L, 64)

        # Line 5: backward input = flipped along temporal dim
        z_bwd = torch.flip(z, dims=[1])             # (B, L, 64)

        # ── Algorithm 5, lines 6-8: Apply encoder blocks ─────────────────
        # Lines 7-8: 2 forward + 2 backward blocks (independent weights)
        for block in self.fwd_blocks:
            z_fwd = block(z_fwd)                    # (B, L, 64)

        for block in self.bwd_blocks:
            z_bwd = block(z_bwd)                    # (B, L, 64)

        # ── Algorithm 5, lines 9-11: Extract last token + concatenate ────
        # Line 9:  last token from forward encoder
        h_fwd = z_fwd[:, -1, :]                     # (B, 64)

        # Line 10: last token from backward encoder
        h_bwd = z_bwd[:, -1, :]                     # (B, 64)

        # Line 11: concatenate along feature dim → (B, 128)
        h = torch.cat([h_fwd, h_bwd], dim=-1)       # (B, 128)

        # ── Algorithm 5, line 12: Classification ─────────────────────────
        logits = self.classifier(h)                 # (B, 256)

        return logits                               # raw logits for CrossEntropyLoss

    # ─────────────────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters. Paper reports ~148,000."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
