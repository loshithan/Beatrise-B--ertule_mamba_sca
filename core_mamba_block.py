"""
Core S6 Mamba Block — Internal SSM computation unit.

This is the INNER Mamba block (without residual connection).
The residual connection is applied OUTSIDE in MambaEncoderBlock,
exactly as shown in Figure 4.4 and Algorithm 5 of the paper.

Paper: "Deep Learning-based Side-Channel Attack: Mamba Approach"
       Beatrise Bertule, Radboud University, January 2026.
       Section 2.4.1 (Mamba Block), Figure 2.1, Figure 4.4.

Architecture (from Figure 2.1 and Section 2.4.1):
    1. Linear projection: d_model → 2 × d_inner  (SSM branch + gate branch)
    2. SSM branch:
        a. Conv1d (local depthwise, causal)
        b. SiLU activation
        c. Selective SSM (S6): input-dependent B, C, Δ
    3. Gate branch: parallel SiLU gate
    4. Element-wise multiply: SSM output ⊙ gate
    5. Linear projection back: d_inner → d_model
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def _ssm_scan(
    deltaA  : torch.Tensor,   # [B, L, d_inner, d_state]
    deltaB_u: torch.Tensor,   # [B, L, d_inner, d_state]
    C       : torch.Tensor,   # [B, L, d_state]
) -> torch.Tensor:            # [B, L, d_inner]
    """
    Sequential SSM recurrence compiled with TorchScript.
    Numerically identical to original — eliminates Python loop overhead.
    h[t] = Ā[t]*h[t-1] + B̄[t],   y[t] = C[t]·h[t]
    """
    B_b     = deltaA.shape[0]
    L       = deltaA.shape[1]
    d_inner = deltaA.shape[2]
    d_state = deltaA.shape[3]
    h = torch.zeros(B_b, d_inner, d_state, device=deltaA.device, dtype=deltaA.dtype)
    y = torch.zeros(B_b, L, d_inner,       device=deltaA.device, dtype=deltaA.dtype)
    for t in range(L):
        h = deltaA[:, t, :, :] * h + deltaB_u[:, t, :, :]
        y[:, t, :] = (h * C[:, t, None, :]).sum(dim=-1)
    return y


class CoreMambaBlock(nn.Module):
    """
    Inner Mamba S6 block WITHOUT residual connection.

    The residual is handled externally by MambaEncoderBlock.

    Default parameters follow standard Mamba implementation (Gu & Dao 2023),
    which the paper builds upon (Section 2.4.1).

    Args:
        d_model  (int):   Input/output feature dimension. Paper uses 64.
        d_state  (int):   SSM hidden state size N. Standard default: 16.
        d_conv   (int):   Causal depthwise conv kernel size. Standard: 4.
        expand   (int):   d_inner = expand * d_model. Standard: 2.
                          → d_inner = 2 * 64 = 128 for paper's d_model=64.
        dt_rank        :  Rank of Δ projection. 'auto' = ceil(d_model / 16).
        dt_min  (float):  Min initial Δ value. Standard: 0.001.
        dt_max  (float):  Max initial Δ value. Standard: 0.1.
    """

    def __init__(
        self,
        d_model : int   = 64,
        d_state : int   = 16,
        d_conv  : int   = 4,
        expand  : int   = 2,
        dt_rank         = 'auto',
        dt_min  : float = 0.001,
        dt_max  : float = 0.1,
    ):
        super().__init__()

        self.d_model  = d_model
        self.d_state  = d_state
        self.d_inner  = expand * d_model          # 128 for d_model=64
        self.dt_rank  = (math.ceil(d_model / 16)
                         if dt_rank == 'auto' else dt_rank)  # 4 for d_model=64

        # ── 1. Input projection: d_model → 2 × d_inner ────────────────────
        # Splits into x_ssm (SSM branch) and z (gate branch)
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # ── 2a. Local causal depthwise conv ────────────────────────────────
        # Captures short-range local dependencies before SSM
        # padding = d_conv - 1 ensures causal (no future leakage after trim)
        self.conv1d   = nn.Conv1d(
            in_channels  = self.d_inner,
            out_channels = self.d_inner,
            kernel_size  = d_conv,
            padding      = d_conv - 1,
            groups       = self.d_inner,   # depthwise
            bias         = True,
        )

        # ── 2c. Selective parameter projections ────────────────────────────
        # x_proj maps d_inner → (dt_rank + d_state + d_state)
        # Output is split into: Δ (low-rank), B, C
        self.x_proj   = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
        )

        # dt_proj expands low-rank Δ back to d_inner
        self.dt_proj  = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialise dt_proj bias so softplus(bias) ≈ Uniform[dt_min, dt_max]
        # This ensures Δ starts in a reasonable range for the ZOH discretisation
        _dt = torch.exp(
            torch.rand(self.d_inner)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=1e-4)
        _inv_dt = _dt + torch.log(-torch.expm1(-_dt))   # softplus⁻¹
        self.dt_proj.bias = nn.Parameter(_inv_dt)
        self.dt_proj.bias._no_weight_decay = True

        # ── SSM fixed parameters ───────────────────────────────────────────
        # A: structured diagonal [d_inner, d_state], stored as log for stability
        # Values: log(1), log(2), ..., log(d_state) repeated d_inner times
        _A = (torch.arange(1, d_state + 1, dtype=torch.float32)
                   .unsqueeze(0)
                   .expand(self.d_inner, -1))           # [d_inner, d_state]
        self.A_log = nn.Parameter(torch.log(_A))
        self.A_log._no_weight_decay = True

        # D: per-channel skip connection weight (x bypasses SSM)
        self.D      = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # ── 5. Output projection: d_inner → d_model ───────────────────────
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # ─────────────────────────────────────────────────────────────────────────

    def _selective_scan(
        self,
        u     : torch.Tensor,   # [B, L, d_inner]
        delta : torch.Tensor,   # [B, L, d_inner]  — positive after softplus
        A     : torch.Tensor,   # [d_inner, d_state] — negative
        B     : torch.Tensor,   # [B, L, d_state]
        C     : torch.Tensor,   # [B, L, d_state]
        D     : torch.Tensor,   # [d_inner]
    ) -> torch.Tensor:          # [B, L, d_inner]
        """
        Sequential selective scan implementing the recurrence:
            h[t] = Ā[t] * h[t-1] + B̄[t]
            y[t] = C[t] · h[t]  +  D * u[t]

        where (ZOH discretisation):
            Ā[t] = exp(Δ[t] * A)
            B̄[t] = Δ[t] * B[t] * u[t]

        This is the O(n * d_state) sequential implementation.
        For production use with long sequences, replace with a CUDA parallel scan.
        For ASCADv1 (L ≈ 233 after conv stride=3 on 700 samples) this is fine.

        Args / Returns: see type hints above.
        """
        B_b, L, d_inner = u.shape
        d_state = A.shape[1]

        # Discretise: Ā[b,t,i,s] = exp(Δ[b,t,i] * A[i,s])
        deltaA = torch.exp(
            delta.unsqueeze(-1) * A[None, None, :, :]
        )                                               # [B, L, d_inner, d_state]

        # B̄[b,t,i,s] = Δ[b,t,i] * B[b,t,s] * u[b,t,i]
        deltaB_u = (
            delta.unsqueeze(-1)    # [B, L, d_inner, 1]
            * B[:, :, None, :]     # [B, L,       1, d_state]
            * u.unsqueeze(-1)      # [B, L, d_inner, 1]
        )                                               # [B, L, d_inner, d_state]

        # ── TorchScript sequential scan ────────────────────────────────────
        # @torch.jit.script compiles the for-loop to TorchScript IR,
        # removing Python interpreter overhead on every forward call.
        # Math is byte-for-byte identical to the original sequential scan.
        y = _ssm_scan(deltaA, deltaB_u, C)              # [B, L, d_inner]
        y = y + u * D[None, None, :]                    # skip connection
        return y

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]
        Returns:
            [B, L, d_model]  — NO residual (added externally)
        """
        B_b, L, _ = x.shape

        # ── 1. Input projection ───────────────────────────────────────────
        xz    = self.in_proj(x)                        # [B, L, 2*d_inner]
        x_ssm, z = xz.chunk(2, dim=-1)                # each [B, L, d_inner]

        # ── 2a. Causal depthwise conv ─────────────────────────────────────
        x_conv = self.conv1d(
            x_ssm.transpose(1, 2)                      # [B, d_inner, L]
        )[:, :, :L]                                    # causal trim → [B, d_inner, L]
        x_conv = F.silu(x_conv).transpose(1, 2)        # [B, L, d_inner]

        # ── 2c. Selective parameters (all input-dependent) ────────────────
        x_dbl  = self.x_proj(x_conv)                   # [B, L, dt_rank + 2*d_state]
        delta_raw, B_mat, C_mat = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # Δ: low-rank → d_inner, then softplus to ensure positivity
        delta = F.softplus(self.dt_proj(delta_raw))    # [B, L, d_inner]

        # A: fixed diagonal structure, always negative
        A = -torch.exp(self.A_log.float())             # [d_inner, d_state]

        # ── Selective scan ────────────────────────────────────────────────
        y = self._selective_scan(
            x_conv.float(), delta.float(),
            A, B_mat.float(), C_mat.float(), self.D.float()
        ).to(x.dtype)                                  # [B, L, d_inner]

        # ── Gate ──────────────────────────────────────────────────────────
        y = y * F.silu(z)                              # element-wise gate

        # ── Output projection ─────────────────────────────────────────────
        return self.out_proj(y)                        # [B, L, d_model]
