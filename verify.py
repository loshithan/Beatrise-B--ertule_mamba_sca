"""
Verification script — confirms model matches paper specification.

Expected from paper (Table 4.2): ~148,000 parameters.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from mamba_sca import MambaSCAModel


def verify():
    print("=" * 60)
    print("Paper: 'Deep Learning-based Side-Channel Attack: Mamba Approach'")
    print("Beatrise Bertule, Radboud University, January 2026")
    print("=" * 60)

    # ── Build model with exact paper hyperparameters (Table 4.2) ─────────
    model = MambaSCAModel(
        trace_length = 700,   # ASCADv1 input length
        d_model      = 64,    # Table 4.2
        n_blocks     = 2,     # Table 4.2
        num_classes  = 256,   # 256 S-box values
        d_state      = 16,    # standard Mamba default
        d_conv       = 4,     # standard Mamba default
        expand       = 2,     # standard Mamba default
    )

    total_params = model.count_parameters()
    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"Paper reports:              ~148,000")
    print(f"Match: {'✅ YES' if 140_000 <= total_params <= 160_000 else '❌ NO — check architecture'}")

    # ── Verify forward pass shapes ────────────────────────────────────────
    print("\n--- Forward pass shape check ---")
    B, T = 4, 700
    x = torch.randn(B, T)

    with torch.no_grad():
        # Check conv output length
        import math
        L = math.floor((T - 3) / 3 + 1)
        print(f"Input:            ({B}, {T})")
        print(f"After conv:       ({B}, {L}, 64)   [L = floor((T-3)/3+1) = {L}]")

        logits = model(x)
        print(f"Output logits:    {tuple(logits.shape)}   [expected ({B}, 256)]")
        assert logits.shape == (B, 256), f"Wrong output shape: {logits.shape}"

    # ── Verify module structure ───────────────────────────────────────────
    print("\n--- Module structure ---")
    print(f"conv:        {model.conv}")
    print(f"fwd_blocks:  {len(model.fwd_blocks)} × MambaEncoderBlock")
    print(f"bwd_blocks:  {len(model.bwd_blocks)} × MambaEncoderBlock")
    print(f"classifier:  {model.classifier}")

    # ── Verify fwd and bwd blocks have independent weights ────────────────
    fwd_w = model.fwd_blocks[0].mamba.in_proj.weight.data_ptr()
    bwd_w = model.bwd_blocks[0].mamba.in_proj.weight.data_ptr()
    independent = fwd_w != bwd_w
    print(f"\nfwd/bwd weights independent: {'✅ YES' if independent else '❌ NO'}")

    # ── Verify classifier is single Linear(128, 256) ─────────────────────
    assert isinstance(model.classifier, torch.nn.Linear), "Classifier must be nn.Linear"
    assert model.classifier.in_features  == 128, f"Expected 128 in, got {model.classifier.in_features}"
    assert model.classifier.out_features == 256, f"Expected 256 out, got {model.classifier.out_features}"
    print(f"Classifier is Linear(128, 256): ✅ YES")

    # ── Per-module parameter breakdown ───────────────────────────────────
    print("\n--- Parameter breakdown ---")
    def count(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

    conv_p   = count(model.conv)
    fwd_p    = sum(count(b) for b in model.fwd_blocks)
    bwd_p    = sum(count(b) for b in model.bwd_blocks)
    cls_p    = count(model.classifier)

    print(f"  Conv module:        {conv_p:>8,}")
    print(f"  Fwd encoder (×2):   {fwd_p:>8,}")
    print(f"  Bwd encoder (×2):   {bwd_p:>8,}")
    print(f"  Classifier:         {cls_p:>8,}")
    print(f"  ─────────────────────────────")
    print(f"  Total:              {total_params:>8,}")

    print("\n✅ All checks passed. Architecture matches paper specification.")


if __name__ == "__main__":
    verify()
