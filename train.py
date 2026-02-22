"""
Training Script — Exact paper hyperparameters.

All values sourced directly from:
    "Deep Learning-based Side-Channel Attack: Mamba Approach"
    Beatrise Bertule, Radboud University Nijmegen, January 2026.

Training config (Tables 4.3, 4.4, Section 4.4.2):
    batch_size   = 768          Table 4.4
    lr           = 5e-3         Table 4.4
    optimizer    = Adam + weight decay  Section 4.4.2
    max_epochs   = 100          Section 4.4.2
    loss         = CrossEntropyLoss    Section 4.3.1
    val_metric   = mean GE over 100 attack runs  Section 4.4.2

Data splits (Table 4.3):
    train  = 40,000 traces
    val    = 10,000 traces
    test   = 10,000 traces

Label preparation (Section 4.4.1, Equation 4.1/4.2):
    y_j = S_box[plaintext_j[i] XOR key_j[i]]   Identity model → 256 classes
    CW-Target  : target byte i = 0
    ASCADv1    : target byte i = 2

Data normalisation (Section 4.4.1):
    StandardScaler fitted on TRAIN set only
    Same transform applied to val and test
    Labels one-hot encoded

Model selection (Section 4.4.2):
    Checkpoint with LOWEST validation GE is saved
    Validation GE = mean key rank across 100 random attack simulations
    Each simulation draws 1,000 random traces from val set

IMPORTANT — what is NOT used for validation (Section 4.4.2, citing Perin et al. [20]):
    - NOT accuracy
    - NOT loss
    Only mean GE is used to select the best checkpoint.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# AES S-box lookup table (Section 2.2.2, Table 2.1)
AES_SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
], dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Label preparation (Section 4.4.1, Equations 4.1 / 4.2)
# ─────────────────────────────────────────────────────────────────────────────

def make_labels(plaintexts: np.ndarray, key: np.ndarray, target_byte: int) -> np.ndarray:
    """
    Compute S-box output labels under the Identity (ID) leakage model.

    Equation 4.1 / 4.2 from paper:
        y_j = S_box[ plaintext_j[i] XOR key_j[i] ]

    Args:
        plaintexts  : [N, 16]  uint8 — one plaintext per trace
        key         : [16]     uint8 — fixed key (same across all traces for both datasets)
        target_byte : int      — byte index i
                                 CW-Target  → i = 0   (Section 4.4.1)
                                 ASCADv1    → i = 2   (Section 4.4.1)
    Returns:
        labels : [N]  int64 in [0, 255]
    """
    intermediate = plaintexts[:, target_byte] ^ key[target_byte]   # XOR
    labels = AES_SBOX[intermediate].astype(np.int64)                # S-box lookup
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Guessing Entropy computation (Section 2.5.7, Equation 2.25)
# ─────────────────────────────────────────────────────────────────────────────

def compute_guessing_entropy(
    model       : nn.Module,
    traces      : torch.Tensor,     # [N, T]  normalised, on CPU
    plaintexts  : np.ndarray,       # [N, 16] uint8
    correct_key : int,              # correct key byte value
    target_byte : int,
    n_runs      : int  = 100,       # paper: 100 attack simulations
    n_traces    : int  = 1000,      # paper: 1,000 traces per simulation
    device      : str  = 'cpu',
    eps         : float = 1e-10,    # numerical stability (Section 2.5.6 footnote)
) -> float:
    """
    Mean Guessing Entropy across n_runs attack simulations.

    Equation 2.25:
        GE = (1/M) * sum_j rank_j(g)

    Attack execution (Section 2.5.6, Equations 2.22-2.24):
        For each key candidate k_i in {0,...,255}:
            h(k_i) = S_box[ plaintext_j[i] XOR k_i ]  (hypothetical label)
            score(k_i) = sum_j log( P_j( h(k_i) ) )   (accumulated log-likelihood)
        g = argsort( scores )  descending
        key_rank = position of correct_key in g

    Args / Returns: see type hints above.
    """
    model.eval()
    N = traces.shape[0]
    ranks = []

    with torch.no_grad():
        # Get all model predictions at once (avoid per-batch overhead)
        all_probs = []
        loader = DataLoader(TensorDataset(traces), batch_size=512, shuffle=False)
        for (batch,) in loader:
            logits = model(batch.to(device))
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        all_probs = np.concatenate(all_probs, axis=0)   # [N, 256]

    # 256 hypothetical labels for all traces and all key candidates
    # hyp_labels[j, k] = S_box[ plaintext_j[target_byte] XOR k ]
    pt_byte = plaintexts[:, target_byte].astype(np.uint8)
    hyp_labels = AES_SBOX[
        pt_byte[:, None] ^ np.arange(256, dtype=np.uint8)[None, :]
    ]                                                   # [N, 256]

    for _ in range(n_runs):
        # Random subsample of n_traces (Section 4.4.3: 1,000 traces)
        idx = np.random.choice(N, size=n_traces, replace=False)
        probs_sub = all_probs[idx]                      # [n_traces, 256]
        hyp_sub   = hyp_labels[idx]                     # [n_traces, 256]

        # Accumulate log-likelihoods for each key candidate — vectorized
        # log_probs_all[j, k] = log( P_j( S_box[p_j XOR k] ) )
        # scores[k] = sum_j log_probs_all[j, k]
        log_probs_all = np.log(
            probs_sub[np.arange(n_traces)[:, None], hyp_sub] + eps   # [n_traces, 256]
        )
        scores = log_probs_all.sum(axis=0)                            # [256]

        # Rank = position of correct key in descending score order
        rank = np.where(np.argsort(scores)[::-1] == correct_key)[0][0]
        ranks.append(rank)

    return float(np.mean(ranks))                        # Equation 2.25


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model           : nn.Module,
    X_train         : np.ndarray,    # [40000, T]  raw traces (before normalisation)
    y_train         : np.ndarray,    # [40000]     int labels 0-255
    X_val           : np.ndarray,    # [10000, T]
    y_val           : np.ndarray,    # [10000]
    pt_val          : np.ndarray,    # [10000, 16] plaintexts for GE computation
    correct_key_val : int,           # correct key byte for val GE
    target_byte     : int,
    save_path       : str = 'best_model.pt',
    device          : str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Train with exact paper hyperparameters (Tables 4.3, 4.4, Section 4.4.2).

    Model selection: checkpoint with LOWEST mean validation GE is saved.
    Neither accuracy nor loss is used for model selection.
    """

    # ── Data normalisation (Section 4.4.1) ───────────────────────────────
    # Fit StandardScaler on train set only, apply to all splits
    mean  = X_train.mean(axis=0, keepdims=True)         # [1, T]
    std   = X_train.std(axis=0, keepdims=True) + 1e-8   # [1, T]

    X_train_n = ((X_train - mean) / std).astype(np.float32)
    X_val_n   = ((X_val   - mean) / std).astype(np.float32)

    # Convert to tensors
    X_train_t = torch.from_numpy(X_train_n)
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    X_val_t   = torch.from_numpy(X_val_n)

    # ── DataLoaders ───────────────────────────────────────────────────────
    # Effective batch_size = 768 (Table 4.4) achieved via gradient accumulation:
    # micro_batch=256 × ACCUM_STEPS=3 = 768 (mathematically identical to a
    # single batch of 768, but uses 1/3 the GPU memory — needed for D_STATE=16)
    MICRO_BATCH = 256
    ACCUM_STEPS = 3     # 3 × 256 = 768  ≡  Table 4.4

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size = MICRO_BATCH,
        shuffle    = True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, torch.from_numpy(y_val.astype(np.int64))),
        batch_size = MICRO_BATCH,
        shuffle    = False,
    )

    # ── Optimiser (Section 4.4.2, Table 4.4) ─────────────────────────────
    # Adam with weight decay (weight_decay value not specified, use standard 1e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = 5e-3,    # Table 4.4 — exact value
        weight_decay = 1e-4,    # decoupled weight decay (AdamW, not L2 mixed into grad)
    )

    # ── Loss (Section 4.3.1) ──────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    best_ge       = float('inf')
    best_epoch    = -1

    # ── Training loop: max 100 epochs (Section 4.4.2) ────────────────────
    for epoch in range(100):    # Section 4.4.2 — exact value

        # ── Train (gradient accumulation: micro×3 = effective batch 768) ──
        model.train()
        train_loss = 0.0
        gnorm_sum  = 0.0
        n_updates  = 0
        optimizer.zero_grad()

        for step, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Scale loss so accumulated gradient == gradient from full batch of 768
            logits = model(X_batch)
            loss   = criterion(logits, y_batch) / ACCUM_STEPS
            loss.backward()
            train_loss += loss.item() * ACCUM_STEPS * len(X_batch)

            # Optimizer step every ACCUM_STEPS micro-batches
            if (step + 1) % ACCUM_STEPS == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                gnorm_sum += gnorm.item()
                n_updates += 1
                optimizer.step()
                optimizer.zero_grad()

        # Flush any remaining accumulated gradients at the end of the epoch
        if (step + 1) % ACCUM_STEPS != 0:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            gnorm_sum += gnorm.item()
            n_updates += 1
            optimizer.step()
            optimizer.zero_grad()

        train_loss /= len(X_train_t)
        avg_gnorm   = gnorm_sum / max(n_updates, 1)

        # ── Eval loss — cross-entropy on unseen validation traces ─────────
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits  = model(X_batch)
                loss    = criterion(logits, y_batch)
                eval_loss += loss.item() * len(X_batch)
        eval_loss /= len(X_val_t)

        # ── Validate with mean GE (Section 4.4.2) ────────────────────────
        # Paper: "mean GE measures the average rank of the correct key byte
        #         across 100 attack executions"
        # Paper: "validation GE was monitored over 100 epochs, and the model
        #         checkpoint that corresponded to the lowest validation GE
        #         — beyond which no further improvement was observed —
        #         was selected for the final evaluation"
        val_ge = compute_guessing_entropy(
            model       = model,
            traces      = X_val_t,
            plaintexts  = pt_val,
            correct_key = correct_key_val,
            target_byte = target_byte,
            n_runs      = 100,      # Section 4.4.2
            n_traces    = 1000,     # Section 4.4.3
            device      = device,
        )

        print(
            f"Epoch {epoch+1:3d}/100 | "
            f"train_loss: {train_loss:.4f} | "
            f"eval_loss: {eval_loss:.4f} | "
            f"gnorm: {avg_gnorm:.4f} | "
            f"val_GE: {val_ge:.4f}"
        )

        # ── Checkpoint: save best GE model ────────────────────────────────
        if val_ge < best_ge:
            best_ge    = val_ge
            best_epoch = epoch + 1
            torch.save({
                'epoch'      : best_epoch,
                'model_state': model.state_dict(),
                'val_ge'     : best_ge,
                'mean_std'   : (mean, std),
            }, save_path)
            print(f"  → Saved checkpoint (val_GE={best_ge:.4f} at epoch {best_epoch})")

    print(f"\nTraining complete. Best val_GE = {best_ge:.4f} at epoch {best_epoch}.")
    return best_ge, best_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Attack phase evaluation (Section 4.4.3, 2.5.7)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_attack(
    model       : nn.Module,
    X_test      : np.ndarray,    # [10000, T]  raw traces
    pt_test     : np.ndarray,    # [10000, 16]
    correct_key : int,
    target_byte : int,
    mean        : np.ndarray,    # from training normalisation
    std         : np.ndarray,
    device      : str = 'cpu',
    eps         : float = 1e-10,
):
    """
    Full attack phase evaluation (Section 4.4.3).

    Computes:
    1. Key rank on all 10,000 test traces
    2. Mean GE + Median GE over 100 attack simulations of 1,000 traces each
    3. GE convergence curve over 1,000 traces

    Returns dict with all metrics.
    """
    # Normalise test set with TRAIN statistics
    X_test_n = ((X_test - mean) / std).astype(np.float32)
    X_test_t = torch.from_numpy(X_test_n)

    model.eval()
    N = X_test_t.shape[0]

    # Get all predictions
    all_probs = []
    with torch.no_grad():
        loader = DataLoader(TensorDataset(X_test_t), batch_size=512, shuffle=False)
        for (batch,) in loader:
            logits = model(batch.to(device))
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0)   # [N, 256]

    pt_byte    = pt_test[:, target_byte].astype(np.uint8)
    hyp_labels = AES_SBOX[
        pt_byte[:, None] ^ np.arange(256, dtype=np.uint8)[None, :]
    ]                                               # [N, 256]

    # ── 1. Key rank on full 10,000 traces — vectorized ───────────────────
    log_probs_full = np.log(
        all_probs[np.arange(N)[:, None], hyp_labels] + eps   # [N, 256]
    )
    scores_full = log_probs_full.sum(axis=0)                  # [256]
    key_rank = int(np.where(np.argsort(scores_full)[::-1] == correct_key)[0][0])

    # ── 2. Mean GE + Median GE over 100 runs — vectorized ────────────────
    ranks = []
    for _ in range(100):
        idx = np.random.choice(N, size=1000, replace=False)
        log_probs_sub = np.log(
            all_probs[idx][np.arange(1000)[:, None], hyp_labels[idx]] + eps   # [1000, 256]
        )
        scores = log_probs_sub.sum(axis=0)                                      # [256]
        rank = int(np.where(np.argsort(scores)[::-1] == correct_key)[0][0])
        ranks.append(rank)

    mean_ge   = float(np.mean(ranks))
    median_ge = float(np.median(ranks))

    # ── 3. GE convergence over 1,000 traces — fully vectorized ───────────
    # cumsum over traces gives running score; argsort each row for the rank
    conv_ranks = []
    for _ in range(100):
        idx = np.random.choice(N, size=1000, replace=False)
        log_probs_sub = np.log(
            all_probs[idx][np.arange(1000)[:, None], hyp_labels[idx]] + eps   # [1000, 256]
        )
        cum_scores  = np.cumsum(log_probs_sub, axis=0)                         # [1000, 256]
        sorted_idx  = np.argsort(cum_scores, axis=1)[:, ::-1]                  # [1000, 256]
        trace_ranks = np.argmax(sorted_idx == correct_key, axis=1)             # [1000]
        conv_ranks.append(trace_ranks)
    ge_convergence = np.mean(conv_ranks, axis=0)                                # [1000]

    ge_conv_value  = float(ge_convergence[-1])
    ge_conv_trace  = int(np.argmax(ge_convergence <= 1.0)) if (ge_convergence <= 1.0).any() else 1000

    print(f"\n=== Attack Evaluation Results ===")
    print(f"Key Rank (full 10k traces): {key_rank}")
    print(f"Median GE (100 runs):       {median_ge:.4f}")
    print(f"Mean GE   (100 runs):       {mean_ge:.4f}")
    print(f"GE Convergence:             {ge_conv_value:.4f} with {ge_conv_trace} traces")

    return {
        'key_rank'      : key_rank,
        'mean_ge'       : mean_ge,
        'median_ge'     : median_ge,
        'ge_convergence': ge_convergence,
        'ge_conv_value' : ge_conv_value,
        'ge_conv_trace' : ge_conv_trace,
    }
