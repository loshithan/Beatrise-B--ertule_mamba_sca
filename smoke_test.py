"""
Smoke test — runs each notebook cell's logic with minimal data to catch runtime errors.
This does NOT train for real; it runs 1 batch / 1 GE run to verify shapes & imports.
"""

import os
import sys
import time
import math
import traceback

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def section(name):
    print(f"\n{'─'*60}")
    print(f"  Testing: {name}")
    print(f"{'─'*60}")

# ══════════════════════════════════════════════════════════════════════
# Cell 1: Imports & Configuration
# ══════════════════════════════════════════════════════════════════════
section("Cell 1 — Imports & Configuration")
try:
    import numpy as np
    import h5py
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from mamba_sca_model import MambaSCAModel

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {DEVICE}")
    print(f"  PyTorch: {torch.__version__}")
    results.append(("Cell 1 — Imports", PASS))
except Exception as e:
    traceback.print_exc()
    results.append(("Cell 1 — Imports", FAIL))

# ══════════════════════════════════════════════════════════════════════
# Cell 2: AES S-box & label helpers
# ══════════════════════════════════════════════════════════════════════
section("Cell 2 — S-box & make_labels")
try:
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

    def make_labels(plaintexts, key, target_byte):
        intermediate = plaintexts[:, target_byte] ^ key[target_byte]
        return AES_SBOX[intermediate].astype(np.int64)

    # Quick test
    dummy_pt = np.random.randint(0, 256, size=(10, 16), dtype=np.uint8)
    dummy_key = np.random.randint(0, 256, size=(16,), dtype=np.uint8)
    labels = make_labels(dummy_pt, dummy_key, 2)
    assert labels.shape == (10,), f"Bad shape: {labels.shape}"
    assert labels.min() >= 0 and labels.max() <= 255
    print(f"  make_labels output shape: {labels.shape}, range: [{labels.min()}, {labels.max()}]")
    results.append(("Cell 2 — S-box / labels", PASS))
except Exception as e:
    traceback.print_exc()
    results.append(("Cell 2 — S-box / labels", FAIL))

# ══════════════════════════════════════════════════════════════════════
# Cell 3: Load ASCADv1 dataset
# ══════════════════════════════════════════════════════════════════════
section("Cell 3 — Load ASCAD.h5")
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'ASCAD.h5')
TARGET_BYTE = 2
try:
    with h5py.File(DATA_PATH, 'r') as f:
        print("  HDF5 structure:")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"    {name}: shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(print_structure)

        X_profiling = np.array(f['Profiling_traces/traces'], dtype=np.float32)
        pt_profiling = np.array(f['Profiling_traces/metadata']['plaintext'])
        key_profiling = np.array(f['Profiling_traces/metadata']['key'])

        X_attack = np.array(f['Attack_traces/traces'], dtype=np.float32)
        pt_attack = np.array(f['Attack_traces/metadata']['plaintext'])
        key_attack = np.array(f['Attack_traces/metadata']['key'])

    print(f"  Profiling: {X_profiling.shape}, Attack: {X_attack.shape}")
    correct_key_byte = int(key_profiling[0][TARGET_BYTE])
    print(f"  Correct key byte: 0x{correct_key_byte:02X}")
    results.append(("Cell 3 — Data loading", PASS))
except Exception as e:
    traceback.print_exc()
    results.append(("Cell 3 — Data loading", FAIL))
    # Can't continue without data
    print("\n\nCANNOT CONTINUE without data. Exiting.")
    for name, status in results:
        print(f"  {status}  {name}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════
# Cell 4: Train/Val/Test splits
# ══════════════════════════════════════════════════════════════════════
section("Cell 4 — Splits")
try:
    N_TRAIN = 40000
    N_VAL   = 10000
    X_train = X_profiling[:N_TRAIN]
    X_val   = X_profiling[N_TRAIN:N_TRAIN + N_VAL]
    X_test  = X_attack
    pt_train = pt_profiling[:N_TRAIN]
    pt_val   = pt_profiling[N_TRAIN:N_TRAIN + N_VAL]
    pt_test  = pt_attack
    key_train = key_profiling[:N_TRAIN]
    key_val   = key_profiling[N_TRAIN:N_TRAIN + N_VAL]

    y_train = make_labels(pt_train, key_train[0], TARGET_BYTE)
    y_val   = make_labels(pt_val,   key_val[0],   TARGET_BYTE)

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}, y_val: {y_val.shape}")
    results.append(("Cell 4 — Splits", PASS))
except Exception as e:
    traceback.print_exc()
    results.append(("Cell 4 — Splits", FAIL))

# ══════════════════════════════════════════════════════════════════════
# Cell 5: Normalise & DataLoaders
# ══════════════════════════════════════════════════════════════════════
section("Cell 5 — Normalise & DataLoaders")
try:
    train_mean = X_train.mean(axis=0, keepdims=True)
    train_std  = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train_n = ((X_train - train_mean) / train_std).astype(np.float32)
    X_val_n   = ((X_val   - train_mean) / train_std).astype(np.float32)

    X_train_t = torch.from_numpy(X_train_n)
    y_train_t = torch.from_numpy(y_train)
    X_val_t   = torch.from_numpy(X_val_n)
    y_val_t   = torch.from_numpy(y_val)

    BATCH_SIZE = 768
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

    total_train_steps = math.ceil(len(X_train_t) / BATCH_SIZE)
    total_val_steps   = math.ceil(len(X_val_t)   / BATCH_SIZE)
    print(f"  Train batches: {total_train_steps}, Val batches: {total_val_steps}")
    results.append(("Cell 5 — Normalise / DataLoaders", PASS))
except Exception as e:
    traceback.print_exc()
    results.append(("Cell 5 — Normalise / DataLoaders", FAIL))

# ══════════════════════════════════════════════════════════════════════
# Cell 6: Build Model
# ══════════════════════════════════════════════════════════════════════
section("Cell 6 — Build Model")
try:
    trace_length = X_train.shape[1]
    model = MambaSCAModel(
        trace_length=trace_length, d_model=64, n_blocks=2,
        num_classes=256, d_state=8, d_conv=4, expand=2,
    )
    model.to(DEVICE)
    total_params = model.count_parameters()
    print(f"  Params: {total_params:,}  (paper: ~148,000)")
    print(f"  Match: {'YES' if 140_000 <= total_params <= 160_000 else 'CLOSE' if 100_000 <= total_params <= 200_000 else 'NO'}")
    results.append(("Cell 6 — Model build", PASS))
except Exception as e:
    traceback.print_exc()
    results.append(("Cell 6 — Model build", FAIL))

# ══════════════════════════════════════════════════════════════════════
# Cell 7: GE function (test with 1 run, 100 traces)
# ══════════════════════════════════════════════════════════════════════
section("Cell 7 — GE function (mini)")
try:
    def compute_guessing_entropy(model, traces, plaintexts, correct_key, target_byte,
                                  n_runs=100, n_traces=1000, device='cpu', eps=1e-10):
        model.eval()
        N = traces.shape[0]
        all_probs = []
        with torch.no_grad():
            loader = DataLoader(TensorDataset(traces), batch_size=512, shuffle=False)
            for (batch,) in loader:
                logits = model(batch.to(device))
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        all_probs = np.concatenate(all_probs, axis=0)
        pt_byte = plaintexts[:, target_byte].astype(np.uint8)
        hyp_labels = AES_SBOX[pt_byte[:, None] ^ np.arange(256, dtype=np.uint8)[None, :]]
        ranks = []
        for _ in range(n_runs):
            idx = np.random.choice(N, size=n_traces, replace=False)
            probs_sub = all_probs[idx]
            hyp_sub   = hyp_labels[idx]
            scores = np.zeros(256, dtype=np.float64)
            for k in range(256):
                log_probs = np.log(probs_sub[np.arange(n_traces), hyp_sub[:, k]] + eps)
                scores[k] = log_probs.sum()
            rank = np.where(np.argsort(scores)[::-1] == correct_key)[0][0]
            ranks.append(rank)
        return float(np.mean(ranks))

    # Test with tiny params: 1 run, 100 traces
    ge = compute_guessing_entropy(
        model, X_val_t[:500], pt_val[:500], correct_key_byte, TARGET_BYTE,
        n_runs=1, n_traces=100, device=DEVICE,
    )
    print(f"  Mini GE (1 run, 100 traces): {ge:.2f}")
    results.append(("Cell 7 — GE function", PASS))
except Exception as e:
    traceback.print_exc()
    results.append(("Cell 7 — GE function", FAIL))

# ══════════════════════════════════════════════════════════════════════
# Cell 8: Training loop (1 batch train + 1 batch val + mini GE)
# ══════════════════════════════════════════════════════════════════════
section("Cell 8 — Training loop (1 batch smoke)")
try:
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # --- 1 training batch ---
    model.train()
    X_batch, y_batch = next(iter(train_loader))
    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
    optimizer.zero_grad()
    logits = model(X_batch)
    loss = criterion(logits, y_batch)
    loss.backward()
    optimizer.step()
    batch_loss = loss.item()
    preds = logits.argmax(dim=-1)
    batch_acc = (preds == y_batch).float().mean().item()
    print(f"  Train batch — loss: {batch_loss:.4f}, acc: {batch_acc:.4f}")

    # --- 1 val batch ---
    model.eval()
    with torch.no_grad():
        X_vb, y_vb = next(iter(val_loader))
        X_vb, y_vb = X_vb.to(DEVICE), y_vb.to(DEVICE)
        vlogits = model(X_vb)
        vloss = criterion(vlogits, y_vb)
        vpreds = vlogits.argmax(dim=-1)
        val_batch_acc = (vpreds == y_vb).float().mean().item()
    print(f"  Val   batch — loss: {vloss.item():.4f}, acc: {val_batch_acc:.4f}")

    # --- Print formatting test ---
    epoch = 1
    avg_train_loss = batch_loss
    avg_val_loss = vloss.item()
    train_acc = batch_acc
    val_acc = val_batch_acc
    val_ge = 127.0  # placeholder
    print()
    print(f"  {'─' * 55}")
    print(f"    Epoch {epoch:3d}/100 complete")
    print(f"    Train Loss : {avg_train_loss:.4f}   |   Train Acc : {train_acc:.4f}")
    print(f"    Val   Loss : {avg_val_loss:.4f}   |   Val   Acc : {val_acc:.4f}")
    print(f"    Val   GE   : {val_ge:.4f}")
    print(f"  {'─' * 55}")

    # --- Checkpoint save/load ---
    SAVE_PATH = os.path.join(PROJECT_DIR, '_smoke_test_ckpt.pt')
    torch.save({
        'epoch': 1,
        'model_state': model.state_dict(),
        'val_ge': val_ge,
        'mean_std': (train_mean, train_std),
        'history': {'train_loss': [batch_loss], 'val_loss': [vloss.item()], 'val_ge': [val_ge]},
    }, SAVE_PATH)

    ckpt = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    ckpt_mean, ckpt_std = ckpt['mean_std']
    print(f"  Checkpoint save/load: OK")
    os.remove(SAVE_PATH)

    results.append(("Cell 8 — Training loop", PASS))
except Exception as e:
    traceback.print_exc()
    results.append(("Cell 8 — Training loop", FAIL))

# ══════════════════════════════════════════════════════════════════════
# Cell 9: Training curves (matplotlib)
# ══════════════════════════════════════════════════════════════════════
section("Cell 9 — Matplotlib plotting")
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend for smoke test
    import matplotlib.pyplot as plt

    history = {'train_loss': [5.5, 5.3, 5.1], 'val_loss': [5.6, 5.4, 5.2], 'val_ge': [127, 120, 115]}
    step_losses = [(1, 5.5), (2, 5.4), (3, 5.3)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot([1,2,3], history['train_loss'], 'b-o')
    axes[0].plot([1,2,3], history['val_loss'], 'r-o')
    axes[1].plot([1,2,3], history['val_ge'], 'g-o')
    steps, losses = zip(*step_losses)
    axes[2].plot(steps, losses, 'b-')
    plt.close(fig)
    print("  Plotting works (non-interactive)")
    results.append(("Cell 9 — Matplotlib", PASS))
except Exception as e:
    traceback.print_exc()
    results.append(("Cell 9 — Matplotlib", FAIL))

# ══════════════════════════════════════════════════════════════════════
# Cell 11: Attack evaluation (mini — 1 run, 100 traces, no convergence)
# ══════════════════════════════════════════════════════════════════════
section("Cell 11 — Attack evaluation (mini)")
try:
    X_test_n = ((X_test[:500] - ckpt_mean) / ckpt_std).astype(np.float32)
    X_test_t = torch.from_numpy(X_test_n)
    N_test = X_test_t.shape[0]
    eps = 1e-10

    model.eval()
    all_probs = []
    with torch.no_grad():
        loader = DataLoader(TensorDataset(X_test_t), batch_size=512, shuffle=False)
        for (batch,) in loader:
            logits = model(batch.to(DEVICE))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    all_probs = np.concatenate(all_probs, axis=0)

    pt_byte_test = pt_test[:500, TARGET_BYTE].astype(np.uint8)
    hyp_labels = AES_SBOX[pt_byte_test[:, None] ^ np.arange(256, dtype=np.uint8)[None, :]]

    # Key rank
    scores_full = np.zeros(256, dtype=np.float64)
    for k in range(256):
        scores_full[k] = np.log(all_probs[np.arange(N_test), hyp_labels[:, k]] + eps).sum()
    key_rank = int(np.where(np.argsort(scores_full)[::-1] == correct_key_byte)[0][0])
    print(f"  Key rank (500 traces): {key_rank}")

    # 1 attack run
    idx = np.random.choice(N_test, size=100, replace=False)
    scores = np.zeros(256, dtype=np.float64)
    for k in range(256):
        scores[k] = np.log(all_probs[idx][np.arange(100), hyp_labels[idx, k]] + eps).sum()
    rank = int(np.where(np.argsort(scores)[::-1] == correct_key_byte)[0][0])
    print(f"  Single run rank (100 traces): {rank}")

    # Mini convergence (10 traces)
    trace_ranks = []
    scores_c = np.zeros(256, dtype=np.float64)
    for t in range(min(10, N_test)):
        k_idx = hyp_labels[t, :]
        log_p = np.log(all_probs[t, k_idx] + eps)
        scores_c += log_p
        r = int(np.where(np.argsort(scores_c)[::-1] == correct_key_byte)[0][0])
        trace_ranks.append(r)
    ge_convergence = np.array(trace_ranks)
    print(f"  Convergence (10 traces): {ge_convergence}")

    results.append(("Cell 11 — Attack eval", PASS))
except Exception as e:
    traceback.print_exc()
    results.append(("Cell 11 — Attack eval", FAIL))

# ══════════════════════════════════════════════════════════════════════
# Cell 13: Summary table formatting
# ══════════════════════════════════════════════════════════════════════
section("Cell 13 — Summary table")
try:
    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    EXPERIMENT SUMMARY                       ║
║  Model             : MambaSCAModel                          ║
║  Parameters         : {total_params:>10,}                          ║
║  Key rank (smoke)   : {key_rank:>10}                          ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(summary)
    results.append(("Cell 13 — Summary table", PASS))
except Exception as e:
    traceback.print_exc()
    results.append(("Cell 13 — Summary table", FAIL))

# ══════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  SMOKE TEST RESULTS")
print("=" * 60)
all_pass = True
for name, status in results:
    print(f"  {status}  {name}")
    if status == FAIL:
        all_pass = False

if all_pass:
    print(f"\n  🎉 ALL {len(results)} CHECKS PASSED — notebook is runtime-safe!")
else:
    fails = sum(1 for _, s in results if s == FAIL)
    print(f"\n  ⚠️ {fails}/{len(results)} checks FAILED — see errors above.")
print("=" * 60)
