"""
evaluate.py — Standalone attack evaluation for the Mamba SCA model.

Usage:
    python evaluate.py --checkpoint models/best_model_bb_epoch100_25.pt
                       --data       data/ASCAD.h5
                       --device     cuda          # or cpu

Outputs:
    results/evaluation_<model_name>.png   — GE convergence + summary bar chart
    results/evaluation_<model_name>.json  — all metrics as JSON
    logs/evaluation_<model_name>.log      — full text log
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt

# ── Make repo importable when run from the project root ──────────────────────
_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)

# Patch relative imports (from .module → from module) so the files can be
# imported directly without a package context — same fix as the Colab notebook.
_PATCH_FILES = ["mamba_sca_model.py", "mamba_encoder_block.py", "core_mamba_block.py"]
_REPLACEMENTS = [
    ("from .mamba_encoder_block", "from mamba_encoder_block"),
    ("from .core_mamba_block",    "from core_mamba_block"),
    ("from .rms_norm",            "from rms_norm"),
]
for _fname in _PATCH_FILES:
    _fpath = os.path.join(_REPO, _fname)
    if os.path.isfile(_fpath):
        with open(_fpath, "r", encoding="utf-8") as f:
            _src = f.read()
        _new = _src
        for _old, _rep in _REPLACEMENTS:
            _new = _new.replace(_old, _rep)
        if _new != _src:
            with open(_fpath, "w", encoding="utf-8") as f:
                f.write(_new)

from mamba_sca_model import MambaSCAModel
from train import evaluate_attack, AES_SBOX


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Mamba SCA — attack evaluation")
    p.add_argument("--checkpoint", default="models/best_model_bb_epoch100_25.pt",
                   help="Path to .pt checkpoint file")
    p.add_argument("--data",       default="data/ASCAD.h5",
                   help="Path to ASCAD.h5 dataset")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu",
                   help="cuda or cpu")
    p.add_argument("--target-byte", type=int, default=2,
                   help="Target AES key byte (ASCADv1=2, CW-Target=0)")
    p.add_argument("--n-runs",     type=int, default=100,
                   help="Number of GE attack simulations")
    p.add_argument("--n-train",    type=int, default=40_000)
    p.add_argument("--n-val",      type=int, default=10_000)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────────────────────────────────────

class Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.terminal = sys.stdout
        self.logfile  = open(path, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.logfile.write(msg)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(ge_convergence, key_rank, mean_ge, median_ge,
                 ge_conv_trace, model_name, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Attack Evaluation — {model_name}", fontsize=13, fontweight="bold")

    # ── Left: GE convergence curve ────────────────────────────────────────
    ax = axes[0]
    ax.plot(range(1, len(ge_convergence) + 1), ge_convergence,
            "b-", linewidth=1.5, label="Mean GE")
    ax.axhline(y=1.0,  color="red",   linestyle="--", alpha=0.7, label="GE = 1")
    ax.axhline(y=0.0,  color="green", linestyle="--", alpha=0.5, label="GE = 0 (perfect)")
    if ge_conv_trace < len(ge_convergence):
        ax.axvline(x=ge_conv_trace, color="orange", linestyle="--", alpha=0.8,
                   label=f"GE ≤ 1 at {ge_conv_trace} traces")
    ax.set_xlabel("Number of Attack Traces")
    ax.set_ylabel("Mean Guessing Entropy")
    ax.set_title("GE Convergence (100 attack simulations)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(ge_convergence))
    ax.set_ylim(bottom=0)

    # ── Right: summary bar chart ──────────────────────────────────────────
    ax2 = axes[1]
    labels  = ["Key Rank\n(10k traces)", "Mean GE\n(100 runs)", "Median GE\n(100 runs)"]
    values  = [key_rank, mean_ge, median_ge]
    colors  = ["steelblue", "darkorange", "seagreen"]
    bars = ax2.bar(labels, values, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(values) * 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.6, label="GE = 1 threshold")
    ax2.set_ylabel("Value")
    ax2.set_title("Attack Metrics Summary")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Derive output names from checkpoint filename ──────────────────────
    model_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    ts         = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name   = f"{model_name}_{ts}"

    log_path    = os.path.join("logs",    f"evaluation_{run_name}.log")
    plot_path   = os.path.join("results", f"evaluation_{run_name}.png")
    json_path   = os.path.join("results", f"evaluation_{run_name}.json")

    os.makedirs("logs",    exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Redirect all print output to both terminal and log file
    tee = Tee(log_path)
    sys.stdout = tee

    try:
        print("=" * 68)
        print(f"  Mamba SCA — Attack Evaluation")
        print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 68)
        print(f"  Checkpoint  : {args.checkpoint}")
        print(f"  Dataset     : {args.data}")
        print(f"  Device      : {args.device}")
        print(f"  Target byte : {args.target_byte}")
        print(f"  GE runs     : {args.n_runs}")
        print("=" * 68)

        # ── Load checkpoint ───────────────────────────────────────────────
        print("\n[1/4] Loading checkpoint …")
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        ckpt_epoch = ckpt.get("epoch",   "?")
        ckpt_vge   = ckpt.get("val_ge",  float("nan"))
        ckpt_mean, ckpt_std = ckpt["mean_std"]
        print(f"  Trained epoch : {ckpt_epoch}")
        print(f"  Val GE (saved): {ckpt_vge:.4f}")

        # ── Build model from checkpoint's saved config, infer d_state ─────
        state = ckpt["model_state"]
        # A_log shape is [d_inner, d_state] — infer d_state
        d_inner = state["fwd_blocks.0.mamba.A_log"].shape[0]   # 128
        d_state = state["fwd_blocks.0.mamba.A_log"].shape[1]   # 8 or 16
        d_model = d_inner // 2                                  # 64 (expand=2)

        model = MambaSCAModel(
            trace_length = 700,
            d_model      = d_model,
            n_blocks     = 2,
            num_classes  = 256,
            d_state      = d_state,
            d_conv       = 4,
            expand       = 2,
        )
        model.load_state_dict(state)
        model.to(args.device)
        model.eval()
        n_params = model.count_parameters()
        print(f"  Model params  : {n_params:,}  (d_model={d_model}, d_state={d_state})")

        # ── Load dataset ──────────────────────────────────────────────────
        print("\n[2/4] Loading dataset …")
        with h5py.File(args.data, "r") as f:
            X_profiling = np.array(f["Profiling_traces/traces"],              dtype=np.float32)
            pt_profiling = np.array(f["Profiling_traces/metadata"]["plaintext"])
            key_profile  = np.array(f["Profiling_traces/metadata"]["key"])
            X_attack     = np.array(f["Attack_traces/traces"],                dtype=np.float32)
            pt_attack    = np.array(f["Attack_traces/metadata"]["plaintext"])

        correct_key_byte = int(key_profile[0][args.target_byte])
        X_test  = X_attack
        pt_test = pt_attack
        print(f"  Attack traces : {X_test.shape}")
        print(f"  Correct key byte {args.target_byte}: 0x{correct_key_byte:02X}  ({correct_key_byte})")

        # ── Run evaluation ────────────────────────────────────────────────
        print("\n[3/4] Running attack evaluation …")
        results = evaluate_attack(
            model       = model,
            X_test      = X_test,
            pt_test     = pt_test,
            correct_key = correct_key_byte,
            target_byte = args.target_byte,
            mean        = ckpt_mean,
            std         = ckpt_std,
            device      = args.device,
        )

        key_rank       = results["key_rank"]
        mean_ge        = results["mean_ge"]
        median_ge      = results["median_ge"]
        ge_convergence = results["ge_convergence"]
        ge_conv_value  = results["ge_conv_value"]
        ge_conv_trace  = results["ge_conv_trace"]

        # ── Plot ──────────────────────────────────────────────────────────
        print("\n[4/4] Saving plot and results …")
        plot_results(ge_convergence, key_rank, mean_ge, median_ge,
                     ge_conv_trace, model_name, plot_path)

        # ── Save JSON ─────────────────────────────────────────────────────
        json_data = {
            "model_name"    : model_name,
            "checkpoint"    : args.checkpoint,
            "dataset"       : args.data,
            "target_byte"   : args.target_byte,
            "ckpt_epoch"    : ckpt_epoch,
            "ckpt_val_ge"   : float(ckpt_vge),
            "n_params"      : n_params,
            "d_model"       : d_model,
            "d_state"       : d_state,
            "key_rank"      : key_rank,
            "mean_ge"       : mean_ge,
            "median_ge"     : median_ge,
            "ge_conv_value" : ge_conv_value,
            "ge_conv_trace" : ge_conv_trace,
            "ge_convergence": ge_convergence.tolist(),
            "timestamp"     : ts,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        print(f"Results saved → {json_path}")

        # ── Final summary ─────────────────────────────────────────────────
        print()
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║                      EVALUATION SUMMARY                         ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        print(f"║  Checkpoint   : {model_name[:50]:<50s}  ║")
        print(f"║  Saved epoch  : {str(ckpt_epoch):<50s}  ║")
        print(f"║  Val GE (ckpt): {ckpt_vge:<50.4f}  ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        print(f"║  Key rank (10k traces)  : {key_rank:<42d}║")
        print(f"║  Mean GE   (100 runs)   : {mean_ge:<42.4f}║")
        print(f"║  Median GE (100 runs)   : {median_ge:<42.4f}║")
        print(f"║  GE at 1000 traces      : {ge_conv_value:<42.4f}║")
        print(f"║  Traces to GE ≤ 1       : {'<1000' if ge_conv_trace < 1000 else '>1000':<42s}║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print(f"\nLog  → {log_path}")
        print(f"Plot → {plot_path}")
        print(f"JSON → {json_path}")

        if key_rank == 0:
            print("\n🎉 Key rank = 0 — correct key is the TOP candidate!")
        elif key_rank <= 5:
            print(f"\n✅ Key rank = {key_rank} — practically broken.")
        else:
            print(f"\n⚠️  Key rank = {key_rank} — model needs more training.")

    finally:
        sys.stdout = tee.terminal
        tee.close()


if __name__ == "__main__":
    main()
