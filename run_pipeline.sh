#!/usr/bin/env bash
# =================================================================
# run_pipeline.sh — Master orchestrator for ESM-Mamba + LR pipeline
#
# Run this on the rig (Ryzen 9 9900X / RTX PRO 4000 Blackwell).
# Expects a Python venv with all dependencies installed.
# =================================================================
set -euo pipefail

echo "============================================"
echo " ESM-Mamba + Logistic Regression Pipeline"
echo " Target: RTX PRO 4000 (24 GB VRAM)"
echo "============================================"
echo ""

# ── Step 0: Sanity checks ───────────────────────────────────────
if ! python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null; then
    echo "ERROR: PyTorch CUDA not available. Install CUDA-enabled PyTorch."
    exit 1
fi
echo "✓ CUDA available"

if ! python -c "import esm" 2>/dev/null; then
    echo "ERROR: fair-esm not installed. Run: pip install fair-esm"
    exit 1
fi
echo "✓ fair-esm installed"

if ! python -c "from mambapy.vim import VMamba" 2>/dev/null; then
    echo "ERROR: mambapy not installed. Run: pip install mambapy"
    exit 1
fi
echo "✓ mambapy installed"
echo ""

# ── Step 1: Generate ESM-2 embeddings (Phase A) ────────────────
if [ -f "Outputs/Pretrained_HIV/ab/.done" ] && [ -f "Outputs/Pretrained_HIV/ag/.done" ]; then
    echo "✓ ESM-2 embeddings already generated (skipping Step 1)"
else
    echo "▶ Step 1: Generating ESM-2 embeddings (GPU-bound, ~10-30 min)..."
    python Pretrained.py
    touch Outputs/Pretrained_HIV/ab/.done
    touch Outputs/Pretrained_HIV/ag/.done
    echo "✓ ESM-2 embeddings generated"
fi
echo ""

# ── Step 2: Train MambaCross + extract v_fused (Phases B-D) ────
if [ -f "cache/.done" ]; then
    echo "✓ v_fused cache already exists (skipping Step 2)"
else
    echo "▶ Step 2: Training MambaCross on 4 splits + extracting v_fused..."
    python 02_train_and_extract_vfused.py
    touch cache/.done
    echo "✓ v_fused cached for all splits"
fi
echo ""

# ── Step 3: Fit Logistic Regression ────────────────────────────
echo "▶ Step 3: Fitting L2-regularised Logistic Regression..."
python 03_fit_logistic_regression.py
echo ""

echo "============================================"
echo " Pipeline complete!"
echo " Results: Results/lr_vs_mlp_comparison.csv"
echo "============================================"
