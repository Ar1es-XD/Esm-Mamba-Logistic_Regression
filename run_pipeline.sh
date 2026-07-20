#!/usr/bin/env bash
# =================================================================
# run_pipeline.sh — Master orchestrator for ESM-Mamba + LR pipeline
#
# Restructured to run the 4 modular self-contained experiments.
# Run this on the rig (Ryzen 9 9900X / RTX PRO 4000 Blackwell).
# =================================================================
set -euo pipefail

echo "=========================================================="
echo " ESM-Mamba + Logistic Regression Pipeline Orchestration"
echo " Target: RTX PRO 4000 (24 GB VRAM)"
echo "=========================================================="
echo ""

# ── Step 0: Sanity checks ───────────────────────────────────────
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null; then
    echo "ERROR: PyTorch CUDA not available. Install CUDA-enabled PyTorch."
    exit 1
fi
echo "✓ CUDA available"

if ! python3 -c "import esm" 2>/dev/null; then
    echo "ERROR: fair-esm not installed. Run: pip install fair-esm"
    exit 1
fi
echo "✓ fair-esm installed"

if ! python3 -c "from mambapy.vim import VMamba" 2>/dev/null; then
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
    python3 shared/Pretrained.py
    touch Outputs/Pretrained_HIV/ab/.done
    touch Outputs/Pretrained_HIV/ag/.done
    echo "✓ ESM-2 embeddings generated"
fi
echo ""

# ── Step 2: Extract and Cache v_fused (Phases B-D) ─────────────
if [ -f "shared/v_fused_cache/.done" ]; then
    echo "✓ v_fused cache already exists (skipping Step 2)"
else
    echo "▶ Step 2: Caching v_fused vectors for all unique pairs (GPU-bound)..."
    python3 extract_vfused_cache.py
    touch shared/v_fused_cache/.done
    echo "✓ v_fused vectors cached successfully"
fi
echo ""

# ── Step 3: Run all experiments ────────────────────────────────
echo "▶ Step 3: Fitting Logistic Regression across all partitions..."
python3 run_all_experiments.py
echo ""

echo "=========================================================="
echo " Pipeline Complete!"
echo " Results aggregated in: summary_results.csv"
echo "=========================================================="
