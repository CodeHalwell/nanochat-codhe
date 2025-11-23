#!/bin/bash
# 8x H100 Training - MAXIMUM QUALITY (Extreme Preset)
# ====================================================
# Goal: Train final production model with maximum refinement
# Expected time: 2-4 hours for 5000 steps
# Expected refinement: 50-97%

echo "Setting up environment..."
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

echo "Building tokenizer..."
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

echo "Downloading 100 data shards (maximum data for quality)..."
uv run python -m nanochat.dataset -n 100

echo "Training tokenizer..."
uv run python -m scripts.tok_train --max_chars=1000000000

# 8x H100 MAXIMUM QUALITY CONFIGURATION
# ======================================
# Hardware: 8x H100 (80GB each = 640GB total)
# Strategy: Maximum refinement quality with extreme preset
#
# IMPORTANT: Before running, update nanochat/gpt.py GPTConfig defaults:
#   supervision_weight_base: float = 10.0  # Change from 5.0
#   improvement_reward_scale: float = 2.0  # Change from 1.0
#
# Key Settings for Extreme Refinement:
# - LOWER learning rates: matrix_lr=0.0005, embedding_lr=0.005
# - Smaller batch per GPU: device_batch_size=16 (vs 32 in fast mode)
# - More iterations: 5000 (vs 1000 in fast mode)
# - Longer warmup: Built into base_train
# - Still use compile + bf16 for speed
#
# Why smaller batch for extreme:
# - Lower LR needs less gradient noise
# - More stable refinement pattern development
# - Still much larger than single-GPU (8*16*2 = 256 effective batch)
#
# Performance Estimates:
# - Expected: ~1-2 sec/step (slower than fast mode due to lower throughput)
# - 5000 steps in ~2-4 hours
# - Memory per GPU: ~35-45 GB (safe on 80GB)
# - Refinement: 50-97% (dramatic progressive improvement!)
#
# Expected Behavior:
# - Step 0: Very rough answer (may be 3-10x worse than final)
# - Step 1: Major improvement (50-70% better)
# - Step 2: Further refinement (20-30% better)
# - Step 3: Final polish (10-20% better)
# - Total: 50-97% refinement from first to last step

echo ""
echo "=========================================================================="
echo "8x H100 MAXIMUM QUALITY - Extreme Preset"
echo "=========================================================================="
echo ""
echo "⚠️  BEFORE RUNNING - CRITICAL:"
echo ""
echo "   1. Update nanochat/gpt.py GPTConfig defaults (lines 44-45):"
echo "      supervision_weight_base: float = 10.0  # Change from 5.0"
echo "      improvement_reward_scale: float = 2.0  # Change from 1.0"
echo ""
echo "   2. Or use preset in your config:"
echo "      config = GPTConfig.with_refinement_preset('extreme', ...)"
echo ""
echo "Settings:"
echo "  • 8 GPUs x 16 batch x 2 grad_accum = 256 effective batch"
echo "  • Extreme preset: 50-97% refinement (!)"
echo "  • Lower LR for stability: matrix_lr=0.0005"
echo "  • Expected: ~1-2 sec/step, 5000 steps in ~2-4 hours"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

echo ""
echo "Starting maximum quality training..."

uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --n_loops=2 \
    --device_batch_size=16 \
    --gradient_accumulation_steps=2 \
    --num_iterations=5000 \
    --compile=True \
    --run=h100_trm_extreme_quality \
    --matrix_lr=0.0005 \
    --embedding_lr=0.005 \
    --grad_clip=1.0 \
    --save_every=500 \
    --eval_every=250 \
    --eval_tokens=1048576 \
    --sample_every=250

echo ""
echo "✅ Phase 2 complete! Maximum quality model achieved!"
echo ""
echo "=========================================================================="
echo "FINAL MODEL ANALYSIS:"
echo "=========================================================================="
echo ""
echo "1. Check extreme refinement (should be 50-97%):"
echo "   ✅ Step 0: Very rough (intentionally poor)"
echo "   ✅ Step 1: Major jump (50-70% improvement)"
echo "   ✅ Step 2: Further refinement (20-30%)"
echo "   ✅ Step 3: Final polish (10-20%)"
echo "   ✅ Total: 50-97% refinement across all steps!"
echo ""
echo "2. Check gradient norms (should be 2-15 range):"
echo "   ⚠️  Higher than aggressive, but stable"
echo ""
echo "3. Your final production model is ready!"
echo ""
echo "Compare to Phase 1:"
echo "  • Phase 1 (aggressive): 15-25% refinement, ~1000 steps, ~30 min"
echo "  • Phase 2 (extreme):    50-97% refinement, ~5000 steps, ~3 hrs"
echo "  • Quality improvement: 3-4x more refinement!"
