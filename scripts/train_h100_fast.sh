#!/bin/bash
# 8x H100 Training - FAST VALIDATION (Aggressive Preset)
# ======================================================
# Goal: Validate TRM refinement quickly, then move to Phase 2
# Expected time: 30-60 minutes for 1000 steps
# Expected refinement: 15-25%

echo "Setting up environment..."
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

echo "Building tokenizer..."
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

echo "Downloading 50 data shards (more data for larger batch)..."
uv run python -m nanochat.dataset -n 50

echo "Training tokenizer..."
uv run python -m scripts.tok_train --max_chars=500000000

# 8x H100 FAST VALIDATION CONFIGURATION
# ======================================
# Hardware: 8x H100 (80GB each = 640GB total)
# Strategy: Maximize throughput with large batches
#
# Key Settings:
# - device_batch_size=32 per GPU (vs 2 on RTX 3090)
# - Total batch = 8 GPUs * 32 = 256 sequences per step
# - gradient_accumulation_steps=2 → effective batch = 512
# - compile=True (CRITICAL for H100 performance)
# - bf16=True (H100s excel at bfloat16)
# - depth=12 (can increase to 24 later if needed)
# - n_loops=2 (same TRM config)
# - Aggressive preset (supervision_weight_base=5.0, reward_scale=1.0)
#
# Performance Estimates:
# - H100 is ~10x faster than RTX 3090 for this workload
# - With 8 GPUs + larger batches: ~50-100x total speedup
# - Expected: ~0.5-1 sec/step (vs ~100 sec on RTX 3090)
# - 1000 steps in ~10-20 minutes (vs 28 hours on RTX 3090!)
# - Memory per GPU: ~40-50 GB (plenty of headroom on 80GB)
#
# Why Aggressive (not Extreme) for Phase 1:
# - Faster iteration to validate architecture
# - 15-25% refinement is already very good
# - Can switch to Extreme in Phase 2 for final model

echo ""
echo "=========================================================================="
echo "8x H100 FAST VALIDATION - Aggressive Preset"
echo "=========================================================================="
echo ""
echo "Settings:"
echo "  • 8 GPUs x 32 batch x 2 grad_accum = 512 effective batch"
echo "  • Aggressive preset: 15-25% refinement"
echo "  • Compile + BF16 for maximum H100 performance"
echo "  • Expected: ~0.5-1 sec/step, 1000 steps in ~10-20 minutes"
echo ""
echo "Starting training..."

uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --n_loops=2 \
    --device_batch_size=32 \
    --gradient_accumulation_steps=2 \
    --num_iterations=1000 \
    --compile=True \
    --run=h100_trm_fast_validation \
    --matrix_lr=0.001 \
    --embedding_lr=0.01 \
    --grad_clip=1.0 \
    --save_every=200 \
    --eval_every=100 \
    --eval_tokens=524288 \
    --sample_every=100

echo ""
echo "✅ Phase 1 complete! (~30-60 minutes)"
echo ""
echo "=========================================================================="
echo "VALIDATION CHECKLIST:"
echo "=========================================================================="
echo ""
echo "1. Check refinement (should be 15-25%):"
echo "   ✅ step_0 > step_1 > step_2 > step_3 (monotonic improvement)"
echo ""
echo "2. Check gradient norms (should be 0.5-1.5):"
echo "   ✅ Stable and not exploding"
echo ""
echo "3. Check loss curve:"
echo "   ✅ Smoothly decreasing"
echo ""
echo "If all looks good → Proceed to Phase 2 (extreme preset)"
echo "See train_h100_extreme.sh for maximum quality training"
