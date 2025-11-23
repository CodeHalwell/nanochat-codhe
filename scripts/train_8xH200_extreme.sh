#!/bin/bash
# 8x H200 Training - PHASE 2: MAXIMUM QUALITY
# ============================================
# Cost: $36.32/hour (8 × $4.54)
# Time: 2-3.5 hours
# Total: $73-127
# Refinement: 50-97%

echo "Setting up environment..."
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

echo "Building tokenizer..."
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

echo "Downloading 100 data shards..."
uv run python -m nanochat.dataset -n 100

echo "Training tokenizer..."
uv run python -m scripts.tok_train --max_chars=1000000000

echo ""
echo "=========================================================================="
echo "8x H200 - PHASE 2: Maximum Quality (Extreme Preset)"
echo "=========================================================================="
echo ""
echo "⚠️  BEFORE RUNNING - Update nanochat/gpt.py lines 44-45:"
echo "   supervision_weight_base: float = 10.0"
echo "   improvement_reward_scale: float = 2.0"
echo ""
echo "Hardware: 8x Nvidia H200"
echo "Cost: $36.32/hour"
echo "Settings:"
echo "  • 8 GPUs × 18 batch × 2 grad_accum = 288 effective batch"
echo "  • Extreme preset: 50-97% refinement"
echo "  • Expected: ~0.8-1.2 sec/step"
echo "  • Estimated time: 2-3.5 hours (5000 steps)"
echo "  • Estimated cost: $73-127"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --n_loops=2 \
    --device_batch_size=18 \
    --gradient_accumulation_steps=2 \
    --num_iterations=5000 \
    --compile=True \
    --run=h200_8x_extreme_quality \
    --matrix_lr=0.0005 \
    --embedding_lr=0.005 \
    --grad_clip=1.0 \
    --save_every=500 \
    --eval_every=250 \
    --eval_tokens=1048576 \
    --sample_every=250

echo ""
echo "✅ Phase 2 complete! Production model with 50-97% refinement achieved!"
echo ""
echo "Total cost: $88-157 (Phase 1 + Phase 2)"
echo "Total time: ~2.5-4 hours"
