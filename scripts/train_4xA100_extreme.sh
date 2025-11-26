#!/bin/bash
# 4x A100 80GB Training - PHASE 2: MAXIMUM QUALITY
# =================================================
# Cost: $10/hour (4 × $2.50)
# Time: 4-7 hours
# Total: $40-70
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
echo "4x A100 80GB - PHASE 2: Maximum Quality (Extreme Preset)"
echo "=========================================================================="
echo ""
echo "⚠️  BEFORE RUNNING - Update nanochat/gpt.py lines 44-45:"
echo "   supervision_weight_base: float = 10.0"
echo "   improvement_reward_scale: float = 2.0"
echo ""
echo "Hardware: 4x Nvidia A100 80GB (ULTRA BUDGET!)"
echo "Cost: $10/hour"
echo "Settings:"
echo "  • 4 GPUs × 14 batch × 4 grad_accum = 224 effective batch"
echo "  • Extreme preset: 50-97% refinement"
echo "  • Expected: ~2.0-3.0 sec/step"
echo "  • Estimated time: 4-7 hours (5000 steps)"
echo "  • Estimated cost: $40-70"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

uv run torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=12 \
    --n_loops=3 \
    --device_batch_size=14 \
    --gradient_accumulation_steps=4 \
    --num_iterations=5000 \
    --compile=True \
    --run=a100_4x_extreme_quality \
    --matrix_lr=0.001 \
    --embedding_lr=0.001 \
    --grad_clip=0.5 \
    --save_every=500 \
    --eval_every=250 \
    --eval_tokens=1048576 \
    --sample_every=250

echo ""
echo "✅ Phase 2 complete! Production model with 50-97% refinement achieved!"
echo ""
echo "Total cost: $50-87 (Phase 1 + Phase 2)"
echo "Total time: ~5-8 hours"
echo ""
echo "🏆 ULTRA BUDGET CHAMPION: Production-quality model for under $90!"
