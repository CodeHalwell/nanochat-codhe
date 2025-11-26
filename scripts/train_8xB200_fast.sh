#!/bin/bash
# 8x B200 Training - PHASE 1: FAST VALIDATION
# ============================================
# Cost: $50/hour (8 × $6.25)
# Time: 20-40 minutes
# Total: $17-33
# Refinement: 15-25%

echo "Setting up environment..."
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

echo "Building tokenizer..."
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

echo "Downloading 50 data shards..."
uv run python -m nanochat.dataset -n 50

echo "Training tokenizer..."
uv run python -m scripts.tok_train --max_chars=500000000

echo ""
echo "=========================================================================="
echo "8x B200 - PHASE 1: Fast Validation (Aggressive Preset)"
echo "=========================================================================="
echo ""
echo "Hardware: 8x Nvidia B200 (bleeding edge, ~2x faster than H100)"
echo "Cost: $50/hour"
echo "Settings:"
echo "  • 8 GPUs × 40 batch × 2 grad_accum = 640 effective batch"
echo "  • Aggressive preset: 15-25% refinement"
echo "  • Expected: ~0.3-0.5 sec/step"
echo "  • Estimated time: 20-40 minutes (1000 steps)"
echo "  • Estimated cost: $17-33"
echo ""
echo "Starting training..."

uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --n_loops=2 \
    --device_batch_size=40 \
    --gradient_accumulation_steps=2 \
    --num_iterations=1000 \
    --compile=True \
    --run=b200_8x_fast_validation \
    --matrix_lr=0.001 \
    --embedding_lr=0.01 \
    --grad_clip=1.0 \
    --save_every=200 \
    --eval_every=100 \
    --eval_tokens=524288 \
    --sample_every=100

echo ""
echo "✅ Phase 1 complete!"
echo ""
echo "Check refinement: Should see 15-25% improvement from step_0 to step_3"
echo ""
echo "Next: If validation successful, run train_8xB200_extreme.sh for maximum quality"
