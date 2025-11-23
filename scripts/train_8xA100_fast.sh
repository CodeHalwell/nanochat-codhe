#!/bin/bash
# 8x A100 80GB Training - PHASE 1: FAST VALIDATION
# =================================================
# Cost: $20/hour (8 × $2.50)
# Time: 45-75 minutes
# Total: $15-25
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
echo "8x A100 80GB - PHASE 1: Fast Validation (Aggressive Preset)"
echo "=========================================================================="
echo ""
echo "Hardware: 8x Nvidia A100 80GB (best value option!)"
echo "Cost: $20/hour"
echo "Settings:"
echo "  • 8 GPUs × 28 batch × 2 grad_accum = 448 effective batch"
echo "  • Aggressive preset: 15-25% refinement"
echo "  • Expected: ~0.7-1.0 sec/step"
echo "  • Estimated time: 45-75 minutes (1000 steps)"
echo "  • Estimated cost: $15-25"
echo ""
echo "Starting training..."

uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --n_loops=2 \
    --device_batch_size=28 \
    --gradient_accumulation_steps=2 \
    --num_iterations=1000 \
    --compile=True \
    --run=a100_8x_fast_validation \
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
echo "Next: If validation successful, run train_8xA100_extreme.sh for maximum quality"
