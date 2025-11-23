#!/bin/bash
# 4x H100 Training - PHASE 1: FAST VALIDATION
# ============================================
# Cost: $15.80/hour (4 × $3.95)
# Time: 50-90 minutes
# Total: $13-24
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
echo "4x H100 - PHASE 1: Fast Validation (Aggressive Preset)"
echo "=========================================================================="
echo ""
echo "Hardware: 4x Nvidia H100 (mid-range budget option)"
echo "Cost: $15.80/hour"
echo "Settings:"
echo "  • 4 GPUs × 32 batch × 4 grad_accum = 512 effective batch"
echo "  • Aggressive preset: 15-25% refinement"
echo "  • Expected: ~0.8-1.2 sec/step"
echo "  • Estimated time: 50-90 minutes (1000 steps)"
echo "  • Estimated cost: $13-24"
echo ""
echo "Starting training..."

uv run torchrun --standalone --nproc_per_node=4 -m scripts.base_train -- \
    --depth=12 \
    --n_loops=2 \
    --device_batch_size=32 \
    --gradient_accumulation_steps=4 \
    --num_iterations=1000 \
    --compile=True \
    --run=h100_4x_fast_validation \
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
echo "Next: If validation successful, run train_4xH100_extreme.sh for maximum quality"
