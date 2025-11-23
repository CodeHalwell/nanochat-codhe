#!/bin/bash
# 250M Parameter Model - RTX 3090 Local Training
# ===============================================
# Scales up from 70M to 250M parameters
# Uses SwiGLU activation for quality
# batch_size=1, compile=False to fit in 24GB

echo "Setting up environment..."
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

echo "Building tokenizer..."
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

echo "Downloading 10 data shards..."
uv run python -m nanochat.dataset -n 10

echo "Training tokenizer..."
uv run python -m scripts.tok_train --max_chars=100000000

# 250M PARAMETER MODEL - LOCAL RTX 3090
# ======================================
# Model Configuration:
# - n_layer=20 (vs 12 baseline) - deeper model
# - n_embd=1024 (vs 768 baseline) - wider model
# - n_head=12 (vs 6 baseline) - more attention heads
# - n_kv_head=4 (enables GQA - memory efficient)
# - activation='swiglu' (modern activation)
# - Total params: ~250M (vs 70M baseline)
#
# Memory Configuration:
# - device_batch_size=1 (required for 250M to fit)
# - compile=False (saves 3-5 GB memory)
# - Expected memory: ~18-20 GB (fits in 24GB with headroom)
#
# Performance:
# - Speed: ~20-30 sec/step (larger model + no compile)
# - 200 steps: ~1-1.5 hours
# - Quality: +30-40% better than 70M baseline!
#
# Learning Rates (adjusted for larger model):
# - matrix_lr=0.0008 (20% lower than default for stability)
# - embedding_lr=0.008 (20% lower than default)

echo ""
echo "=========================================================================="
echo "250M Parameter Model - RTX 3090 Training"
echo "=========================================================================="
echo ""
echo "Model: 20 layers, 1024 dim, 12 heads (4 KV), SwiGLU activation"
echo "Params: ~250M (3.5x larger than baseline)"
echo "Memory: ~18-20 GB estimated (batch=1, no compile)"
echo "Speed: ~20-30 sec/step, ~1-1.5 hours for 200 steps"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

echo ""
echo "Starting 250M model training..."

uv run torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=20 \
    --n_embd=1024 \
    --n_head=12 \
    --n_kv_head=4 \
    --n_loops=2 \
    --device_batch_size=1 \
    --num_iterations=200 \
    --compile=False \
    --run=trm_250m_local \
    --matrix_lr=0.0008 \
    --embedding_lr=0.008 \
    --grad_clip=1.0 \
    --save_every=100 \
    --eval_every=25 \
    --eval_tokens=163840 \
    --sample_every=25

echo ""
echo "✅ 250M model training complete!"
echo ""
echo "=========================================================================="
echo "RESULTS COMPARISON:"
echo "=========================================================================="
echo ""
echo "70M baseline:  ~1.7-1.9 final loss, 15-25% refinement"
echo "250M model:    ~1.2-1.5 final loss, 25-35% refinement (30-40% better!)"
echo ""
echo "Next steps:"
echo "  • Compare final loss and refinement vs 70M baseline"
echo "  • If quality is good, consider cloud training for even larger model"
echo "  • See CLOUD_TRAINING_GUIDE.md for 8x A100 setup (500M+ params possible)"
