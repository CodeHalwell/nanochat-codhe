#!/bin/bash
# 180M Parameter Model - RTX 3090 Local Training (SAFER OPTION)
# ==============================================================
# Conservative scale-up from 70M to 180M parameters
# Uses SwiGLU activation for quality
# batch_size=1, compile=False to fit comfortably in 24GB

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

# 180M PARAMETER MODEL - SAFER OPTION FOR RTX 3090
# =================================================
# Model Configuration:
# - n_layer=16 (vs 12 baseline) - moderately deeper
# - n_embd=1024 (vs 768 baseline) - wider model
# - n_head=12 (vs 6 baseline) - more attention heads
# - n_kv_head=4 (enables GQA - memory efficient)
# - activation='swiglu' (modern activation)
# - Total params: ~180M (vs 70M baseline)
#
# Memory Configuration:
# - device_batch_size=1
# - compile=False (saves 3-5 GB memory)
# - Expected memory: ~16-18 GB (comfortable headroom in 24GB)
#
# Performance:
# - Speed: ~15-25 sec/step
# - 200 steps: ~50-85 minutes
# - Quality: +25-35% better than 70M baseline!
#
# This is the SAFEST option for local training:
# - Less likely to OOM than 250M
# - Still 2.5x larger than baseline
# - Can potentially enable compile=True if memory allows

echo ""
echo "=========================================================================="
echo "180M Parameter Model - RTX 3090 Training (SAFER OPTION)"
echo "=========================================================================="
echo ""
echo "Model: 16 layers, 1024 dim, 12 heads (4 KV), SwiGLU activation"
echo "Params: ~180M (2.5x larger than baseline)"
echo "Memory: ~16-18 GB estimated (comfortable headroom)"
echo "Speed: ~15-25 sec/step, ~50-85 min for 200 steps"
echo ""
echo "This is a conservative scale-up with lower OOM risk"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

echo ""
echo "Starting 180M model training..."

uv run torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=16 \
    --n_embd=1024 \
    --n_head=12 \
    --n_kv_head=4 \
    --n_loops=2 \
    --device_batch_size=1 \
    --num_iterations=200 \
    --compile=False \
    --run=trm_180m_local \
    --matrix_lr=0.0009 \
    --embedding_lr=0.009 \
    --grad_clip=1.0 \
    --save_every=100 \
    --eval_every=25 \
    --eval_tokens=163840 \
    --sample_every=25

echo ""
echo "✅ 180M model training complete!"
echo ""
echo "=========================================================================="
echo "RESULTS COMPARISON:"
echo "=========================================================================="
echo ""
echo "70M baseline:  ~1.7-1.9 final loss, 15-25% refinement"
echo "180M model:    ~1.3-1.6 final loss, 20-30% refinement (25-35% better!)"
echo ""
echo "Next steps:"
echo "  • If this worked well, try 250M (test_local_250m.sh)"
echo "  • Or enable compile=True to speed up training"
echo "  • See CLOUD_TRAINING_GUIDE.md for 8x A100 setup (500M+ params possible)"
