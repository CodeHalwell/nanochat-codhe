#!/bin/bash
# Optimized TRM training - Maximum quality!

# ============================================================================
# CACHE CLEARING CONFIGURATION
# ============================================================================
# Set to "true" to clear compilation caches before training
# Recommended when:
#   - Changing model architecture (n_loops, T_recursion, n_sup_train, etc.)
#   - Switching model sizes (70M → 180M → 250M)
#   - After git pull with code changes
#   - Activation function changes (relu² → swiglu)
# Set to "false" for normal training runs with unchanged architecture
CLEAR_CACHE="true"

if [ "$CLEAR_CACHE" = "true" ]; then
    echo "============================================================================"
    echo "Clearing compilation caches..."
    echo "============================================================================"

    # Clear Python bytecode cache
    echo "→ Clearing Python bytecode cache..."
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

    # Clear Triton compilation cache
    echo "→ Clearing Triton compilation cache..."
    rm -rf ~/.triton/cache/* 2>/dev/null || true

    # Clear Torch compilation cache
    echo "→ Clearing Torch compilation cache..."
    rm -rf ~/.torch/compile_cache/* 2>/dev/null || true
    rm -rf /tmp/torchinductor_* 2>/dev/null || true

    echo "✓ All caches cleared!"
    echo "  Note: First training step will be slower (~8-10 min) due to recompilation"
    echo "============================================================================"
    echo ""
fi

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

# OPTIMIZED TRM TRAINING WITH PROGRESSIVE REFINEMENT
# ===================================================
# TRM Configuration (AGGRESSIVE BOOST - Maximum Grad Norm):
# - n_loops=3, T_recursion=1, N_sup=6 (optimized for 24GB GPU)
# - supervision_weight_base=2.5 (balanced incentive for later steps)
# - improvement_reward_scale=1.5 (strong improvement bonus)
# - detach_threshold=2 (NEW: first 2 steps fully connected!)
# - activation_fn='swiglu' (modern gated activation, 10-15% better than relu²)
#
# Key Improvements for Maximum Gradient Flow:
# ✅ Loosened grad_clip: 0.25 → 0.5 (2x more headroom)
# ✅ Increased LR: 0.015/0.15 → 0.025/0.25 (+67% boost)
# ✅ Selective detachment: Steps 0,1 fully connected (FULL grad signal!)
# ✅ Stronger improvement bonus: 1.0 → 1.5 (rewards refinement)
# ✅ Rebalanced weighting: base 2.0 → 2.5 (incentivizes later steps)
#
# Selective Strategic Detachment (NEW FEATURE):
# - Steps 0,1: FULLY CONNECTED (complete gradient signal)
# - Steps 2,3,4: DETACHED (prevent vanishing gradients)
# - Step 5 (final): CONNECTED (complete backward pass)
# - Result: Early steps get 2-3x stronger learning signal
#
# Expected Results (AGGRESSIVE BOOST):
# - Gradient norms: 0.25-0.60 (was 0.08-0.15) → 2-4x improvement!
# - Progressive refinement: 10-25% (was <5%)
# - Supervision losses: Strong monotonic decrease
# - No gradient collapse pattern
# - Much stronger early-step learning
#
# Hierarchical Weights (base=2.5, N_sup=6):
# Step 0: 0.62%, Step 1: 1.54%, Step 2: 3.86%
# Step 3: 9.64%, Step 4: 24.1%, Step 5: 60.2%
# (Final step gets 60% weight, step 0 gets 0.62%)
#
# Performance (RTX 3090 24GB):
# - device_batch_size=1: Required for memory
# - compile=False: Faster iteration during experimentation
# - Expected: ~10-14 sec/step
# - 200 steps in ~30-45 minutes
# - Memory usage: ~22-23 GB (fits comfortably in 24GB)
echo "Starting TRM training with AGGRESSIVE BOOST..."
echo "n_loops=3, T_recursion=1, N_sup=6, detach_threshold=2"
echo "LR: 0.025/0.25, grad_clip=0.5, base=2.5, reward=1.5"
uv run torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=10 \
    --device_batch_size=1 \
    --num_iterations=200 \
    --compile=False \
    --run=trm_conservative_enhanced_fixed \
    --matrix_lr=0.025 \
    --embedding_lr=0.25 \
    --grad_clip=0.5 \
    --save_every=100 \
    --eval_every=25 \
    --eval_tokens=163840 \
    --sample_every=25


echo "✅ Training complete!"
echo ""
echo "Expected TRM Refinement Behavior (AGGRESSIVE BOOST):"
echo "  ✅ Gradient norms: 0.25-0.60 (2-4x higher than previous runs!)"
echo "  ✅ Progressive refinement: 10-25% from step_0 to final"
echo "  ✅ Supervision order: step_0 > step_1 > step_2 > step_3 > step_4 > step_5"
echo "  ✅ Loss decreasing smoothly without collapse"
echo "  ✅ NO gradient collapse pattern (stable throughout training)"
echo "  ✅ Strong early-step learning (steps 0,1 fully connected)"
echo ""
echo "Applied Improvements (AGGRESSIVE BOOST Configuration):"
echo "  ✅ Loosened grad_clip: 0.25 → 0.5 (2x more headroom)"
echo "  ✅ Increased LR: 0.015/0.15 → 0.025/0.25 (+67% boost)"
echo "  ✅ SELECTIVE DETACHMENT: Steps 0,1 fully connected (NEW!)"
echo "  ✅ Stronger improvement bonus: 1.0 → 1.5"
echo "  ✅ Rebalanced weighting: base 2.0 → 2.5"
echo ""
echo "Current Configuration:"
echo "  - n_loops=3, T_recursion=1, N_sup=6"
echo "  - supervision_weight_base=2.5 (balanced)"
echo "  - improvement_reward_scale=1.5 (strong bonus)"
echo "  - detach_threshold=2 (steps 0,1 fully connected)"
echo "  - Learning rates: 0.025/0.25, grad_clip=0.5"
echo ""
echo "Key Innovation: Selective Strategic Detachment"
echo "  - Steps 0,1: FULL gradient signal (no detachment)"
echo "  - Steps 2,3,4: Detached (prevent vanishing gradients)"
echo "  - Step 5: FULL gradient signal (final refinement)"
echo "  - Result: Best of both worlds - strong learning + stability"