#!/bin/bash
# TRM Training - EXTREME REFINEMENT (Maximum Quality)
# ====================================================
# This configuration prioritizes refinement quality over speed
# Expected: 50-97% progressive refinement across supervision steps

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

# EXTREME TRM REFINEMENT CONFIGURATION
# =====================================
# TRM Settings (requires modifying gpt.py GPTConfig defaults):
# - supervision_weight_base=10.0 (extreme: final step weighted 1000x more!)
# - improvement_reward_scale=2.0 (double reward for improvements)
# - n_loops=2, N_sup=4
#
# Training Settings for Maximum Refinement:
# - VERY LOW LR: matrix_lr=0.0005, embedding_lr=0.005 (10x lower than aggressive)
# - Slower convergence allows refinement pattern to develop fully
# - device_batch_size=1 (more stable gradients, less noise)
# - compile=False (more deterministic, easier to debug)
# - More iterations: 500+ recommended (refinement develops gradually)
# - Longer warmup: Built into base_train
#
# Expected Results:
# - 50-97% progressive refinement (dramatic improvement across steps!)
# - Step 0 may be intentionally poor (rough draft)
# - Each step dramatically improves: step_0 >> step_1 >> step_2 >> step_3
# - Gradient norms: 2-15 range (higher than aggressive, monitor carefully)
# - Slower training: ~15-20 sec/step, 500 steps in ~2-3 hours
#
# Memory Usage:
# - ~20-21 GB with batch_size=1, compile=False
# - Safe for 24GB card with headroom for OS
#
# IMPORTANT: Before running, update nanochat/gpt.py GPTConfig defaults:
#   supervision_weight_base: float = 10.0  # Change from 5.0
#   improvement_reward_scale: float = 2.0  # Change from 1.0
#
# Or use the preset method in your training script:
#   config = GPTConfig.with_refinement_preset('extreme', ...)

echo ""
echo "=========================================================================="
echo "EXTREME REFINEMENT MODE - Maximum Quality Configuration"
echo "=========================================================================="
echo ""
echo "⚠️  BEFORE RUNNING:"
echo "   1. Update nanochat/gpt.py GPTConfig defaults:"
echo "      supervision_weight_base: float = 10.0"
echo "      improvement_reward_scale: float = 2.0"
echo ""
echo "   OR modify base_train to use:"
echo "      config = GPTConfig.with_refinement_preset('extreme', ...)"
echo ""
echo "   2. This will take 2-3 hours for 500 steps"
echo "      (vs ~20-25 min for aggressive preset)"
echo ""
echo "   3. Expected: 50-97% refinement (vs 15-25% for aggressive)"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

echo ""
echo "Starting EXTREME refinement training..."
echo "Settings: lr=0.0005/0.005, batch=1, compile=False, 500 iterations"
uv run torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=12 \
    --n_loops=2 \
    --device_batch_size=1 \
    --num_iterations=500 \
    --compile=False \
    --run=trm_extreme_refinement \
    --matrix_lr=0.0005 \
    --embedding_lr=0.005 \
    --grad_clip=1.0 \
    --save_every=100 \
    --eval_every=50 \
    --eval_tokens=163840 \
    --sample_every=50

echo ""
echo "✅ Extreme refinement training complete!"
echo ""
echo "=========================================================================="
echo "ANALYSIS CHECKLIST:"
echo "=========================================================================="
echo ""
echo "1. Check supervision order (should be DRAMATIC improvement):"
echo "   ✅ Step 0: High loss (rough draft, may be 3-10x worse than final)"
echo "   ✅ Step 1: Major improvement (50-70% better than step 0)"
echo "   ✅ Step 2: Further improvement (20-30% better than step 1)"
echo "   ✅ Step 3: Final polish (10-20% better than step 2)"
echo ""
echo "2. Check gradient norms:"
echo "   ✅ Should be in 2-15 range (higher than aggressive)"
echo "   ⚠️  If >20: reduce LR further or increase grad_clip"
echo "   ⚠️  If NaN: too aggressive, use aggressive preset instead"
echo ""
echo "3. Check total refinement:"
echo "   ✅ 50-70%: Good extreme refinement"
echo "   ✅ 70-90%: Excellent extreme refinement"
echo "   ✅ 90-97%: Outstanding extreme refinement!"
echo "   ⚠️  <30%: May need more training steps or check config"
echo ""
echo "4. Compare to aggressive preset:"
echo "   • Aggressive: 15-25% refinement, ~20-25 min for 200 steps"
echo "   • Extreme:    50-97% refinement, ~2-3 hrs for 500 steps"
echo "   • Trade-off: 3-4x longer training for 3-4x more refinement"
echo ""
echo "See TRM_REFINEMENT_GUIDE.md for detailed analysis."
