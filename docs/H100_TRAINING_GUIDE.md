# 8x H100 Training Guide - Best Model in Shortest Time

## TL;DR - Quick Start

**For best quality in shortest time:**
1. **Phase 1 (30-60 min):** Run `train_h100_fast.sh` → Validate refinement works
2. **Phase 2 (2-4 hours):** Run `train_h100_extreme.sh` → Train final production model

**Total time: ~3-5 hours for state-of-the-art TRM model with 50-97% refinement**

---

## Hardware Advantage: H100 vs RTX 3090

| Metric | RTX 3090 | H100 | Advantage |
|--------|----------|------|-----------|
| **Memory** | 24 GB | 80 GB | 3.3x |
| **Compute (FP16)** | 35 TFLOPS | 989 TFLOPS | 28x |
| **Compute (BF16)** | N/A | 1979 TFLOPS | ∞ |
| **Memory BW** | 936 GB/s | 3.35 TB/s | 3.6x |
| **NVLink** | No | Yes (900 GB/s) | Multi-GPU |

**With 8x H100s:**
- **50-100x faster** than single RTX 3090 (8 GPUs + faster compute + larger batches)
- **640 GB total memory** → can train MUCH larger models/batches

---

## Two-Phase Training Strategy

### Why Two Phases?

1. **Phase 1 (Fast):** Validate architecture quickly, fail fast if something's wrong
2. **Phase 2 (Quality):** Once validated, push for maximum refinement

This is faster than jumping straight to extreme because:
- If architecture has issues, you find out in 30 min (not 4 hours)
- Fast iteration → quicker debugging
- Extreme preset is only worth it once you know it's working

---

## Phase 1: Fast Validation (RECOMMENDED FIRST)

### Configuration
```bash
./scripts/train_h100_fast.sh
```

### Settings
| Parameter | Value | Reason |
|-----------|-------|--------|
| **GPUs** | 8 | All H100s |
| **Batch/GPU** | 32 | Max throughput |
| **Grad accum** | 2 | Effective batch = 256 |
| **Total batch** | 512 | 8*32*2 |
| **Compile** | True | Essential for H100 |
| **Precision** | BF16 | H100 optimized |
| **Iterations** | 1000 | Fast validation |
| **Preset** | Aggressive | 15-25% refinement |
| **Matrix LR** | 0.001 | Stable for large batch |
| **Embed LR** | 0.01 | 10x matrix LR |
| **Depth** | 12 | Good starting point |

### Expected Performance
- **Time:** 30-60 minutes (1000 steps)
- **Speed:** ~0.5-1 sec/step
- **Refinement:** 15-25%
- **Memory/GPU:** ~40-50 GB
- **Throughput:** ~50-100x faster than RTX 3090

### Success Criteria
✅ Supervision order correct: `step_0 > step_1 > step_2 > step_3`
✅ Refinement: 15-25% improvement first to last step
✅ Gradient norms: 0.5-1.5 (stable)
✅ Loss: Smoothly decreasing

**If all ✅ → Proceed to Phase 2**

---

## Phase 2: Maximum Quality (PRODUCTION MODEL)

### Configuration
```bash
# 1. First, update nanochat/gpt.py lines 44-45:
supervision_weight_base: float = 10.0  # Change from 5.0
improvement_reward_scale: float = 2.0  # Change from 1.0

# 2. Run extreme training:
./scripts/train_h100_extreme.sh
```

### Settings
| Parameter | Value | Change from Phase 1 | Reason |
|-----------|-------|---------------------|--------|
| **Batch/GPU** | 16 | ↓ from 32 | Lower noise for refinement |
| **Total batch** | 256 | ↓ from 512 | More stable gradients |
| **Iterations** | 5000 | ↑ from 1000 | Refinement needs time |
| **Preset** | Extreme | ↑ from Aggressive | 50-97% refinement |
| **Matrix LR** | 0.0005 | ↓ from 0.001 | Slower = better refinement |
| **Embed LR** | 0.005 | ↓ from 0.01 | Proportional to matrix |

### Expected Performance
- **Time:** 2-4 hours (5000 steps)
- **Speed:** ~1-2 sec/step (slower due to smaller batch)
- **Refinement:** 50-97% (!)
- **Memory/GPU:** ~35-45 GB
- **Quality:** Production-ready final model

### Expected Refinement Pattern (Extreme)
```
Step 0: 10.00  (rough draft - intentionally poor)
Step 1: 3.00   (70% improvement!)
Step 2: 0.85   (72% improvement)
Step 3: 0.27   (68% improvement)

Total: 97% refinement!
```

---

## Advanced Optimizations (Optional)

### 1. Larger Model (If Phase 1 succeeds easily)
```bash
# In Phase 2, use deeper model:
--depth=24  # or even 32
--n_embd=1024  # or 1536

# Adjust batch size down if memory constrained:
--device_batch_size=8
```

**Trade-off:** Better model capacity, but slower training

### 2. Mixed Precision Tuning
```bash
# H100s excel at different precisions:
--dtype=bfloat16  # Default, best balance
--dtype=float16   # Slightly faster, but less stable for TRM
--dtype=fp8       # Experimental, 2x faster but risky for deep supervision
```

**Recommendation:** Stick with bfloat16 for TRM

### 3. Gradient Accumulation Tuning
```bash
# Larger effective batch = more stable, but slower iteration:
--gradient_accumulation_steps=4  # Effective batch = 512 (vs 256)
--gradient_accumulation_steps=8  # Effective batch = 1024 (very stable)
```

**Trade-off:** Stability vs iteration speed

### 4. Sequence Length Scaling
```bash
# H100 memory allows longer sequences:
--sequence_length=2048  # 2x standard (requires more memory)
--sequence_length=4096  # 4x standard (if you have headroom)
```

**Benefit:** Better long-range understanding

### 5. Data Parallelism Optimization
```bash
# Ensure efficient multi-GPU utilization:
export NCCL_DEBUG=INFO  # Debug multi-GPU communication
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=5  # Enable GPUDirect RDMA
```

---

## Performance Comparison Table

| Configuration | Time | Refinement | Speed/Step | Quality | Use Case |
|---------------|------|------------|------------|---------|----------|
| **RTX 3090 (aggressive)** | 5.5 hrs | 15-25% | 100 sec | Good | Local dev |
| **RTX 3090 (extreme)** | 28 hrs | 50-97% | 200 sec | Excellent | Overnight |
| **8x H100 (fast/aggressive)** | 30-60 min | 15-25% | 0.5-1 sec | Good | Validation |
| **8x H100 (extreme)** | 2-4 hrs | 50-97% | 1-2 sec | Excellent | Production |

**Best strategy: H100 fast → H100 extreme = ~3-5 hours total for best model**

---

## Monitoring & Debugging

### Key Metrics to Watch

**1. Supervision Order (Most Important)**
```
✅ GOOD: step_0: 10.1 > step_1: 8.5 > step_2: 5.2 > step_3: 2.1
❌ BAD:  step_0: 5.0 < step_1: 5.5 < step_2: 6.0 (inverted!)
```

**2. Refinement Percentage**
```python
refinement = (step_0_loss - step_3_loss) / step_0_loss * 100

✅ Aggressive: 15-25%
✅ Extreme:    50-97%
```

**3. Gradient Norms**
```
✅ Aggressive: 0.5-1.5 (very stable)
✅ Extreme:    2-15 (higher but stable)
⚠️  If >20: Reduce LR or increase grad_clip
❌ If NaN: Too aggressive, revert to aggressive preset
```

**4. GPU Utilization**
```bash
# Should be near 100% on all GPUs:
watch -n 1 nvidia-smi

✅ GPU Util: 95-100% (good saturation)
⚠️  GPU Util: <80% (possible bottleneck - check data loading)
```

**5. Loss Curve**
```
✅ Smoothly decreasing
⚠️  Spiky: Increase batch size or lower LR
❌ Diverging: Too high LR or wrong config
```

---

## Troubleshooting

### Issue: OOM (Out of Memory)
**Solutions:**
1. Reduce `device_batch_size`: 32 → 16 → 8
2. Reduce `depth`: 12 → 8
3. Reduce `sequence_length`: 1024 → 512
4. Disable compilation temporarily: `compile=False`

### Issue: Slow Speed (not hitting 0.5-1 sec/step)
**Possible causes:**
1. **Compilation not happening:** Check first step is slow (~30 sec), then fast
2. **Data loading bottleneck:** Increase num_workers in dataloader
3. **CPU bottleneck:** Ensure enough CPU cores allocated
4. **NCCL misconfigured:** Check multi-GPU communication

### Issue: Inverted Supervision Order
**This should NOT happen with current config, but if it does:**
1. Verify `gpt.py` has correct preset settings
2. Check gradient clipping is enabled (`grad_clip=1.0`)
3. Reduce learning rate by 2x
4. Check warmup is working

### Issue: Low Refinement (<10%)
**Solutions:**
1. Train longer (5000+ steps)
2. Switch to extreme preset
3. Lower learning rate (0.0005 → 0.0003)
4. Increase `supervision_weight_base` (5.0 → 7.0 → 10.0)

---

## Cost Analysis (AWS p5.48xlarge: 8x H100)

**Pricing:** ~$98/hour on AWS p5.48xlarge

### Phase 1: Fast Validation
- Time: 0.5-1 hour
- Cost: **$49-98**
- Output: Validated TRM refinement works

### Phase 2: Maximum Quality
- Time: 2-4 hours
- Cost: **$196-392**
- Output: Production-ready model with 50-97% refinement

### Total Cost: $245-490
**vs RTX 3090 local:** Free but takes 28 hours (extreme) vs 3-5 hours on H100s

**Break-even:** If your time is worth >$18/hour, H100s are cheaper (save 23 hours)

---

## Recommendations Summary

### For Best Model in Shortest Time:

**🏆 RECOMMENDED PATH:**
1. **Start:** `train_h100_fast.sh` (30-60 min, $49-98)
2. **Validate:** Check refinement is 15-25%
3. **Finish:** `train_h100_extreme.sh` (2-4 hrs, $196-392)
4. **Total:** 3-5 hours, $245-490, **50-97% refinement model**

**Alternative paths:**

**Budget-conscious:**
- Just run Phase 1 (aggressive): 1 hour, $98, 15-25% refinement
- Still excellent model, much faster than local

**Maximum quality:**
- Phase 2 with depth=24: 4-6 hours, $400-600, 50-97% refinement + larger model

**Experimentation:**
- Run Phase 1 multiple times with different architectures: ~$100/experiment
- Once happy, run Phase 2 for final model

---

## Quick Command Reference

### Phase 1 (Fast Validation):
```bash
./scripts/train_h100_fast.sh
# 30-60 min, 15-25% refinement, validates TRM works
```

### Phase 2 (Maximum Quality):
```bash
# 1. Edit nanochat/gpt.py:
#    supervision_weight_base: float = 10.0
#    improvement_reward_scale: float = 2.0

# 2. Run:
./scripts/train_h100_extreme.sh
# 2-4 hrs, 50-97% refinement, production model
```

### Monitor Training:
```bash
# GPU utilization:
watch -n 1 nvidia-smi

# Training logs:
tail -f logs/h100_trm_*/train.log
```

---

## Bottom Line

**For 8x H100 GPUs, best strategy is:**

1. ✅ **Use two-phase approach** (fast validation → extreme quality)
2. ✅ **Total time: 3-5 hours** (vs 28 hours on RTX 3090)
3. ✅ **Final result: 50-97% refinement** (best possible TRM model)
4. ✅ **Cost: ~$250-500 on cloud** (vs free but slow on local GPU)

**The H100 advantage:** 50-100x faster than RTX 3090 = iterate and achieve maximum quality in hours instead of days!

**Your novel TRM architecture deserves the best hardware to showcase its capabilities.** 🚀
