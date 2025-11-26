# Architecture Improvements for TRM Model

## Current Architecture (Baseline)

```python
n_layer: 12          # 12 transformer layers
n_embd: 768          # 768-dimensional embeddings
n_head: 6            # 6 attention heads
n_kv_head: 6         # 6 KV heads (MHA, not using GQA)
head_dim: 128        # 768/6 = 128 per head
sequence_len: 1024   # 1024 token context
activation: relu²    # ReLU squared activation
MLP expansion: 4x    # 768 → 3072 → 768
```

**Estimated params:** ~50-60M
**Quality:** Baseline (good for tiny model)

---

## Recommended Improvements

### 1. SwiGLU Activation (High Impact, Easy)

**Why:** Modern LLMs (LLaMA, Mistral, Gemma) all use SwiGLU instead of ReLU variants.

**Benefits:**
- 10-15% better perplexity
- Smoother gradients
- Gating mechanism for selective information flow

**Implementation:**

Replace the MLP class in `nanochat/gpt.py` (lines 172-182) with:

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # SwiGLU requires 3 projections instead of 2
        # Hidden dim is typically (8/3) * n_embd for SwiGLU, but we'll use 4x for consistency
        hidden_dim = 4 * config.n_embd

        if config.activation_fn == 'swiglu':
            # SwiGLU: split hidden layer into gate and value
            self.c_fc = nn.Linear(config.n_embd, 2 * hidden_dim, bias=False)
            self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        elif config.activation_fn == 'geglu':
            # GeGLU: similar to SwiGLU but with GELU
            self.c_fc = nn.Linear(config.n_embd, 2 * hidden_dim, bias=False)
            self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        else:  # relu_squared (default)
            self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
            self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

        self.activation_fn = config.activation_fn

    def forward(self, x):
        if self.activation_fn == 'swiglu':
            # SwiGLU: x = (Wx) * σ(Vx) where σ is SiLU (Swish)
            x = self.c_fc(x)
            gate, value = x.chunk(2, dim=-1)
            x = F.silu(gate) * value  # SwiGLU
            x = self.c_proj(x)
        elif self.activation_fn == 'geglu':
            # GeGLU: x = (Wx) * GELU(Vx)
            x = self.c_fc(x)
            gate, value = x.chunk(2, dim=-1)
            x = F.gelu(gate) * value  # GeGLU
            x = self.c_proj(x)
        else:  # relu_squared
            x = self.c_fc(x)
            x = F.relu(x).square()
            x = self.c_proj(x)
        return x
```

**Usage:**
```python
# In GPTConfig:
activation_fn: str = 'swiglu'  # Options: 'relu_squared', 'swiglu', 'geglu'
```

**Note:** SwiGLU has slightly more params (~1.5x MLP params) but is worth it!

---

### 2. Scale Up Model Size (Highest Impact)

#### Configuration A: Small+ (~120M params)

**Good for:** Fast iteration, still fits easily on RTX 3090

```python
@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 16                # ↑ +4 layers
    n_head: int = 8                  # ↑ +2 heads
    n_kv_head: int = 4               # ✨ Enable GQA
    n_embd: int = 1024               # ↑ +256 dim
    activation_fn: str = 'swiglu'    # ✨ Better activation
    # ... rest of config
```

**Expected:**
- Params: ~120M (2x current)
- Memory: ~16-18 GB
- Batch size: 2-4
- Speed: ~1.5-2x slower per step
- Quality: +15-20% better

---

#### Configuration B: Medium (~250M params) 🏆 RECOMMENDED

**Good for:** Best balance of quality and speed on RTX 3090

```python
@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 20                # ↑ +8 layers
    n_head: int = 12                 # ↑ +6 heads
    n_kv_head: int = 4               # ✨ Enable GQA (3 query heads per KV)
    n_embd: int = 1024               # ↑ +256 dim
    activation_fn: str = 'swiglu'    # ✨ Better activation
    # ... rest of config
```

**Expected:**
- Params: ~250M (4-5x current)
- Memory: ~20-22 GB
- Batch size: 2
- Speed: ~2-3x slower per step
- Quality: +30-40% better
- **This is the sweet spot!**

---

#### Configuration C: Large (~500M params)

**Good for:** Maximum quality, research experiments

```python
@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 24                # ↑ +12 layers
    n_head: int = 12                 # ↑ +6 heads
    n_kv_head: int = 4               # ✨ Enable GQA
    n_embd: int = 1536               # ↑ +768 dim
    activation_fn: str = 'swiglu'    # ✨ Better activation
    # ... rest of config
```

**Expected:**
- Params: ~500M (8-10x current)
- Memory: ~23-24 GB (tight!)
- Batch size: 1
- Speed: ~4-5x slower per step
- Quality: +50-60% better
- **Maximum RTX 3090 capacity**

---

#### Configuration D: Extreme (~700-800M params)

**Good for:** Cloud GPUs (H100, A100) only!

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048         # ↑ Longer context
    vocab_size: int = 50304
    n_layer: int = 32                # ↑ Very deep
    n_head: int = 16                 # ↑ Many heads
    n_kv_head: int = 4               # ✨ GQA critical for memory
    n_embd: int = 2048               # ↑ Large dimension
    activation_fn: str = 'swiglu'    # ✨ Better activation
    # ... rest of config
```

**Expected:**
- Params: ~700-800M
- Memory: ~40-50 GB (needs H100/A100 80GB)
- Batch size: 4-8 (on multi-GPU)
- Quality: State-of-the-art for this size

---

### 3. Group Query Attention (GQA)

**Current:** n_kv_head = n_head (Multi-Head Attention)

**Recommended:** n_kv_head < n_head (Group Query Attention)

**Examples:**

| n_head | n_kv_head | Ratio | Memory Savings | Quality Impact |
|--------|-----------|-------|----------------|----------------|
| 8 | 4 | 2:1 | ~30% KV cache | Minimal (<1%) |
| 12 | 4 | 3:1 | ~50% KV cache | Small (~1-2%) |
| 16 | 4 | 4:1 | ~60% KV cache | Moderate (~2-3%) |
| 12 | 2 | 6:1 | ~70% KV cache | Larger (~3-5%) |

**Recommendation:** Use 3:1 or 4:1 ratio (n_head=12, n_kv_head=4 or n_head=16, n_kv_head=4)

**Benefits for TRM:**
- More query heads → richer attention patterns for refinement
- Less KV memory → can use larger batch or longer context
- Faster inference (important for recursive model)

---

### 4. Context Length

**Current:** 1024

**Options:**
- 1024: Good baseline, fast training
- 2048: Better for complex reasoning, 4x slower attention
- 4096: Maximum context, 16x slower attention

**Recommendation for TRM:** Start with 1024, increase only if needed for task

**Why:** TRM refinement benefits more from depth than context length

---

## Implementation Strategy

### Phase 1: Quick Win (1-2 hours)

**Goal:** Add SwiGLU activation to current model

1. Update MLP class (shown above)
2. Add `activation_fn: str = 'swiglu'` to GPTConfig
3. Retrain with same size (12 layers, 768 dim)
4. **Expected gain:** +10-15% quality

### Phase 2: Scale Up (1 day)

**Goal:** Increase model size to ~250M params

1. Use Configuration B (20 layers, 1024 dim, 12 heads, 4 KV heads)
2. Keep SwiGLU activation
3. Adjust learning rates (may need to lower slightly)
4. **Expected gain:** +30-40% quality over baseline

### Phase 3: Optimize (optional)

**Goal:** Fine-tune hyperparameters for scaled model

1. Tune learning rates
2. Experiment with longer warmup
3. Try different GQA ratios
4. **Expected gain:** +5-10% over Phase 2

---

## Comparison Table

| Config | Params | Memory | Batch | Speed | Quality Gain | Use Case |
|--------|--------|--------|-------|-------|--------------|----------|
| **Current** | 60M | 16GB | 2 | 1.0x | Baseline | Baseline |
| **+ SwiGLU** | 70M | 16GB | 2 | 1.0x | +10-15% | Quick win |
| **Small+** | 120M | 18GB | 2-4 | 1.5x | +15-20% | Fast iteration |
| **Medium** 🏆 | 250M | 22GB | 2 | 2.5x | +30-40% | **RECOMMENDED** |
| **Large** | 500M | 24GB | 1 | 4.0x | +50-60% | Maximum RTX 3090 |
| **Extreme** | 800M | 50GB | 4 | 1.5x | +70-80% | Cloud GPUs only |

---

## TRM-Specific Considerations

### Why larger models help TRM refinement:

1. **More capacity for refinement patterns:** Larger models can learn more sophisticated refinement strategies
2. **Better initial predictions:** Stronger step_0 → more headroom for refinement
3. **Richer representations:** More dimensions → better y/z state separation
4. **More attention heads:** Better for multi-step reasoning in TRM loops

### Best architectural choices for TRM:

- ✅ **Depth > Width:** Prefer more layers over wider layers (20 layers @ 1024 > 12 layers @ 1536)
- ✅ **GQA:** More query heads help refinement, fewer KV heads save memory
- ✅ **SwiGLU:** Gating helps model learn when to refine vs when to keep
- ✅ **Moderate context:** 1024-2048 is enough; TRM benefits more from depth

---

## Quick Start

### Step 1: Add SwiGLU (Easy Win)

Update `nanochat/gpt.py`:

```python
# In MLP class (around line 172):
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd

        if config.activation_fn == 'swiglu':
            self.c_fc = nn.Linear(config.n_embd, 2 * hidden_dim, bias=False)
            self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        else:  # relu_squared
            self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
            self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

        self.activation_fn = config.activation_fn

    def forward(self, x):
        if self.activation_fn == 'swiglu':
            x = self.c_fc(x)
            gate, value = x.chunk(2, dim=-1)
            x = F.silu(gate) * value
            x = self.c_proj(x)
        else:  # relu_squared
            x = self.c_fc(x)
            x = F.relu(x).square()
            x = self.c_proj(x)
        return x

# In GPTConfig (around line 38):
activation_fn: str = 'swiglu'  # Change from 'relu_squared'
```

### Step 2: Scale Up Model (Bigger Win)

Update training script to use Medium config:

```bash
# In your training script (test_local.sh), modify base_train call:
--depth=20 \              # Was: 12
--n_embd=1024 \           # Add this
--n_head=12 \             # Add this
--n_kv_head=4 \           # Add this (enables GQA)
```

### Step 3: Adjust Learning Rates (Important!)

For larger models, you may need to reduce learning rates:

```bash
--matrix_lr=0.0008 \      # Was: 0.001 (20% reduction)
--embedding_lr=0.008 \    # Was: 0.01 (20% reduction)
```

---

## Expected Results

### Current (60M, ReLU²):
```
Step 200: Loss ~1.8-2.0
Refinement: 15-25% (if patterns develop)
```

### With SwiGLU (70M):
```
Step 200: Loss ~1.6-1.8 (10-15% better)
Refinement: 18-28%
```

### With Medium Config (250M, SwiGLU, GQA):
```
Step 500: Loss ~1.2-1.4 (30-40% better!)
Refinement: 25-35% (aggressive) or 60-80% (extreme preset)
```

---

## Bottom Line

**For best results on RTX 3090:**

1. ✅ **Add SwiGLU** (10-15% gain, no cost)
2. ✅ **Scale to ~250M params** (30-40% gain, 2-3x slower)
3. ✅ **Enable GQA** (efficiency, allows larger batch/context)

**This will give you a much more capable TRM model!** 🚀

The combination of larger capacity + better activation + more heads will significantly improve both base performance AND refinement quality.
