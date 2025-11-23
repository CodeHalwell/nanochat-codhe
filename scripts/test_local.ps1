# Fast local test for Recursive Model on a single GPU (or CPU) - Windows PowerShell

# 1. Build Tokenizer (Rust)
Write-Host "Building tokenizer..."
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# 2. Download 1 shard of data (if not exists)
Write-Host "Downloading 1 data shard..."
uv run python -m nanochat.dataset -n 1

# 3. Train Tokenizer (fast, on 10M chars)
Write-Host "Training tokenizer (fast)..."
uv run python -m scripts.tok_train --max_chars=10000000

# 4. Train Recursive Model (Tiny)
# depth=2, n_loops=3, 50 iterations
Write-Host "Starting training..."
# Note: Using python -m torch.distributed.run is often more reliable on Windows than torchrun directly
uv run python -m torch.distributed.run --standalone --nproc_per_node=1 -m scripts.base_train -- `
    --depth=2 `
    --n_loops=3 `
    --device_batch_size=8 `
    --num_iterations=50 `
    --run=dummy `
    --save_every=-1 `
    --eval_every=50 `
    --sample_every=50

Write-Host "Test complete! If you see loss decreasing and samples generated, it works."
