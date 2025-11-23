"""
Find optimal TRM configuration for progressive refinement.
Test different n_sup and loss weighting strategies.
"""
import torch
from nanochat.gpt import GPT, GPTConfig

def get_lr_schedule(step, warmup_steps=50, base_lr=1.0):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr

def test_config(n_sup, n_loops, use_weighting=False):
    """Test a specific TRM configuration"""
    torch.manual_seed(42)

    config = GPTConfig(
        sequence_len=32,
        vocab_size=100,
        n_layer=3,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        n_loops=n_loops,
        n_sup_train=n_sup,
        activation_fn='relu_squared'
    )

    model = GPT(config)
    model.init_weights()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

    idx = torch.randint(0, 100, (2, 32))
    targets = torch.randint(0, 100, (2, 32))

    for i in range(80):
        lr = get_lr_schedule(i, warmup_steps=50, base_lr=0.003)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()
        loss, aux = model(idx, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        x = model.transformer.wte(idx)
        x = torch.nn.functional.rms_norm(x, (x.size(-1),))
        y, z = x.clone(), torch.zeros_like(x)

        losses = []
        for sup_step in range(n_sup):
            for block in model.transformer.h:
                cos_sin = model.cos[:, :idx.size(1)], model.sin[:, :idx.size(1)]
                y, z = block(x, y, z, cos_sin, kv_cache=None)

            logits = model.lm_head(torch.nn.functional.rms_norm(y, (y.size(-1),)))
            logits = 15 * torch.tanh(logits / 15)
            step_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            losses.append(step_loss.item())

    # Check if monotonically improving
    is_refining = all(losses[j] >= losses[j+1] for j in range(len(losses)-1))
    refinement = (losses[0] - losses[-1]) / losses[0] * 100 if losses[0] > 0 else 0

    return losses, is_refining, refinement

def main():
    print("="*80)
    print("FINDING OPTIMAL TRM CONFIGURATION")
    print("="*80)
    print()

    configs = [
        (2, 1, "n_sup=2, n_loops=1 (minimal)"),
        (2, 2, "n_sup=2, n_loops=2 (balanced)"),
        (3, 1, "n_sup=3, n_loops=1"),
        (3, 2, "n_sup=3, n_loops=2 (current)"),
        (4, 1, "n_sup=4, n_loops=1"),
        (4, 2, "n_sup=4, n_loops=2"),
    ]

    results = []

    for n_sup, n_loops, desc in configs:
        losses, is_refining, refinement = test_config(n_sup, n_loops)
        results.append((desc, losses, is_refining, refinement))

        status = "✅ REFINING" if is_refining else "⚠️  mixed"
        print(f"{status} | {desc:30s} | refinement: {refinement:+.2f}%")
        print(f"         losses: [{', '.join([f'{l:.4f}' for l in losses])}]")
        print()

    print()
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Find best refining config
    refining_configs = [(desc, ref) for desc, _, is_ref, ref in results if is_ref]

    if refining_configs:
        best = max(refining_configs, key=lambda x: x[1])
        print(f"✅ BEST REFINING CONFIG: {best[0]}")
        print(f"   Refinement: {best[1]:.2f}%")
        print()
        print("🎯 USE THIS CONFIGURATION for your training!")
    else:
        # Find config with best refinement even if not perfect
        best = max(results, key=lambda x: x[3])
        print(f"⚠️  No perfectly refining config found")
        print(f"   Best partial: {best[0]} ({best[3]:.2f}% refinement)")
        print()
        print("💡 Suggestions:")
        print("   - Use n_sup=2 or 3 (fewer supervision steps)")
        print("   - Lower learning rate further")
        print("   - Increase n_loops for more computation per step")

if __name__ == "__main__":
    main()
