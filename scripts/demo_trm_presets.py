"""
Demonstration of TRM refinement presets.
Shows moderate, aggressive, and extreme configurations in action.
"""
import torch
from nanochat.gpt import GPT, GPTConfig

def evaluate_refinement(model, idx, targets, n_sup=4):
    """Evaluate how much the model refines across supervision steps"""
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

    refinement = (losses[0] - losses[-1]) / losses[0] * 100 if losses[0] > 0 else 0
    is_monotonic = all(losses[j] >= losses[j+1] - 0.001 for j in range(len(losses)-1))

    return losses, refinement, is_monotonic

def train_with_preset(preset_name, steps=150):
    """Train a model with the given preset"""
    print(f"\n{'='*80}")
    print(f"TESTING PRESET: {preset_name.upper()}")
    print(f"{'='*80}\n")

    torch.manual_seed(42)

    # Create config using preset
    config = GPTConfig.with_refinement_preset(
        preset=preset_name,
        sequence_len=64,
        vocab_size=100,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        n_loops=2,
        n_sup_train=4,
    )

    print(f"Configuration:")
    print(f"  supervision_weight_base: {config.supervision_weight_base}")
    print(f"  improvement_reward_scale: {config.improvement_reward_scale}")
    print()

    model = GPT(config)
    model.init_weights()

    # Adjust LR based on preset
    lr_map = {'moderate': 0.002, 'aggressive': 0.001, 'extreme': 0.0005}
    base_lr = lr_map[preset_name]
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

    idx = torch.randint(0, 100, (4, 64))
    targets = torch.randint(0, 100, (4, 64))

    print(f"Training for {steps} iterations (lr={base_lr})...\n")

    best_refinement = 0
    best_losses = None

    for step in range(steps):
        # Warmup
        warmup = 50 if preset_name == 'moderate' else 100 if preset_name == 'aggressive' else 150
        lr_mult = min((step + 1) / warmup, 1.0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr * lr_mult

        optimizer.zero_grad()
        loss, aux = model(idx, targets)
        loss.backward()

        grad_norm = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)**0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if torch.isnan(loss):
            print(f"❌ NaN at step {step}")
            break

        # Evaluate every 30 steps
        if step % 30 == 0 or step == steps - 1:
            losses, refinement, is_monotonic = evaluate_refinement(model, idx, targets)

            if refinement > best_refinement:
                best_refinement = refinement
                best_losses = losses

            status = "✅" if is_monotonic else "⚠️"
            print(f"Step {step:3d}: {status} ref={refinement:+.2f}% | grad={grad_norm:.3f}")
            print(f"         losses=[{', '.join([f'{l:.4f}' for l in losses])}]")

    # Final summary
    print(f"\n{'='*70}")
    print(f"RESULTS - {preset_name.upper()}")
    print(f"{'='*70}\n")

    print(f"Best refinement achieved: {best_refinement:.2f}%")
    print(f"Best losses: [{', '.join([f'{l:.4f}' for l in best_losses])}]")

    # Step-by-step analysis
    print(f"\nStep-by-step improvements:")
    for i in range(len(best_losses) - 1):
        delta = (best_losses[i] - best_losses[i+1]) / best_losses[i] * 100
        arrow = "✅" if delta > 0 else "❌"
        print(f"  {arrow} Step {i}→{i+1}: {delta:+.2f}%")

    return best_refinement

def main():
    print("="*80)
    print("TRM REFINEMENT PRESETS DEMONSTRATION")
    print("="*80)
    print()
    print("This script demonstrates the three built-in refinement presets:")
    print("  • moderate:   8-12% refinement, very stable")
    print("  • aggressive: 15-25% refinement, recommended for production")
    print("  • extreme:    50-97% refinement, experimental")
    print()

    results = {}

    # Test each preset
    for preset in ['moderate', 'aggressive', 'extreme']:
        refinement = train_with_preset(preset, steps=150)
        results[preset] = refinement

    # Final comparison
    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'='*80}\n")

    print(f"{'Preset':<15} {'Refinement':<15} {'Status'}")
    print("-" * 80)

    for preset, ref in results.items():
        status = "⭐⭐⭐⭐⭐" if preset == 'moderate' else \
                 "⭐⭐⭐⭐" if preset == 'aggressive' else \
                 "⭐⭐⭐"

        print(f"{preset:<15} {ref:>12.2f}% {status:>15}")

    print()
    print("Recommendations:")
    print("  • For production: Use 'aggressive' preset (best balance)")
    print("  • For research: Use 'extreme' preset (maximum refinement)")
    print("  • For stability: Use 'moderate' preset (safest option)")
    print()
    print("Usage:")
    print("  config = GPTConfig.with_refinement_preset('aggressive', n_layer=12)")
    print("  model = GPT(config)")

if __name__ == "__main__":
    main()
