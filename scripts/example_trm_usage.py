"""
Quick-start example for using TRM with progressive refinement.

Shows how to:
1. Create a model with different refinement presets
2. Train with proper gradient clipping and warmup
3. Monitor refinement progress
"""
import torch
from nanochat.gpt import GPT, GPTConfig

def main():
    print("="*80)
    print("TRM QUICK START EXAMPLE")
    print("="*80)
    print()

    # ========================================================================
    # Option 1: Use default settings (aggressive preset built-in)
    # ========================================================================
    print("Option 1: Default configuration (aggressive refinement)")
    print("-" * 80)

    config_default = GPTConfig(
        sequence_len=1024,
        vocab_size=50304,
        n_layer=12,
        n_embd=768,
        n_loops=2,
        n_sup_train=4,
        # supervision_weight_base=5.0,      # Default
        # improvement_reward_scale=1.0,     # Default
    )

    print(f"  supervision_weight_base: {config_default.supervision_weight_base}")
    print(f"  improvement_reward_scale: {config_default.improvement_reward_scale}")
    print(f"  Expected refinement: 15-25%")
    print()

    # ========================================================================
    # Option 2: Use preset for easy configuration
    # ========================================================================
    print("Option 2: Using refinement presets")
    print("-" * 80)

    # Moderate preset (8-12% refinement, very stable)
    config_moderate = GPTConfig.with_refinement_preset(
        'moderate',
        n_layer=12,
        n_embd=768,
        n_loops=2,
    )
    print(f"  Moderate: weight_base={config_moderate.supervision_weight_base}, "
          f"reward_scale={config_moderate.improvement_reward_scale}")
    print(f"           Expected: 8-12% refinement (very stable)")

    # Aggressive preset (15-25% refinement, recommended)
    config_aggressive = GPTConfig.with_refinement_preset(
        'aggressive',
        n_layer=12,
        n_embd=768,
        n_loops=2,
    )
    print(f"  Aggressive: weight_base={config_aggressive.supervision_weight_base}, "
          f"reward_scale={config_aggressive.improvement_reward_scale}")
    print(f"             Expected: 15-25% refinement (RECOMMENDED)")

    # Extreme preset (50-97% refinement, experimental)
    config_extreme = GPTConfig.with_refinement_preset(
        'extreme',
        n_layer=12,
        n_embd=768,
        n_loops=2,
    )
    print(f"  Extreme: weight_base={config_extreme.supervision_weight_base}, "
          f"reward_scale={config_extreme.improvement_reward_scale}")
    print(f"          Expected: 50-97% refinement (requires careful tuning)")
    print()

    # ========================================================================
    # Option 3: Custom configuration
    # ========================================================================
    print("Option 3: Custom configuration")
    print("-" * 80)

    config_custom = GPTConfig(
        n_layer=12,
        n_embd=768,
        n_loops=2,
        n_sup_train=4,
        supervision_weight_base=7.0,      # Custom: between aggressive and extreme
        improvement_reward_scale=1.5,     # Custom: stronger than default
    )
    print(f"  Custom: weight_base={config_custom.supervision_weight_base}, "
          f"reward_scale={config_custom.improvement_reward_scale}")
    print(f"         Expected: 30-50% refinement")
    print()

    # ========================================================================
    # Training Loop Example
    # ========================================================================
    print("="*80)
    print("TRAINING LOOP EXAMPLE (CRITICAL SETTINGS)")
    print("="*80)
    print()

    print("```python")
    print("# Create model with your chosen preset")
    print("config = GPTConfig.with_refinement_preset('aggressive', n_layer=12)")
    print("model = GPT(config)")
    print("model.init_weights()")
    print()
    print("# IMPORTANT: Use appropriate learning rate for the preset")
    print("# moderate:   lr=0.002")
    print("# aggressive: lr=0.001 (RECOMMENDED)")
    print("# extreme:    lr=0.0005 (slower convergence allows more refinement)")
    print("optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)")
    print()
    print("# Training loop")
    print("warmup_steps = 100  # Use more for extreme (150-200)")
    print("for step in range(num_steps):")
    print("    # Warmup schedule (IMPORTANT!)")
    print("    lr_mult = min((step + 1) / warmup_steps, 1.0)")
    print("    for group in optimizer.param_groups:")
    print("        group['lr'] = 0.001 * lr_mult")
    print()
    print("    optimizer.zero_grad()")
    print("    loss, aux = model(idx, targets)")
    print("    loss.backward()")
    print()
    print("    # CRITICAL: Gradient clipping prevents explosion")
    print("    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)")
    print()
    print("    optimizer.step()")
    print()
    print("    # Monitor refinement")
    print("    if step % 100 == 0:")
    print("        print(f'Supervision losses: {aux}')")
    print("        # aux contains: step_0, step_2, step_4, final")
    print("        # Expect: step_0 > step_2 > final (later steps better)")
    print("```")
    print()

    # ========================================================================
    # Quick comparison
    # ========================================================================
    print("="*80)
    print("PRESET COMPARISON TABLE")
    print("="*80)
    print()
    print(f"{'Preset':<12} {'Weight Base':<12} {'Reward Scale':<13} {'Refinement':<15} {'LR':<10} {'Stability'}")
    print("-" * 95)
    print(f"{'moderate':<12} {3.0:<12} {0.5:<13} {'8-12%':<15} {0.002:<10} {'⭐⭐⭐⭐⭐'}")
    print(f"{'aggressive':<12} {5.0:<12} {1.0:<13} {'15-25%':<15} {0.001:<10} {'⭐⭐⭐⭐'}")
    print(f"{'extreme':<12} {10.0:<12} {2.0:<13} {'50-97%':<15} {0.0005:<10} {'⭐⭐⭐'}")
    print()

    print("="*80)
    print("RECOMMENDATION")
    print("="*80)
    print()
    print("🎯 For most use cases:")
    print("   config = GPTConfig.with_refinement_preset('aggressive', n_layer=12)")
    print("   lr = 0.001, grad_clip = 1.0, warmup = 100")
    print()
    print("🔬 For maximum refinement research:")
    print("   config = GPTConfig.with_refinement_preset('extreme', n_layer=12)")
    print("   lr = 0.0005, grad_clip = 1.0, warmup = 200")
    print()
    print("See TRM_REFINEMENT_GUIDE.md for detailed documentation.")

if __name__ == "__main__":
    main()
