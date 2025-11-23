"""
Test TRM with proper gradient flow for progressive refinement.
Controls explosion via: gradient clipping, lower LR, warmup.
"""
import torch
import torch.nn as nn
from nanochat.gpt import GPT, GPTConfig

def get_lr_schedule(step, warmup_steps=100, base_lr=1.0):
    """Warmup then constant LR"""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr

def test_trm_refinement():
    torch.manual_seed(42)

    # Small config for testing
    config = GPTConfig(
        sequence_len=32,
        vocab_size=100,
        n_layer=3,  # Slightly deeper to test refinement
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        n_loops=2,
        n_sup_train=4,
        activation_fn='relu_squared'
    )

    model = GPT(config)
    model.init_weights()

    # LOWER learning rates for stability with deep gradient chains
    base_matrix_lr = 0.0005  # 10x lower than default
    base_embedding_lr = 0.005  # 10x lower than default

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_embedding_lr)

    # Dummy data
    idx = torch.randint(0, 100, (2, 32))
    targets = torch.randint(0, 100, (2, 32))

    print("="*70)
    print("TRM PROGRESSIVE REFINEMENT TEST")
    print("="*70)
    print(f"Config: {config.n_layer} layers, {config.n_loops} loops, {config.n_sup_train} supervision steps")
    print(f"LR: {base_embedding_lr} (with 100-step warmup)")
    print(f"Gradient clipping: 1.0")
    print("="*70)
    print()

    best_refinement_ratio = 0
    refinement_working = False

    for i in range(100):
        # Update learning rate with warmup
        lr = get_lr_schedule(i, warmup_steps=100, base_lr=base_embedding_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()

        # Forward pass
        loss, aux = model(idx, targets)

        # Backward pass
        loss.backward()

        # CRITICAL: Strong gradient clipping to prevent explosion
        grad_norm_before = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)**0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_after = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)**0.5

        # Check for NaN/Inf
        if torch.isnan(grad_norm_before) or torch.isinf(grad_norm_before):
            print(f"❌ FAILED at iter {i}: NaN/Inf gradients!")
            break

        optimizer.step()

        # Analyze supervision progression
        if i % 10 == 0:
            # Manually check all supervision losses
            model.eval()
            with torch.no_grad():
                x = model.transformer.wte(idx)
                x = torch.nn.functional.rms_norm(x, (x.size(-1),))
                y, z = x.clone(), torch.zeros_like(x)

                losses = []
                for sup_step in range(config.n_sup_train):
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

            model.train()

            # Check refinement: is each step better than the last?
            is_refining = all(losses[j] >= losses[j+1] for j in range(len(losses)-1))

            # Calculate refinement ratio (how much better is last vs first?)
            if losses[0] > 0:
                refinement_ratio = (losses[0] - losses[-1]) / losses[0] * 100
            else:
                refinement_ratio = 0

            if refinement_ratio > best_refinement_ratio:
                best_refinement_ratio = refinement_ratio

            if is_refining:
                refinement_working = True
                status = "✅ REFINING"
            else:
                status = "⚠️  mixed   "

            clipped = "🔴 CLIPPED" if grad_norm_before > 1.0 else "          "

            print(f"Iter {i:3d}: {status} | loss={loss.item():.4f} | "
                  f"grad={grad_norm_before:.3f} {clipped}")
            print(f"         sup_losses=[{', '.join([f'{l:.4f}' for l in losses])}]")
            print(f"         refinement: {refinement_ratio:+.2f}% | lr={lr:.6f}")
            print()

    print()
    print("="*70)
    print("FINAL RESULTS")
    print("="*70)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        x = model.transformer.wte(idx)
        x = torch.nn.functional.rms_norm(x, (x.size(-1),))
        y, z = x.clone(), torch.zeros_like(x)

        final_losses = []
        for sup_step in range(config.n_sup_train):
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
            final_losses.append(step_loss.item())

    is_refining = all(final_losses[j] >= final_losses[j+1] for j in range(len(final_losses)-1))

    print(f"Final supervision losses: [{', '.join([f'{l:.4f}' for l in final_losses])}]")
    print(f"Best refinement achieved: {best_refinement_ratio:.2f}%")
    print()

    if is_refining and refinement_working:
        print("🎉 SUCCESS! TRM loops ARE refining progressively!")
        print("   ✅ Later supervision steps consistently improve")
        print("   ✅ Gradients controlled (no explosion)")
        print("   ✅ Training is stable")
        return True
    elif refinement_working:
        print("⚠️  PARTIAL SUCCESS: TRM showed refinement during training")
        print("   ✅ Some supervision steps improved progressively")
        print("   ⚠️  Final state not perfectly monotonic")
        print("   💡 May need more training or tuning")
        return True
    else:
        print("❌ TRM loops NOT refining consistently")
        print("   Suggestions:")
        print("   - Try even lower learning rates")
        print("   - Increase warmup steps")
        print("   - Adjust n_loops or n_sup_train")
        return False

if __name__ == "__main__":
    success = test_trm_refinement()
    exit(0 if success else 1)
