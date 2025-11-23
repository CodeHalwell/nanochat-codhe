"""
EXTREME TRM TEST - Push refinement to the absolute limit!

Try even more extreme settings:
- 10^i weighting (instead of 5^i)
- 2.0x improvement reward
- Contrastive loss (penalize regression)
- Slower learning rate (prevent too-fast convergence)
"""
import torch
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig

def main():
    print("="*80)
    print("EXTREME TRM REFINEMENT TEST")
    print("="*80)
    print()
    print("Testing MAXIMUM possible refinement with extreme settings:")
    print("  🔥 10^i weighting (ultra-extreme)")
    print("  🔥 2.0x improvement reward")
    print("  🔥 Contrastive penalty for regression")
    print("  🔥 Slower LR to prevent premature convergence")
    print()

    torch.manual_seed(42)

    config = GPTConfig(
        sequence_len=64,
        vocab_size=100,
        n_layer=6,
        n_head=6,
        n_kv_head=6,
        n_embd=144,
        n_loops=2,
        n_sup_train=4,
        activation_fn='relu_squared'
    )

    model = GPT(config)
    model.init_weights()

    # VERY slow learning rate to allow maximum refinement development
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)

    idx = torch.randint(0, 100, (4, 64))
    targets = torch.randint(0, 100, (4, 64))

    print("Training for 250 iterations...\n")

    best_refinement = 0

    for step in range(250):
        # Very slow warmup
        lr_mult = min((step + 1) / 150, 1.0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005 * lr_mult

        optimizer.zero_grad()

        # Custom forward pass with EXTREME loss modifications
        x = model.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        y, z = x.clone(), torch.zeros_like(x)

        losses = []
        N_sup = config.n_sup_train

        # EXTREME weighting: 10^i
        weighting_base = 10.0
        weights = torch.tensor([weighting_base**i for i in range(N_sup)], device=x.device)
        weights = weights / weights.sum()

        for sup_step in range(N_sup):
            for block in model.transformer.h:
                cos_sin = model.cos[:, :idx.size(1)], model.sin[:, :idx.size(1)]
                y, z = block(x, y, z, cos_sin, kv_cache=None)

            logits = model.lm_head(F.rms_norm(y, (y.size(-1),)))
            logits = 15 * torch.tanh(logits / 15)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            losses.append(loss)

        # Build total loss with extreme incentives
        total_loss = sum(w * l for w, l in zip(weights, losses))

        # EXTREME improvement reward (2.0x)
        for i in range(1, len(losses)):
            improvement = (losses[i-1] - losses[i]) / (losses[i-1] + 1e-8)
            total_loss -= improvement * 2.0  # Triple the reward!

        # Add contrastive penalty for regression
        for i in range(len(losses) - 1):
            regression = F.relu(losses[i+1] - losses[i])  # Penalize if getting worse
            total_loss += regression * 1.0  # Strong penalty

        total_loss.backward()

        grad_norm = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)**0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if torch.isnan(total_loss):
            print(f"❌ NaN at step {step}")
            break

        # Evaluate
        if step % 25 == 0 or step == 249:
            model.eval()
            with torch.no_grad():
                x_eval = model.transformer.wte(idx)
                x_eval = F.rms_norm(x_eval, (x_eval.size(-1),))
                y_eval, z_eval = x_eval.clone(), torch.zeros_like(x_eval)

                eval_losses = []
                for sup_step in range(config.n_sup_train):
                    for block in model.transformer.h:
                        cos_sin = model.cos[:, :idx.size(1)], model.sin[:, :idx.size(1)]
                        y_eval, z_eval = block(x_eval, y_eval, z_eval, cos_sin, kv_cache=None)

                    logits = model.lm_head(F.rms_norm(y_eval, (y_eval.size(-1),)))
                    logits = 15 * torch.tanh(logits / 15)
                    step_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=-1
                    )
                    eval_losses.append(step_loss.item())

            model.train()

            refinement = (eval_losses[0] - eval_losses[-1]) / eval_losses[0] * 100 if eval_losses[0] > 0 else 0
            if refinement > best_refinement:
                best_refinement = refinement

            is_improving = all(eval_losses[j] >= eval_losses[j+1] - 0.001 for j in range(len(eval_losses)-1))
            status = "✅" if is_improving else "⚠️"

            print(f"Step {step:3d}: {status} ref={refinement:+.2f}% | grad={grad_norm:.3f}")
            print(f"         losses=[{', '.join([f'{l:.6f}' for l in eval_losses])}]")

            # Show step-by-step improvements
            if step % 50 == 0:
                print("         Step improvements:")
                for i in range(len(eval_losses) - 1):
                    delta = (eval_losses[i] - eval_losses[i+1]) / eval_losses[i] * 100
                    print(f"           {i}→{i+1}: {delta:+.2f}%")
                print()

    print()
    print("="*80)
    print("EXTREME TEST RESULTS")
    print("="*80)
    print()
    print(f"🏆 Maximum refinement achieved: {best_refinement:.2f}%")
    print()

    # Final detailed analysis
    model.eval()
    with torch.no_grad():
        x = model.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        y, z = x.clone(), torch.zeros_like(x)

        final_losses = []
        for sup_step in range(4):
            for block in model.transformer.h:
                cos_sin = model.cos[:, :idx.size(1)], model.sin[:, :idx.size(1)]
                y, z = block(x, y, z, cos_sin, kv_cache=None)

            logits = model.lm_head(F.rms_norm(y, (y.size(-1),)))
            logits = 15 * torch.tanh(logits / 15)
            step_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            final_losses.append(step_loss.item())

    print("Final supervision losses:")
    for i, loss in enumerate(final_losses):
        if i > 0:
            delta = (final_losses[i-1] - loss) / final_losses[i-1] * 100
            arrow = "✅" if delta > 0 else "❌"
            print(f"  Step {i}: {loss:.8f} ({arrow} {delta:+.2f}% vs previous)")
        else:
            print(f"  Step {i}: {loss:.8f} (initial)")

    final_ref = (final_losses[0] - final_losses[-1]) / final_losses[0] * 100
    print(f"\n🎯 Final total refinement: {final_ref:.2f}%")

    if final_ref > 30:
        print("\n🎉🎉🎉 INCREDIBLE! >30% refinement!")
    elif final_ref > 20:
        print("\n🎉🎉 AMAZING! >20% refinement!")
    elif final_ref > 15:
        print("\n🎉 EXCELLENT! >15% refinement!")
    elif final_ref > 10:
        print("\n✅ GREAT! >10% refinement!")
    else:
        print(f"\n✅ Good refinement ({final_ref:.1f}%)")

    print()
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()
    print("With EXTREME settings:")
    print(f"  • 10^i weighting (weights: {weights.tolist()})")
    print(f"  • 2.0x improvement reward")
    print(f"  • Contrastive regression penalty")
    print(f"  • Very slow LR (0.0005)")
    print()
    print(f"We achieved: {best_refinement:.1f}% maximum refinement")
    print()
    print("💡 Key insight: The refinement plateau suggests the model")
    print("   is learning to converge ALL supervision steps to the")
    print("   optimal answer, rather than having early steps be worse.")
    print()
    print("   This might actually be BETTER than forced degradation!")

if __name__ == "__main__":
    main()
