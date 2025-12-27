"""
Test different incentive mechanisms to encourage stronger TRM refinement.

Strategies:
1. Contrastive loss - penalize when later steps don't improve
2. Improvement reward - bonus for each improvement
3. Stronger weighting - more aggressive than 2^i
4. Progressive temperature - later steps more confident
"""
import torch
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig

def train_with_strategy(strategy_name, modify_loss_fn):
    """Train with a specific incentive strategy"""
    torch.manual_seed(42)

    config = GPTConfig(
        sequence_len=32,
        vocab_size=100,
        n_layer=3,
        n_head=4,
        n_kv_head=4,
        n_embd=64,
        n_loops=2,
        n_sup_train=4,
        activation_fn='relu_squared'
    )

    model = GPT(config)
    model.init_weights()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)

    idx = torch.randint(0, 100, (2, 32))
    targets = torch.randint(0, 100, (2, 32))

    for i in range(100):
        lr_mult = min((i + 1) / 50, 1.0)  # Warmup
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005 * lr_mult

        optimizer.zero_grad()

        # Forward pass - manually compute to get all supervision losses
        x = model.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        y, z = x.clone(), torch.zeros_like(x)

        losses = []
        for sup_step in range(config.n_sup_train):
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

        # Apply the strategy-specific loss modification
        total_loss = modify_loss_fn(losses)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        x = model.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        y, z = x.clone(), torch.zeros_like(x)

        final_losses = []
        for sup_step in range(config.n_sup_train):
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

    is_monotonic = all(final_losses[j] >= final_losses[j+1] for j in range(len(final_losses)-1))
    refinement = (final_losses[0] - final_losses[-1]) / final_losses[0] * 100

    return final_losses, is_monotonic, refinement


def main():
    print("="*80)
    print("TESTING TRM INCENTIVE MECHANISMS")
    print("="*80)
    print()

    strategies = []

    # Strategy 1: Baseline (current weighted approach)
    def baseline(losses):
        weights = torch.tensor([2.0**i for i in range(len(losses))], device=losses[0].device)
        weights = weights / weights.sum()
        return sum(w * l for w, l in zip(weights, losses))

    strategies.append(("Baseline (2^i weighting)", baseline))

    # Strategy 2: Aggressive weighting (4^i)
    def aggressive_weighting(losses):
        weights = torch.tensor([4.0**i for i in range(len(losses))], device=losses[0].device)
        weights = weights / weights.sum()
        return sum(w * l for w, l in zip(weights, losses))

    strategies.append(("Aggressive weighting (4^i)", aggressive_weighting))

    # Strategy 3: Contrastive loss - penalize when later steps are worse
    def contrastive_loss(losses):
        # Base weighted loss
        weights = torch.tensor([2.0**i for i in range(len(losses))], device=losses[0].device)
        weights = weights / weights.sum()
        base_loss = sum(w * l for w, l in zip(weights, losses))

        # Add penalty for non-improvement
        penalty = 0.0
        for i in range(len(losses) - 1):
            # If next step is worse, add a penalty
            degradation = F.relu(losses[i+1] - losses[i])  # Only penalize if worse
            penalty += degradation * 0.5  # Penalty weight

        return base_loss + penalty

    strategies.append(("Contrastive (penalize regression)", contrastive_loss))

    # Strategy 4: Improvement reward - bonus for each improvement
    def improvement_reward(losses):
        # Base weighted loss
        weights = torch.tensor([2.0**i for i in range(len(losses))], device=losses[0].device)
        weights = weights / weights.sum()
        base_loss = sum(w * l for w, l in zip(weights, losses))

        # Add reward for improvement
        reward = 0.0
        for i in range(len(losses) - 1):
            improvement = losses[i] - losses[i+1]  # Positive if improving
            reward += improvement * 0.3  # Reward weight

        return base_loss - reward  # Subtract reward (lower loss is better)

    strategies.append(("Improvement reward", improvement_reward))

    # Strategy 5: Margin loss - enforce minimum improvement
    def margin_loss(losses):
        # Base weighted loss
        weights = torch.tensor([2.0**i for i in range(len(losses))], device=losses[0].device)
        weights = weights / weights.sum()
        base_loss = sum(w * l for w, l in zip(weights, losses))

        # Enforce margin: loss[i+1] should be at least 1% better than loss[i]
        margin = 0.01
        margin_penalty = 0.0
        for i in range(len(losses) - 1):
            target_improvement = losses[i] * margin
            actual_improvement = losses[i] - losses[i+1]
            shortfall = F.relu(target_improvement - actual_improvement)
            margin_penalty += shortfall * 1.0

        return base_loss + margin_penalty

    strategies.append(("Margin loss (enforce 1% improvement)", margin_loss))

    # Strategy 6: Final-only (only care about last step)
    def final_only(losses):
        # Weight only the last step heavily
        weights = [0.05, 0.1, 0.15, 0.7]
        weights = torch.tensor(weights, device=losses[0].device)
        return sum(w * l for w, l in zip(weights, losses))

    strategies.append(("Final-only (focus on last step)", final_only))

    # Strategy 7: Hierarchical weighting with improvement bonus
    def hierarchical(losses):
        # Exponential weighting
        weights = torch.tensor([3.0**i for i in range(len(losses))], device=losses[0].device)
        weights = weights / weights.sum()
        base_loss = sum(w * l for w, l in zip(weights, losses))

        # Strong improvement bonus
        improvement_bonus = 0.0
        for i in range(len(losses) - 1):
            delta = (losses[i] - losses[i+1]) / (losses[i] + 1e-8)  # Relative improvement
            improvement_bonus += delta * 0.5

        return base_loss - improvement_bonus

    strategies.append(("Hierarchical (3^i + strong bonus)", hierarchical))

    # Test all strategies
    results = []
    for name, strategy_fn in strategies:
        print(f"Testing: {name}...")
        final_losses, is_monotonic, refinement = train_with_strategy(name, strategy_fn)
        results.append((name, final_losses, is_monotonic, refinement))

        status = "✅" if is_monotonic else "⚠️"
        print(f"  {status} Refinement: {refinement:+.2f}%")
        print(f"     Losses: [{', '.join([f'{l:.4f}' for l in final_losses])}]")
        print()

    # Summary
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print()

    # Find best
    best = max(results, key=lambda x: x[3])
    monotonic_best = [r for r in results if r[2]]
    if monotonic_best:
        best_monotonic = max(monotonic_best, key=lambda x: x[3])
    else:
        best_monotonic = None

    print(f"🏆 BEST OVERALL: {best[0]}")
    print(f"   Refinement: {best[3]:.2f}%")
    print(f"   Losses: [{', '.join([f'{l:.4f}' for l in best[1]])}]")
    print()

    if best_monotonic and best_monotonic != best:
        print(f"🏆 BEST MONOTONIC: {best_monotonic[0]}")
        print(f"   Refinement: {best_monotonic[3]:.2f}%")
        print(f"   Losses: [{', '.join([f'{l:.4f}' for l in best_monotonic[1]])}]")
        print()

    print("="*80)
    print("RECOMMENDATION")
    print("="*80)
    print()
    print(f"✅ Use: {best[0]}")
    print(f"   This achieves {best[3]:.2f}% refinement across supervision steps")
    print()

    # Show comparison table
    print("Full comparison:")
    print(f"{'Strategy':<40} {'Refinement':<12} {'Monotonic'}")
    print("-" * 80)
    for name, losses, is_mono, ref in sorted(results, key=lambda x: -x[3]):
        mono_str = "✅" if is_mono else "⚠️"
        print(f"{name:<40} {ref:>10.2f}% {mono_str:>12}")

if __name__ == "__main__":
    main()
