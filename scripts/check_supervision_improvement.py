import torch
from nanochat.gpt import GPT, GPTConfig

def check_supervision_improvement():
    torch.manual_seed(42)

    config = GPTConfig(
        sequence_len=32,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        n_loops=2,
        n_sup_train=4,
        activation_fn='relu_squared'
    )

    model = GPT(config)
    model.init_weights()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Dummy data
    idx = torch.randint(0, 100, (2, 32))
    targets = torch.randint(0, 100, (2, 32))

    print("Checking if TRM supervision loops are improving:\n")
    print("Expected behavior: Loss should DECREASE as we go through more supervision steps")
    print("i.e., step_0 > step_1 > step_2 > step_3 (later steps = more computation = better)\n")

    for i in range(20):
        optimizer.zero_grad()

        # Manually compute all supervision losses
        model.eval()
        with torch.no_grad():
            # Temporarily enable training mode to get all supervision losses
            model.train()

            # Get embeddings
            x = model.transformer.wte(idx)
            x = torch.nn.functional.rms_norm(x, (x.size(-1),))

            # Initialize states
            y, z = x.clone(), torch.zeros_like(x)

            # Compute loss at each supervision step
            losses = []
            for sup_step in range(config.n_sup_train):
                # Run through all blocks
                for block in model.transformer.h:
                    cos_sin = model.cos[:, :idx.size(1)], model.sin[:, :idx.size(1)]
                    y, z = block(x, y, z, cos_sin, kv_cache=None)

                # Compute logits and loss
                logits = model.lm_head(torch.nn.functional.rms_norm(y, (y.size(-1),)))
                logits = 15 * torch.tanh(logits / 15)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1
                )
                losses.append(loss.item())

                # DON'T detach for this analysis - we want to see the natural progression

        # Now do actual training step
        model.train()
        optimizer.zero_grad()
        loss, aux = model(idx, targets)
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            # Check if losses are improving across supervision steps
            is_improving = all(losses[j] >= losses[j+1] for j in range(len(losses)-1))
            trend = "✅ IMPROVING" if is_improving else "❌ NOT improving"

            print(f"Iter {i:2d}: {trend}")
            print(f"  Losses: [{', '.join([f'{l:.4f}' for l in losses])}]")

            # Show the deltas
            deltas = [losses[j+1] - losses[j] for j in range(len(losses)-1)]
            print(f"  Deltas: [{', '.join([f'{d:+.4f}' for d in deltas])}]")
            print(f"  (Negative delta = improvement)\n")

    print("\n" + "="*60)
    print("FINAL CHECK:")
    print("="*60)
    is_improving = all(losses[j] >= losses[j+1] for j in range(len(losses)-1))
    if is_improving:
        print("✅ YES! TRM loops ARE improving (losses decrease with more supervision)")
    else:
        print("❌ NO! TRM loops are NOT consistently improving")
        print("   This suggests the supervision mechanism may need adjustment.")
    print(f"\nFinal supervision losses: [{', '.join([f'{l:.4f}' for l in losses])}]")

if __name__ == "__main__":
    check_supervision_improvement()
