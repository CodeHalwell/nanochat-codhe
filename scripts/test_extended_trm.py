import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig

def test_extended_trm():
    torch.manual_seed(42)

    # Tiny config
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

    print("Extended training test (50 iterations):\n")

    for i in range(50):
        optimizer.zero_grad()
        loss, aux = model(idx, targets)

        if i % 10 == 0:
            print(f"Iter {i:3d}: loss={loss.item():.4f} | "
                  f"sup_losses=[{aux['step_0'].item():.4f}, _, "
                  f"{aux['step_2'].item():.4f}, {aux['final'].item():.4f}]")

        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"ERROR: NaN/Inf detected at iteration {i}!")
            break

        loss.backward()

        # Check gradients
        grad_norm = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)**0.5
        if i % 10 == 0:
            print(f"         grad_norm={grad_norm.item():.4f}\n")

        if torch.isnan(grad_norm):
            print(f"ERROR: NaN gradient at iteration {i}!")
            break

        optimizer.step()

    print("✅ Extended test completed successfully!")
    print(f"Final loss: {loss.item():.4f}")
    print(f"Supervision order: {aux['step_0'].item():.4f} > {aux['step_2'].item():.4f} > {aux['final'].item():.4f}")

    # Verify supervision order is correct
    assert aux['step_0'].item() > aux['step_2'].item() > aux['final'].item(), \
        "Supervision order should be decreasing!"
    print("✅ Supervision order is CORRECT!")

if __name__ == "__main__":
    test_extended_trm()
