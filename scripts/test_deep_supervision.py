import torch
from nanochat.gpt import GPT, GPTConfig

def test_deep_supervision():
    print("Testing Deep Supervision...")
    # Setup model with recursion
    config = GPTConfig(n_layer=1, n_head=4, n_kv_head=4, n_embd=64, n_loops=3)
    model = GPT(config)
    
    # Create dummy inputs and targets
    idx = torch.randint(0, config.vocab_size, (2, 10))
    targets = torch.randint(0, config.vocab_size, (2, 10))
    
    # Forward pass with targets (should trigger deep supervision)
    loss = model(idx, targets=targets)
    
    print(f"Loss computed: {loss.item()}")
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
    
    # Verify that it works without targets (inference)
    logits = model(idx)
    print(f"Inference logits shape: {logits.shape}")
    assert logits.shape == (2, 10, config.vocab_size)

if __name__ == "__main__":
    test_deep_supervision()
