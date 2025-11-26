
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig

def debug_trm():
    torch.manual_seed(42)
    
    # Tiny config
    config = GPTConfig(
        sequence_len=32,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_kv_head=2,  # Must match n_head for this test
        n_embd=32,
        n_loops=2,
        n_sup_train=4,
        activation_fn='relu_squared'
    )
    
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Dummy data
    idx = torch.randint(0, 100, (2, 32))
    targets = torch.randint(0, 100, (2, 32))
    
    print("Initial training step:")
    model.train()
    
    for i in range(5):
        optimizer.zero_grad()
        loss, aux = model(idx, targets)
        
        print(f"Iter {i}:")
        print(f"  Total Loss: {loss.item():.4f}")
        print(f"  Step 0: {aux['step_0'].item():.4f}")
        print(f"  Step 1: {aux.get('step_1', aux['final']).item():.4f}") # step_1 might not be in aux if logic is weird
        print(f"  Step 2: {aux['step_2'].item():.4f}")
        print(f"  Step 3: {aux['step_3'].item() if 'step_3' in aux else aux['final'].item():.4f}")
        
        loss.backward()
        
        # Check gradients
        print(f"  Grad norm wte: {model.transformer.wte.weight.grad.norm().item():.4f}")
        print(f"  Grad norm head: {model.lm_head.weight.grad.norm().item():.4f}")
        
        optimizer.step()

if __name__ == "__main__":
    debug_trm()
