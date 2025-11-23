import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import Engine

def test_recursive_model():
    print("Testing Recursive Model...")
    config = GPTConfig(n_layer=2, n_head=4, n_kv_head=4, n_embd=64, n_loops=3)
    model = GPT(config)
    
    # Check if RecursiveBlock is used
    print(f"Model structure: {model.transformer.h[0]}")
    assert model.transformer.h[0].n_loops == 3
    
    # Run forward pass
    idx = torch.randint(0, config.vocab_size, (1, 10))
    logits = model(idx)
    print(f"Forward pass successful. Logits shape: {logits.shape}")

def test_bayesian_refinement():
    print("\nTesting Bayesian Refinement...")
    config = GPTConfig(n_layer=2, n_head=4, n_kv_head=4, n_embd=64, n_loops=3)
    model = GPT(config)
    tokenizer = type('obj', (object,), {'encode': lambda s: [1, 2, 3], 'decode': lambda t: "test", 'encode_special': lambda s: 0, 'get_bos_token_id': lambda: 1})
    engine = Engine(model, tokenizer)
    
    tokens = [1, 2, 3]
    
    # Mock sample_next_token to avoid randomness issues in test
    # We just want to see if it runs without error
    
    print("Running generate with bayesian_refine=True...")
    try:
        generator = engine.generate(tokens, max_tokens=5, bayesian_refine=True, refine_top_k=5)
        for tokens, masks in generator:
            pass
        print("Bayesian Refinement generation successful.")
    except Exception as e:
        print(f"Bayesian Refinement failed: {e}")
        raise e

if __name__ == "__main__":
    test_recursive_model()
    test_bayesian_refinement()
