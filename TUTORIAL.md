# nanochat: In-Depth Tutorial

## Table of Contents
1. [Introduction](#introduction)
2. [Repository Architecture Overview](#repository-architecture-overview)
3. [The Complete Training Pipeline](#the-complete-training-pipeline)
4. [Core Components Deep Dive](#core-components-deep-dive)
5. [Model Architecture (GPT)](#model-architecture-gpt)
6. [Tokenization System](#tokenization-system)
7. [Data Loading and Processing](#data-loading-and-processing)
8. [Evaluation Framework](#evaluation-framework)
9. [Inference and Serving](#inference-and-serving)
10. [Customization Patterns](#customization-patterns)
11. [How It All Fits Together](#how-it-all-fits-together)

---

## Introduction

**nanochat** is a full-stack implementation of a Large Language Model (LLM) system similar to ChatGPT, designed to be minimal, hackable, and complete. The entire project—from tokenizer training to web serving—is contained in approximately 8,000 lines of clean, readable code.

### What Makes nanochat Unique?

1. **Complete Pipeline**: Unlike many ML projects that focus on just one aspect (e.g., model architecture or training), nanochat implements the entire stack from raw text to a chat interface
2. **Budget-Conscious**: Designed to train capable models for $100-$1000, making it accessible for learning and experimentation
3. **Single-Node Training**: Optimized to run on a single 8xH100 GPU node
4. **Minimal Dependencies**: Clean dependency tree, mostly PyTorch and standard Python libraries
5. **Educational**: Written to be understood, not just to work

### Key Terminology

Before diving in, let's clarify some terms you'll encounter:

- **Pretraining/Base Training**: Teaching the model language modeling on raw internet text
- **Midtraining**: Intermediate training phase that teaches conversation format and tool use
- **SFT (Supervised Fine-Tuning)**: Final training to improve response quality
- **RL (Reinforcement Learning)**: Optional training using rewards to improve specific capabilities
- **Tokens**: Text broken into chunks (usually subwords) that the model processes
- **Context Window**: How much text the model can "see" at once (typically 2048 tokens)
- **Parameters**: The weights in the neural network (e.g., 561M parameters for d20 model)
- **FLOPs**: Floating Point Operations, a measure of computational cost

---

## Repository Architecture Overview

### High-Level Structure

```
nanochat/
├── nanochat/          # Core library: model, tokenizer, training utilities
├── scripts/           # Executable training and inference scripts
├── tasks/             # Evaluation datasets and metrics
├── rustbpe/           # Fast Rust tokenizer implementation
├── dev/               # Development utilities and examples
└── tests/             # Test suite
```

### The Three Pillars

nanochat is built around three main pillars:

1. **Core Library (`nanochat/`)**: Reusable components like the GPT model, data loaders, optimizers
2. **Training Scripts (`scripts/`)**: Executable programs for each training stage
3. **Evaluation Tasks (`tasks/`)**: Standardized benchmarks to measure model quality

### Design Philosophy

**Why scripts instead of a framework?**
- Each script is self-contained and readable top-to-bottom
- Easy to understand what happens when you run `python -m scripts.base_train`
- No hidden configuration magic or complex object hierarchies
- Fork-friendly: modify one script without breaking others

**Why minimal dependencies?**
- Easier to understand what's happening under the hood
- Less likely to break due to dependency updates
- Faster to install and get started
- More educational value

---

## The Complete Training Pipeline

The training pipeline consists of four main stages, each building on the previous one:

### Stage 1: Tokenization (scripts/tok_train.py)

**What**: Train a BPE (Byte Pair Encoding) tokenizer to convert text to integers

**Why**: Neural networks work with numbers, not text. The tokenizer learns an efficient mapping from text to integers.

**Input**: ~2 billion characters of raw text
**Output**: A vocabulary of 65,536 tokens (2^16)

**Key Decisions**:
- Vocab size of 65,536 balances efficiency vs. memory
- Custom Rust implementation for ~100x speedup over pure Python
- GPT-4-style splitting pattern (respects word boundaries)

**What Happens**:
1. Start with 256 base tokens (all possible bytes)
2. Iteratively merge the most frequent byte pairs
3. Continue until reaching vocab size of 65,536
4. Special tokens added for conversation formatting

### Stage 2: Base Training / Pretraining (scripts/base_train.py)

**What**: Teach the model to predict the next token in raw internet text

**Why**: This is where the model learns language, facts about the world, and reasoning patterns

**Input**: 
- ~54 billion characters of web text (FineWeb dataset)
- For d20 model: 240 data shards × 250M chars/shard = 60B chars

**Output**: A "base model" that can continue any text prompt

**Key Decisions**:
- Chinchilla ratio: 20 tokens per parameter (561M params × 20 = 11.2B tokens)
- d20 architecture: 20 layers, 561M parameters
- Hybrid optimizer: Muon for weights, AdamW for embeddings
- 2048 token context window

**What Happens**:
1. Load tokenizer
2. Initialize GPT model with random weights
3. Stream training data, tokenize on-the-fly
4. For each batch:
   - Forward pass: predict next token probabilities
   - Compute cross-entropy loss
   - Backward pass: calculate gradients
   - Update weights with optimizer
5. Periodically evaluate on CORE benchmark
6. Save checkpoint every few thousand steps

**Cost Breakdown**:
- d20 model on 8×H100: ~1 hour, ~$24
- d26 model on 8×H100: ~12 hours, ~$300
- d32 model on 8×H100: ~33 hours, ~$800

### Stage 3: Midtraining (scripts/mid_train.py)

**What**: Teach the model conversation format, special tokens, and tool use

**Why**: Base models only know how to continue text. Midtraining teaches them to follow conversation structure and use tools.

**Input**:
- Base model checkpoint
- Mix of tasks:
  - SmolTalk: general conversation data
  - Multiple choice questions (ARC, MMLU)
  - Tool use examples (calculator, code execution)
  - Identity conversations (personality)

**Output**: A "mid model" that understands conversation format

**Key Decisions**:
- Much shorter training than base (few thousand steps)
- Introduces special tokens: `<|user_start|>`, `<|assistant_start|>`, etc.
- Mixes multiple data sources to teach various capabilities
- Still predicts next token, but now on structured conversations

**What Happens**:
1. Load base model checkpoint
2. Load conversation datasets
3. Format conversations with special tokens:
   ```
   <|user_start|>Hello!<|user_end|>
   <|assistant_start|>Hi there!<|assistant_end|>
   ```
4. Train to predict tokens in this format
5. Model learns when to respond, when to use tools, etc.

### Stage 4: Supervised Fine-Tuning / SFT (scripts/chat_sft.py)

**What**: Polish the model on high-quality conversation examples

**Why**: Midtraining teaches format; SFT teaches quality responses

**Input**:
- Mid model checkpoint
- High-quality conversations from SmolTalk

**Output**: The final "SFT model" ready for deployment

**Key Decisions**:
- Even shorter training (couple thousand steps)
- Higher learning rate since we're close to optimal
- Focus on quality over quantity of data

**What Happens**:
1. Load mid model checkpoint
2. Load curated conversation data
3. Train with same next-token prediction objective
4. Model learns better response patterns, style, and quality
5. Final checkpoint saved

### Stage 5 (Optional): Reinforcement Learning (scripts/chat_rl.py)

**What**: Use outcome-based rewards to improve specific capabilities

**Why**: Some tasks (like math) benefit from optimizing for correctness rather than just likelihood

**Input**:
- SFT model checkpoint
- GSM8K math problems with correct answers

**Output**: An "RL model" specialized for math reasoning

**Key Decisions**:
- Currently only implemented for GSM8K math problems
- Uses self-generated solutions + reward signal
- More complex than supervised training

**What Happens**:
1. Load SFT model checkpoint
2. For each math problem:
   - Generate multiple solution attempts
   - Check if answer is correct (reward = +1 or -1)
   - Use REINFORCE algorithm to increase probability of correct solutions
3. Model learns to favor reasoning paths that lead to correct answers

---

## Core Components Deep Dive

### 1. GPT Model (nanochat/gpt.py)

**Purpose**: The neural network that does all the thinking

**Architecture Choices**:

```python
@dataclass
class GPTConfig:
    sequence_len: int = 1024    # context window (default)
    vocab_size: int = 50304     # vocabulary size (default)
    n_layer: int = 12           # number of transformer blocks (default)
    n_head: int = 6             # attention heads per layer (default)
    n_kv_head: int = 6          # key/value heads (GQA) (default)
    n_embd: int = 768           # embedding dimension (default)
```

**Note**: These are base defaults in GPTConfig. Actual training typically uses:
- `sequence_len = 2048` (set via `max_seq_len` in training scripts)
- `vocab_size = 65536` (determined by trained tokenizer, 2^16)
- Other parameters derived from `depth` argument (e.g., depth=20 → n_layer=20)

**Key Features**:

1. **Rotary Embeddings (RoPE)**
   - **What**: Encode position information directly in attention mechanism
   - **Why**: More efficient than learned position embeddings, better extrapolation
   - **How**: Rotate query/key vectors based on their position

2. **RMSNorm**
   - **What**: Simplified layer normalization without learnable parameters
   - **Why**: Faster and works just as well
   - **Code**: `F.rms_norm(x, (x.size(-1),))`

3. **Untied Embeddings**
   - **What**: Separate weights for token embedding and output projection
   - **Why**: More flexible, often better performance
   - **Traditional**: Share weights between input embedding and output layer
   - **nanochat**: Separate `wte` (token embedding) and `lm_head` (output projection)

4. **Group Query Attention (GQA)**
   - **What**: Multiple query heads share key/value heads
   - **Why**: Reduces memory and computation during inference
   - **Example**: 12 query heads might share 4 key/value heads

5. **ReLU² Activation**
   - **What**: `relu(x)²` instead of GELU or SiLU
   - **Why**: Simpler, faster, works well

6. **QK Normalization**
   - **What**: Normalize query and key vectors before attention
   - **Why**: More stable training, prevents attention collapse

**Forward Pass Overview**:
```python
def forward(self, idx):
    # 1. Token embedding
    x = self.transformer.wte(idx)  # (batch, seq_len, n_embd)
    
    # 2. Process through transformer blocks
    for block in self.transformer.h:
        x = block(x)  # attention + MLP
    
    # 3. Output projection
    logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
    
    return logits
```

### 2. Optimizers

nanochat uses two different optimizers for different parameter types:

#### Muon (nanochat/muon.py)
- **For**: Weight matrices in attention and MLP layers
- **Why**: More efficient for large matrices
- **Key Idea**: Use momentum in the orthogonal complement space
- **Benefits**: Often converges faster than Adam for transformers

#### AdamW (nanochat/adamw.py)
- **For**: Embedding and unembedding layers
- **Why**: Better for sparse updates (not all tokens appear in each batch)
- **Key Features**: 
  - Adaptive learning rates per parameter
  - Weight decay decoupled from gradient updates

**Why Two Optimizers?**
- Different parameter types have different optimization landscapes
- Embeddings are sparse (only updated for tokens present)
- Weight matrices benefit from momentum-based methods
- Best of both worlds: efficiency where it matters

### 3. Data Loader (nanochat/dataloader.py)

**Purpose**: Efficiently stream training data from disk to GPU

**Key Challenges**:
1. Training data too large to fit in memory (60GB+)
2. Need to tokenize on-the-fly
3. Must support distributed training across multiple GPUs
4. Should be resumable if training crashes

**Solution**:

```python
def tokenizing_distributed_data_loader(B, T, split):
    """
    B: batch size (e.g., 64)
    T: sequence length (e.g., 2048)
    split: "train" or "val"
    """
    while True:  # infinite loop for multi-epoch training
        for parquet_file in data_files:
            # Each GPU reads different row groups
            for row_group in my_row_groups:
                texts = load_texts(row_group)
                tokens = tokenize(texts)
                
                # Yield batches of shape (B, T)
                for batch in create_batches(tokens, B, T):
                    yield batch
```

**Key Features**:

1. **Streaming**: Never loads all data into memory
2. **Distributed**: Each GPU reads different data
3. **Tokenizing**: Text converted to tokens on-the-fly
4. **Infinite**: Loops over data for multiple epochs
5. **Resumable**: Can save/load position in data stream

**Data Format**:
- Raw data stored in Parquet files (compressed columnar format)
- Each shard: ~250M characters, ~100MB on disk
- Total dataset: 1822 shards = 455GB of text

### 4. Checkpoint Manager (nanochat/checkpoint_manager.py)

**Purpose**: Save and load model weights during training

**Why Important**:
- Training takes hours/days - need to resume if crashes
- Want to save best model, not just latest
- Need to load models for evaluation or inference

**What's Saved**:
```python
checkpoint = {
    'model': model.state_dict(),           # all weights and biases
    'optimizer': optimizer.state_dict(),   # optimizer state
    'config': config,                      # model architecture
    'step': iteration_number,              # training progress
    'train_state': dataloader_state,       # where in dataset
}
```

**File Organization**:
```
~/.cache/nanochat/
├── base_step10000.pt      # pretraining checkpoint
├── mid_step2000.pt        # midtraining checkpoint
└── sft_step1000.pt        # final SFT checkpoint
```

### 5. Engine (nanochat/engine.py)

**Purpose**: Efficient inference with KV caching

**The Problem**:
- Naive inference: recompute everything for each new token
- Very slow and wasteful

**The Solution - KV Caching**:
```python
# First token: compute everything
q, k, v = compute_qkv(token_0)
cache = {"k": k, "v": v}
output_0 = attention(q, k, v)

# Second token: reuse previous k, v
q, k, v = compute_qkv(token_1)
k_all = concat(cache["k"], k)  # reuse!
v_all = concat(cache["v"], v)  # reuse!
output_1 = attention(q, k_all, v_all)
```

**Why It Works**:
- Attention looks at all previous tokens
- Those don't change, so we can cache their keys/values
- Only compute new key/value for new token
- Speeds up inference by ~10-100x

**Engine Features**:

1. **KV Cache Management**: Automatically grows cache as generation continues
2. **Batching**: Process multiple requests in parallel
3. **Tool Use**: Can execute Python code and insert results
4. **Temperature/Top-k Sampling**: Configurable randomness in generation

### 6. Tokenizer (nanochat/tokenizer.py)

**Purpose**: Convert between text and token IDs

**Two Implementations**:

1. **Training (rustbpe/)**: Fast Rust implementation for tokenizer training
2. **Inference (tiktoken)**: Efficient C++ implementation from OpenAI

**Why Two?**
- Training is rare, complex, needs flexibility → Rust implementation
- Inference is frequent, needs pure speed → tiktoken (C++)
- Best tool for each job

**Special Tokens**:
```python
SPECIAL_TOKENS = [
    "<|bos|>",              # beginning of sequence
    "<|user_start|>",       # user message starts
    "<|user_end|>",         # user message ends
    "<|assistant_start|>",  # assistant response starts
    "<|assistant_end|>",    # assistant response ends
    "<|python_start|>",     # tool use starts
    "<|python_end|>",       # tool use ends
    "<|output_start|>",     # tool output starts
    "<|output_end|>",       # tool output ends
]
```

**These enable**:
- Structured conversations
- Tool use (calculator, code execution)
- Clear boundaries between user and assistant

### 7. Report (nanochat/report.py)

**Purpose**: Generate a comprehensive report card for your model

**What It Tracks**:
- Model size and architecture
- Training time and cost
- Evaluation scores on all benchmarks
- Sample generations
- System information

**Output**: `report.md` file with complete training summary

---

## Model Architecture (GPT)

### The Transformer Block

Each layer in the model contains two main components:

```python
class Block(nn.Module):
    def __init__(self, config):
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(norm(x))  # attention with residual
        x = x + self.mlp(norm(x))   # feedforward with residual
        return x
```

### 1. Attention Mechanism

**Purpose**: Let each token "look at" previous tokens to gather context

**How It Works**:

```python
def attention(x):
    # 1. Project to queries, keys, values
    q = x @ W_q  # what am I looking for?
    k = x @ W_k  # what do I contain?
    v = x @ W_v  # what do I communicate?
    
    # 2. Compute attention scores
    scores = q @ k.T / sqrt(d_k)
    
    # 3. Mask future tokens (causal attention)
    scores = mask_future(scores)
    
    # 4. Softmax to get weights
    weights = softmax(scores)
    
    # 5. Weighted sum of values
    output = weights @ v
    
    return output
```

**Why It's Powerful**:
- Each token can gather information from any previous token
- Learned patterns: pronouns refer to nouns, subjects need verbs, etc.
- Multi-head attention: learn multiple patterns simultaneously

**Rotary Position Embeddings**:
```python
def apply_rotary_emb(x, cos, sin):
    # Rotate query/key vectors based on position
    # This encodes "token at position 5" without explicit position embeddings
    x1, x2 = x.split(x.shape[-1] // 2, dim=-1)
    return torch.cat([x1 * cos + x2 * sin,
                      x2 * cos - x1 * sin], dim=-1)
```

### 2. Feedforward Network (MLP)

**Purpose**: Process each token independently after attention

**Structure**:
```python
class MLP(nn.Module):
    def __init__(self, config):
        n_embd = config.n_embd
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)  # expand
        self.fc2 = nn.Linear(4 * n_embd, n_embd)  # compress
    
    def forward(self, x):
        x = self.fc1(x)
        x = relu_squared(x)  # activation
        x = self.fc2(x)
        return x
```

**Why 4x Expansion?**
- More parameters for learning
- Standard in transformer literature
- Good balance of capacity vs. efficiency

### 3. Residual Connections

**The Problem**: Deep networks hard to train (vanishing gradients)

**The Solution**: Skip connections
```python
x = x + attention(x)  # can learn or skip attention
x = x + mlp(x)        # can learn or skip MLP
```

**Why It Works**:
- Gradient flows directly through residual path
- Each layer only needs to learn the "delta"
- Easier to train very deep networks (20-100+ layers)

### 4. Layer Normalization

**Purpose**: Stabilize training by normalizing activations

**RMSNorm (used in nanochat)**:
```python
def norm(x):
    return x / sqrt(mean(x^2) + epsilon)
```

**Why RMSNorm vs LayerNorm**:
- Simpler (no learned parameters)
- Faster computation
- Works just as well in practice
- Used by LLaMA and other modern models

### Full Model Overview

```
Input text → Tokenizer → Token IDs

Token IDs → Token Embedding → Initial vectors
            ↓
         Norm Layer
            ↓
    ┌─────────────┐
    │ Block 1     │
    │ - Attention │
    │ - MLP       │
    └─────────────┘
            ↓
    ┌─────────────┐
    │ Block 2     │
    │ - Attention │
    │ - MLP       │
    └─────────────┘
            ↓
         ... (x20 for d20 model)
            ↓
         Norm Layer
            ↓
    Output Projection → Logits over vocab
            ↓
         Softmax → Probabilities
            ↓
        Sample → Next token
```

---

## Tokenization System

### Why Tokenization?

**The Problem**: Neural networks process fixed-size vectors, but text is variable-length strings

**Solutions Considered**:

1. **Character-level**: One character = one token
   - ❌ Very long sequences (slow)
   - ❌ Model must learn to combine characters into words

2. **Word-level**: One word = one token
   - ❌ Huge vocabulary (millions of words)
   - ❌ Can't handle new words
   - ❌ Inefficient for morphology (run, runs, running all separate)

3. **Byte-Pair Encoding (BPE)**: ✅ Best of both worlds
   - ✅ Reasonable sequence length
   - ✅ Can represent any text (fallback to bytes)
   - ✅ Common words = single token, rare words = few tokens

### BPE Algorithm

**Training Process**:

```python
# Start with base vocabulary (all bytes)
vocab = {byte: byte for byte in range(256)}  # 256 tokens

# Training corpus
text = "the the the cat sat on the mat"

while len(vocab) < target_size:  # e.g., 65536
    # 1. Find most frequent byte pair
    pairs = count_pairs(text)  # {"th": 100, "he": 95, ...}
    most_frequent = max(pairs)  # "th"
    
    # 2. Merge this pair into a new token
    new_token = len(vocab)
    vocab["th"] = new_token
    
    # 3. Replace in text
    text = text.replace("th", new_token)
    
    # 4. Repeat until target vocabulary size
```

**Example**:
```
Original: "the cat"
After 1000 merges: "the" "cat"
After 5000 merges: "the" "cat"
After 65000 merges: "the" "cat"

# Common words become single tokens!
```

### Why Rust Implementation?

**The Problem**: Pure Python BPE training is very slow
- Must scan through billions of characters
- Many string operations
- Takes hours/days in Python

**The Solution**: Rust implementation (rustbpe/)
- ~100x faster than Python
- Trains 65k-token tokenizer on 2B chars in minutes
- Same algorithm, just faster execution

**Integration**:
```python
# Python calls Rust code
import rustbpe

# Train tokenizer
rustbpe.train_bpe(
    input_file="data.txt",
    output_file="tokenizer.json",
    vocab_size=65536,
)

# Use for inference (via tiktoken)
tokenizer = tiktoken.get_encoding("nanochat")
tokens = tokenizer.encode("Hello world")
```

### Special Tokens Deep Dive

**Problem**: How does model know when user is talking vs. assistant?

**Solution**: Special tokens that mark boundaries

**Example Conversation**:
```
<|user_start|>
What is 2+2?
<|user_end|>
<|assistant_start|>
Let me calculate that.
<|python_start|>
2+2
<|python_end|>
<|output_start|>
4
<|output_end|>
The answer is 4!
<|assistant_end|>
```

**Why This Works**:
- Model learns these tokens during midtraining
- Learns to generate assistant tokens after user tokens
- Learns to use tools (python) when appropriate
- Learns to incorporate tool output into response

---

## Data Loading and Processing

### The Data Pipeline

```
Raw Internet Text (FineWeb)
    ↓
Parquet Files on Disk (compressed, columnar)
    ↓
Data Loader (streams files)
    ↓
Tokenizer (text → token IDs)
    ↓
Batches (B, T) shape
    ↓
GPU for Training
```

### FineWeb Dataset

**What**: 15 trillion tokens of filtered web text from Common Crawl

**Why**: High quality, diverse, covers many topics

**Filtering**:
- Removes duplicates
- Removes low-quality content
- Removes harmful content
- Keeps educational, informative text

**Size**: 
- Full dataset: ~15TB
- nanochat typically uses: 60GB (for d20) to 500GB (for d32)

### Parquet Format

**Why Parquet?**
- Columnar storage: only read "text" column
- Compressed: ~3-4x smaller than raw text
- Fast seeks: can jump to any row group
- Standard format: works with many tools

**Structure**:
```
shard_0000.parquet (100MB compressed → 250M chars)
├── Row Group 0 (1024 rows)
│   └── text column
├── Row Group 1 (1024 rows)
│   └── text column
└── ... (~200 row groups per shard)
```

### Distributed Loading

**Challenge**: 8 GPUs need different data

**Solution**: Each GPU reads different row groups
```python
# GPU 0 reads row groups: 0, 8, 16, 24, ...
# GPU 1 reads row groups: 1, 9, 17, 25, ...
# GPU 2 reads row groups: 2, 10, 18, 26, ...
# ...
# GPU 7 reads row groups: 7, 15, 23, 31, ...

for shard in shards:
    for row_group_idx in range(rank, num_row_groups, world_size):
        data = read_row_group(shard, row_group_idx)
        yield data
```

**Benefits**:
- No duplicate data across GPUs
- Efficient use of bandwidth
- Simple to implement

### Tokenization During Loading

**Why On-the-Fly?**
- Pre-tokenizing 60GB+ of text takes time and disk space
- Different models may need different tokenizers
- Easier to update tokenizer

**How It Works**:
```python
def data_loader():
    for text_batch in read_texts():
        # Batch tokenization (faster than one-by-one)
        token_batch = tokenizer.encode_batch(text_batch)
        
        # Pack into sequences of length T
        for i in range(0, len(token_batch), T):
            yield token_batch[i:i+T]
```

**Optimization**: 
- Tokenize in batches (128 texts at once)
- Use multiple threads for tokenization
- Overlap tokenization with GPU training

### Resumable Training

**The Problem**: If training crashes at step 5000, don't want to start from step 0

**The Solution**: Save data loader state
```python
state = {
    'parquet_idx': 42,      # which shard
    'row_group_idx': 153,   # which row group
}

# On resume:
loader = create_loader(resume_state=state)
# Starts from approximately the same point
```

**Note**: Resumption is approximate
- Might skip a few documents
- Avoids complexity of exact resumption
- In practice, doesn't matter much (so much data)

---

## Evaluation Framework

### Why Evaluate?

**Goals**:
1. Measure progress during training
2. Compare different models
3. Identify strengths and weaknesses
4. Avoid overfitting (check val performance)

### Core Evaluation Metrics

#### 1. CORE Score (scripts/base_eval.py)

**What**: Composite score from DCLM paper measuring base language modeling

**Tasks**:
- HellaSwag: Commonsense reasoning
- PIQA: Physical reasoning
- ARC-Easy: Science questions
- OpenbookQA: Fact-based questions
- WinoGrande: Pronoun resolution

**Why**: Good single number for base model quality

**How**: Average accuracy across tasks

**Good Scores**:
- CORE 0.22: Better than random (0.25 is random for 4-choice)
- CORE 0.35: GPT-2 level
- CORE 0.50: Pretty good

#### 2. Bits Per Byte (scripts/base_loss.py)

**What**: How well model compresses text

**Intuition**: 
- Lower = better compression = better understanding
- Random model: ~8 bits/byte (no compression)
- Good model: ~1-2 bits/byte

**Why It's Useful**:
- More fine-grained than accuracy
- Measures all tokens, not just answers
- Correlates with downstream performance

#### 3. Task-Specific Evaluations (scripts/chat_eval.py)

After chat training, evaluate on specific tasks:

**ARC (AI2 Reasoning Challenge)**:
- Multiple choice science questions
- Tests factual knowledge and reasoning

**GSM8K (Grade School Math)**:
- Math word problems
- Tests arithmetic and reasoning
- Answer must be exact number

**HumanEval**:
- Python coding problems
- Tests code generation ability
- Pass@1: percentage that pass test cases

**MMLU (Massive Multitask Language Understanding)**:
- Multiple choice questions on 57 subjects
- Tests broad world knowledge

**ChatCORE**:
- Conversational version of CORE tasks
- Tests chat-formatted reasoning

### How Evaluation Works

#### Generative Tasks (e.g., GSM8K, HumanEval)

```python
def evaluate_generative(task, model):
    scores = []
    for example in task:
        # 1. Format problem as prompt
        prompt = format_problem(example)
        
        # 2. Generate solution
        response = model.generate(prompt, max_tokens=256)
        
        # 3. Extract answer
        answer = extract_answer(response)
        
        # 4. Check correctness
        correct = task.check(answer, example.solution)
        scores.append(correct)
    
    return mean(scores)
```

**Example GSM8K**:
```
Problem: "Janet has 3 apples and buys 2 more. How many does she have?"
Generation: "Janet starts with 3 apples. She buys 2 more. 3 + 2 = 5. She has 5 apples. #### 5"
Extract: "5"
Correct: Yes!
```

#### Multiple Choice Tasks (e.g., ARC, MMLU)

```python
def evaluate_multiple_choice(task, model):
    scores = []
    for example in task:
        # 1. Format as multiple choice
        prompt = f"{example.question}\n"
        prompt += f"A) {example.choices[0]}\n"
        prompt += f"B) {example.choices[1]}\n"
        prompt += f"C) {example.choices[2]}\n"
        prompt += f"D) {example.choices[3]}\n"
        
        # 2. Get probability of each choice
        probs = []
        for choice in ['A', 'B', 'C', 'D']:
            prob = model.get_prob(prompt + choice)
            probs.append(prob)
        
        # 3. Pick most likely
        predicted = argmax(probs)
        
        # 4. Check correctness
        correct = (predicted == example.answer)
        scores.append(correct)
    
    return mean(scores)
```

### Task Implementations

All tasks follow a common interface:

```python
class Task:
    @property
    def eval_type(self):
        # "generative" or "categorical"
        return "categorical"
    
    def __len__(self):
        # Number of examples
        return 1000
    
    def __getitem__(self, idx):
        # Get example as Conversation object
        return self.examples[idx]
    
    def evaluate(self, problem, completion):
        # Check if completion is correct
        return is_correct
```

**Example: ARC Task** (tasks/arc.py)
```python
class ARCEasy(Task):
    def __init__(self):
        # Load dataset from HuggingFace
        self.data = load_dataset("allenai/ai2_arc", "ARC-Easy")
    
    @property
    def eval_type(self):
        return "categorical"  # multiple choice
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Format as conversation
        return Conversation([
            Message(role="user", content=item["question"]),
            Message(role="assistant", content=item["answer"])
        ])
```

### Evaluation During Training

**Base Training**:
- CORE score every 5000 steps
- Bits per byte every 1000 steps
- Helps track if learning is progressing

**Midtraining**:
- Full eval at the end
- ARC, MMLU, ChatCORE, HumanEval
- Checks conversation ability

**SFT**:
- Full eval at the end
- Should see small bump in all scores
- Checks quality improvements

### Reading the Report

Final report includes table like:

```
| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2219   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy        | -        | 0.3561   | 0.3876   | -        |
| GSM8K           | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval       | -        | 0.0671   | 0.0854   | -        |
| MMLU            | -        | 0.3111   | 0.3151   | -        |
| ChatCORE        | -        | 0.0730   | 0.0884   | -        |
```

**How to Read**:
- BASE column: After pretraining (raw language modeling)
- MID column: After midtraining (conversation format)
- SFT column: After supervised finetuning (quality polish)
- RL column: After reinforcement learning (specialized for GSM8K)

**Trends to Expect**:
- BASE → MID: Might drop slightly (learning new format)
- MID → SFT: Small improvements across board
- SFT → RL: Large improvement on RL task (GSM8K), maybe regression on others

---

## Inference and Serving

### Three Interfaces

nanochat provides three ways to interact with your trained model:

1. **CLI** (scripts/chat_cli.py): Simple command-line interface
2. **Web UI** (scripts/chat_web.py): ChatGPT-like interface in browser
3. **API** (scripts/chat_web.py): Programmatic access

### 1. CLI Interface

**Basic Usage**:
```bash
# Interactive mode
python -m scripts.chat_cli

# One-shot mode
python -m scripts.chat_cli -p "Why is the sky blue?"
```

**How It Works**:
```python
def cli_chat(prompt):
    # 1. Load model
    model = load_model("sft_step1000.pt")
    engine = Engine(model)
    
    # 2. Format prompt
    conversation = [
        {"role": "user", "content": prompt}
    ]
    tokens = format_conversation(conversation)
    
    # 3. Generate response
    response_tokens = []
    for token in engine.generate(tokens):
        response_tokens.append(token)
        print(tokenizer.decode([token]), end="", flush=True)
        
        # Stop if end token
        if token == assistant_end_token:
            break
    
    return tokenizer.decode(response_tokens)
```

**Features**:
- Streaming output (tokens appear as generated)
- Multi-turn conversations (maintains history)
- Tool use (can execute Python code)
- Temperature and top-k control

### 2. Web Interface

**Launch**:
```bash
python -m scripts.chat_web
```

**Access**: Open browser to `http://localhost:8000`

**Architecture**:
```
Frontend (nanochat/ui.html)
    ↕ HTTP/SSE
Backend (FastAPI server)
    ↕
Engine (inference)
    ↕
Model (GPU)
```

**Features**:
- ChatGPT-like UI
- Streaming responses
- Multi-turn conversations
- Mobile-friendly
- No external dependencies (self-contained HTML)

**Implementation Highlights**:

```python
@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    # 1. Validate request
    validate_request(request)
    
    # 2. Format messages
    tokens = format_conversation(request.messages)
    
    # 3. Generate streaming response
    async def generate():
        for token in engine.generate(tokens):
            chunk = {
                "choices": [{
                    "delta": {"content": decode(token)}
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return StreamingResponse(generate())
```

### 3. Multi-GPU Serving

**The Problem**: One model per GPU, but many users

**The Solution**: Worker pool with multiple GPUs

```bash
python -m scripts.chat_web --num-gpus 4
```

**Architecture**:
```
User Requests
    ↓
Load Balancer (FastAPI)
    ↓
┌──────────────────────────┐
│ GPU 0: Model + Engine    │
│ GPU 1: Model + Engine    │
│ GPU 2: Model + Engine    │
│ GPU 3: Model + Engine    │
└──────────────────────────┘
```

**How It Works**:
```python
class WorkerPool:
    def __init__(self, num_gpus):
        self.workers = []
        for gpu_id in range(num_gpus):
            worker = Worker(gpu_id)
            self.workers.append(worker)
        self.queue = asyncio.Queue()
    
    async def submit(self, request):
        # Find available worker
        worker = await self.get_available_worker()
        
        # Process request
        response = await worker.generate(request)
        
        return response
```

**Benefits**:
- Higher throughput (multiple requests in parallel)
- Better GPU utilization
- Automatic load balancing

### Generation Parameters

**Temperature**: Controls randomness
```python
# Temperature = 0.0: Always pick most likely token (deterministic)
# Temperature = 1.0: Sample proportional to probability (balanced)
# Temperature = 2.0: More random, creative but less coherent

logits = model(tokens)
probs = softmax(logits / temperature)
next_token = sample(probs)
```

**Top-k**: Only consider top-k most likely tokens
```python
# Top-k = 1: Greedy (always pick most likely)
# Top-k = 50: Consider top 50 tokens
# Top-k = vocab_size: Consider all tokens

logits = model(tokens)
top_k_logits, top_k_indices = topk(logits, k)
probs = softmax(top_k_logits)
next_token = top_k_indices[sample(probs)]
```

**Max Tokens**: Limit response length
```python
max_tokens = 256  # Stop after 256 tokens

response = []
for _ in range(max_tokens):
    token = generate_next()
    response.append(token)
    if token == end_token:
        break
```

### Tool Use

**Capability**: Model can execute Python code

**Example**:
```
User: What is 123 * 456?
Assistant: <|python_start|>123 * 456<|python_end|>
<|output_start|>56088<|output_end|>
The answer is 56088.
```

**How It Works**:

1. **Detection**: Model generates `<|python_start|>` token
2. **Extraction**: Extract code between python tags
3. **Execution**: Run code in safe sandbox
4. **Insertion**: Insert result with `<|output_start|>` tags
5. **Continuation**: Model continues generation with result

**Code**:
```python
def execute_tool(code):
    try:
        # Safe eval with limited builtins
        result = eval(code, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"
```

**Safety**:
- No file access
- No network access
- Timeout after 3 seconds
- Limited to mathematical expressions

---

## Customization Patterns

### Adding Personality (Identity Conversations)

**Goal**: Give your model a unique personality or identity

**How**:

1. **Generate Synthetic Data** (dev/gen_synthetic_data.py):
```python
def generate_identity_data(name, backstory, traits):
    conversations = []
    
    # Questions about identity
    conversations.append({
        "user": "What's your name?",
        "assistant": f"I'm {name}!"
    })
    
    # Incorporate personality traits
    conversations.append({
        "user": "Tell me about yourself",
        "assistant": f"{backstory}. {traits}"
    })
    
    return conversations
```

2. **Mix into Midtraining**:
```python
# In mid_train.py, add your identity data
identity_data = load("identity_conversations.jsonl")
other_data = load("smoltalk.jsonl")

# Mix 5% identity, 95% other
training_data = mix([
    (identity_data, 0.05),
    (other_data, 0.95)
])
```

3. **Result**: Model adopts personality in conversations

**Example Identities**:
- Pirate chatbot (talks like pirate)
- Helpful librarian (very formal and informative)
- Enthusiastic teacher (uses lots of exclamation marks)

### Adding New Capabilities

**Goal**: Teach model new task (e.g., counting letters)

**Reference**: See [Guide: counting r in strawberry](https://github.com/karpathy/nanochat/discussions/164)

**Steps**:

1. **Create Training Data**:
```python
def generate_counting_data():
    words = ["strawberry", "apple", "banana", ...]
    data = []
    
    for word in words:
        for letter in "abcdefghijklmnopqrstuvwxyz":
            count = word.count(letter)
            data.append({
                "user": f"How many {letter}'s in {word}?",
                "assistant": f"Let me count: {word}. The letter {letter} appears {count} times."
            })
    
    return data
```

2. **Add to Midtraining/SFT**:
- Include in training data mix
- Higher weight if capability is critical

3. **Evaluate**:
- Create test set
- Measure accuracy on counting task

4. **Iterate**:
- Increase data if accuracy low
- Vary phrasing in training data
- Add chain-of-thought reasoning

### Adjusting Model Size

**Goal**: Train larger or smaller model

**Key Parameter**: `--depth` in training scripts

**Model Sizes**:
```python
# d12: ~150M parameters
torchrun --nproc_per_node=8 -m scripts.base_train --depth=12

# d20: ~561M parameters (speedrun default)
torchrun --nproc_per_node=8 -m scripts.base_train --depth=20

# d26: ~1.1B parameters (GPT-2 scale)
torchrun --nproc_per_node=8 -m scripts.base_train --depth=26 --device_batch_size=16

# d32: ~1.9B parameters ($1000 tier)
torchrun --nproc_per_node=8 -m scripts.base_train --depth=32 --device_batch_size=16
```

**Scaling Rules**:

1. **Chinchilla Law**: 20 tokens per parameter
   - d20 (561M params): 11.2B tokens
   - d26 (1.1B params): 22B tokens
   - d32 (1.9B params): 38B tokens

2. **Data Shards Needed**:
   ```
   tokens_needed = params * 20
   chars_needed = tokens_needed * 4.8  # tokenizer compression
   shards_needed = chars_needed / 250M  # chars per shard
   ```

3. **Memory Management**:
   - If OOM, reduce `--device_batch_size`
   - Code automatically does more gradient accumulation
   - Same result, just slower

**Example d26 Changes**:
```bash
# speedrun.sh modifications for d26:

# 1. Download more data (450 shards instead of 240)
python -m nanochat.dataset -n 450

# 2. Use depth=26 and reduce batch size to fit in memory
torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    --depth=26 --device_batch_size=16

# 3. Keep batch size reduced for midtraining too
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train \
    --device_batch_size=16
```

### Custom Evaluation Tasks

**Goal**: Evaluate model on your own task

**Steps**:

1. **Create Task Class** (tasks/mytask.py):
```python
from tasks.common import Task, Conversation, Message

class MyTask(Task):
    def __init__(self):
        # Load your data
        self.examples = load_my_data()
    
    @property
    def eval_type(self):
        return "generative"  # or "categorical"
    
    def num_examples(self):
        return len(self.examples)
    
    def get_example(self, idx):
        ex = self.examples[idx]
        return Conversation([
            Message(role="user", content=ex["question"]),
            Message(role="assistant", content=ex["answer"])
        ])
    
    def evaluate(self, problem, completion):
        # Check if completion is correct
        return completion.strip() == problem.answer.strip()
```

2. **Add to Evaluation Script**:
```python
# In scripts/chat_eval.py
from tasks.mytask import MyTask

# Add to task list
tasks = {
    "MyTask": MyTask(),
    "ARC": ARCEasy(),
    # ... other tasks
}
```

3. **Run Evaluation**:
```bash
torchrun --nproc_per_node=8 -m scripts.chat_eval -a MyTask
```

### Using Different Datasets

**Goal**: Train on your own data instead of FineWeb

**For Pretraining**:

1. **Prepare Data**:
   - Convert to Parquet format
   - Each file should have "text" column
   - Split into ~100MB shards

2. **Update Dataset Path**:
```python
# In nanochat/dataset.py
def list_parquet_files():
    # Point to your data directory
    return glob("/path/to/your/data/*.parquet")
```

3. **Train as Normal**

**For Finetuning**:

1. **Format Conversations**:
```jsonl
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
{"messages": [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Goodbye!"}]}
```

2. **Use CustomJSON Task**:
```python
# In training script
from tasks.customjson import CustomJSON

dataset = CustomJSON("my_conversations.jsonl")
```

---

## How It All Fits Together

### The Big Picture

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  1. TOKENIZER TRAINING                              │
│     Raw text → BPE algorithm → Vocabulary           │
│                                                     │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│                                                     │
│  2. BASE PRETRAINING                                │
│     FineWeb text → Tokenize → Train GPT →           │
│     Base model that predicts next token             │
│                                                     │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│                                                     │
│  3. MIDTRAINING                                     │
│     Conversation data → Format with special tokens  │
│     → Continue training → Chat-aware model          │
│                                                     │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│                                                     │
│  4. SUPERVISED FINETUNING (SFT)                     │
│     High-quality conversations → Polish responses   │
│     → Production-ready chat model                   │
│                                                     │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│                                                     │
│  5. (OPTIONAL) REINFORCEMENT LEARNING               │
│     Generate + reward → Improve specific tasks      │
│                                                     │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│                                                     │
│  6. DEPLOYMENT                                      │
│     Load model → Engine with KV cache →             │
│     CLI / Web UI / API                              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### File Dependencies

Understanding what imports what:

```
scripts/base_train.py
├── nanochat.gpt (GPT, GPTConfig)
│   ├── nanochat.muon (Muon optimizer)
│   └── nanochat.adamw (AdamW optimizer)
├── nanochat.dataloader (data streaming)
│   ├── nanochat.dataset (parquet files)
│   └── nanochat.tokenizer (text → tokens)
│       └── rustbpe (fast BPE training)
├── nanochat.checkpoint_manager (save/load)
├── nanochat.core_eval (CORE benchmark)
└── nanochat.common (utilities)

scripts/chat_web.py
├── nanochat.engine (inference with KV cache)
│   └── nanochat.gpt (model architecture)
├── nanochat.checkpoint_manager (load model)
└── nanochat.tokenizer (tokens ↔ text)
```

### Data Flow During Training

```
Disk (Parquet files)
    ↓ read_row_group()
List of text strings
    ↓ tokenizer.encode_batch()
List of token sequences
    ↓ pack_sequences()
Batches of shape (B, T)
    ↓ to(device)
Batches on GPU
    ↓ model.forward()
Logits (B, T, vocab_size)
    ↓ cross_entropy_loss()
Scalar loss value
    ↓ loss.backward()
Gradients for all parameters
    ↓ optimizer.step()
Updated model weights
```

### Data Flow During Inference

```
User input text
    ↓ tokenizer.encode()
Token IDs
    ↓ engine.prefill()
Initial KV cache + first prediction
    ↓ sample_token()
Next token ID
    ↓ while not done:
        ├─ engine.step(token)  # add to KV cache
        ├─ sample_token()      # get next token
        └─ decode(token)       # convert to text
    ↓
Complete response text
```

### Computation Distribution (8 GPUs)

```
GPU 0: ├────────────────────────┤
       │ model copy 0           │
       │ batch slice 0          │
       └────────────────────────┘
GPU 1: ├────────────────────────┤
       │ model copy 1           │
       │ batch slice 1          │
       └────────────────────────┘
       ... (6 more GPUs)
       
Each GPU:
1. Has full model copy
2. Processes 1/8 of batch
3. Computes gradients
4. All-reduces gradients (average across GPUs)
5. Updates its model copy (all end up identical)
```

### Checkpoint Progression

```
~/.cache/nanochat/
├── tokenizer.json              # After tok_train
├── base_step0000.pt            # Initial random weights
├── base_step5000.pt            # Midway through pretraining
├── base_step10000.pt           # Final base model
├── mid_step2000.pt             # After midtraining
├── sft_step1000.pt             # After SFT (final)
└── rl_step500.pt               # After RL (optional)
```

**Which checkpoint for what**:
- Base model (`base_step*.pt`): Language modeling, completion
- Mid model (`mid_step*.pt`): Conversations, tool use
- SFT model (`sft_step*.pt`): Production chat (default)
- RL model (`rl_step*.pt`): Specialized tasks (GSM8K)

### Common Workflows

**Workflow 1: Quick Experiment**
```bash
# Train tiny model on CPU
python -m scripts.base_train --depth=4 --max_seq_len=512 \
    --device_batch_size=1 --num_iterations=100

# Chat with it
python -m scripts.chat_cli
```

**Workflow 2: $100 Speedrun**
```bash
# Run complete pipeline
bash speedrun.sh

# Chat via web
python -m scripts.chat_web
```

**Workflow 3: Custom Personality**
```bash
# Generate identity data
python dev/gen_synthetic_data.py > my_identity.jsonl

# Modify speedrun.sh to use your data
# Then run:
bash speedrun.sh

# Result: Model with your custom personality
```

**Workflow 4: Evaluate on New Task**
```bash
# Create task in tasks/mytask.py
# Then evaluate:
torchrun --nproc_per_node=8 -m scripts.chat_eval -a MyTask
```

### Key Hyperparameters

**Model Architecture**:
- `depth`: Number of transformer layers (bigger = more capable, slower)
- `max_seq_len`: Context window (2048 is good balance)
- `n_head`: Attention heads (6-12 for small models)

**Training**:
- `device_batch_size`: Per-GPU batch size (tune to not OOM)
- `total_batch_size`: Effective batch size (bigger = more stable, but slower)
- `num_iterations`: Training steps (set via Chinchilla ratio)

**Optimization**:
- `matrix_lr`: Learning rate for Muon (0.01-0.05)
- `embedding_lr`: Learning rate for embeddings (0.1-0.5)
- `warmup_iters`: Learning rate warmup (100-500)

**Inference**:
- `temperature`: Randomness (0.0 = deterministic, 1.0 = balanced)
- `top_k`: Sampling pool size (50-200)
- `max_tokens`: Response length limit (256-1024)

### Performance Optimization

**Training Speed**:
1. Use 8 GPUs instead of 1: 8x faster
2. Use H100 instead of A100: 2-3x faster
3. Increase `device_batch_size`: Better GPU utilization
4. Use `torch.compile()`: 10-20% speedup (future)

**Inference Speed**:
1. KV caching: ~10-100x speedup (already implemented)
2. Batching: Process multiple requests together
3. Quantization: 2-4x speedup with slight quality loss (not implemented)
4. Flash Attention: 2-3x speedup (future)

**Memory Optimization**:
1. Reduce `device_batch_size`: Use gradient accumulation instead
2. Use GQA: Fewer KV heads = less cache memory
3. Shorter context: Smaller KV cache
4. Gradient checkpointing: Trade compute for memory (not implemented)

---

## Conclusion

### What You've Learned

You now understand:

1. **Architecture**: How transformers work (attention, MLP, residuals)
2. **Training Pipeline**: Tokenization → Pretraining → Midtraining → SFT → RL
3. **Data Systems**: Streaming, distributed loading, on-the-fly tokenization
4. **Evaluation**: CORE, task-specific benchmarks, measuring progress
5. **Inference**: KV caching, serving, tool use
6. **Customization**: Personality, new tasks, scaling

### Next Steps

**To Learn More**:
1. Read through the actual code (it's short and readable!)
2. Train a tiny model on CPU to see everything work
3. Modify a training script to experiment
4. Add a custom evaluation task
5. Create a unique personality for your model

**To Go Deeper**:
1. Read the Chinchilla paper for scaling laws
2. Study the Transformer paper ("Attention Is All You Need")
3. Explore the DCLM paper for dataset curation
4. Learn about RLHF and PPO for RL details
5. Read about Flash Attention for optimization

**To Contribute**:
1. Improve documentation
2. Add new evaluation tasks
3. Optimize training speed
4. Add new features (quantization, etc.)
5. Fix bugs

### Key Takeaways

1. **LLMs are simple**: At the core, just predicting next token
2. **Data is critical**: Quality and quantity both matter
3. **Training is expensive**: But $100-1000 can get you far
4. **Evaluation is essential**: Can't improve what you don't measure
5. **End-to-end understanding**: Having full pipeline is powerful

### Philosophy

nanochat demonstrates that:
- Clarity beats complexity
- Minimalism enables understanding
- End-to-end control is valuable
- Open source fosters learning
- Small models teach big lessons

### Resources

- **Original Repository**: https://github.com/karpathy/nanochat
- **This Fork**: https://github.com/CodeHalwell/nanochat-codhe
- **Discussions**: Check GitHub Discussions in the original repo for guides and help
- **DeepWiki**: https://deepwiki.com/karpathy/nanochat for AI-powered exploration (original repo)
- **Wandb**: Use for experiment tracking
- **Lambda**: Recommended GPU provider

### Final Words

nanochat is more than a ChatGPT clone—it's an educational journey through modern LLM technology. Every line of code teaches something about how these systems work. The best way to learn is to run it, break it, fix it, and make it your own.

Happy hacking! 🚀

---

## Appendix: Quick Reference

### Common Commands

```bash
# Install dependencies
uv sync --extra gpu

# Train tokenizer
python -m scripts.tok_train --max_chars=2000000000

# Train base model
torchrun --nproc_per_node=8 -m scripts.base_train --depth=20

# Evaluate base model
torchrun --nproc_per_node=8 -m scripts.base_eval

# Train mid model
torchrun --nproc_per_node=8 -m scripts.mid_train

# Train SFT model
torchrun --nproc_per_node=8 -m scripts.chat_sft

# Evaluate chat model
torchrun --nproc_per_node=8 -m scripts.chat_eval

# Chat via CLI
python -m scripts.chat_cli

# Chat via web
python -m scripts.chat_web

# Complete pipeline
bash speedrun.sh
```

### File Locations

```bash
# Cache directory
~/.cache/nanochat/

# Data shards
~/.cache/nanochat/data/shard_*.parquet

# Checkpoints
~/.cache/nanochat/*_step*.pt

# Tokenizer
~/.cache/nanochat/tokenizer.json

# Report
./report.md (copied to current directory)
```

### Troubleshooting

**OOM / Out of Memory**:
- Reduce `--device_batch_size`
- Reduce `--depth` (smaller model)
- Reduce `--max_seq_len`

**Slow Training**:
- Use more GPUs (--nproc_per_node=8)
- Increase `--device_batch_size` (if memory allows)
- Use faster GPUs (H100 > A100)

**Poor Performance**:
- Train longer (more iterations)
- Use more data (more shards)
- Increase model size (higher depth)
- Check evaluation scores during training

**Import Errors**:
- Run `uv sync --extra gpu`
- Build rustbpe: `uv run maturin develop --release`
- Activate venv: `source .venv/bin/activate`

**Distribution Issues**:
- Check all GPUs visible: `nvidia-smi`
- Use correct nproc_per_node value
- Set `OMP_NUM_THREADS=1`

---

*This tutorial was created to help understand the nanochat codebase deeply. For the most up-to-date information, always refer to the actual code and official documentation.*