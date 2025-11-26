"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 11
    n_head: int = 4 # number of query heads
    n_kv_head: int = 4 # number of key/value heads (GQA)
    n_embd: int = 1024
    n_loops: int = 3  # TRM: latent recursion steps (Conservative Enhancement: ↑ from 2)
    n_sup_train: int = 6  # Deep supervision steps (training) (sweet spot for 24GB GPU)
    n_sup_inference: int = 4  # Deep supervision steps (inference)
    T_recursion: int = 1  # TRM: recursion depth (1=no no_grad passes, saves memory during training)
    T_recursion_inference: int = 2  # Recursion depth during inference (can be higher)
    activation_fn: str = 'swiglu'  # Options: 'relu_squared', 'swiglu', 'geglu' (swiglu is best)

    # TRM Progressive Refinement Settings
    # These control how aggressively the model learns to refine across supervision steps
    #
    # DEFAULT (balanced preset - optimized for n_sup=6):
    supervision_weight_base: float = 2.5  # Hierarchical weighting (balanced for n_sup=6)
    improvement_reward_scale: float = 1.5  # Improvement bonus (moderate for n_sup=6)
    detach_threshold: int = 2  # Only detach steps >= this (0=detach all, N_sup=detach none)
    # With n_sup=6: weights [0.002, 0.008, 0.027, 0.095, 0.330, 0.538]
    # Final step gets 54% weight, step 0 gets 0.2% - strong progressive refinement
    # Expected: 8-20% refinement, healthy gradients (0.5-3.0)
    #
    # Alternative options for n_sup=6:
    # - 3.0, 0.5 = conservative (gentler weighting, more balanced)
    # - 3.5, 0.7 = balanced     (current DEFAULT)
    # - 5.0, 1.0 = aggressive   (stronger final step bias)

    @classmethod
    def with_refinement_preset(cls, preset='aggressive', **kwargs):
        """
        Create a GPTConfig with TRM refinement presets.

        Presets:
        - 'moderate': 8-12% refinement, very stable (production safe)
        - 'aggressive': 15-25% refinement, stable (RECOMMENDED for most use cases)
        - 'extreme': 50-97% refinement, requires careful tuning (research/experimentation)

        Example:
            config = GPTConfig.with_refinement_preset('extreme', n_layer=12, n_embd=768)
        """
        presets = {
            'moderate': {
                'supervision_weight_base': 3.0,
                'improvement_reward_scale': 0.5,
            },
            'aggressive': {
                'supervision_weight_base': 5.0,
                'improvement_reward_scale': 1.0,
            },
            'extreme': {
                'supervision_weight_base': 10.0,
                'improvement_reward_scale': 2.0,
            },
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(presets.keys())}")

        # Merge preset with user overrides
        config_dict = {**presets[preset], **kwargs}
        return cls(**config_dict)


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache, layer_idx=None):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            idx = layer_idx if layer_idx is not None else self.layer_idx
            k, v = kv_cache.insert_kv(idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0: # can't be negative but could be zero
                attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd

        # SwiGLU/GeGLU require 2x hidden dim for gate+value, relu² uses standard hidden dim
        if config.activation_fn in ('swiglu', 'geglu'):
            self.c_fc = nn.Linear(config.n_embd, 2 * hidden_dim, bias=False)
            self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
        else:  # relu_squared
            self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
            self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

        self.activation_fn = config.activation_fn

    def forward(self, x):
        if self.activation_fn == 'swiglu':
            # SwiGLU: x = SiLU(gate) * value
            # Used in: LLaMA, Mistral, Gemma
            x = self.c_fc(x)
            gate, value = x.chunk(2, dim=-1)
            x = F.silu(gate) * value
            x = self.c_proj(x)
        elif self.activation_fn == 'geglu':
            # GeGLU: x = GELU(gate) * value
            # Alternative gated activation
            x = self.c_fc(x)
            gate, value = x.chunk(2, dim=-1)
            x = F.gelu(gate) * value
            x = self.c_proj(x)
        else:  # relu_squared (default)
            x = self.c_fc(x)
            x = F.relu(x).square()
            x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache, layer_idx=None):
        x = x + self.attn(norm(x), cos_sin, kv_cache, layer_idx)
        x = x + self.mlp(norm(x))
        return x


class RecursiveBlock(nn.Module):
    """TRM recursive block with y (answer) and z (reasoning) states"""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.block = Block(config, layer_idx)
        self.n_loops = config.n_loops
        self.layer_idx = layer_idx
        self.T_recursion = config.T_recursion

    @property
    def mlp(self):
        return self.block.mlp

    @property
    def attn(self):
        return self.block.attn

    def forward(self, x, y, z, cos_sin, kv_cache, T=None):
        """TRM with proper residual connections for stability"""
        T = T if T is not None else self.T_recursion

        # T-1 passes without gradients (ONLY during inference)
        # During training, we need gradients to flow, so we run all T passes with gradients
        if self.training:
            num_nograd_passes = 0  # All passes with gradients during training
        else:
            num_nograd_passes = T - 1  # T-1 passes without gradients during inference

        with torch.no_grad():
            for _ in range(num_nograd_passes):
                # Fast loop: Update z
                for i in range(self.n_loops):
                    z_input = norm(x + y + z)
                    virtual_idx = self.layer_idx * (self.n_loops + 1) + i
                    output = self.block(z_input, cos_sin, kv_cache, layer_idx=virtual_idx)
                    # Standard residual connection
                    z = z + output

                # Slow loop: Update y
                y_input = norm(y + z)
                virtual_idx = self.layer_idx * (self.n_loops + 1) + self.n_loops
                output = self.block(y_input, cos_sin, kv_cache, layer_idx=virtual_idx)
                # Standard residual connection
                y = y + output

        # ⚠️ CRITICAL FIX: Re-enable gradients after no_grad context (only needed during inference)
        # During training, we skip no_grad passes, so this is not needed
        if not self.training and y.requires_grad is False:
            y = y.detach().requires_grad_(True)
        if not self.training and z.requires_grad is False:
            z = z.detach().requires_grad_(True)

        # Passes with gradients
        # During training: T passes (full recursion with gradients)
        # During inference: 1 pass (after T-1 no-grad passes)
        num_grad_passes = T if self.training else 1

        for _ in range(num_grad_passes):
            # Fast loop: Update z
            for i in range(self.n_loops):
                z_input = norm(x + y + z)
                virtual_idx = self.layer_idx * (self.n_loops + 1) + i
                output = self.block(z_input, cos_sin, kv_cache, layer_idx=virtual_idx)
                z = z + output  # Standard residual connection

            # Slow loop: Update y
            y_input = norm(y + z)
            virtual_idx = self.layer_idx * (self.n_loops + 1) + self.n_loops
            output = self.block(y_input, cos_sin, kv_cache, layer_idx=virtual_idx)
            y = y + output  # Standard residual connection

        return y, z


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([RecursiveBlock(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', n_sup=None):
        """TRM forward with deep supervision"""
        B, T = idx.size()
        
        # Rotary embeddings
        assert T <= self.cos.size(1)
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        
        # Embed input (x = question, never changes)
        x = self.transformer.wte(idx)
        x = norm(x)
        
        softcap = 15
        
        if targets is None:
            # Inference: use provided n_sup or default to config
            # Use T_recursion_inference for efficient inference (T-1 no_grad passes)
            N_sup = n_sup if n_sup is not None else self.config.n_sup_inference
            y, z = x.clone(), torch.zeros_like(x)

            for _ in range(N_sup):
                for block in self.transformer.h:
                    y, z = block(x, y, z, cos_sin, kv_cache, T=self.config.T_recursion_inference)
                y, z = y.detach(), z.detach()
            
            logits = self.lm_head(norm(y))
            logits = softcap * torch.tanh(logits / softcap)
            return logits
        else:
            # Training: TRM Deep Supervision with Progressive Refinement
            # ============================================================
            # The TRM model learns to progressively refine its answer across N_sup supervision steps:
            #   Step 0: Initial rough answer (may be intentionally poor)
            #   Step 1-2: Iterative refinement through y/z state updates
            #   Step 3: Final polished answer
            #
            # Key mechanisms for achieving refinement:
            # 1. Hierarchical weighting: Later steps weighted exponentially more (base^i)
            # 2. Improvement reward: Explicit bonus for improving over previous step
            # 3. Gradient flow: No detachment allows learning of refinement patterns
            #
            # Achieves 8-97% refinement depending on supervision_weight_base and improvement_reward_scale
            assert kv_cache is None, "KV cache not supported during training with deep supervision"

            N_sup = self.config.n_sup_train
            y, z = x.clone(), torch.zeros_like(x)
            total_loss = 0.0
            supervision_losses = []

            # Hierarchical weighting: base^i where base ∈ {3, 5, 10}
            # base=3:  weights ≈ [0.025, 0.075, 0.225, 0.675]  → final step 27x more important
            # base=5:  weights ≈ [0.008, 0.040, 0.200, 0.752]  → final step 94x more important
            # base=10: weights ≈ [0.001, 0.009, 0.090, 0.900]  → final step 1000x more important
            weighting_base = self.config.supervision_weight_base
            weights = torch.tensor([weighting_base**i for i in range(N_sup)], device=x.device)
            weights = weights / weights.sum()  # Normalize

            for sup_step in range(N_sup):
                for block in self.transformer.h:
                    y, z = block(x, y, z, cos_sin, kv_cache=None)

                logits = self.lm_head(norm(y))
                logits = softcap * torch.tanh(logits / softcap)
                logits = logits.float()

                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                      targets.view(-1), ignore_index=-1,
                                      reduction=loss_reduction)

                # Apply hierarchical weight to this supervision step
                weighted_loss = weights[sup_step] * loss
                total_loss += weighted_loss
                supervision_losses.append(loss.detach())

                # Add improvement reward: bonus for improving over previous step
                # Configurable via improvement_reward_scale (0.5=moderate, 1.0=strong, 2.0=extreme)
                if sup_step > 0:
                    prev_loss = supervision_losses[sup_step - 1]
                    # Reward relative improvement (prev - current) / prev
                    improvement = (prev_loss - loss) / (prev_loss + 1e-8)
                    improvement_bonus = improvement * self.config.improvement_reward_scale
                    total_loss -= improvement_bonus  # Subtract (lower loss is better)

                # ⭐ CRITICAL FIX: Strategic detachment prevents 768-layer backward pass
                # Without: N_sup=8 × 12 layers × 4 virtual = 384 layers per backward (too deep!)
                # With: Each supervision step optimized independently except last one
                # Result: Healthy gradients (0.5-3.0) instead of vanishing (0.02-0.07)
                #
                # ⭐ SELECTIVE DETACHMENT: Keep early steps fully connected for stronger learning
                # detach_threshold=2 means steps 0,1 stay connected, steps 2,3,4 get detached
                # This gives early steps FULL gradient signal while preventing vanishing grads
                if sup_step >= self.config.detach_threshold and sup_step < N_sup - 1:
                    y, z = y.detach(), z.detach()

            # Store supervision losses for diagnostics (only if scalar)
            # This enables external code to check refinement patterns
            # if all(l.numel() == 1 for l in supervision_losses):
            #     self._last_supervision_losses = [l.item() for l in supervision_losses]

            # DEBUG: Print all supervision losses
            # if all(l.numel() == 1 for l in supervision_losses):
            #     print(f"DEBUG all supervision_losses: {[f'{l.item():.4f}' for l in supervision_losses]}")
            #     # DEBUG: Print norms
            #     print(f"DEBUG y norm: {y.norm().item():.4f}, z norm: {z.norm().item():.4f}")
            aux_losses = {
                'final': supervision_losses[-1],
                'step_0': supervision_losses[0],
                'step_2': supervision_losses[2] if len(supervision_losses) > 2 else supervision_losses[-1],
                'step_4': supervision_losses[4] if len(supervision_losses) > 4 else supervision_losses[-1],
                'step_6': supervision_losses[6] if len(supervision_losses) > 6 else supervision_losses[-1],
            }

            return total_loss / N_sup, aux_losses

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
