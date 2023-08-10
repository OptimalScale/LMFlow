from functools import partial

import torch
import transformers
import transformers.models.llama.modeling_llama

class CondenseRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, pi_ratio, ntk_ratio, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        
        self.dim = dim
        
        self.ntk_ratio = ntk_ratio
        max_position_embeddings *= ntk_ratio
        base = base * ntk_ratio ** (dim / (dim-2)) #Base change formula

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        self.pi_ratio = pi_ratio
        max_position_embeddings *= pi_ratio
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype) / pi_ratio
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            
            base = self.base * (
                (self.ntk_ratio * seq_len / self.max_position_embeddings) - (self.ntk_ratio - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(x.device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype) / self.pi_ratio
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False)
            
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def replace_llama_with_condense(pi_ratio, ntk_ratio):
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = partial(CondenseRotaryEmbedding, pi_ratio=pi_ratio, ntk_ratio=ntk_ratio)