import torch
import transformers
import transformers.models.llama.modeling_llama


class ScaledRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        self.scale = 2
        max_position_embeddings = max_position_embeddings * self.scale

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )

        
        t /= self.scale

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        
        #self.window = torch.hann_window(emb.shape[-1])
        self.register_buffer(
            "cos_cached", (emb.cos())[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin())[None, None, :, :], persistent=False
        )

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(
                self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype
            )
            t *= self.scale
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer(
                "cos_cached", emb.cos()[None, None, :, :], persistent=False
            )
            self.register_buffer(
                "sin_cached", emb.sin()[None, None, :, :], persistent=False
            )
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

old_init = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__

def ntk_scaled_init(self, dim, max_position_embeddings=2048, base=10000, device=None):
    #The method is just these three lines
    a = 8 #Alpha value
    max_position_embeddings = max_position_embeddings * a

    base = base * a ** (dim / (dim-2)) #Base change formula

    old_init(self, dim, max_position_embeddings, base, device)
    
def replace_llama_rope_with_scaled_rope():
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = (
        ScaledRotaryEmbedding
    )

def repalce_llama_rope_init_with_scaled_rope_init():
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__ = ntk_scaled_init