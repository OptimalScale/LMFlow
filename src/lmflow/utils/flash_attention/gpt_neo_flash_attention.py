from typing import List, Optional, Tuple

import torch
import transformers
from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input

def _attn(self, query, key, value, attention_mask=None, head_mask=None):
    # (batch, head, seq_length, head_features)
    query = query.to(torch.bfloat16)
    key = key.to(torch.bfloat16)
    query = query * torch.sqrt(torch.tensor(self.head_dim))
    qkv = torch.stack(
        [query, key, value], dim=2
    )# [bsz, nh, 3, t, hd]
    qkv = qkv.transpose(1,3)## [bsz, q_len, 3, nh, hd]
    bsz = qkv.shape[0]
    q_len = qkv.shape[1]
    
    attention_mask = torch.where(attention_mask == -0.0, True, False)
    key_padding_mask = rearrange(attention_mask, "b () () s -> b s") if attention_mask is not None else None
    if key_padding_mask is None:
        qkv = rearrange(qkv, "b s ... -> (b s) ...")
        max_s = q_len
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
        )
        output = flash_attn_unpadded_qkvpacked_func(
            qkv, cu_q_lens, max_s, self.attn_dropout.p if self.training else 0.0 , softmax_scale=None, causal=True
        )# attention compute
        output = rearrange(output, "(b s) ... -> b s ...", b=bsz)
    else:
        nheads = qkv.shape[-2]
        x = rearrange(qkv, "b s three h d -> b s (three h d)")
        x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(
            x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
        )
        output_unpad = flash_attn_unpadded_qkvpacked_func(
            x_unpad, cu_q_lens, max_s, self.attn_dropout.p if self.training else 0.0, softmax_scale=None, causal=True
        )
        output = rearrange(
            pad_input(
                rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len
            ),
            "b s (h d) -> b s h d",
            h=nheads,
        )
    
    return output, None

def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
    
    assert head_mask is None, "head_mask is not supported"
    assert not output_attentions, "output_attentions is not supported"
    assert not use_cache, "use_cache is not supported"
    
    query = self.q_proj(hidden_states)
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    query = self._split_heads(query, self.num_heads, self.head_dim)
    key = self._split_heads(key, self.num_heads, self.head_dim)
    value = self._split_heads(value, self.num_heads, self.head_dim)
    
    if layer_past is not None:
        past_key = layer_past[0]
        past_value = layer_past[1]
        key = torch.cat((past_key, key), dim=-2)
        value = torch.cat((past_value, value), dim=-2)
        
    present = None
    attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
    new_shape = attn_output.size()[:-2] + (self.num_heads * self.head_dim,)
    attn_output = attn_output.view(new_shape)
    attn_output = self.out_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)

    return outputs  # a, present, (attentions)
        
def replace_gpt_neo_attn_with_flash_attn():
    transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention._attn = _attn
    transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention.forward = forward