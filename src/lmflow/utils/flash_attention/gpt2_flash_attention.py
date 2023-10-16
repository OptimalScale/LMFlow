from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from einops import rearrange

#try to import flash_attn 2.x.x, if not, import flash_attn 1.x.x
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func
except:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func

from flash_attn.bert_padding import unpad_input, pad_input

def _attn(self, query, key, value, attention_mask=None, head_mask=None):
    # (batch, head, seq_length, head_features)
    assert head_mask is None, "head_mask is not supported"
    query = query.to(torch.bfloat16)
    key = key.to(torch.bfloat16)
    value = value.to(torch.bfloat16)
    qkv = torch.stack(
        [query, key, value], dim=2
    )# [bsz, nh, 3, t, hd]
    qkv = qkv.transpose(1,3)## [bsz, q_len, 3, nh, hd]
    bsz = qkv.shape[0]
    q_len = qkv.shape[1]
    #attention_mask = torch.where(attention_mask != 0.0, True, False)
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

def _merge_heads(self, tensor, num_heads, attn_head_size):
    """
    Merges attn_head_size dim and num_attn_heads dim into hidden_size
    """
    new_shape = tensor.size()[:-2] + (self.num_heads * self.head_dim,)
    return tensor.view(new_shape)

def replace_gpt2_attn_with_flash_attn():
    transformers.models.gpt2.modeling_gpt2.GPT2Attention._attn = _attn
    transformers.models.gpt2.modeling_gpt2.GPT2Attention._merge_heads = _merge_heads