from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

import transformers
from transformers.models.bloom.modeling_bloom import dropout_add

from einops import rearrange

# from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input

from .triton_flash_attention import flash_attn_qkvpacked_func, flash_attn_func

def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):  
        dtype = hidden_states.dtype
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, q_length, _, _ = query_layer.shape
        bsz, q_len = batch_size, q_length
        # breakpoint()
        # query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        # key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
        # value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        # breakpoint()
        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        # _, _, kv_length = key_layer.shape

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        reshaped_alibi = rearrange(alibi, '(b h) one s-> b h one s', h = self.num_heads)
        # old_attention_mask = attention_mask.bool()
        
        # attention_mask = (1.0 - attention_mask)
        # attention_mask = attention_mask[:, None, None, :].bool()
        # attention_mask = attention_mask.triu(1)
        # reshaped_alibi_masked = reshaped_alibi.masked_fill(attention_mask, -1e9)
        reshaped_alibi_masked = reshaped_alibi * self.beta
        # breakpoint()
        # import pickle
        # with open('/public/home/liuwei4/code/LMFlow/unittest_bloom_alibi_flsh.pkl', 'wb') as f:
        #     pickle.dump(alibi, f)
        # breakpoint()
        # reshaped_query_layer = query_layer.reshape(batch_size, self.num_heads, query_layer.shape[1], query_layer.shape[2]).permute(0, 2, 1, 3)
        # reshaped_key_layer = key_layer.reshape(batch_size, self.num_heads, key_layer.shape[1], key_layer.shape[2]).permute(0, 3, 1, 2)
        # reshaped_value_layer = value_layer.reshape(batch_size, self.num_heads, value_layer.shape[1], value_layer.shape[2]).permute(0, 2, 1, 3)
        reshaped_query_layer = query_layer
        reshaped_key_layer = key_layer
        reshaped_value_layer = value_layer
        # offset_key_layer = self.inv_norm_factor * reshaped_key_layer + self.beta * (torch.linalg.pinv(reshaped_query_layer.permute(0,2,1,3).float()) * alibi.view(batch_size, alibi.shape[0]//batch_size, alibi.shape[1], alibi.shape[2])).permute(0, 3, 1, 2).to(dtype)
        # [batch_size * num_heads, q_length, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
        # matmul_result = alibi.baddbmm(
        #     batch1=query_layer,
        #     batch2=key_layer,
        #     beta=self.beta,
        #     alpha=self.inv_norm_factor,
        # )
        # breakpoint()
        qkv = torch.concat([reshaped_query_layer.unsqueeze(2), reshaped_key_layer.unsqueeze(2), reshaped_value_layer.unsqueeze(2)], dim = 2)

        # key_padding_mask = None
        
        # assert key_padding_mask is None, "Custom attention mask is not supported yet in this version of Bloom with flash attention.\
        #                                     Therefore, we encourage you to set batch_size=1. Plan to support this in the future."
        # if key_padding_mask is None:
            # qkv = rearrange(qkv, "b s ... -> (b s) ...")
            # max_s = q_len
            # cu_q_lens = torch.arange(
            #     0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
            # )
            # output = flash_attn_unpadded_qkvpacked_func(
            #     qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
            # )
            # output = flash_attn_func(reshaped_query_layer.contiguous(),
            #                          reshaped_key_layer.contiguous(),
            #                          reshaped_value_layer.contiguous(),
            #                          reshaped_alibi.contiguous(), True, None)
            # output = flash_attn_qkvpacked_func(
            #     qkv, reshaped_alibi, True, None
            # )
            # output = rearrange(output, "(b s) ... -> b s ...", b=bsz)
        # else:
        #     nheads = qkv.shape[-2]
            # x = rearrange(qkv, "b s three h d -> b s (three h d)")
            # x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
            # x_unpad = rearrange(
            #     x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
            # )
        # breakpoint()
        output = flash_attn_qkvpacked_func(
                qkv, reshaped_alibi_masked, True, self.inv_norm_factor
            )
        # breakpoint()
        # output = flash_attn_qkvpacked_func(
        #         qkv, reshaped_alibi_masked, True, None
        #     )
            # output_unpad = flash_attn_unpadded_qkvpacked_func(
            #     x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
            # )
            # output = rearrange(
            #     pad_input(
            #         rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len
            #     ),
            #     "b s (h d) -> b s h d",
            #     h=nheads,
            # )
        # breakpoint()
        output = rearrange(output, 'b s h d -> (b h) s d')
        
        ################For unitest#################
               # change view to [batch_size, num_heads, q_length, kv_length]
        # attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

        # # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        # input_dtype = attention_scores.dtype
        # # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        # if input_dtype == torch.float16:
        #     attention_scores = attention_scores.to(torch.float)
        # attn_weights = torch.masked_fill(attention_scores, old_attention_mask, torch.finfo(attention_scores.dtype).min)
        # attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)

        # # [batch_size, num_heads, q_length, kv_length]
        # attention_probs = self.attention_dropout(attention_probs)

        # if head_mask is not None:
        #     attention_probs = attention_probs * head_mask

        # # change view [batch_size x num_heads, q_length, kv_length]
        # attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)

        # # matmul: [batch_size * num_heads, q_length, head_dim]
        # torch_values = torch.bmm(attention_probs_reshaped, value_layer)
        
        ################For unitest#################
        # breakpoint()
        
        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = self._merge_heads(output)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)
        # breakpoint()
        # import pickle
        # with open("bloom_flsh.pkl", 'ab') as f:
        #     pickle.dump(output_tensor, f)
        
        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (context_layer,)

        return outputs


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_attn_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.BoolTensor:

        return attention_mask

def replace_bloom_attn_with_flash_attn():
    transformers.models.bloom.modeling_bloom.BloomModel._prepare_attn_mask = (
        _prepare_attn_mask
    )
    transformers.models.bloom.modeling_bloom.BloomAttention.forward = forward