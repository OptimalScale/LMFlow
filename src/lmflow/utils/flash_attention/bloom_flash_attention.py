from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

import transformers
from transformers.models.bloom.modeling_bloom import dropout_add

from einops import rearrange

from .triton_flash_attention import flash_attn_qkvpacked_func

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

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        reshaped_alibi = rearrange(alibi, '(b h) one s-> b h one s', h = self.num_heads)
        reshaped_alibi = reshaped_alibi * self.beta
        
        attention_mask = (1.0 - attention_mask)
        attention_mask = attention_mask[:, None, None, :].bool()
        reshaped_alibi_masked = reshaped_alibi.masked_fill(attention_mask, -1e9)

        reshaped_query_layer = query_layer
        reshaped_key_layer = key_layer
        reshaped_value_layer = value_layer

        qkv = torch.concat([reshaped_query_layer.unsqueeze(2), reshaped_key_layer.unsqueeze(2), reshaped_value_layer.unsqueeze(2)], dim = 2)

        output = flash_attn_qkvpacked_func(
                qkv, reshaped_alibi_masked, True, self.inv_norm_factor
            )

        output = rearrange(output, 'b s h d -> (b h) s d')
        
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