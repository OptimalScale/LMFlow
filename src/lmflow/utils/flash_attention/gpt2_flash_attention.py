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


def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
    

        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        bsz, q_len, _ = hidden_states.size()

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        #TODO Should we support?
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        assert use_cache is False, "Use cache is not supported"
        present = None
        # if use_cache is True:
        #     present = (key, value)
        # else:
        #     present = None

        assert self.reorder_and_upcast_attn is False, "reorder_and_upcast_attn is not supported yet"

        qkv = torch.stack([query, key, value], dim = 2)
        qkv = qkv.transpose(1, 3)   # [bsz, seq_len, 3, heads, hiddens_per_head]
        
        # breakpoint()
        key_padding_mask = attention_mask
        # key_padding_mask = None
        # breakpoint()
        if key_padding_mask is None:
            qkv = rearrange(qkv, "b s ... -> (b s) ...")
            max_s = q_len
            cu_q_lens = torch.arange(
                0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
            )
            output = flash_attn_unpadded_qkvpacked_func(
                qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
            )
            output = rearrange(output, "(b s) ... -> b s ...", b=bsz)
        else:
            # flip in flash attention
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask = (1.0 - key_padding_mask)
            key_padding_mask = key_padding_mask.squeeze(1).squeeze(1)
            nheads = qkv.shape[-2]
            x = rearrange(qkv, "b s three h d -> b s (three h d)")
            x_unpad, indices, cu_q_lens, max_s = unpad_input(x, key_padding_mask)
            x_unpad = rearrange(
                x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
            )
            output_unpad = flash_attn_unpadded_qkvpacked_func(
                x_unpad, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
            )
            output = rearrange(
                pad_input(
                    rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len
                ),
                "b s (h d) -> b s h d",
                h=nheads,
            )
        # if self.reorder_and_upcast_attn:
        #     attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        # else:
        #     attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        output = rearrange(output, 'b s h d -> b h s d')
        attn_output = self._merge_heads(output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        
        assert output_attentions is False, "output attentions is not supported yet"
        # if output_attentions:
        #     outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask


def replace_gpt2_attn_with_flash_attn():
    # transformers.models.gpt2.modeling_gpt2.LlamaModel._prepare_decoder_attention_mask = (
    #     _prepare_decoder_attention_mask
    # )
    transformers.models.gpt2.modeling_gpt2.GPT2Attention.forward = forward