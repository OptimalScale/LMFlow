:py:mod:`lmflow.utils.flash_attention.llama_flash_attention`
============================================================

.. py:module:: lmflow.utils.flash_attention.llama_flash_attention


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   lmflow.utils.flash_attention.llama_flash_attention.forward
   lmflow.utils.flash_attention.llama_flash_attention._prepare_decoder_attention_mask
   lmflow.utils.flash_attention.llama_flash_attention.replace_llama_attn_with_flash_attn



.. py:function:: forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor]] = None, output_attentions: bool = False, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]

   
   Input shape: Batch x Time x Channel

   attention_mask: [bsz, q_len]















   ..
       !! processed by numpydoc !!

.. py:function:: _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length)


.. py:function:: replace_llama_attn_with_flash_attn()


