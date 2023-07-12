:py:mod:`lmflow.utils.flash_attention.triton_flash_attention`
=============================================================

.. py:module:: lmflow.utils.flash_attention.triton_flash_attention

.. autoapi-nested-parse::

   *Experimental* implementation of FlashAttention in Triton.
   Tested with triton==2.0.0.dev20221202.
   Triton 2.0 has a new backend (MLIR) but seems like it doesn't yet work for head dimensions
   other than 64:
   https://github.com/openai/triton/blob/d376020f90002757eea3ea9475d4f7cfc2ec5ead/python/triton/ops/flash_attention.py#L207
   We'll update this implementation with the new Triton backend once this is fixed.

   We use the FlashAttention implementation from Phil Tillet a starting point.
   https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

   Changes:
   - Implement both causal and non-causal attention.
   - Implement both self-attention and cross-attention.
   - Support arbitrary seqlens (not just multiples of 128), for both forward and backward.
   - Support all head dimensions up to 128 (not just 16, 32, 64, 128), for both forward and backward.
   - Support attention bias.
   - Speed up the forward pass a bit, and only store the LSE instead of m and l.
   - Make the backward for d=128 much faster by reducing register spilling.
   - Optionally parallelize the backward pass across seqlen_k, to deal with the case of
   small batch size * nheads.

   Caution:
   - This is an *experimental* implementation. The forward pass should be quite robust but
   I'm not 100% sure that the backward pass doesn't have race conditions (due to the Triton compiler).
   - This implementation has only been tested on A100.
   - If you plan to use headdim other than 64 and 128, you should test for race conditions
   (due to the Triton compiler), as done in tests/test_flash_attn.py
   "test_flash_attn_triton_race_condition". I've tested and fixed many race conditions
   for different head dimensions (40, 48, 64, 128, 80, 88, 96), but I'm still not 100% confident
   that there are none left for other head dimensions.

   Differences between this Triton version and the CUDA version:
   - Triton version doesn't support dropout.
   - Triton forward is generally faster than CUDA forward, while Triton backward is
   generally slower than CUDA backward. Overall Triton forward + backward is slightly slower
   than CUDA forward + backward.
   - Triton version doesn't support different sequence lengths in a batch (i.e., RaggedTensor/NestedTensor).
   - Triton version supports attention bias, while CUDA version doesn't.

   ..
       !! processed by numpydoc !!


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   lmflow.utils.flash_attention.triton_flash_attention.FlashAttnQKVPackedFunc
   lmflow.utils.flash_attention.triton_flash_attention.FlashAttnKVPackedFunc
   lmflow.utils.flash_attention.triton_flash_attention.FlashAttnFunc



Functions
~~~~~~~~~

.. autoapisummary::

   lmflow.utils.flash_attention.triton_flash_attention._fwd_kernel
   lmflow.utils.flash_attention.triton_flash_attention._bwd_preprocess_do_o_dot
   lmflow.utils.flash_attention.triton_flash_attention._bwd_store_dk_dv
   lmflow.utils.flash_attention.triton_flash_attention._bwd_kernel_one_col_block
   lmflow.utils.flash_attention.triton_flash_attention.init_to_zero
   lmflow.utils.flash_attention.triton_flash_attention._bwd_kernel
   lmflow.utils.flash_attention.triton_flash_attention._flash_attn_forward
   lmflow.utils.flash_attention.triton_flash_attention._flash_attn_backward



Attributes
~~~~~~~~~~

.. autoapisummary::

   lmflow.utils.flash_attention.triton_flash_attention.flash_attn_qkvpacked_func
   lmflow.utils.flash_attention.triton_flash_attention.flash_attn_kvpacked_func
   lmflow.utils.flash_attention.triton_flash_attention.flash_attn_func


.. py:function:: _fwd_kernel(Q, K, V, Bias, Out, Lse, TMP, softmax_scale, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn, stride_bb, stride_bh, stride_bm, stride_ob, stride_oh, stride_om, nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim, CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K, BIAS_TYPE: triton.language.constexpr, IS_CAUSAL: triton.language.constexpr, BLOCK_HEADDIM: triton.language.constexpr, EVEN_M: triton.language.constexpr, EVEN_N: triton.language.constexpr, EVEN_HEADDIM: triton.language.constexpr, BLOCK_M: triton.language.constexpr, BLOCK_N: triton.language.constexpr)


.. py:function:: _bwd_preprocess_do_o_dot(Out, DO, Delta, stride_ob, stride_oh, stride_om, stride_dob, stride_doh, stride_dom, nheads, seqlen_q, seqlen_q_rounded, headdim, BLOCK_M: triton.language.constexpr, BLOCK_HEADDIM: triton.language.constexpr)


.. py:function:: _bwd_store_dk_dv(dk_ptrs, dv_ptrs, dk, dv, offs_n, offs_d, seqlen_k, headdim, EVEN_M: triton.language.constexpr, EVEN_N: triton.language.constexpr, EVEN_HEADDIM: triton.language.constexpr)


.. py:function:: _bwd_kernel_one_col_block(start_n, Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale, stride_qm, stride_kn, stride_vn, stride_bm, stride_dom, stride_dqm, stride_dkn, stride_dvn, seqlen_q, seqlen_k, headdim, ATOMIC_ADD: triton.language.constexpr, BIAS_TYPE: triton.language.constexpr, IS_CAUSAL: triton.language.constexpr, BLOCK_HEADDIM: triton.language.constexpr, EVEN_M: triton.language.constexpr, EVEN_N: triton.language.constexpr, EVEN_HEADDIM: triton.language.constexpr, BLOCK_M: triton.language.constexpr, BLOCK_N: triton.language.constexpr)


.. py:function:: init_to_zero(name)


.. py:function:: _bwd_kernel(Q, K, V, Bias, DO, DQ, DK, DV, LSE, D, softmax_scale, stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn, stride_vb, stride_vh, stride_vn, stride_bb, stride_bh, stride_bm, stride_dob, stride_doh, stride_dom, stride_dqb, stride_dqh, stride_dqm, stride_dkb, stride_dkh, stride_dkn, stride_dvb, stride_dvh, stride_dvn, nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim, CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K, BIAS_TYPE: triton.language.constexpr, IS_CAUSAL: triton.language.constexpr, BLOCK_HEADDIM: triton.language.constexpr, SEQUENCE_PARALLEL: triton.language.constexpr, EVEN_M: triton.language.constexpr, EVEN_N: triton.language.constexpr, EVEN_HEADDIM: triton.language.constexpr, BLOCK_M: triton.language.constexpr, BLOCK_N: triton.language.constexpr)


.. py:function:: _flash_attn_forward(q, k, v, bias=None, causal=False, softmax_scale=None)


.. py:function:: _flash_attn_backward(do, q, k, v, o, lse, dq, dk, dv, bias=None, causal=False, softmax_scale=None)


.. py:class:: FlashAttnQKVPackedFunc(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`

   
   Base class to create custom `autograd.Function`

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)















   ..
       !! processed by numpydoc !!
   .. py:method:: forward(ctx, qkv, bias=None, causal=False, softmax_scale=None)
      :staticmethod:

      
      qkv: (batch, seqlen, 3, nheads, headdim)
      bias: optional, shape broadcastible to (batch, nheads, seqlen, seqlen).
          For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen).
          ALiBi mask for non-causal would have shape (1, nheads, seqlen, seqlen)
















      ..
          !! processed by numpydoc !!

   .. py:method:: backward(ctx, do)
      :staticmethod:

      
      Defines a formula for differentiating the operation with backward mode
      automatic differentiation (alias to the vjp function).

      This function is to be overridden by all subclasses.

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computated w.r.t. the
      output.















      ..
          !! processed by numpydoc !!


.. py:data:: flash_attn_qkvpacked_func
   

   

.. py:class:: FlashAttnKVPackedFunc(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`

   
   Base class to create custom `autograd.Function`

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)















   ..
       !! processed by numpydoc !!
   .. py:method:: forward(ctx, q, kv, bias=None, causal=False, softmax_scale=None)
      :staticmethod:

      
      q: (batch, seqlen_q, nheads, headdim)
      kv: (batch, seqlen_k, 2, nheads, headdim)
      bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
          For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
          ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
















      ..
          !! processed by numpydoc !!

   .. py:method:: backward(ctx, do)
      :staticmethod:

      
      Defines a formula for differentiating the operation with backward mode
      automatic differentiation (alias to the vjp function).

      This function is to be overridden by all subclasses.

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computated w.r.t. the
      output.















      ..
          !! processed by numpydoc !!


.. py:data:: flash_attn_kvpacked_func
   

   

.. py:class:: FlashAttnFunc(*args, **kwargs)

   Bases: :py:obj:`torch.autograd.Function`

   
   Base class to create custom `autograd.Function`

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)















   ..
       !! processed by numpydoc !!
   .. py:method:: forward(ctx, q, k, v, bias=None, causal=False, softmax_scale=None)
      :staticmethod:

      
      q: (batch_size, seqlen_q, nheads, headdim)
      k, v: (batch_size, seqlen_k, nheads, headdim)
      bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
          For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
          ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
















      ..
          !! processed by numpydoc !!

   .. py:method:: backward(ctx, do)
      :staticmethod:

      
      Defines a formula for differentiating the operation with backward mode
      automatic differentiation (alias to the vjp function).

      This function is to be overridden by all subclasses.

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computated w.r.t. the
      output.















      ..
          !! processed by numpydoc !!


.. py:data:: flash_attn_func
   

   

