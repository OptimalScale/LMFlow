:py:mod:`lmflow.pipeline.base_aligner`
======================================

.. py:module:: lmflow.pipeline.base_aligner

.. autoapi-nested-parse::

   BaseTuner: a subclass of BasePipeline.

   ..
       !! processed by numpydoc !!


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   lmflow.pipeline.base_aligner.BaseAligner




.. py:class:: BaseAligner(*args, **kwargs)

   Bases: :py:obj:`lmflow.pipeline.base_pipeline.BasePipeline`

   
   A subclass of BasePipeline which is alignable.
















   ..
       !! processed by numpydoc !!
   .. py:method:: _check_if_alignable(model, dataset, reward_model)


   .. py:method:: align(model, dataset, reward_model)
      :abstractmethod:



