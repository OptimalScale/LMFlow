:py:mod:`lmflow.pipeline.base_tuner`
====================================

.. py:module:: lmflow.pipeline.base_tuner

.. autoapi-nested-parse::

   BaseTuner: a subclass of BasePipeline.

   ..
       !! processed by numpydoc !!


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   lmflow.pipeline.base_tuner.BaseTuner




.. py:class:: BaseTuner(*args, **kwargs)

   Bases: :py:obj:`lmflow.pipeline.base_pipeline.BasePipeline`

   
   A subclass of BasePipeline which is tunable.
















   ..
       !! processed by numpydoc !!
   .. py:method:: _check_if_tunable(model, dataset)


   .. py:method:: tune(model, dataset)
      :abstractmethod:



