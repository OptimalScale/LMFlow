:py:mod:`lmflow.pipeline.utils.peft_trainer`
============================================

.. py:module:: lmflow.pipeline.utils.peft_trainer

.. autoapi-nested-parse::

   Trainer for Peft models

   ..
       !! processed by numpydoc !!


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   lmflow.pipeline.utils.peft_trainer.PeftTrainer
   lmflow.pipeline.utils.peft_trainer.PeftSavingCallback




.. py:class:: PeftTrainer(*args, **kwargs)

   Bases: :py:obj:`transformers.Trainer`

   
   Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.

   Args:
       model ([`PreTrainedModel`] or `torch.nn.Module`, *optional*):
           The model to train, evaluate or use for predictions. If not provided, a `model_init` must be passed.

           <Tip>

           [`Trainer`] is optimized to work with the [`PreTrainedModel`] provided by the library. You can still use
           your own models defined as `torch.nn.Module` as long as they work the same way as the 🤗 Transformers
           models.

           </Tip>

       args ([`TrainingArguments`], *optional*):
           The arguments to tweak for training. Will default to a basic instance of [`TrainingArguments`] with the
           `output_dir` set to a directory named *tmp_trainer* in the current directory if not provided.
       data_collator (`DataCollator`, *optional*):
           The function to use to form a batch from a list of elements of `train_dataset` or `eval_dataset`. Will
           default to [`default_data_collator`] if no `tokenizer` is provided, an instance of
           [`DataCollatorWithPadding`] otherwise.
       train_dataset (`torch.utils.data.Dataset` or `torch.utils.data.IterableDataset`, *optional*):
           The dataset to use for training. If it is a [`~datasets.Dataset`], columns not accepted by the
           `model.forward()` method are automatically removed.

           Note that if it's a `torch.utils.data.IterableDataset` with some randomization and you are training in a
           distributed fashion, your iterable dataset should either use a internal attribute `generator` that is a
           `torch.Generator` for the randomization that must be identical on all processes (and the Trainer will
           manually set the seed of this `generator` at each epoch) or have a `set_epoch()` method that internally
           sets the seed of the RNGs used.
       eval_dataset (Union[`torch.utils.data.Dataset`, Dict[str, `torch.utils.data.Dataset`]), *optional*):
            The dataset to use for evaluation. If it is a [`~datasets.Dataset`], columns not accepted by the
            `model.forward()` method are automatically removed. If it is a dictionary, it will evaluate on each
            dataset prepending the dictionary key to the metric name.
       tokenizer ([`PreTrainedTokenizerBase`], *optional*):
           The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs to the
           maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
           interrupted training or reuse the fine-tuned model.
       model_init (`Callable[[], PreTrainedModel]`, *optional*):
           A function that instantiates the model to be used. If provided, each call to [`~Trainer.train`] will start
           from a new instance of the model as given by this function.

           The function may have zero argument, or a single one containing the optuna/Ray Tune/SigOpt trial object, to
           be able to choose different architectures according to hyper parameters (such as layer count, sizes of
           inner layers, dropout probabilities etc).
       compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
           The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
           a dictionary string to metric values.
       callbacks (List of [`TrainerCallback`], *optional*):
           A list of callbacks to customize the training loop. Will add those to the list of default callbacks
           detailed in [here](callback).

           If you want to remove one of the default callbacks used, use the [`Trainer.remove_callback`] method.
       optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*): A tuple
           containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your model
           and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
       preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
           A function that preprocess the logits right before caching them at each evaluation step. Must take two
           tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
           by this function will be reflected in the predictions received by `compute_metrics`.

           Note that the labels (second parameter) will be `None` if the dataset does not have them.

   Important attributes:

       - **model** -- Always points to the core model. If using a transformers model, it will be a [`PreTrainedModel`]
         subclass.
       - **model_wrapped** -- Always points to the most external model in case one or more other modules wrap the
         original model. This is the model that should be used for the forward pass. For example, under `DeepSpeed`,
         the inner model is wrapped in `DeepSpeed` and then again in `torch.nn.DistributedDataParallel`. If the inner
         model hasn't been wrapped, then `self.model_wrapped` is the same as `self.model`.
       - **is_model_parallel** -- Whether or not a model has been switched to a model parallel mode (different from
         data parallelism, this means some of the model layers are split on different GPUs).
       - **place_model_on_device** -- Whether or not to automatically place the model on the device - it will be set
         to `False` if model parallel or deepspeed is used, or if the default
         `TrainingArguments.place_model_on_device` is overridden to return `False` .
       - **is_in_train** -- Whether or not a model is currently running `train` (e.g. when `evaluate` is called while
         in `train`)















   ..
       !! processed by numpydoc !!
   .. py:method:: _save_checkpoint(_, trial, metrics=None)

      
      Don't save base model, optimizer etc.
      but create checkpoint folder (needed for saving adapter) 
















      ..
          !! processed by numpydoc !!


.. py:class:: PeftSavingCallback

   Bases: :py:obj:`transformers.trainer_callback.TrainerCallback`

   
   Correctly save PEFT model and not full model 
















   ..
       !! processed by numpydoc !!
   .. py:method:: _save(model, folder)


   .. py:method:: on_train_end(args: transformers.training_args.TrainingArguments, state: transformers.trainer_callback.TrainerState, control: transformers.trainer_callback.TrainerControl, **kwargs)

      
      Save final best model adapter 
















      ..
          !! processed by numpydoc !!

   .. py:method:: on_epoch_end(args: transformers.training_args.TrainingArguments, state: transformers.trainer_callback.TrainerState, control: transformers.trainer_callback.TrainerControl, **kwargs)

      
      Save intermediate model adapters in case of interrupted training 
















      ..
          !! processed by numpydoc !!

   .. py:method:: on_save(args: transformers.training_args.TrainingArguments, state: transformers.trainer_callback.TrainerState, control: transformers.trainer_callback.TrainerControl, **kwargs)

      
      Event called after a checkpoint save.
















      ..
          !! processed by numpydoc !!


