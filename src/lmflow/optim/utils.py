from typing import Optional, Any, Tuple

from transformers import PreTrainedModel
from transformers.utils import is_sagemaker_mp_enabled

import lmflow.optim.optimizers as optim
from lmflow.args import OptimizerNames, TrainingArguments

def create_customized_optimizer(self, base_trainer_class, model_args):
    class CustomizedOptimTrainer(base_trainer_class):

        @staticmethod
        def get_optimizer_cls_and_kwargs(
            args: TrainingArguments,
            model: Optional[PreTrainedModel] = None,
        ) -> Tuple[Any, Any]:
            # parse args.optim_args
            optim_args = {}
            if args.customized_optim_args:
                for mapping in args.customized_optim_args.replace(" ", "").split(","):
                    key, value = mapping.split("=")
                    optim_args[key] = value

            optimizer_kwargs = {"lr": args.learning_rate}

            if args.customized_optim == OptimizerNames.DUMMY:
                optimizer_cls = optim.Dummy
                dummy_kwargs = {
                    "betas": (args.optim_dummy_beta1, args.optim_dummy_beta2),
                }
                optimizer_kwargs.update(dummy_kwargs)
            elif args.customized_optim == OptimizerNames.ADABELIEF:
                optimizer_cls = optim.AdaBelief
                adabelief_kwargs = {
                    "betas": (args.optim_beta1, args.optim_beta2),
                    "weight_decay": (args.optim_weight_decay)
                }
                optimizer_kwargs.update(adabelief_kwargs)
            elif args.customized_optim == OptimizerNames.ADABOUND:
                optimizer_cls = optim.AdaBound
                adabound_kwargs = {
                    "betas": (args.optim_beta1, args.optim_beta2),
                    "weight_decay": (args.optim_weight_decay)
                }
                optimizer_kwargs.update(adabound_kwargs)
            elif args.customized_optim == OptimizerNames.LARS:
                optimizer_cls = optim.LARS
                lars_kwargs = {
                    "momentum": (args.optim_momentum),
                    "weight_decay": (args.optim_weight_decay),
                }
                optimizer_kwargs.update(lars_kwargs)
            elif args.customized_optim == OptimizerNames.LAMB:
                optimizer_cls = optim.Lamb
                lamb_kwargs = {
                    "betas": (args.optim_beta1, args.optim_beta2),
                    "weight_decay": (args.optim_weight_decay),
                }
                optimizer_kwargs.update(lamb_kwargs)
            elif args.customized_optim == OptimizerNames.ADAMAX:
                optimizer_cls = optim.Adamax
                adamax_kwargs = {
                    "betas": (args.optim_beta1, args.optim_beta2),
                    "weight_decay": (args.optim_weight_decay),
                }
                optimizer_kwargs.update(adamax_kwargs)
            elif args.customized_optim == OptimizerNames.NADAM:
                optimizer_cls = optim.NAdam
                nadam_kwargs = {
                    "betas": (args.optim_beta1, args.optim_beta2),
                    "weight_decay": (args.optim_weight_decay),
                }
                optimizer_kwargs.update(nadam_kwargs)
            elif args.customized_optim == OptimizerNames.RADAM:
                optimizer_cls = optim.RAdam
                radam_kwargs = {
                    "betas": (args.optim_beta1, args.optim_beta2),
                    "weight_decay": (args.optim_weight_decay),
                }
                optimizer_kwargs.update(radam_kwargs)
            elif args.customized_optim == OptimizerNames.ADAMP:
                optimizer_cls = optim.AdamP
                adamp_kwargs = {
                    "betas": (args.optim_beta1, args.optim_beta2),
                    "weight_decay": (args.optim_weight_decay),
                }
                optimizer_kwargs.update(adamp_kwargs)
            elif args.customized_optim == OptimizerNames.SGDP:
                optimizer_cls = optim.SGDP
                sgdp_kwargs = {
                    "momentum": (args.optim_momentum),
                    "weight_decay": (args.optim_weight_decay),
                }
                optimizer_kwargs.update(sgdp_kwargs)
            elif args.customized_optim == OptimizerNames.YOGI:
                optimizer_cls = optim.Yogi
                yogi_kwargs = {
                    "betas": (args.optim_beta1, args.optim_beta2),
                    "weight_decay": (args.optim_weight_decay),
                }
                optimizer_kwargs.update(yogi_kwargs)
            elif args.customized_optim == OptimizerNames.SOPHIA:
                optimizer_cls = optim.SophiaG
                sophia_kwargs = {
                    "betas": (args.optim_beta1, args.optim_beta2),
                    "weight_decay": (args.optim_weight_decay),
                }
                optimizer_kwargs.update(sophia_kwargs)
            elif args.customized_optim == OptimizerNames.ADAM:
                optimizer_cls = optim.Adam
                adam_kwargs = {
                    "betas": (args.optim_beta1, args.optim_beta2),
                }
                optimizer_kwargs.update(adam_kwargs)
            elif args.customized_optim == OptimizerNames.NOVOGRAD:
                optimizer_cls = optim.NovoGrad
                novograd_kwargs = {
                    "betas": (args.optim_beta1, args.optim_beta2),
                    "weight_decay": (args.optim_weight_decay),
                }
                optimizer_kwargs.update(novograd_kwargs)
            elif args.customized_optim == OptimizerNames.ADADELTA:
                optimizer_cls = optim.Adadelta
                adadelta_kwargs = {
                }
                optimizer_kwargs.update(adadelta_kwargs)
            elif args.customized_optim == OptimizerNames.ADAGRAD:
                optimizer_cls = optim.AdaGrad
                adagrad_kwargs = {
                }
                optimizer_kwargs.update(adagrad_kwargs)
            elif args.customized_optim == OptimizerNames.ADAMW_SCHEDULE_FREE:
                optimizer_cls = optim.AdamWScheduleFree
                adamw_schedule_free_kwargs = {
                    "betas": (args.optim_beta1, args.optim_beta2),
                    "weight_decay": (args.optim_weight_decay),
                }
                optimizer_kwargs.update(adamw_schedule_free_kwargs)
            elif args.customized_optim == OptimizerNames.SGD_SCHEDULE_FREE:
                optimizer_cls = optim.SGDScheduleFree
                sgd_schedule_free_kwargs = {
                    "momentum": (args.optim_momentum),
                    "weight_decay": (args.optim_weight_decay),
                }
                optimizer_kwargs.update(sgd_schedule_free_kwargs)
            elif args.customized_optim == OptimizerNames.ADAN:
                optimizer_cls = optim.Adan
                adan_kwargs = {
                    "betas": (args.optim_beta1, args.optim_beta2, args.optim_beta3),
                    "weight_decay": (args.optim_weight_decay),
                }
                optimizer_kwargs.update(adan_kwargs)
            else:
                raise ValueError(
                    f"Trainer cannot instantiate unsupported optimizer: "
                    f" {args.customized_optim}"
                )
            return optimizer_cls, optimizer_kwargs

        def create_optimizer(self):
            opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

            if self.optimizer is None:
                decay_parameters = self.get_decay_parameter_names(opt_model)
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                                if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                                if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

                optimizer_cls, optimizer_kwargs = CustomizedOptimTrainer.get_optimizer_cls_and_kwargs(self.args, opt_model)

                # Overwrite `params` in case it's created by
                # `get_optimizer_cls_and_kwargs` e.g. for GaLore optimizer.
                if "params" in optimizer_kwargs:
                    optimizer_grouped_parameters = optimizer_kwargs.pop(
                        "params"
                    )

                # For layer-wise dummy optimizers we overwrite
                # optimizer_grouped_parameters with `optimizer_dict` to
                # avoid arguments conflicts.
                if "optimizer_dict" in optimizer_kwargs:
                    optimizer_grouped_parameters = optimizer_kwargs.pop(
                        "optimizer_dict"
                    )

                self.optimizer = optimizer_cls(
                    optimizer_grouped_parameters,
                    **optimizer_kwargs
                )
            if is_sagemaker_mp_enabled():
                self.optimizer = smp.DistributedOptimizer(self.optimizer)
                
    return CustomizedOptimTrainer