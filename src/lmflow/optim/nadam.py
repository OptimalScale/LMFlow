from typing import cast, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
class NAdam(Optimizer): 
    def __init__(
        self,
        params,
        lr: float = 2e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum_decay: float = 4e-3,
        decoupled_weight_decay: bool = False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ):  # noqa: D107
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= momentum_decay:
            raise ValueError(f"Invalid momentum_decay value: {momentum_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum_decay=momentum_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):  # noqa: D105
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("decoupled_weight_decay", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0:
                    if not torch.is_tensor(p_state["step"]):
                        step_val = float(p_state["step"])
                        p_state["step"] = (
                            torch.tensor(
                                step_val, dtype=_get_scalar_dtype(), device=p.device
                            )
                            if group["capturable"]
                            else torch.tensor(step_val, dtype=_get_scalar_dtype())
                        )
                    if not torch.is_tensor(p_state["mu_product"]):
                        mu_prod_val = p_state["mu_product"]
                        p_state["mu_product"] = (
                            torch.tensor(
                                mu_prod_val, dtype=_get_scalar_dtype(), device=p.device
                            )
                            if group["capturable"]
                            else torch.tensor(mu_prod_val, dtype=_get_scalar_dtype())
                        )

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        mu_products,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("NAdam does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` and `mu_product` on CPU if capturable is False.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state["step"] = (
                        torch.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                        if group["capturable"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype())
                    )
                    state["mu_product"] = (
                        torch.ones((), dtype=_get_scalar_dtype(), device=p.device)
                        if group["capturable"]
                        else torch.tensor(1.0, dtype=_get_scalar_dtype())
                    )
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                mu_products.append(state["mu_product"])
                state_steps.append(state["step"])
        return has_complex

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            mu_products: List[Tensor] = []
            state_steps: List[Tensor] = []
            beta1, beta2 = cast(Tuple[float, float], group["betas"])

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                mu_products,
                state_steps,
            )

            nadam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                mu_products,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                momentum_decay=group["momentum_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                decoupled_weight_decay=group["decoupled_weight_decay"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                has_complex=has_complex,
            )

        return loss

