from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
class Adamax(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 2e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        foreach: Optional[bool] = None,
        *,
        maximize: bool = False,
        differentiable: bool = False,
        capturable: bool = False,
    ):
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

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
            capturable=capturable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            group.setdefault("capturable", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val, dtype=_get_scalar_dtype(), device=p.device
                        )
                        if group["capturable"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _init_group(
        self, group, params_with_grad, grads, exp_avgs, exp_infs, state_steps
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("Adamax does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state["step"] = (
                    torch.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                    if group["capturable"]
                    else torch.tensor(0.0, dtype=_get_scalar_dtype())
                )
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                state["exp_inf"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

            exp_avgs.append(state["exp_avg"])
            exp_infs.append(state["exp_inf"])
            state_steps.append(state["step"])

        return has_complex

    def step(self, closure=None):
        """Performs a single optimization step.

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
            exp_infs: List[Tensor] = []
            state_steps: List[Tensor] = []

            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            foreach = group["foreach"]
            maximize = group["maximize"]
            differentiable = group["differentiable"]
            capturable = group["capturable"]

            has_complex = self._init_group(
                group, params_with_grad, grads, exp_avgs, exp_infs, state_steps
            )

            adamax(
                params_with_grad,
                grads,
                exp_avgs,
                exp_infs,
                state_steps,
                eps=eps,
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                weight_decay=weight_decay,
                foreach=foreach,
                maximize=maximize,
                differentiable=differentiable,
                capturable=capturable,
                has_complex=has_complex,
            )

        return loss
