import math
from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
from torch.nn.utils import clip_grad_norm_
import logging
import abc
import sys

logger = logging.getLogger(__name__)
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta("ABC", (), {})


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, and correct_bias, as saved in
                # the constructor).
                #
                # 1- Update first and second moments of the gradients.
                # 2- Apply bias correction.
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given as the pseudo-code in the project description).
                # 3- Update parameters (p.data).
                # 4- After that main gradient-based update, update again using weight decay
                #    (incorporating the learning rate again).

                device = "cuda" if torch.cuda.is_available() else "cpu"

                if "t" not in state.keys():
                    state["t"] = 0
                    state["mt"] = torch.zeros(p.data.shape).to(device)
                    state["vt"] = torch.zeros(p.data.shape).to(device)
                beta1, beta2 = group["betas"]

                state["t"] = state["t"] + 1
                state["mt"] = state["mt"].mul(beta1) + grad.mul(1 - beta1)
                state["vt"] = state["vt"].mul(beta2) + torch.mul(grad, grad).mul(
                    1 - beta2
                )

                state["alpha"] = (group["lr"] * math.sqrt(1 - beta2 ** state["t"])) / (
                    1 - beta1 ** state["t"]
                )

                p.data = p.data.sub(
                    state["mt"]
                    .div(torch.sqrt(state["vt"]).add_(group["eps"]))
                    .mul(state["alpha"])
                )
                p.data = p.data.sub(p.data.mul(group["lr"]).mul(group["weight_decay"]))

        return loss


class BertAdam(Optimizer):
    """Implements BERT version of Adam algorithm with weight decay fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate of 1. (no warmup regardless of warmup setting). Default: -1
        schedule: schedule to use for the warmup (see above).
            Can be `'warmup_linear'`, `'warmup_constant'`, `'warmup_cosine'`, `'none'`, `None` or a `_LRSchedule` object (see below).
            If `None` or `'none'`, learning rate is always kept constant.
            Default : `'warmup_linear'`
        b1: Adams b1. Default: 0.9
        b2: Adams b2. Default: 0.999
        e: Adams epsilon. Default: 1e-6
        weight_decay: Weight decay. Default: 0.01
        max_grad_norm: Maximum norm for the gradients (-1 means no clipping). Default: 1.0
    """

    def __init__(
        self,
        params,
        lr=required,
        warmup=-1,
        t_total=-1,
        schedule="warmup_linear",
        b1=0.9,
        b2=0.999,
        e=1e-6,
        weight_decay=0.01,
        max_grad_norm=1.0,
        **kwargs
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not isinstance(schedule, _LRSchedule) and schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))
        if not 0.0 <= b1 < 1.0:
            raise ValueError(
                "Invalid b1 parameter: {} - should be in [0.0, 1.0[".format(b1)
            )
        if not 0.0 <= b2 < 1.0:
            raise ValueError(
                "Invalid b2 parameter: {} - should be in [0.0, 1.0[".format(b2)
            )
        if not e >= 0.0:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(e))
        # initialize schedule object
        if not isinstance(schedule, _LRSchedule):
            schedule_type = SCHEDULES[schedule]
            schedule = schedule_type(warmup=warmup, t_total=t_total)
        else:
            if warmup != -1 or t_total != -1:
                logger.warning(
                    "warmup and t_total on the optimizer are ineffective when _LRSchedule object is provided as schedule. "
                    "Please specify custom warmup and t_total in _LRSchedule object."
                )
        defaults = dict(
            lr=lr,
            schedule=schedule,
            b1=b1,
            b2=b2,
            e=e,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        super(BertAdam, self).__init__(params, defaults)

    def get_lr(self):
        lr = []
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    return [0]
                lr_scheduled = group["lr"]
                lr_scheduled *= group["schedule"].get_lr(state["step"])
                lr.append(lr_scheduled)
        return lr

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["next_m"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["next_v"] = torch.zeros_like(p.data)

                next_m, next_v = state["next_m"], state["next_v"]
                beta1, beta2 = group["b1"], group["b2"]

                # Add grad clipping
                if group["max_grad_norm"] > 0:
                    clip_grad_norm_(p, group["max_grad_norm"])

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                next_m.mul_(beta1).add_(1 - beta1, grad)
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                update = next_m / (next_v.sqrt() + group["e"])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                if group["weight_decay"] > 0.0:
                    update += group["weight_decay"] * p.data

                lr_scheduled = group["lr"]
                lr_scheduled *= group["schedule"].get_lr(state["step"])

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state["step"] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss


class SophiaG(Optimizer):
    """
    Sophia: Second-order Clipped Stochastic Optimization.
    Using Sophia with the Gauss-Newton-Bartlett estimate of the Hessian.state["hessian"]

    https://arxiv.org/pdf/2305.14342.pdf
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.1,
        eps: float = 1e-15,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= rho:
            raise ValueError("Invalid rho value: {} - should be >= 0.0".format(rho))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {} - should be >= 0.0".format(weight_decay)
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            eps=eps,
        )
        super(SophiaG, self).__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self, bs: int):
        for group in self.param_groups:
            _, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                # B · ^g ⊙ ^g
                # Update the hessian estimate (moving average)
                state["hessian"].mul_(beta2).addcmul_(
                    p.grad, p.grad, value=bs - bs * beta2
                )

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                grad = p.grad

                if grad is None:
                    continue

                if grad.is_sparse:
                    raise RuntimeError("Sophia does not support sparse gradients")

                # State should be stored in this dictionary
                state = self.state[p]

                # Init state variables
                if len(state) == 0:
                    state["step"] = torch.zeros(
                        (1,), dtype=torch.float, device=p.device
                    )
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian"] = torch.zeros_like(p)

                # Access hyperparameters from the `group` dictionary
                beta1, _ = group["betas"]
                rho = group["rho"]
                lr = group["lr"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                exp_avg = state["exp_avg"]
                hess = state["hessian"]

                # Calculation of new weights
                state["step"] += 1

                # 1 - Perform stepweight decay
                p.data.mul_(1 - lr * weight_decay)

                # 2 - Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 3 - Decay the hessian running average coefficient
                # Clipping the hessian.
                ratio = (exp_avg / (rho * hess + eps)).clamp(-1, 1)
                p.data.add_(ratio, alpha=-lr)

        return loss


class SophiaH(Optimizer):
    """
    Sophia: Second-order Clipped Stochastic Optimization.
    Using Sophia with the Hutchinson estimate of the Hessian.state["hessian"]

    https://arxiv.org/pdf/2305.14342.pdf
    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.96, 0.99),
        rho: float = 1e-2,
        weight_decay: float = 0.0,
        eps: float = 1e-12,
        update_period: int = 10,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= rho:
            raise ValueError("Invalid rho value: {} - should be >= 0.0".format(rho))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {} - should be >= 0.0".format(weight_decay)
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        if not 0 < update_period:
            raise ValueError(
                "Invalid update_period value: {} - should be > 0".format(update_period)
            )

        self.update_period = update_period

        defaults = dict(
            lr=lr,
            betas=betas,
            rho=rho,
            weight_decay=weight_decay,
            eps=eps,
            update_period=update_period,
        )
        super(SophiaH, self).__init__(params, defaults)

    def _update_hessian(self):
        for group in self.param_groups:
            _, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                # draw u from N(0, I)
                u = torch.randn_like(p.grad)

                # Compute < grad, u >
                # Differentiate < grad, u > w.r.t. p
                hvp = torch.autograd.grad(p.grad, p, grad_outputs=u, retain_graph=True)[
                    0
                ]

                # u ⊙ hvp
                state["hessian"].mul_(beta2).addcmul_(u, hvp, value=1 - beta2)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        step = self.param_groups[0].get("step", 1)
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if step % self.update_period == 0:
            self._update_hessian()

        for group in self.param_groups:
            if "step" in group:
                group["step"] += 1
            else:
                group["step"] = 1

            for p in group["params"]:
                grad = p.grad

                if grad is None:
                    continue

                if grad.is_sparse:
                    raise RuntimeError("Sophia does not support sparse gradients")

                # State should be stored in this dictionary
                state = self.state[p]

                # Init state variables
                if len(state) == 0:
                    state["step"] = torch.zeros(
                        (1,), dtype=torch.float, device=p.device
                    )
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian"] = torch.zeros_like(p)

                # Access hyperparameters from the `group` dictionary
                beta1, _ = group["betas"]
                rho = group["rho"]
                lr = group["lr"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                exp_avg = state["exp_avg"]
                hess = state["hessian"]

                # Calculation of new weights
                state["step"] += 1

                # 1 - Perform stepweight decay
                p.data.mul_(1 - lr * weight_decay)

                # 2 - Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 3 - Decay the hessian running average coefficient
                # Clipping the hessian.
                ratio = (exp_avg / torch.clip(hess, min=eps)).clamp(-rho, rho)
                p.data.add_(ratio, alpha=-lr)

        return loss
